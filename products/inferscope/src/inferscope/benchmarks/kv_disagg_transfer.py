"""KV disaggregated transfer profiling — NIXL transfer overhead measurement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferscope.benchmarks.catalog import materialize_workload
from inferscope.benchmarks.models import ChatMessage, WorkloadPack, WorkloadRequest
from inferscope.benchmarks.openai_replay import run_openai_replay
from inferscope.benchmarks.procedural import ProceduralWorkloadOptions


def _seed_pack_name_for_model(model_name: str) -> str:
    """Pick a procedural seed pack name based on the target model."""
    return "kimi-k2-long-context-coding" if "kimi" in model_name.lower() else "coding-long-context"


@dataclass
class TransferPoint:
    """Transfer metrics at a specific sequence length."""
    isl: int
    transfer_latency_ms: float | None = None
    transfer_bandwidth_gbps: float | None = None
    decode_idle_fraction: float | None = None
    nixl_failures: float = 0.0
    kvbm_offload_events: float = 0.0
    kvbm_onboard_events: float = 0.0
    confidence_kind: str = "direct"


@dataclass
class KVDisaggTransferResult:
    """Result of disaggregated KV transfer profiling."""
    model_name: str
    topology: str
    transfer_curve: list[TransferPoint] = field(default_factory=list)
    avg_decode_idle_fraction: float | None = None
    transport_type: str = ""  # "nixl_rdma" | "nixl_nvlink" | "p2p_nccl" | "lmcache"
    support_tier: str = ""
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "topology": self.topology,
            "transfer_curve": [
                {
                    "isl": t.isl,
                    "transfer_latency_ms": t.transfer_latency_ms,
                    "transfer_bandwidth_gbps": round(t.transfer_bandwidth_gbps, 2) if t.transfer_bandwidth_gbps else None,
                    "decode_idle_fraction": round(t.decode_idle_fraction, 3) if t.decode_idle_fraction else None,
                    "nixl_failures": t.nixl_failures,
                    "confidence_kind": t.confidence_kind,
                }
                for t in self.transfer_curve
            ],
            "avg_decode_idle_fraction": round(self.avg_decode_idle_fraction, 3) if self.avg_decode_idle_fraction else None,
            "transport_type": self.transport_type,
            "support_tier": self.support_tier,
            "warnings": self.warnings,
            "summary": self.summary,
        }


def _build_disagg_transfer_pack(model_name: str, isl: int) -> WorkloadPack:
    """Build a disagg-transfer workload pack with a single prompt shaped to the
    labeled ISL.

    Closes the snapshot v1.0.0 P0 bug `kv_phase_runner_synthetic_prompt_size`.
    Earlier drafts produced ~9-token prompts regardless of `isl`.
    """
    options = ProceduralWorkloadOptions(
        request_count=1,
        input_tokens=isl,
        output_tokens=64,
        seed=isl,
    )
    materialized = materialize_workload(_seed_pack_name_for_model(model_name), options=options)
    materialized = materialized.model_copy(
        update={
            "name": f"kv-disagg-transfer-{isl}",
            "description": "Live KV disaggregated transfer probe",
            "workload_class": "kv_disagg_transfer",
            "model": model_name,
            "concurrency": 1,
            "stream": True,
        }
    )
    for request in materialized.requests:
        request.metadata = {
            **(request.metadata or {}),
            "phase": "kv_disagg_transfer",
            "isl": isl,
            "approx_context_tokens": isl,
        }
    return materialized


async def run_kv_disagg_transfer(
    endpoint: str,
    model_name: str,
    *,
    topology: str,
    isl_list: list[int] | None = None,
    metrics_endpoint: str | None = None,
    capture_metrics: bool = True,
    client: Any | None = None,
) -> KVDisaggTransferResult:
    """Run a live disaggregated KV transfer probe."""

    if isl_list is None:
        isl_list = [4096, 8192, 32768]

    result = KVDisaggTransferResult(
        model_name=model_name,
        topology=topology,
        support_tier="live_probe",
    )

    idle_fractions: list[float] = []
    for isl in isl_list:
        artifact = await run_openai_replay(
            _build_disagg_transfer_pack(model_name, isl),
            endpoint,
            model=model_name,
            metrics_endpoint=metrics_endpoint,
            capture_metrics=capture_metrics,
            client=client,
        )
        metrics = (artifact.metrics_after.normalized_metrics if artifact.metrics_after else {}) or {}
        disagg = metrics.get("disaggregation", {})
        idle_fraction = float(disagg.get("decode_idle_fraction", 0.0) or 0.0)
        idle_fractions.append(idle_fraction)
        result.transfer_curve.append(
            TransferPoint(
                isl=isl,
                transfer_latency_ms=disagg.get("nixl_transfer_latency_ms"),
                transfer_bandwidth_gbps=float(disagg.get("transfer_bandwidth_gbps", 0.0) or 0.0),
                decode_idle_fraction=idle_fraction,
                nixl_failures=float(disagg.get("nixl_transfer_failures", 0.0) or 0.0),
                kvbm_offload_events=float(disagg.get("kvbm_offload_d2h", 0.0) or 0.0),
                kvbm_onboard_events=float(disagg.get("kvbm_onboard_h2d", 0.0) or 0.0),
                confidence_kind="direct",
            )
        )

    if idle_fractions:
        result.avg_decode_idle_fraction = sum(idle_fractions) / len(idle_fractions)
    result.transport_type = "nixl_rdma"
    result.summary = f"Live disaggregated transfer probe captured {len(result.transfer_curve)} sequence lengths."
    return result
