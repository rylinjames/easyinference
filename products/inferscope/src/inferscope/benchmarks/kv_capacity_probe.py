"""KV cache capacity probing — binary search for max concurrent requests at each ISL."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from inferscope.benchmarks.catalog import materialize_workload
from inferscope.benchmarks.models import WorkloadPack, WorkloadRequest, ChatMessage
from inferscope.benchmarks.openai_replay import run_openai_replay
from inferscope.benchmarks.procedural import ProceduralWorkloadOptions
from inferscope.optimization.memory_planner import plan_memory, MemoryPlan
from inferscope.models.registry import ModelVariant
from inferscope.hardware.gpu_profiles import GPUProfile

logger = logging.getLogger(__name__)

# Map phase-runner model names to seed packs that materialize_procedural_workload
# knows how to expand. The Kimi production lane uses the dedicated seed; every
# other coding model falls back to coding-long-context.
_SEED_PACK_FOR_KIMI = "kimi-k2-long-context-coding"
_SEED_PACK_FOR_CODING = "coding-long-context"


def _seed_pack_name_for_model(model_name: str) -> str:
    """Pick a procedural seed pack name based on the target model."""
    name_lower = model_name.lower()
    if "kimi" in name_lower:
        return _SEED_PACK_FOR_KIMI
    return _SEED_PACK_FOR_CODING


def _build_capacity_probe_pack(model_name: str, isl: int, concurrency: int) -> WorkloadPack:
    """Build a capacity-probe workload pack with prompts shaped to the labeled ISL.

    Closes the snapshot v1.0.0 P0 bug `kv_phase_runner_synthetic_prompt_size`.
    Earlier drafts of this helper produced ~22-token prompts regardless of
    `isl`, so the capacity-probe report was unfaithful by orders of magnitude.

    The fix routes through `materialize_workload` (which calls
    `materialize_procedural_workload` under the hood) so the prompts are
    shaped to ~`isl` tokens via `_shape_context`. The materialized pack
    inherits the seed's tool catalog and request structure; we override
    `model` and `name` to match the phase runner's caller-facing identity
    so downstream artifact metadata still reflects which model the probe
    targeted.
    """
    seed_pack_name = _seed_pack_name_for_model(model_name)
    options = ProceduralWorkloadOptions(
        request_count=concurrency,
        input_tokens=isl,
        output_tokens=64,
        seed=isl,  # deterministic per-ISL reproducibility
    )
    materialized = materialize_workload(seed_pack_name, options=options)

    # Override identity fields so the artifact metadata reflects the
    # phase-runner caller (not the upstream seed pack).
    materialized = materialized.model_copy(
        update={
            "name": f"kv-capacity-probe-{isl}",
            "description": "Live KV capacity probe",
            "workload_class": "kv_capacity_probe",
            "model": model_name,
            "concurrency": concurrency,
            "stream": True,
        }
    )
    # Tag every request with the phase-runner metadata that downstream
    # consumers (audit, comparison, observability) key off of.
    for idx, request in enumerate(materialized.requests):
        request.metadata = {
            **(request.metadata or {}),
            "phase": "kv_capacity_probe",
            "isl": isl,
            "approx_context_tokens": isl,
            "probe_concurrency": concurrency,
            "probe_request_index": idx,
        }
    return materialized


@dataclass
class CapacityPoint:
    """Single point on the capacity curve."""
    isl: int
    max_concurrent: int
    degradation_type: str  # "none" | "reliability" | "kv_saturation" | "queue_saturation" | "transport"
    kv_memory_gb: float
    kv_usage_at_max: float
    ttft_baseline_ms: float | None = None
    ttft_at_max_ms: float | None = None
    error_rate_at_max: float = 0.0
    confidence_kind: str = "derived"


@dataclass
class KVCapacityProbeResult:
    """Result of a KV capacity probe across multiple ISLs."""
    model_name: str
    gpu_name: str
    tp: int
    precision: str
    kv_precision: str
    capacity_curve: list[CapacityPoint] = field(default_factory=list)
    memory_plan: dict[str, Any] = field(default_factory=dict)
    support_tier: str = ""
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "gpu_name": self.gpu_name,
            "tp": self.tp,
            "precision": self.precision,
            "kv_precision": self.kv_precision,
            "capacity_curve": [
                {
                    "isl": p.isl,
                    "max_concurrent": p.max_concurrent,
                    "degradation_type": p.degradation_type,
                    "kv_memory_gb": round(p.kv_memory_gb, 2),
                    "kv_usage_at_max": round(p.kv_usage_at_max, 3),
                    "ttft_baseline_ms": p.ttft_baseline_ms,
                    "ttft_at_max_ms": p.ttft_at_max_ms,
                    "error_rate_at_max": round(p.error_rate_at_max, 4),
                    "confidence_kind": p.confidence_kind,
                }
                for p in self.capacity_curve
            ],
            "memory_plan": self.memory_plan,
            "support_tier": self.support_tier,
            "warnings": self.warnings,
            "summary": self.summary,
        }


def estimate_capacity_curve(
    model: ModelVariant,
    gpu: GPUProfile,
    *,
    tp: int = 1,
    precision: str = "fp8",
    kv_precision: str = "bf16",
    isl_list: list[int] | None = None,
) -> KVCapacityProbeResult:
    """Estimate KV capacity curve from memory math (no live endpoint needed).

    This is the planning-mode capacity estimator. For live probing,
    use run_kv_capacity_probe() which sends actual requests.
    """
    if isl_list is None:
        isl_list = [1024, 4096, 8192, 32768, 65536, 131072]

    mem = plan_memory(model, gpu, num_gpus=tp, tp=tp, precision=precision, kv_precision=kv_precision)
    result = KVCapacityProbeResult(
        model_name=model.name,
        gpu_name=gpu.name,
        tp=tp,
        precision=precision,
        kv_precision=kv_precision,
        memory_plan=mem.to_dict(),
        support_tier=model.serving.get("support_tier", "unknown"),
    )

    if not mem.fits:
        result.warnings.append("Model does not fit in GPU memory at this configuration.")
        result.summary = "Model does not fit — no capacity curve generated."
        return result

    # Use per-GPU budget and per-GPU token cost to avoid TP double-counting.
    # mem.kv_cache_budget_gb is total across TP shards; kv_cache_per_token_bytes is un-sharded.
    per_gpu_budget_gb = mem.kv_cache_budget_gb / max(tp, 1)
    kv_budget_bytes = per_gpu_budget_gb * (1024**3)
    kv_per_token = mem.kv_cache_per_token_bytes / max(tp, 1)
    deltanet_state = model.serving.get("deltanet_state_bytes_per_seq_bf16", 0)
    # DeltaNet state is also sharded across TP
    deltanet_state_per_gpu = deltanet_state / max(tp, 1)

    for isl in sorted(isl_list):
        if kv_per_token <= 0:
            break
        per_seq_kv = kv_per_token * isl
        per_seq_total = per_seq_kv + deltanet_state_per_gpu
        if per_seq_total <= 0:
            break
        max_conc = int(kv_budget_bytes / per_seq_total)
        if max_conc < 1:
            max_conc = 0
            degradation = "kv_saturation"
        else:
            degradation = "none"

        kv_gb = (per_seq_total * max(max_conc, 1)) / (1024**3)
        usage = kv_gb / mem.kv_cache_budget_gb if mem.kv_cache_budget_gb > 0 else 1.0

        result.capacity_curve.append(CapacityPoint(
            isl=isl,
            max_concurrent=max_conc,
            degradation_type=degradation,
            kv_memory_gb=kv_gb,
            kv_usage_at_max=min(usage, 1.0),
            confidence_kind="derived" if mem.estimation_mode == "exact" else "heuristic",
        ))

    if result.capacity_curve:
        points = result.capacity_curve
        result.summary = (
            f"Capacity curve: {points[0].max_concurrent} sessions at {points[0].isl // 1024}K → "
            f"{points[-1].max_concurrent} sessions at {points[-1].isl // 1024}K context. "
            f"KV budget: {per_gpu_budget_gb:.1f} GB/GPU, {kv_per_token:.0f} bytes/token/GPU."
            + (f" DeltaNet state: {deltanet_state_per_gpu / (1024**2):.1f} MB/seq/GPU." if deltanet_state else "")
        )

    if model.serving.get("warnings"):
        result.warnings.extend(model.serving["warnings"])
    if mem.estimation_mode != "exact":
        result.warnings.append(f"KV estimation mode: {mem.estimation_mode}")

    return result


async def run_kv_capacity_probe(
    endpoint: str,
    model_name: str,
    *,
    metrics_endpoint: str | None = None,
    tp: int = 1,
    precision: str = "fp8",
    kv_precision: str = "bf16",
    gpu_name: str = "",
    isl_list: list[int] | None = None,
    concurrency_candidates: list[int] | None = None,
    ttft_slo_ms: float = 5000.0,
    capture_metrics: bool = True,
    client: Any | None = None,
) -> KVCapacityProbeResult:
    """Run a live KV capacity probe against an endpoint."""

    if isl_list is None:
        isl_list = [1024, 4096, 8192, 32768]
    if concurrency_candidates is None:
        concurrency_candidates = [1, 2, 4, 8, 16]

    result = KVCapacityProbeResult(
        model_name=model_name,
        gpu_name=gpu_name,
        tp=tp,
        precision=precision,
        kv_precision=kv_precision,
        support_tier="live_probe",
    )

    for isl in sorted(isl_list):
        best_point = CapacityPoint(
            isl=isl,
            max_concurrent=0,
            degradation_type="reliability",
            kv_memory_gb=0.0,
            kv_usage_at_max=0.0,
            confidence_kind="direct",
        )

        for concurrency in sorted(set(concurrency_candidates)):
            artifact = await run_openai_replay(
                _build_capacity_probe_pack(model_name, isl, concurrency),
                endpoint,
                model=model_name,
                metrics_endpoint=metrics_endpoint,
                capture_metrics=capture_metrics,
                concurrency=concurrency,
                client=client,
            )
            summary = artifact.summary
            metrics = (artifact.metrics_after.normalized_metrics if artifact.metrics_after else {}) or {}
            cache = metrics.get("cache", {})
            succeeded = summary.failed == 0 and (
                summary.ttft_p99_ms is None or summary.ttft_p99_ms <= ttft_slo_ms
            )
            if not succeeded:
                best_point.degradation_type = "latency" if summary.failed == 0 else "reliability"
                break

            best_point = CapacityPoint(
                isl=isl,
                max_concurrent=concurrency,
                degradation_type="none",
                kv_memory_gb=0.0,
                kv_usage_at_max=float(cache.get("kv_usage", 0.0) or 0.0),
                ttft_baseline_ms=summary.ttft_avg_ms,
                ttft_at_max_ms=summary.ttft_p99_ms or summary.ttft_avg_ms,
                error_rate_at_max=(summary.failed / summary.total_requests) if summary.total_requests else 0.0,
                confidence_kind="direct",
            )

        result.capacity_curve.append(best_point)

    if result.capacity_curve:
        first = result.capacity_curve[0]
        last = result.capacity_curve[-1]
        result.summary = (
            f"Live capacity probe from {first.isl} to {last.isl} tokens. "
            f"Max concurrency ranged from {first.max_concurrent} to {last.max_concurrent}."
        )

    return result
