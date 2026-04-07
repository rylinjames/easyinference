"""KV cache pressure ramp — degradation profiling at increasing load levels."""

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
class PressurePoint:
    """Metrics at a single pressure level."""
    pressure_pct: float
    concurrency: int
    ttft_p50_ms: float | None = None
    ttft_p99_ms: float | None = None
    tpot_p50_ms: float | None = None
    tpot_p99_ms: float | None = None
    error_rate: float = 0.0
    preemption_count: float = 0.0
    kv_cache_usage: float = 0.0
    queue_depth_avg: float = 0.0
    throughput_tps: float = 0.0
    confidence_kind: str = "direct"


@dataclass
class KVPressureProfileResult:
    """Result of a pressure ramp across multiple load levels."""
    model_name: str
    isl: int
    base_capacity: int
    pressure_points: list[PressurePoint] = field(default_factory=list)
    cliff_pressure_pct: float | None = None
    cliff_type: str = ""  # "preemption" | "latency" | "error" | "queue"
    support_tier: str = ""
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "isl": self.isl,
            "base_capacity": self.base_capacity,
            "pressure_points": [
                {
                    "pressure_pct": p.pressure_pct,
                    "concurrency": p.concurrency,
                    "ttft_p50_ms": p.ttft_p50_ms,
                    "ttft_p99_ms": p.ttft_p99_ms,
                    "error_rate": round(p.error_rate, 4),
                    "preemption_count": p.preemption_count,
                    "kv_cache_usage": round(p.kv_cache_usage, 3),
                    "queue_depth_avg": round(p.queue_depth_avg, 1),
                    "throughput_tps": round(p.throughput_tps, 1),
                    "confidence_kind": p.confidence_kind,
                }
                for p in self.pressure_points
            ],
            "cliff_pressure_pct": self.cliff_pressure_pct,
            "cliff_type": self.cliff_type,
            "support_tier": self.support_tier,
            "warnings": self.warnings,
            "summary": self.summary,
        }


def _build_pressure_ramp_pack(model_name: str, isl: int, concurrency: int) -> WorkloadPack:
    """Build a pressure-ramp workload pack with prompts shaped to the labeled ISL.

    Closes the snapshot v1.0.0 P0 bug `kv_phase_runner_synthetic_prompt_size`.
    Earlier drafts produced ~6-token prompts regardless of `isl`. The fix
    routes through `materialize_workload` to shape the prompts to ~`isl` tokens.
    """
    options = ProceduralWorkloadOptions(
        request_count=concurrency,
        input_tokens=isl,
        output_tokens=64,
        seed=isl,
    )
    materialized = materialize_workload(_seed_pack_name_for_model(model_name), options=options)
    materialized = materialized.model_copy(
        update={
            "name": f"kv-pressure-ramp-{isl}-{concurrency}",
            "description": "Live KV pressure ramp",
            "workload_class": "kv_pressure_ramp",
            "model": model_name,
            "concurrency": concurrency,
            "stream": True,
        }
    )
    for idx, request in enumerate(materialized.requests):
        request.metadata = {
            **(request.metadata or {}),
            "phase": "kv_pressure_ramp",
            "isl": isl,
            "approx_context_tokens": isl,
            "probe_concurrency": concurrency,
            "probe_request_index": idx,
        }
    return materialized


async def run_kv_pressure_ramp(
    endpoint: str,
    model_name: str,
    *,
    isl: int,
    base_capacity: int,
    pressure_levels: list[float] | None = None,
    metrics_endpoint: str | None = None,
    capture_metrics: bool = True,
    client: Any | None = None,
) -> KVPressureProfileResult:
    """Run a live KV pressure ramp across increasing load levels."""

    if pressure_levels is None:
        pressure_levels = [0.5, 0.75, 0.9, 1.0, 1.1]

    result = KVPressureProfileResult(
        model_name=model_name,
        isl=isl,
        base_capacity=base_capacity,
        support_tier="live_probe",
    )

    for level in pressure_levels:
        concurrency = max(1, round(base_capacity * level))
        artifact = await run_openai_replay(
            _build_pressure_ramp_pack(model_name, isl, concurrency),
            endpoint,
            model=model_name,
            metrics_endpoint=metrics_endpoint,
            capture_metrics=capture_metrics,
            concurrency=concurrency,
            client=client,
        )
        metrics = (artifact.metrics_after.normalized_metrics if artifact.metrics_after else {}) or {}
        cache = metrics.get("cache", {})
        latency = metrics.get("latency", {})
        reliability = metrics.get("reliability", {})
        queue = metrics.get("request_state", {})

        point = PressurePoint(
            pressure_pct=level,
            concurrency=concurrency,
            ttft_p50_ms=artifact.summary.ttft_avg_ms,
            ttft_p99_ms=artifact.summary.ttft_p99_ms,
            tpot_p50_ms=latency.get("itl_avg_ms"),
            tpot_p99_ms=latency.get("itl_avg_ms"),
            error_rate=(artifact.summary.failed / artifact.summary.total_requests) if artifact.summary.total_requests else 0.0,
            preemption_count=float(reliability.get("kv_active_blocks", 0.0) or 0.0),
            kv_cache_usage=float(cache.get("kv_usage", 0.0) or 0.0),
            queue_depth_avg=float(queue.get("waiting", 0.0) or 0.0),
            throughput_tps=float(metrics.get("throughput", {}).get("gen_throughput_tps", 0.0) or 0.0),
            confidence_kind="direct",
        )
        result.pressure_points.append(point)

        if result.cliff_pressure_pct is None and (point.error_rate > 0.0 or (point.ttft_p99_ms or 0.0) > 5000.0):
            result.cliff_pressure_pct = level
            result.cliff_type = "error" if point.error_rate > 0.0 else "latency"

    if result.pressure_points:
        result.summary = f"Pressure ramp captured {len(result.pressure_points)} load levels at ISL {isl}."
    return result
