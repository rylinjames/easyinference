"""KV cache capacity probing — binary search for max concurrent requests at each ISL."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from inferscope.benchmarks.models import WorkloadPack, WorkloadRequest, ChatMessage
from inferscope.optimization.memory_planner import plan_memory, MemoryPlan
from inferscope.models.registry import ModelVariant
from inferscope.hardware.gpu_profiles import GPUProfile

logger = logging.getLogger(__name__)


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
