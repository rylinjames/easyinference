"""Memory planning — KV cache + weight memory math.

Calculates exact memory breakdown for a model on a given GPU configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferscope.hardware.gpu_profiles import GPUProfile
from inferscope.models.registry import ModelVariant
from inferscope.optimization.platform_policy import resolve_platform_traits


@dataclass
class MemoryPlan:
    """Detailed memory breakdown for a serving deployment."""

    weight_gb: float = 0.0
    kv_cache_per_token_bytes: float = 0.0
    kv_cache_budget_gb: float = 0.0
    activation_overhead_gb: float = 2.0  # Conservative estimate
    total_gpu_memory_gb: float = 0.0
    usable_memory_gb: float = 0.0
    max_tokens_in_cache: int = 0
    max_concurrent_sequences: int = 0
    max_context_length: int = 0
    platform_overflow_tier: str = "gpu_only"
    overflow_memory_gb: float = 0.0
    overflow_bandwidth_gb_s: float = 0.0
    fits: bool = False
    notes: list[str] = None  # type: ignore[assignment]
    estimation_mode: str = "exact"  # "exact" | "heuristic"
    assumptions: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []
        if self.assumptions is None:
            self.assumptions = []

    def to_dict(self) -> dict:
        return {
            "weight_gb": round(self.weight_gb, 2),
            "kv_cache_per_token_bytes": round(self.kv_cache_per_token_bytes, 1),
            "kv_cache_budget_gb": round(self.kv_cache_budget_gb, 2),
            "activation_overhead_gb": round(self.activation_overhead_gb, 2),
            "total_gpu_memory_gb": round(self.total_gpu_memory_gb, 2),
            "usable_memory_gb": round(self.usable_memory_gb, 2),
            "max_tokens_in_cache": self.max_tokens_in_cache,
            "max_concurrent_sequences": self.max_concurrent_sequences,
            "max_context_length": self.max_context_length,
            "platform_overflow_tier": self.platform_overflow_tier,
            "overflow_memory_gb": round(self.overflow_memory_gb, 2),
            "overflow_bandwidth_gb_s": round(self.overflow_bandwidth_gb_s, 2),
            "fits": self.fits,
            "notes": self.notes,
            "estimation_mode": self.estimation_mode,
            "assumptions": self.assumptions,
        }


def plan_memory(
    model: ModelVariant,
    gpu: GPUProfile,
    num_gpus: int = 1,
    tp: int = 1,
    precision: str = "fp16",
    kv_precision: str = "auto",
    gpu_memory_utilization: float = 0.92,
    max_context: int = 0,
) -> MemoryPlan:
    """Calculate memory plan for serving a model on a GPU configuration.

    Args:
        model: Model variant with architecture details
        gpu: GPU hardware profile
        num_gpus: Total GPUs available
        tp: Tensor parallelism degree
        precision: Weight precision (fp16, fp8, fp4, int8, int4)
        kv_precision: KV cache precision (auto resolves to fp16 or fp8)
        gpu_memory_utilization: Fraction of GPU memory to use
        max_context: Max context length (0 = model default)
    """
    plan = MemoryPlan()
    traits = resolve_platform_traits(gpu)

    # Resolve KV precision
    if kv_precision == "auto":
        kv_precision = "fp8_e4m3" if gpu.fp8_support else "fp16"

    # Total GPU memory available (across TP shards on one node)
    plan.total_gpu_memory_gb = gpu.memory_gb * min(tp, num_gpus)
    plan.usable_memory_gb = plan.total_gpu_memory_gb * gpu_memory_utilization

    # Weight memory (divided by TP)
    total_weight_gb = model.weight_gb(precision)
    plan.weight_gb = total_weight_gb / tp

    # Activation overhead (rough estimate: 1-3 GB depending on model size)
    plan.activation_overhead_gb = min(3.0, max(1.0, model.params_total_b / 100))

    # KV cache budget = usable - weights - activations
    per_gpu_usable = gpu.memory_gb * gpu_memory_utilization
    per_gpu_weight = total_weight_gb / tp
    per_gpu_kv_budget = per_gpu_usable - per_gpu_weight - plan.activation_overhead_gb
    plan.kv_cache_budget_gb = max(0, per_gpu_kv_budget * tp)  # Total across TP shards

    # KV cache per token (all layers)
    kv_per_token_per_layer = model.kv_cache_bytes_per_token(kv_precision)

    # Hybrid attention: only some layers have KV cache
    kv_layers = model.serving.get("kv_layers", model.layers)
    if isinstance(kv_layers, int) and kv_layers < model.layers:
        plan.notes.append(
            f"Hybrid attention: {kv_layers}/{model.layers} layers have standard KV cache. "
            f"Remaining {model.layers - kv_layers} layers use fixed recurrent state."
        )

    plan.kv_cache_per_token_bytes = kv_per_token_per_layer * kv_layers

    # Max tokens in cache
    if plan.kv_cache_per_token_bytes > 0 and plan.kv_cache_budget_gb > 0:
        # KV cache is sharded across TP, so per-GPU budget × bytes per token / tp
        kv_budget_bytes = per_gpu_kv_budget * (1024**3)
        per_gpu_kv_per_token = plan.kv_cache_per_token_bytes / tp
        plan.max_tokens_in_cache = int(kv_budget_bytes / per_gpu_kv_per_token)
    else:
        plan.max_tokens_in_cache = 0

    # Max context length
    context = max_context if max_context > 0 else model.context_length
    plan.max_context_length = min(context, plan.max_tokens_in_cache) if plan.max_tokens_in_cache > 0 else 0

    # Max concurrent sequences (at average context length)
    avg_context = min(4096, context)  # Assume average 4K context for concurrency calc
    if avg_context > 0 and plan.max_tokens_in_cache > 0:
        plan.max_concurrent_sequences = plan.max_tokens_in_cache // avg_context

    # Note DeltaNet fixed state overhead (capacity probe handles per-ISL math)
    deltanet_state = model.serving.get("deltanet_state_bytes_per_seq_bf16", 0)
    if deltanet_state:
        plan.notes.append(
            f"DeltaNet recurrent state: {deltanet_state / (1024**2):.1f} MB/sequence (fixed, "
            "independent of sequence length). Capacity probes account for this per-sequence overhead."
        )

    # Does it fit?
    plan.fits = per_gpu_kv_budget > 0
    plan.platform_overflow_tier = traits.overflow_tier
    if traits.is_grace:
        plan.overflow_memory_gb = traits.grace_memory_gb
        plan.overflow_bandwidth_gb_s = traits.grace_memory_bandwidth_gb_s
        plan.notes.append(
            "Coherent Grace overflow is available for KV spill/staging, but HBM fit is still evaluated independently."
        )

    # Notes
    if not plan.fits:
        plan.notes.append(
            f"Model weights ({per_gpu_weight:.1f} GB/GPU) exceed usable memory "
            f"({per_gpu_usable:.1f} GB/GPU). Increase TP or use more aggressive quantization."
        )

    if gpu.fp8_support and precision in ("fp16", "bf16") and model.params_total_b > 13:
        plan.notes.append(
            f"Consider FP8 quantization — would reduce weights from "
            f"{total_weight_gb:.0f} GB to {model.weight_gb('fp8'):.0f} GB"
        )

    if gpu.fp4_support and precision not in ("fp4", "mxfp4", "int4"):
        plan.notes.append(
            f"GPU supports {'NVFP4' if gpu.fp4_format == 'NVFP4' else 'MXFP4'} — "
            f"would reduce weights to {model.weight_gb('fp4'):.0f} GB (eval quality first)"
        )

    if model.attention_type == "MLA":
        plan.notes.append(
            f"MLA attention compresses KV cache ~{model.serving.get('compression_ratio', 32)}x — "
            f"much larger effective context than standard GQA"
        )
    if not plan.fits and traits.is_grace:
        plan.notes.append(
            "Grace coherent memory can extend session capacity, "
            "but model weights still need an HBM-resident serving plan."
        )

    # Set estimation mode from model metadata
    kv_estimation_mode = model.serving.get("kv_estimation_mode", "exact")
    if kv_estimation_mode in ("heuristic", "hybrid_exact"):
        plan.estimation_mode = kv_estimation_mode
        plan.assumptions.append(
            f"KV cache math for {model.name} is {kv_estimation_mode} — "
            + ("exact for attention layers, fixed state for DeltaNet layers."
               if kv_estimation_mode == "hybrid_exact"
               else "actual usage may differ due to non-standard attention.")
        )
    if model.attention_type == "hybrid":
        plan.assumptions.append(
            "Model uses hybrid attention; some layers have compressed state instead of full KV. "
            "Calculated KV budget is an upper-bound estimate."
        )

    return plan
