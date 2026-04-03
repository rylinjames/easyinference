"""Pre-flight validation — checks if a serving config will work before deployment.

Validates TP divisibility, memory fit, format compatibility, known bugs, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inferscope.hardware.gpu_profiles import GPUProfile
from inferscope.models.registry import ModelVariant
from inferscope.optimization.memory_planner import plan_memory
from inferscope.optimization.platform_policy import (
    EngineSupportTier,
    resolve_engine_support,
    resolve_preferred_tp,
)
from inferscope.optimization.serving_profile import WorkloadMode
from inferscope.production_target import (
    is_target_engine,
    is_target_gpu,
    is_target_model,
    target_profile_summary,
)


@dataclass
class ValidationResult:
    """Result of pre-flight validation."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


def validate_config(
    model: ModelVariant,
    gpu: GPUProfile,
    tp: int = 1,
    quantization: str = "auto",
    engine: str = "dynamo",
) -> ValidationResult:
    """Validate a serving configuration before deployment.

    Checks:
    - TP divides num_attention_heads and num_kv_heads
    - Model fits in GPU memory
    - Quantization format is supported by GPU
    - Known engine/model/GPU incompatibilities
    """
    result = ValidationResult()

    if not is_target_model(model):
        result.valid = False
        result.errors.append("Supported models are limited to Kimi-K2.5.")
    if not is_target_gpu(gpu):
        result.valid = False
        result.errors.append("Supported GPUs are limited to H100, H200, B200, and B300 variants.")
    if not is_target_engine(engine):
        result.valid = False
        result.errors.append("InferScope's production lane only supports the Dynamo engine.")
    result.info.append(target_profile_summary())

    tp_min = model.serving.get("tp_min")
    if isinstance(tp_min, int) and tp < tp_min:
        result.valid = False
        result.errors.append(f"TP={tp} is below the model minimum tp_min={tp_min}.")

    # --- TP divisibility ---
    if model.kv_heads > 0 and tp > 1 and model.kv_heads % tp != 0:
        result.valid = False
        result.errors.append(
            f"TP={tp} does not evenly divide num_kv_heads={model.kv_heads}. "
            f"Valid TP values: {[i for i in range(1, model.kv_heads + 1) if model.kv_heads % i == 0]}"
        )

    # --- Quantization compatibility ---
    if quantization == "auto":
        quantization = "fp4" if model.name == "Kimi-K2.5" and gpu.fp4_support else "fp8" if gpu.fp8_support else "bf16"
    normalized_precision = "fp4" if quantization in ("nvfp4", "mxfp4") else quantization

    if normalized_precision in ("fp8", "fp8_e4m3") and not gpu.fp8_support:
        result.valid = False
        result.errors.append(
            f"FP8 quantization requires Hopper or Blackwell native FP8 support. {gpu.name} does not support it."
        )

    if quantization in ("nvfp4", "fp4", "mxfp4") and not gpu.fp4_support:
        result.valid = False
        result.errors.append(
            f"FP4 quantization requires Blackwell (NVFP4) or CDNA4 (MXFP4). "
            f"{gpu.name} ({gpu.architecture}) does not support FP4."
        )

    if quantization == "nvfp4" and gpu.fp4_format == "MXFP4":
        result.valid = False
        result.errors.append(
            f"NVFP4 is NVIDIA Blackwell only. {gpu.name} supports MXFP4, not NVFP4. Use MXFP4 quantization instead."
        )

    if quantization == "mxfp4" and gpu.fp4_format == "NVFP4":
        result.valid = False
        result.errors.append(
            f"MXFP4 is AMD CDNA4 native. {gpu.name} supports NVFP4, not MXFP4. Use NVFP4 quantization instead."
        )

    # --- Memory fit ---
    precision = normalized_precision if normalized_precision not in ("auto",) else "fp16"
    mem = plan_memory(model=model, gpu=gpu, num_gpus=tp, tp=tp, precision=precision)
    if not mem.fits:
        result.valid = False
        result.errors.append(
            f"Model does not fit: weights need {mem.weight_gb:.1f} GB/GPU, "
            f"but only {mem.usable_memory_gb / tp:.1f} GB usable per GPU "
            f"(after {gpu.memory_gb} GB × 0.92 utilization / TP={tp})."
        )
    else:
        result.info.append(
            f"Memory: {mem.weight_gb:.1f} GB weights/GPU, "
            f"{mem.kv_cache_budget_gb:.1f} GB KV cache budget, "
            f"~{mem.max_concurrent_sequences} max concurrent sequences"
        )
        if mem.platform_overflow_tier != "gpu_only":
            result.info.append(
                f"Platform overflow advisory: {mem.platform_overflow_tier} (+{mem.overflow_memory_gb:.0f} GB)."
            )

    # --- Engine-specific checks ---
    support = resolve_engine_support(engine, gpu, multi_node=tp > 1)
    if support.tier == EngineSupportTier.UNSUPPORTED:
        result.valid = False
        result.errors.append(support.reason)
    elif support.tier != EngineSupportTier.RECOMMENDED:
        result.warnings.append(support.reason)

    if model.name == "Kimi-K2.5" and gpu.architecture == "Hopper" and tp < 4:
        result.warnings.append(
            "Kimi-K2.5 on Hopper is reliability-sensitive below TP=4 because "
            "KV headroom disappears quickly under long contexts."
        )
    result.info.append("LMCache with sticky session routing is assumed for the supported production deployment.")

    preferred_tp, preferred_reason = resolve_preferred_tp(
        model,
        gpu,
        num_gpus=max(tp, 1),
        precision=normalized_precision,
        workload=WorkloadMode.CODING,
    )
    if preferred_tp is not None and preferred_tp != tp and preferred_reason:
        result.warnings.append(f"Platform/model hint prefers TP={preferred_tp}. {preferred_reason}")

    return result
