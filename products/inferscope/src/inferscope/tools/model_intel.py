"""Model intelligence tools — profiles, validation, capacity estimation."""

from __future__ import annotations

from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.models.registry import get_model_variant
from inferscope.optimization.memory_planner import plan_memory
from inferscope.production_target import (
    PRODUCTION_TARGET_NAME,
    is_target_gpu,
    is_target_model,
    supported_gpu_aliases,
    supported_model_names,
    target_profile_summary,
)
from inferscope.optimization.validator import validate_config
from inferscope.security import InputValidationError, validate_model_name


def get_model_profile(model: str) -> dict:
    """Return serving profile for a model: architecture, memory, model class, serving hints."""
    try:
        model = validate_model_name(model)
    except InputValidationError as e:
        return {"error": str(e), "confidence": 0.0}

    variant = get_model_variant(model)
    if variant is None or not is_target_model(variant):
        available = supported_model_names()
        return {
            "error": f"Unsupported model: '{model}'",
            "available_models": available,
            "summary": target_profile_summary(),
            "confidence": 0.0,
            "evidence": "lookup_failure",
        }

    return {
        "model": variant.to_dict(),
        "summary": (
            f"{variant.name}: {variant.params_total_b}B params "
            f"({'active ' + str(variant.params_active_b) + 'B, ' if variant.model_type == 'moe' else ''}"
            f"{variant.model_type}, {variant.attention_type}, "
            f"{variant.context_length // 1024}K context, "
            f"class={variant.model_class.value})"
        ),
        "confidence": 1.0,
        "evidence": "model_knowledge_base",
    }


def validate_serving_config(
    model: str,
    gpu: str,
    tp: int = 1,
    quantization: str = "auto",
    engine: str = "dynamo",
) -> dict:
    """Pre-flight check: does this config work?

    Checks TP divisibility, memory fit, format compatibility, known bugs.
    """
    variant = get_model_variant(model)
    if variant is None or not is_target_model(variant):
        return {
            "error": f"Unsupported model: '{model}'",
            "available_models": supported_model_names(),
            "confidence": 0.0,
        }

    gpu_profile = get_gpu_profile(gpu)
    if gpu_profile is None or not is_target_gpu(gpu_profile):
        return {
            "error": f"Unsupported GPU: '{gpu}'",
            "available_gpus": supported_gpu_aliases(),
            "confidence": 0.0,
        }

    result = validate_config(
        model=variant,
        gpu=gpu_profile,
        tp=tp,
        quantization=quantization,
        engine=engine,
    )

    return {
        "validation": result.to_dict(),
        "model": variant.name,
        "gpu": gpu_profile.name,
        "tp": tp,
        "quantization": quantization,
        "engine": engine,
        "summary": (
            f"{'✅ Valid' if result.valid else '❌ Invalid'}: "
            f"{variant.name} on {gpu_profile.name} TP={tp} {quantization} ({engine})"
            + (f" — {len(result.errors)} error(s)" if result.errors else "")
            + (f", {len(result.warnings)} warning(s)" if result.warnings else "")
        ),
        "confidence": 0.95 if result.valid else 0.9,
        "evidence": "pre_flight_validation",
    }


def estimate_capacity(
    model: str,
    gpu: str,
    num_gpus: int = 1,
    quantization: str = "auto",
    max_context: int = 0,
) -> dict:
    """Calculate max concurrent users, max context length, and KV cache budget.

    Uses exact per-layer KV cache formulas from model profile.
    """
    variant = get_model_variant(model)
    if variant is None or not is_target_model(variant):
        return {
            "error": f"Unsupported model: '{model}'",
            "available_models": supported_model_names(),
            "confidence": 0.0,
        }

    gpu_profile = get_gpu_profile(gpu)
    if gpu_profile is None or not is_target_gpu(gpu_profile):
        return {
            "error": f"Unsupported GPU: '{gpu}'",
            "available_gpus": supported_gpu_aliases(),
            "confidence": 0.0,
        }

    # Resolve quantization
    if quantization == "auto":
        if gpu_profile.fp8_support:
            quantization = "fp8"
        elif variant.params_total_b > 30:
            quantization = "awq"
        else:
            quantization = "bf16"

    # KV precision
    kv_prec = "fp8_e4m3" if gpu_profile.fp8_support else "fp16"

    # Calculate with TP = num_gpus (simple case)
    tp = num_gpus
    mem = plan_memory(
        model=variant,
        gpu=gpu_profile,
        num_gpus=num_gpus,
        tp=tp,
        precision=quantization,
        kv_precision=kv_prec,
        max_context=max_context,
    )

    return {
        "capacity": mem.to_dict(),
        "model": variant.name,
        "gpu": gpu_profile.name,
        "num_gpus": num_gpus,
        "quantization": quantization,
        "target_profile": PRODUCTION_TARGET_NAME,
        "summary": (
            f"{variant.name} on {num_gpus}× {gpu_profile.name} ({quantization}): "
            f"{'✅ fits' if mem.fits else '❌ does not fit'}, "
            f"{mem.weight_gb:.1f} GB weights/GPU, "
            f"{mem.kv_cache_budget_gb:.1f} GB KV budget, "
            f"~{mem.max_concurrent_sequences} max concurrent @ 4K avg context"
            + (
                f", overflow tier={mem.platform_overflow_tier} (+{mem.overflow_memory_gb:.0f} GB advisory)"
                if mem.platform_overflow_tier != "gpu_only"
                else ""
            )
        ),
        "confidence": 0.85,
        "evidence": "memory_model_calculation",
    }
