"""Recommendation tools — production-target config generation and planning."""

from __future__ import annotations

from typing import Any

from inferscope.hardware.gpu_profiles import GPUProfile, get_gpu_profile
from inferscope.models.registry import ModelVariant, get_model_variant
from inferscope.optimization.platform_policy import (
    resolve_engine_support,
    resolve_preferred_precision,
    resolve_preferred_tp,
)
from inferscope.optimization.recommender import recommend
from inferscope.optimization.serving_profile import WorkloadMode
from inferscope.production_target import (
    PRODUCTION_TARGET_NAME,
    is_target_gpu,
    is_target_model,
    normalize_target_workload_class,
    supported_gpu_aliases,
    supported_model_names,
    target_profile_summary,
)
from inferscope.security import (
    InputValidationError,
    validate_gpu_name,
    validate_model_name,
    validate_positive_int,
)

TargetPair = tuple[ModelVariant, GPUProfile]


def _safe_lookup(
    model: str,
    gpu: str,
) -> tuple[TargetPair | None, dict[str, Any] | None]:
    """Validate and look up model + GPU. Returns (variant, gpu_profile) or (None, error_dict)."""
    try:
        model = validate_model_name(model)
    except InputValidationError as e:
        return None, {"error": str(e), "confidence": 0.0}
    try:
        gpu = validate_gpu_name(gpu)
    except InputValidationError as e:
        return None, {"error": str(e), "confidence": 0.0}

    variant = get_model_variant(model)
    if variant is None or not is_target_model(variant):
        return (
            None,
            {
                "error": f"Unsupported model: '{model}'",
                "available_models": supported_model_names(),
                "summary": target_profile_summary(),
                "confidence": 0.0,
            },
        )

    gpu_profile = get_gpu_profile(gpu)
    if gpu_profile is None or not is_target_gpu(gpu_profile):
        return (
            None,
            {
                "error": f"Unsupported GPU: '{gpu}'",
                "available_gpus": supported_gpu_aliases(),
                "summary": target_profile_summary(),
                "confidence": 0.0,
            },
        )

    return (variant, gpu_profile), None


def recommend_config(
    model: str,
    gpu: str,
    workload: str = "coding",
    num_gpus: int = 1,
    engine: str = "dynamo",
    *,
    has_rdma: bool = False,
    rdma_type: str = "",
    node_count: int = 1,
) -> dict:
    """Generate the supported Dynamo serving config for the production target.

    Cluster fabric parameters (`has_rdma`, `rdma_type`, `node_count`) flow
    through to the recommender DAG and into the engine compiler. With
    `has_rdma=True`, Dynamo split-topology configures
    `DYNAMO_KV_TRANSPORT="rdma"` and the vLLM NixlConnector branch
    becomes reachable. Closes the snapshot v1.0.0 P0 bug
    `recommender_inventory_missing_rdma`.
    """
    resolved, error = _safe_lookup(model, gpu)
    if error is not None:
        return error
    if resolved is None:
        return {"error": "Unable to resolve model and GPU.", "confidence": 0.0}
    variant, gpu_profile = resolved

    normalized_workload = normalize_target_workload_class(workload)
    if normalized_workload not in {"coding", "chat"}:
        return {
            "error": "Supported workloads are coding and chat.",
            "summary": target_profile_summary(),
            "confidence": 0.0,
        }

    try:
        num_gpus = validate_positive_int(num_gpus, "num_gpus", max_value=1024)
    except InputValidationError as e:
        return {"error": str(e), "confidence": 0.0}

    try:
        workload_mode = WorkloadMode.CHAT if normalized_workload == "chat" else WorkloadMode.CODING
        profile, engine_config, mem_plan = recommend(
            model=variant,
            gpu=gpu_profile,
            num_gpus=num_gpus,
            workload=workload_mode,
            engine=engine,
            has_rdma=has_rdma,
            rdma_type=rdma_type,
            node_count=node_count,
        )
    except ValueError as e:
        return {"error": str(e), "summary": target_profile_summary(), "confidence": 0.0}

    return {
        "serving_profile": profile.to_dict(),
        "engine_config": engine_config.to_dict(),
        "memory_plan": mem_plan.to_dict(),
        "target_profile": PRODUCTION_TARGET_NAME,
        "summary": (
            f"Recommended Dynamo config: {variant.name} on {num_gpus}× {gpu_profile.name} | "
            f"TP={profile.topology.tp} DP={profile.topology.dp} | "
            f"Precision={profile.precision.weights} | LMCache={profile.cache.lmcache_mode} | "
            f"{'fits' if mem_plan.fits else 'does not fit'}"
        ),
        "launch_command": engine_config.command,
        "confidence": 0.9 if mem_plan.fits else 0.5,
        "evidence": "production_target_recommendation",
    }


def recommend_engine(
    model: str,
    gpu: str,
    workload: str = "coding",
    num_gpus: int = 1,
    multi_node: bool = False,
) -> dict:
    """Return the only supported engine lane with rationale."""
    resolved, error = _safe_lookup(model, gpu)
    if error is not None:
        return error
    if resolved is None:
        return {"error": "Unable to resolve model and GPU.", "confidence": 0.0}
    variant, gpu_profile = resolved

    normalized_workload = normalize_target_workload_class(workload)
    if normalized_workload != "coding":
        return {"error": "Supported workload is coding.", "confidence": 0.0}

    support = resolve_engine_support("dynamo", gpu_profile, multi_node=multi_node or num_gpus > 1)
    ranking = {
        "engine": "dynamo",
        "rank": 0,
        "rationale": support.reason,
        "best_for": ("Long-context coding on Kimi-K2.5 with LMCache and observability-first MCP benchmarking"),
        "support_tier": support.tier.value,
        "support_reason": support.reason,
    }
    return {
        "rankings": [ranking],
        "model": variant.name,
        "gpu": gpu_profile.name,
        "workload": "coding",
        "selected_engine": "dynamo",
        "summary": f"Top pick: dynamo — {support.reason}",
        "target_profile": PRODUCTION_TARGET_NAME,
        "confidence": 0.95,
        "evidence": "production_engine_policy",
    }


def suggest_parallelism(model: str, gpu: str, num_gpus: int) -> dict:
    """Recommend the supported TP/DP layout for the production target."""
    resolved, error = _safe_lookup(model, gpu)
    if error is not None:
        return error
    if resolved is None:
        return {"error": "Unable to resolve model and GPU.", "confidence": 0.0}
    variant, gpu_profile = resolved

    try:
        num_gpus = validate_positive_int(num_gpus, "num_gpus", max_value=1024)
    except InputValidationError as e:
        return {"error": str(e), "confidence": 0.0}

    precision, precision_reason = resolve_preferred_precision(
        variant,
        gpu_profile,
        WorkloadMode.CODING,
        num_gpus,
    )
    tp, tp_reason = resolve_preferred_tp(
        variant,
        gpu_profile,
        num_gpus,
        precision.weights,
        WorkloadMode.CODING,
    )
    if tp is None:
        return {"error": tp_reason or "Unable to derive TP.", "confidence": 0.0}

    dp = max(1, num_gpus // tp)
    return {
        "suggestions": [
            {
                "precision": precision.weights,
                "tp": tp,
                "pp": 1,
                "dp": dp,
                "ep": 1,
                "notes": [precision_reason, tp_reason or ""],
            }
        ],
        "model": variant.name,
        "gpu": gpu_profile.name,
        "num_gpus": num_gpus,
        "target_profile": PRODUCTION_TARGET_NAME,
        "summary": (
            f"{variant.name} on {num_gpus}× {gpu_profile.name}: "
            f"{precision.weights.upper()} with TP={tp}, DP={dp}, EP=1 "
            "for Dynamo + LMCache coding."
        ),
        "confidence": 0.9,
        "evidence": "target_parallelism_policy",
    }
