"""Production benchmark and MCP contract for the narrowed InferScope surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping

from inferscope.hardware.gpu_profiles import GPUProfile, get_gpu_profile
from inferscope.models.registry import ModelVariant, get_model_variant

if TYPE_CHECKING:
    from inferscope.benchmarks.models import BenchmarkArtifact

PRODUCTION_TARGET_NAME = "dynamo_long_context_coding"
SUPPORTED_MODEL = "Kimi-K2.5"
SUPPORTED_ENGINE = "dynamo"
SUPPORTED_BENCHMARK_ENGINES = ("vllm", "dynamo")
SUPPORTED_WORKLOAD_MODE = "coding"
SUPPORTED_WORKLOAD_MODES = ("coding", "chat")
SUPPORTED_WORKLOAD_PACK = "kimi-k2-long-context-coding"
SUPPORTED_TOPOLOGY_MODES = ("single_endpoint", "prefill_decode_split")
SUPPORTED_CACHE_STRATEGIES = ("lmcache",)
SUPPORTED_GPU_CANONICAL = ("h100", "h200", "b200", "b300")
SUPPORTED_GPU_NAMES = {
    "H100 SXM",
    "H100 NVL",
    "H100 PCIe",
    "H200 SXM",
    "H200 NVL",
    "B200",
    "B300",
}
NON_GOALS = [
    "Generic benchmark matrix or workload catalog discovery.",
    "Benchmark strategy planners and stack-plan materialization as public product surfaces.",
    "A broad multi-engine benchmark toolbox beyond the Dynamo production lane and vLLM comparison lane.",
    "A generic GPU/model reference MCP surface that competes with the diagnostics workflow.",
]

# =============================================================================
# Support tier system
# =============================================================================

SupportTier = Literal["production_validated", "benchmark_supported", "planning_preview"]
BenchmarkPhase = Literal[
    "benchmark_replay",
    "kv_capacity_probe",
    "kv_pressure_profile",
    "kv_cache_behavior",
    "kv_disagg_transfer",
]


@dataclass(frozen=True)
class ModelSupportContract:
    """Defines what a model is allowed to do within InferScope."""

    model_name: str
    tier: SupportTier
    allowed_workloads: tuple[str, ...]
    allowed_experiments: tuple[str, ...]
    allowed_phases: tuple[str, ...]  # BenchmarkPhase values
    recommendation_scope: str  # "full_operator" | "benchmark_only" | "planning_only"
    kv_estimation_mode: str  # "exact" | "heuristic"
    warnings: tuple[str, ...] = ()


_ALL_LIVE_PHASES: tuple[str, ...] = (
    "benchmark_replay",
    "kv_capacity_probe",
    "kv_pressure_profile",
    "kv_cache_behavior",
    "kv_disagg_transfer",
)

_PLANNING_ONLY_PHASES: tuple[str, ...] = ("kv_math",)

# --- Tier-specific constants ---
PRODUCTION_VALIDATED_MODELS = ("Kimi-K2.5",)
BENCHMARK_SUPPORTED_MODELS = ("Qwen3-Coder-480B-A35B-Instruct", "Qwen3-Coder-30B-A3B-Instruct", "Qwen3-Coder-Next")
PLANNING_PREVIEW_MODELS: tuple[str, ...] = ()

PRODUCTION_WORKLOAD_PACKS = ("kimi-k2-long-context-coding",)
BENCHMARK_WORKLOAD_PACKS = ("qwen3-coder-kv-stress",)

PRODUCTION_EXPERIMENTS = (
    "dynamo-aggregated-lmcache-kimi-k2",
    "vllm-disagg-prefill-lmcache",
    "dynamo-disagg-lmcache-kimi-k2",
)
BENCHMARK_EXPERIMENTS = (
    "dynamo-aggregated-lmcache-qwen3-coder-480b",
    "dynamo-disagg-lmcache-qwen3-coder-480b",
    "vllm-disagg-prefill-nixl-qwen3-coder-30b",
)

# --- Model support registry ---
_MODEL_CONTRACTS: dict[str, ModelSupportContract] = {
    "Kimi-K2.5": ModelSupportContract(
        model_name="Kimi-K2.5",
        tier="production_validated",
        allowed_workloads=PRODUCTION_WORKLOAD_PACKS,
        allowed_experiments=PRODUCTION_EXPERIMENTS,
        allowed_phases=_ALL_LIVE_PHASES,
        recommendation_scope="full_operator",
        kv_estimation_mode="exact",
    ),
    "Qwen3-Coder-480B-A35B-Instruct": ModelSupportContract(
        model_name="Qwen3-Coder-480B-A35B-Instruct",
        tier="benchmark_supported",
        allowed_workloads=BENCHMARK_WORKLOAD_PACKS + PRODUCTION_WORKLOAD_PACKS,
        allowed_experiments=BENCHMARK_EXPERIMENTS,
        allowed_phases=_ALL_LIVE_PHASES,
        recommendation_scope="benchmark_only",
        kv_estimation_mode="exact",
    ),
    "Qwen3-Coder-30B-A3B-Instruct": ModelSupportContract(
        model_name="Qwen3-Coder-30B-A3B-Instruct",
        tier="benchmark_supported",
        allowed_workloads=BENCHMARK_WORKLOAD_PACKS + PRODUCTION_WORKLOAD_PACKS,
        allowed_experiments=BENCHMARK_EXPERIMENTS,
        allowed_phases=_ALL_LIVE_PHASES,
        recommendation_scope="benchmark_only",
        kv_estimation_mode="exact",
    ),
    "Qwen3-Coder-Next": ModelSupportContract(
        model_name="Qwen3-Coder-Next",
        tier="benchmark_supported",
        allowed_workloads=BENCHMARK_WORKLOAD_PACKS + PRODUCTION_WORKLOAD_PACKS,
        allowed_experiments=BENCHMARK_EXPERIMENTS,
        allowed_phases=_ALL_LIVE_PHASES,
        recommendation_scope="benchmark_only",
        kv_estimation_mode="hybrid_exact",
        warnings=(
            "Hybrid attention: 12/48 layers use standard KV cache, 36 use Gated DeltaNet.",
            "FP8 KV cache not supported — BF16 KV required.",
        ),
    ),
}

SUPPORTED_MODELS = PRODUCTION_VALIDATED_MODELS + BENCHMARK_SUPPORTED_MODELS + PLANNING_PREVIEW_MODELS
SUPPORTED_WORKLOAD_PACKS = PRODUCTION_WORKLOAD_PACKS + BENCHMARK_WORKLOAD_PACKS
SUPPORTED_EXPERIMENTS = PRODUCTION_EXPERIMENTS + BENCHMARK_EXPERIMENTS
DEFAULT_EXPERIMENT = SUPPORTED_EXPERIMENTS[0]

RELIABILITY_MIN_SUCCESS_RATE = 0.99
RELIABILITY_REQUIRE_METRICS_CAPTURE_COMPLETE = True
RELIABILITY_MAX_FAILED_SESSIONS = 0
RELIABILITY_WARNING_QUEUE_DEPTH = 10
RELIABILITY_WARNING_MIGRATIONS = 1
RELIABILITY_WARNING_KV_USAGE = 0.9

FRONTEND_METRICS = [
    "dynamo_frontend_inflight_requests",
    "dynamo_frontend_queued_requests",
    "dynamo_frontend_time_to_first_token_seconds",
    "dynamo_frontend_inter_token_latency_seconds",
    "dynamo_frontend_request_duration_seconds",
    "dynamo_frontend_disconnected_clients",
    "dynamo_frontend_model_migration_total",
]
BACKEND_METRICS = [
    "dynamo_component_inflight_requests",
    "dynamo_component_request_duration_seconds",
    "dynamo_component_requests_total",
]
KV_METRICS = [
    "dynamo_component_kvstats_gpu_cache_usage_percent",
    "dynamo_component_kvstats_gpu_prefix_cache_hit_rate",
    "dynamo_component_kvstats_active_blocks",
    "dynamo_component_kvstats_total_blocks",
]
TRACE_ENV_VARS = [
    "DYN_LOGGING_JSONL",
    "OTEL_EXPORT_ENABLED",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_SERVICE_NAME",
]
REQUEST_HEADERS = ["x-request-id", "x-session-id"]


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _compact(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def supported_model_names(tier: str | None = None) -> list[str]:
    """Return supported model names, optionally filtered by tier."""
    if tier is None:
        return list(SUPPORTED_MODELS)
    return [name for name, c in _MODEL_CONTRACTS.items() if c.tier == tier]


def supported_gpu_aliases() -> list[str]:
    return list(SUPPORTED_GPU_CANONICAL)


def supported_benchmark_engines() -> list[str]:
    return list(SUPPORTED_BENCHMARK_ENGINES)


def supported_probe_workload_packs(tier: str | None = None) -> list[str]:
    if tier is None:
        return list(SUPPORTED_WORKLOAD_PACKS)
    return [
        wp
        for c in _MODEL_CONTRACTS.values() if c.tier == tier
        for wp in c.allowed_workloads
    ]


def supported_experiment_names(tier: str | None = None) -> list[str]:
    if tier is None:
        return list(SUPPORTED_EXPERIMENTS)
    return [
        exp
        for c in _MODEL_CONTRACTS.values() if c.tier == tier
        for exp in c.allowed_experiments
    ]


def normalize_target_workload_class(value: str) -> str:
    normalized = _normalize(value)
    if normalized in {"coding_long_context", "coding", "coding_agent", "long_context_coding"}:
        return "coding"
    if normalized in {"chat", "conversational"}:
        return "chat"
    return normalized


def resolve_supported_model(model_name: str) -> ModelVariant | None:
    """Resolve any supported model (across all tiers)."""
    if not model_name.strip():
        return None
    model = get_model_variant(model_name)
    if model is None:
        return None
    # Check if this model has a support contract
    contract = resolve_model_support_contract(model.name)
    if contract is None:
        return None
    return model


def resolve_supported_gpu(gpu_name: str) -> GPUProfile | None:
    """Resolve a supported Hopper or Blackwell production GPU."""
    if not gpu_name.strip():
        return None
    gpu = get_gpu_profile(gpu_name)
    if gpu is None or gpu.name not in SUPPORTED_GPU_NAMES:
        return None
    return gpu


def is_target_engine(engine: str | None) -> bool:
    if engine is None:
        return False
    return _normalize(engine) == SUPPORTED_ENGINE


def is_target_model(model: ModelVariant | str | None) -> bool:
    if model is None:
        return False
    name = model.name if hasattr(model, "name") else str(model)
    return resolve_supported_model(name) is not None or _compact(name) == _compact(SUPPORTED_MODEL)


def is_target_gpu(gpu: GPUProfile | str | None) -> bool:
    if gpu is None:
        return False
    name = gpu.name if hasattr(gpu, "name") else str(gpu)
    return resolve_supported_gpu(name) is not None or _compact(name) in {_compact(v) for v in SUPPORTED_GPU_CANONICAL}


def is_target_workload(value: str) -> bool:
    return normalize_target_workload_class(value) in set(SUPPORTED_WORKLOAD_MODES)


def supports_topology(mode: str) -> bool:
    return _normalize(mode) in {_normalize(value) for value in SUPPORTED_TOPOLOGY_MODES}


def supports_cache_strategy(strategy: str) -> bool:
    return _normalize(strategy) in {_normalize(value) for value in SUPPORTED_CACHE_STRATEGIES}


def resolve_model_support_contract(model_name: str) -> ModelSupportContract | None:
    """Look up the support contract for a model by name."""
    if not model_name.strip():
        return None
    # Canonicalize through the model registry first
    variant = get_model_variant(model_name)
    if variant is not None and variant.name in _MODEL_CONTRACTS:
        return _MODEL_CONTRACTS[variant.name]
    # Direct match fallback
    if model_name in _MODEL_CONTRACTS:
        return _MODEL_CONTRACTS[model_name]
    # Case-insensitive compact match
    compact = _compact(model_name)
    for key, contract in _MODEL_CONTRACTS.items():
        if _compact(key) == compact:
            return contract
    return None


def support_tier_for_model(model_name: str) -> str | None:
    """Return the support tier for a model, or None if unsupported."""
    contract = resolve_model_support_contract(model_name)
    return contract.tier if contract else None


def is_model_in_tier(model_name: str, tier: str) -> bool:
    """Check if a model belongs to a specific support tier."""
    return support_tier_for_model(model_name) == tier


def target_profile_summary() -> str:
    return (
        "InferScope production scope: Kimi-K2.5 with Dynamo + LMCache for long-context coding on H100/H200/B200/B300; "
        "vLLM exists only as the comparison benchmark lane."
    )


def _minimum_tp_for_lane(model: ModelVariant) -> int:
    hinted = model.serving.get("tp_min")
    if isinstance(hinted, int) and hinted > 0:
        return hinted
    return 1


def _minimum_tp_for_gpu(model: ModelVariant, gpu: GPUProfile | None) -> int:
    if gpu is None:
        for key in ("tp_fp8_h200", "tp_fp8_b200", "tp_fp8_b300", "tp_fp8_h100"):
            hinted = model.serving.get(key)
            if isinstance(hinted, int) and hinted > 0:
                return hinted
        return _minimum_tp_for_lane(model)

    name = gpu.name.lower()
    candidate_keys: list[str] = []
    if "h100" in name:
        candidate_keys.append("tp_fp8_h100")
    elif "h200" in name:
        candidate_keys.append("tp_fp8_h200")
    elif gpu.name == "B200":
        if gpu.fp4_support:
            candidate_keys.append("tp_fp4_b200")
        candidate_keys.append("tp_fp8_b200")
    elif gpu.name == "B300":
        if gpu.fp4_support:
            candidate_keys.append("tp_fp4_b300")
        candidate_keys.append("tp_fp8_b300")

    for key in candidate_keys:
        hinted = model.serving.get(key)
        if isinstance(hinted, int) and hinted > 0:
            return hinted
    return _minimum_tp_for_lane(model)


def required_gpus_for_topology(
    *,
    model_name: str,
    gpu_name: str = "",
    topology_mode: str,
) -> int:
    """Return the minimum practical GPU count for the target topology."""
    model = resolve_supported_model(model_name) or resolve_supported_model(SUPPORTED_MODEL)
    gpu = resolve_supported_gpu(gpu_name) if gpu_name else None
    tp_min = _minimum_tp_for_gpu(model, gpu) if model is not None else 1
    normalized_topology = topology_mode.strip().lower() or "single_endpoint"
    if normalized_topology == "prefill_decode_split":
        return tp_min * 2
    return tp_min


def validate_production_target(
    *,
    model_name: str = "",
    gpu_name: str = "",
    workload: str = "",
    engine: str = "",
    num_gpus: int = 0,
    topology_mode: str = "",
) -> list[str]:
    """Return user-facing validation errors for unsupported MCP benchmark targets."""
    errors: list[str] = []

    if model_name and resolve_supported_model(model_name) is None:
        errors.append(
            f"Model '{model_name}' is not supported. Supported models: "
            f"{', '.join(SUPPORTED_MODELS)}."
        )

    if gpu_name and resolve_supported_gpu(gpu_name) is None:
        errors.append(
            "InferScope MCP currently supports Hopper/Blackwell production GPUs only: H100, H200, B200, B300."
        )

    if workload:
        normalized = normalize_target_workload_class(workload)
        if normalized not in set(SUPPORTED_WORKLOAD_MODES):
            errors.append(
                "InferScope MCP supports coding and chat workflows only; got "
                f"'{workload}'."
            )

    if engine:
        normalized_engine = _normalize(engine)
        if normalized_engine not in {"", "auto", *SUPPORTED_BENCHMARK_ENGINES}:
            errors.append("InferScope MCP currently supports vLLM comparison and NVIDIA Dynamo only.")

    if topology_mode and not supports_topology(topology_mode):
        errors.append(
            f"InferScope MCP supports only {', '.join(SUPPORTED_TOPOLOGY_MODES)} topologies; got '{topology_mode}'."
        )

    if num_gpus:
        required_gpus = required_gpus_for_topology(
            model_name=model_name or SUPPORTED_MODEL,
            gpu_name=gpu_name,
            topology_mode=topology_mode,
        )
        if required_gpus and num_gpus < required_gpus:
            topology_label = topology_mode or "single_endpoint"
            errors.append(
                f"InferScope MCP requires at least {required_gpus} GPU(s) for "
                f"{topology_label} serving on this target; got {num_gpus}."
            )

    return errors


def build_production_contract() -> dict[str, Any]:
    """Return the supported deployment, benchmarking, and observability contract."""
    return {
        "name": PRODUCTION_TARGET_NAME,
        "scope_summary": target_profile_summary(),
        "model": SUPPORTED_MODEL,
        "models": list(SUPPORTED_MODELS),
        "engine": SUPPORTED_ENGINE,
        "benchmark_engines": list(SUPPORTED_BENCHMARK_ENGINES),
        "workload_mode": SUPPORTED_WORKLOAD_MODE,
        "workload_modes": list(SUPPORTED_WORKLOAD_MODES),
        "workload_pack": SUPPORTED_WORKLOAD_PACK,
        "workload_packs": list(SUPPORTED_WORKLOAD_PACKS),
        "experiments": list(SUPPORTED_EXPERIMENTS),
        "default_experiment": DEFAULT_EXPERIMENT,
        "supported_gpu_aliases": list(SUPPORTED_GPU_CANONICAL),
        "non_goals": list(NON_GOALS),
        "observability": {
            "frontend_metrics": list(FRONTEND_METRICS),
            "backend_metrics": list(BACKEND_METRICS),
            "kv_metrics": list(KV_METRICS),
            "trace_env_vars": list(TRACE_ENV_VARS),
            "request_headers": list(REQUEST_HEADERS),
            "backend_system_port_env": "DYN_SYSTEM_PORT",
            "summary": (
                "Scrape frontend and worker metrics separately. Frontend exposes queue, "
                "TTFT, ITL, disconnects, and migration counters; worker system ports "
                "expose request duration and KV stats."
            ),
        },
        "reliability": {
            "minimum_success_rate": RELIABILITY_MIN_SUCCESS_RATE,
            "require_metrics_capture_complete": RELIABILITY_REQUIRE_METRICS_CAPTURE_COMPLETE,
            "max_failed_sessions": RELIABILITY_MAX_FAILED_SESSIONS,
            "warning_queue_depth": RELIABILITY_WARNING_QUEUE_DEPTH,
            "warning_migrations": RELIABILITY_WARNING_MIGRATIONS,
            "warning_kv_usage": RELIABILITY_WARNING_KV_USAGE,
        },
        "topologies": [
            {
                "name": "aggregated",
                "experiment": SUPPORTED_EXPERIMENTS[0],
                "min_gpus": required_gpus_for_topology(
                    model_name=SUPPORTED_MODEL,
                    gpu_name="H200 SXM",
                    topology_mode="single_endpoint",
                ),
                "summary": "Single Dynamo frontend plus one aggregated Kimi worker group with LMCache enabled.",
            },
            {
                "name": "disaggregated_comparison",
                "experiment": SUPPORTED_EXPERIMENTS[1],
                "min_gpus": required_gpus_for_topology(
                    model_name=SUPPORTED_MODEL,
                    gpu_name="H200 SXM",
                    topology_mode="prefill_decode_split",
                ),
                "summary": "vLLM disaggregated LMCache comparison lane for the same Kimi long-context coding workload.",
            },
            {
                "name": "disaggregated_production",
                "experiment": SUPPORTED_EXPERIMENTS[2],
                "min_gpus": required_gpus_for_topology(
                    model_name=SUPPORTED_MODEL,
                    gpu_name="H200 SXM",
                    topology_mode="prefill_decode_split",
                ),
                "summary": "Dynamo frontend plus separate prefill/decode workers with LMCache and KV-aware routing.",
            },
        ],
    }


def build_benchmark_readiness_summary(artifact: BenchmarkArtifact) -> dict[str, Any]:
    """Summarize benchmark reliability and observability in production terms."""
    total_requests = artifact.summary.total_requests or 0
    succeeded = artifact.summary.succeeded
    success_rate = (succeeded / total_requests) if total_requests else 0.0
    observed_runtime = artifact.run_plan.get("observed_runtime", {}) if artifact.run_plan else {}
    reliability = observed_runtime.get("reliability", {}) if isinstance(observed_runtime, dict) else {}
    observability = observed_runtime.get("observability", {}) if isinstance(observed_runtime, dict) else {}

    issues: list[str] = []
    if success_rate < RELIABILITY_MIN_SUCCESS_RATE:
        issues.append(
            f"Request success rate is {success_rate:.1%}, below the {RELIABILITY_MIN_SUCCESS_RATE:.0%} production target."
        )
    if RELIABILITY_REQUIRE_METRICS_CAPTURE_COMPLETE and not artifact.summary.metrics_capture_complete:
        issues.append("Metrics capture was incomplete across declared targets.")
    if observability.get("missing_metric_prefixes"):
        issues.append("Required observability prefixes were missing from at least one metrics target.")
    if observability.get("targets_with_errors"):
        issues.append("At least one metrics target failed to scrape cleanly.")

    failed_sessions = int(reliability.get("failed_sessions", 0) or 0)
    if failed_sessions > RELIABILITY_MAX_FAILED_SESSIONS:
        issues.append(f"{failed_sessions} session(s) failed, which breaks coding-session reliability guarantees.")

    observability_gaps = len(observability.get("missing_metric_prefixes", []))
    return {
        "ready": not issues,
        "success_rate": round(success_rate, 4),
        "metrics_capture_complete": artifact.summary.metrics_capture_complete,
        "metrics_targets_total": artifact.summary.metrics_targets_total,
        "metrics_targets_with_errors": artifact.summary.metrics_targets_with_errors,
        "failed_sessions": failed_sessions,
        "failure_types": reliability.get("failure_types", {}),
        "missing_metric_prefixes": observability.get("missing_metric_prefixes", []),
        "issues": issues,
        "summary": (
            f"success={success_rate:.1%}, "
            f"metrics_complete={artifact.summary.metrics_capture_complete}, "
            f"failed_sessions={failed_sessions}, "
            f"observability_gaps={observability_gaps}"
        ),
    }
