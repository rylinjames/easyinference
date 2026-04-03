"""Shared resolution helpers for the narrowed InferScope probe surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from inferscope.benchmarks.catalog import load_experiment, materialize_workload
from inferscope.benchmarks.experiments import (
    BenchmarkExecutionProfile,
    BenchmarkExperimentSpec,
    BenchmarkGoodputSLO,
    BenchmarkRunPlan,
    build_run_plan,
    parse_metrics_target_overrides,
)
from inferscope.benchmarks.models import WorkloadPack
from inferscope.benchmarks.procedural import ProceduralWorkloadOptions
from inferscope.benchmarks.support import BenchmarkSupportProfile, assess_benchmark_support
from inferscope.production_target import (
    DEFAULT_EXPERIMENT,
    SUPPORTED_EXPERIMENTS,
    SUPPORTED_MODEL,
    SUPPORTED_WORKLOAD_PACKS,
    validate_production_target,
)


class ProbeResolutionError(ValueError):
    """Base error for narrowed probe-plan resolution."""


class ProductionTargetValidationError(ProbeResolutionError):
    """Raised when a request falls outside the supported InferScope contract."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class BenchmarkSupportError(ProbeResolutionError):
    """Raised when support validation rejects a requested probe."""

    def __init__(self, message: str, support: BenchmarkSupportProfile | None = None):
        self.support = support
        super().__init__(message)


@dataclass(frozen=True)
class ResolvedProbePlan:
    """Fully resolved probe execution inputs for CLI or MCP entrypoints."""

    workload_reference: str
    workload_pack: WorkloadPack
    experiment_spec: BenchmarkExperimentSpec
    run_plan: BenchmarkRunPlan
    support: BenchmarkSupportProfile


def build_probe_procedural_options(
    *,
    synthetic_requests: int | None = None,
    synthetic_input_tokens: int | None = None,
    synthetic_output_tokens: int | None = None,
    synthetic_seed: int = 42,
    context_file: str = "",
    allow_context_file: bool,
) -> ProceduralWorkloadOptions | None:
    """Resolve optional procedural expansion settings for probe workloads."""
    if context_file and not allow_context_file:
        raise ProbeResolutionError(
            "context_file is not supported from MCP tools; use the CLI for local context-file expansion"
        )
    if not any(
        value is not None and value != ""
        for value in (
            synthetic_requests,
            synthetic_input_tokens,
            synthetic_output_tokens,
            context_file,
        )
    ):
        return None
    return ProceduralWorkloadOptions(
        request_count=synthetic_requests,
        input_tokens=synthetic_input_tokens,
        output_tokens=synthetic_output_tokens,
        seed=synthetic_seed,
        context_file=(context_file or None),
    )


def build_probe_execution_profile(
    *,
    request_rate: float | None = None,
    arrival_model: str = "immediate",
    arrival_shape: float | None = None,
    warmup_requests: int = 0,
    goodput_slo: dict[str, Any] | None = None,
) -> BenchmarkExecutionProfile:
    """Build a replay execution profile for the narrowed probe surface."""
    resolved_arrival_model: Literal["immediate", "poisson", "gamma"] = "immediate"
    if request_rate not in (None, 0.0) and arrival_model in {"immediate", "poisson", "gamma"}:
        resolved_arrival_model = cast(Literal["immediate", "poisson", "gamma"], arrival_model)
    return BenchmarkExecutionProfile(
        request_rate_rps=request_rate,
        arrival_model=resolved_arrival_model,
        arrival_shape=arrival_shape,
        warmup_requests=warmup_requests,
        goodput_slo=BenchmarkGoodputSLO.model_validate(goodput_slo or {}),
    )


def _request_context_tokens(request: Any) -> int | None:
    metadata = getattr(request, "metadata", {})
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("approx_context_tokens")
    return value if isinstance(value, int) else None


def _apply_metrics_target_fallbacks(
    experiment_spec: BenchmarkExperimentSpec,
    *,
    request_endpoint: str,
    metrics_endpoint: str | None,
    overrides: dict[str, str],
) -> dict[str, str]:
    fallback_endpoint = (metrics_endpoint or request_endpoint).strip()
    if not fallback_endpoint:
        return overrides
    resolved = dict(overrides)
    for target in experiment_spec.metrics_targets:
        if target.endpoint_source != "named_override":
            continue
        override_key = (target.override_key or target.name).strip()
        if override_key and override_key not in resolved:
            resolved[override_key] = fallback_endpoint
    return resolved


def resolve_probe_plan(
    workload: str,
    endpoint: str,
    *,
    experiment: str = "",
    gpu: str = "",
    num_gpus: int | None = None,
    metrics_endpoint: str | None = None,
    metrics_target: list[str] | None = None,
    metrics_target_overrides: dict[str, str] | None = None,
    concurrency: int | None = None,
    request_rate: float | None = None,
    arrival_model: str = "immediate",
    arrival_shape: float | None = None,
    warmup_requests: int = 0,
    goodput_slo: dict[str, Any] | None = None,
    strict_support: bool = True,
    synthetic_requests: int | None = None,
    synthetic_input_tokens: int | None = None,
    synthetic_output_tokens: int | None = None,
    synthetic_seed: int = 42,
    context_file: str = "",
    allow_context_file: bool,
) -> ResolvedProbePlan:
    """Resolve a supported InferScope workload into an executable probe plan."""
    procedural_options = build_probe_procedural_options(
        synthetic_requests=synthetic_requests,
        synthetic_input_tokens=synthetic_input_tokens,
        synthetic_output_tokens=synthetic_output_tokens,
        synthetic_seed=synthetic_seed,
        context_file=context_file,
        allow_context_file=allow_context_file,
    )

    input_workload_pack = materialize_workload(workload, options=procedural_options)
    if input_workload_pack.name not in SUPPORTED_WORKLOAD_PACKS:
        raise ProductionTargetValidationError(
            [
                "InferScope exposes only the Kimi-K2.5 long-context coding workload pack as a benchmark probe.",
            ]
        )

    selected_experiment = experiment or DEFAULT_EXPERIMENT
    experiment_spec = load_experiment(selected_experiment)
    if experiment_spec.name not in SUPPORTED_EXPERIMENTS:
        raise ProductionTargetValidationError(
            [
                "InferScope exposes only the Kimi-targeted vLLM and Dynamo benchmark probes.",
            ]
        )
    if input_workload_pack.name != experiment_spec.workload:
        raise ProbeResolutionError(
            f"Workload '{input_workload_pack.name}' does not match experiment "
            f"'{experiment_spec.name}' workload '{experiment_spec.workload}'"
        )

    workload_reference = experiment_spec.workload
    workload_pack = materialize_workload(workload_reference, options=procedural_options)
    selected_model_name = experiment_spec.model or workload_pack.model or SUPPORTED_MODEL
    topology_mode = experiment_spec.topology.mode
    production_errors = validate_production_target(
        model_name=selected_model_name,
        gpu_name=gpu,
        workload=workload_pack.workload_class,
        engine=experiment_spec.engine,
        num_gpus=(num_gpus or 0),
        topology_mode=topology_mode,
    )
    if production_errors:
        raise ProductionTargetValidationError(production_errors)

    support = assess_benchmark_support(
        model_name=selected_model_name,
        gpu_name=gpu,
        num_gpus=num_gpus,
        engine_name=experiment_spec.engine,
        workload=workload_pack,
        experiment=experiment_spec,
        prompt_tokens=max(
            (
                value
                for request in workload_pack.requests
                if (value := _request_context_tokens(request)) is not None
            ),
            default=0,
        )
        or None,
    )
    if strict_support and support.status == "unsupported":
        error_messages = [issue.message for issue in support.issues if issue.severity == "error"]
        raise BenchmarkSupportError(
            "; ".join(error_messages) or "Unsupported benchmark configuration",
            support=support,
        )

    resolved_metrics_target_overrides = (
        metrics_target_overrides
        if metrics_target_overrides is not None
        else parse_metrics_target_overrides(metrics_target)
    )
    resolved_metrics_target_overrides = _apply_metrics_target_fallbacks(
        experiment_spec,
        request_endpoint=endpoint,
        metrics_endpoint=metrics_endpoint,
        overrides=resolved_metrics_target_overrides,
    )
    run_plan = build_run_plan(
        workload_pack,
        endpoint,
        workload_ref=workload_reference,
        experiment=experiment_spec,
        concurrency=concurrency,
        metrics_endpoint=metrics_endpoint,
        metrics_target_overrides=resolved_metrics_target_overrides,
        execution=build_probe_execution_profile(
            request_rate=request_rate,
            arrival_model=arrival_model,
            arrival_shape=arrival_shape,
            warmup_requests=warmup_requests,
            goodput_slo=goodput_slo,
        ),
        support=support,
    )
    return ResolvedProbePlan(
        workload_reference=workload_reference,
        workload_pack=workload_pack,
        experiment_spec=experiment_spec,
        run_plan=run_plan,
        support=support,
    )
