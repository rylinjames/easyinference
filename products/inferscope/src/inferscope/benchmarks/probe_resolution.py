"""Shared resolution helpers for the narrowed InferScope probe surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from inferscope.benchmarks.catalog import (
    load_experiment,
    materialize_workload,
    resolve_experiment_reference,
    resolve_workload_reference,
)
from inferscope.benchmarks.experiments import (
    BenchmarkExecutionProfile,
    BenchmarkExperimentSpec,
    BenchmarkGoodputSLO,
    BenchmarkRunPlan,
    build_run_plan,
    parse_metrics_target_overrides,
)
from inferscope.benchmarks.models import BenchmarkSourceReference, WorkloadPack
from inferscope.benchmarks.preflight import validate_benchmark_preflight
from inferscope.benchmarks.procedural import ProceduralWorkloadOptions
from inferscope.benchmarks.support import BenchmarkSupportProfile, assess_benchmark_support
from inferscope.models.registry import get_model_variant
from inferscope.production_target import (
    DEFAULT_EXPERIMENT,
    SUPPORTED_EXPERIMENTS,
    SUPPORTED_MODEL,
    SUPPORTED_WORKLOAD_PACKS,
    build_lane_reference,
    resolve_model_support_contract,
    supported_configuration_hint,
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
    workload_source: BenchmarkSourceReference
    experiment_source: BenchmarkSourceReference
    experiment_spec: BenchmarkExperimentSpec
    run_plan: BenchmarkRunPlan
    support: BenchmarkSupportProfile


def _source_reference(reference: str | Path, resolved_path: Path) -> BenchmarkSourceReference:
    raw_reference = str(reference)
    return BenchmarkSourceReference(
        reference=raw_reference,
        resolved_path=str(resolved_path),
        source_kind=("file" if Path(raw_reference).exists() else "builtin"),
    )


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
            "context_file is not supported from MCP tools; use the CLI for local context-file expansion. "
            "Workaround: rerun the same workload through "
            "`uv run inferscope benchmark-plan ... --context-file ...` locally."
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


def _default_experiment_for_workload(workload_pack: WorkloadPack) -> str:
    model_name = workload_pack.model or SUPPORTED_MODEL
    contract = resolve_model_support_contract(model_name)
    if contract is not None:
        for candidate in contract.allowed_experiments:
            if load_experiment(candidate).workload == workload_pack.name:
                return candidate
        if contract.allowed_experiments:
            return contract.allowed_experiments[0]
    return DEFAULT_EXPERIMENT


def _canonical_model_name(name: str | None) -> str | None:
    if not name:
        return None
    variant = get_model_variant(name)
    return variant.name if variant is not None else name.strip()


def _validate_workload_experiment_contract(
    workload_pack: WorkloadPack,
    experiment_spec: BenchmarkExperimentSpec,
) -> None:
    errors: list[str] = []
    workload_model_name = _canonical_model_name(workload_pack.model)
    experiment_model_name = _canonical_model_name(experiment_spec.model)
    selected_model_name = experiment_model_name or workload_model_name
    selected_variant = get_model_variant(selected_model_name) if selected_model_name else None

    if workload_model_name and experiment_model_name and workload_model_name != experiment_model_name:
        errors.append(
            "Workload pack model "
            f"'{workload_model_name}' does not match experiment model '{experiment_model_name}'."
        )

    if selected_variant is not None:
        model_class = selected_variant.model_class.value
        if workload_pack.target_model_classes and model_class not in workload_pack.target_model_classes:
            errors.append(
                "Selected model "
                f"'{selected_variant.name}' resolves to class '{model_class}', "
                "which is not allowed by the workload pack target_model_classes."
            )
        if experiment_spec.target_model_classes and model_class not in experiment_spec.target_model_classes:
            errors.append(
                "Selected model "
                f"'{selected_variant.name}' resolves to class '{model_class}', "
                "which is not allowed by the experiment target_model_classes."
            )

    if workload_pack.target_model_classes and experiment_spec.target_model_classes:
        shared_model_classes = set(workload_pack.target_model_classes) & set(experiment_spec.target_model_classes)
        if not shared_model_classes:
            errors.append(
                "Workload pack and experiment declare disjoint target_model_classes. "
                "Choose a matching lane or fix the workload/experiment metadata."
            )

    if errors:
        raise ProbeResolutionError("; ".join(errors))


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
    model_artifact_path: str = "",
    artifact_manifest: str = "",
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

    workload_source_path = resolve_workload_reference(workload)
    workload_source = _source_reference(workload, workload_source_path)
    input_workload_pack = materialize_workload(workload, options=procedural_options)
    if input_workload_pack.name not in SUPPORTED_WORKLOAD_PACKS:
        raise ProductionTargetValidationError(
            [
                f"Workload pack '{input_workload_pack.name}' is not in the shipped InferScope benchmark catalog. "
                f"Supported workload packs today: {', '.join(SUPPORTED_WORKLOAD_PACKS)}. "
                "Workaround: choose one of those built-ins for the supported probe path. "
                f"{supported_configuration_hint()}",
            ]
        )

    selected_experiment = experiment or _default_experiment_for_workload(input_workload_pack)
    experiment_source_path = resolve_experiment_reference(selected_experiment)
    experiment_source = _source_reference(selected_experiment, experiment_source_path)
    experiment_spec = load_experiment(selected_experiment)
    if experiment_spec.name not in SUPPORTED_EXPERIMENTS:
        raise ProductionTargetValidationError(
            [
                f"Experiment '{experiment_spec.name}' is not in the shipped InferScope benchmark catalog. "
                f"Supported experiments today: {', '.join(SUPPORTED_EXPERIMENTS)}. "
                f"Workaround: use '{selected_experiment}' only if it appears in that catalog. "
                f"{supported_configuration_hint()}",
            ]
        )
    if input_workload_pack.name != experiment_spec.workload:
        raise ProbeResolutionError(
            f"Workload '{input_workload_pack.name}' does not match experiment "
            f"'{experiment_spec.name}' workload '{experiment_spec.workload}'. "
            "Workaround: keep the workload and experiment in the same supported lane."
        )
    _validate_workload_experiment_contract(input_workload_pack, experiment_spec)

    workload_reference = str(workload)
    workload_pack = input_workload_pack
    selected_model_name = experiment_spec.model or workload_pack.model or SUPPORTED_MODEL
    topology_mode = experiment_spec.topology.mode
    preflight_validation = validate_benchmark_preflight(
        model_name=selected_model_name,
        gpu_name=gpu,
        num_gpus=num_gpus,
        engine_name=experiment_spec.engine,
        topology_mode=topology_mode,
        model_artifact_path=model_artifact_path,
        artifact_manifest=artifact_manifest,
    )
    if not preflight_validation.valid:
        raise ProbeResolutionError("; ".join(preflight_validation.errors))
    if strict_support:
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
            (
                "; ".join(error_messages)
                or "Unsupported benchmark configuration. "
                "Workaround: use a supported GPU/engine pair from the current InferScope lane."
            ),
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
        workload_source=workload_source,
        experiment_source=experiment_source,
        reference_lane=build_lane_reference(
            model_name=selected_model_name,
            workload_pack=workload_pack.name,
            experiment_name=experiment_spec.name,
        ),
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
        preflight_validation=preflight_validation,
    )
    return ResolvedProbePlan(
        workload_reference=workload_reference,
        workload_pack=workload_pack,
        workload_source=workload_source,
        experiment_source=experiment_source,
        experiment_spec=experiment_spec,
        run_plan=run_plan,
        support=support,
    )
