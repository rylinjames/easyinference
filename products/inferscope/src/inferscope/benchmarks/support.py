"""Benchmark support assessment for GPU/model/engine/topology compatibility."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from inferscope.benchmarks.models import WorkloadPack
from inferscope.hardware.gpu_profiles import GPUProfile, get_gpu_profile
from inferscope.models.registry import ModelVariant, get_model_variant
from inferscope.optimization.platform_policy import (
    EngineSupportTier,
    PlatformTraits,
    resolve_engine_support,
    resolve_platform_traits,
)

if TYPE_CHECKING:
    from inferscope.benchmarks.experiments import BenchmarkExperimentSpec

SupportSeverity = Literal["error", "warning", "info"]
SupportComponent = Literal["gpu", "model", "engine", "topology", "cache", "transport", "context"]
SupportStatus = Literal["supported", "degraded", "unsupported", "unknown"]


class BenchmarkSupportIssue(BaseModel):
    """One compatibility issue discovered during benchmark planning."""

    model_config = ConfigDict(extra="forbid")

    severity: SupportSeverity
    code: str
    message: str
    component: SupportComponent


class BenchmarkSupportProfile(BaseModel):
    """Resolved support profile for a benchmark run or stack plan."""

    model_config = ConfigDict(extra="forbid")

    status: SupportStatus
    gpu: str | None = None
    gpu_isa: str | None = None
    gpu_architecture: str | None = None
    gpu_memory_gb: float | None = None
    num_gpus: int | None = None
    model: str | None = None
    model_class: str | None = None
    model_context_length: int | None = None
    engine: str | None = None
    engine_support_tier: str | None = None
    platform_family: str | None = None
    fp8_support: bool | None = None
    fp4_support: bool | None = None
    issues: list[BenchmarkSupportIssue] = Field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        return self.status in {"supported", "degraded"}


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _approx_token_count_from_text(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, math.ceil(len(value) / 4))
    if isinstance(value, list):
        return sum(_approx_token_count_from_text(item) for item in value)
    if isinstance(value, dict):
        return sum(_approx_token_count_from_text(item) for item in value.values())
    return max(1, math.ceil(len(str(value)) / 4))


def _estimate_prompt_tokens(workload: WorkloadPack | None) -> int | None:
    if workload is None or not workload.requests:
        return None
    estimates: list[int] = []
    for request in workload.requests:
        approx = request.metadata.get("approx_context_tokens")
        if isinstance(approx, int) and approx > 0:
            estimates.append(approx)
            continue
        total = 0
        for message in request.messages:
            total += _approx_token_count_from_text(message.role)
            total += _approx_token_count_from_text(message.content)
        estimates.append(max(1, total))
    return max(estimates) if estimates else None


def _append_issue(
    issues: list[BenchmarkSupportIssue],
    *,
    severity: SupportSeverity,
    code: str,
    message: str,
    component: SupportComponent,
) -> None:
    issues.append(BenchmarkSupportIssue(severity=severity, code=code, message=message, component=component))


def _family_aliases(traits: PlatformTraits) -> set[str]:
    aliases = {traits.family.value}
    if traits.is_hopper:
        aliases.add("hopper")
    if traits.is_hopper_pcie:
        aliases.add("hopper_pcie")
    if traits.is_blackwell:
        aliases.add("blackwell")
    if traits.is_b300:
        aliases.add("blackwell_ultra")
    if traits.is_grace:
        aliases.add("grace")
    if traits.is_gh200:
        aliases.update({"hopper_grace", "gh200"})
    if traits.is_gb200:
        aliases.update({"blackwell_grace", "gb200"})
    if traits.is_gb300:
        aliases.update({"blackwell_ultra_grace", "gb300", "blackwell_grace"})
    return aliases


def _matches_gpu_family(targets: list[str], traits: PlatformTraits) -> bool:
    if not targets:
        return True
    normalized_targets = {_normalize(value) for value in targets}
    return bool(normalized_targets & _family_aliases(traits))


def _matches_model_class(targets: list[str], model: ModelVariant | None) -> bool:
    if not targets or model is None:
        return True
    normalized_targets = {_normalize(value) for value in targets}
    return _normalize(model.model_class.value) in normalized_targets


def _engine_name(engine_name: str, experiment: BenchmarkExperimentSpec | None) -> str:
    if engine_name.strip():
        return engine_name.strip().lower()
    if experiment is not None:
        return experiment.engine
    return ""


def _resolve_context_tokens(prompt_tokens: int | None, workload: WorkloadPack | None) -> int | None:
    if prompt_tokens is not None and prompt_tokens > 0:
        return prompt_tokens
    return _estimate_prompt_tokens(workload)


def assess_benchmark_support(
    *,
    model_name: str,
    gpu_name: str = "",
    num_gpus: int | None = None,
    engine_name: str = "",
    workload: WorkloadPack | None = None,
    experiment: BenchmarkExperimentSpec | None = None,
    prompt_tokens: int | None = None,
    has_rdma: bool | None = None,
) -> BenchmarkSupportProfile:
    """Resolve concrete GPU/model/engine compatibility for a benchmark lane or run."""

    issues: list[BenchmarkSupportIssue] = []
    selected_engine = _engine_name(engine_name, experiment)

    model_variant = get_model_variant(model_name) if model_name else None
    if not model_name:
        _append_issue(
            issues,
            severity="warning",
            code="missing_model",
            message="No model identity was supplied; benchmark support can only be assessed partially.",
            component="model",
        )
    elif model_variant is None:
        _append_issue(
            issues,
            severity="error",
            code="unknown_model",
            message=f"Unknown model '{model_name}'.",
            component="model",
        )

    gpu_profile: GPUProfile | None = None
    traits: PlatformTraits | None = None
    if gpu_name:
        gpu_profile = get_gpu_profile(gpu_name)
        if gpu_profile is None:
            _append_issue(
                issues,
                severity="error",
                code="unknown_gpu",
                message=f"Unknown GPU '{gpu_name}'.",
                component="gpu",
            )
        else:
            traits = resolve_platform_traits(gpu_profile)
    else:
        _append_issue(
            issues,
            severity="warning",
            code="missing_gpu",
            message="No GPU identity was supplied; support gating is advisory only.",
            component="gpu",
        )

    if selected_engine and gpu_profile is not None:
        engine_support = resolve_engine_support(
            selected_engine,
            gpu_profile,
            multi_node=(experiment.topology.mode != "single_endpoint") if experiment is not None else False,
        )
        if engine_support.tier == EngineSupportTier.UNSUPPORTED:
            _append_issue(
                issues,
                severity="error",
                code="unsupported_engine",
                message=engine_support.reason,
                component="engine",
            )
        elif engine_support.tier == EngineSupportTier.PREVIEW:
            _append_issue(
                issues,
                severity="warning",
                code="preview_engine",
                message=engine_support.reason,
                component="engine",
            )
    elif not selected_engine:
        _append_issue(
            issues,
            severity="warning",
            code="missing_engine",
            message="No engine identity was supplied; engine-specific support cannot be enforced.",
            component="engine",
        )

    if experiment is not None and traits is not None:
        resolved_gpu_name = gpu_profile.name if gpu_profile is not None else (gpu_name or "unknown")
        if not _matches_gpu_family(experiment.target_gpu_families, traits):
            _append_issue(
                issues,
                severity="error",
                code="experiment_gpu_family_mismatch",
                message=(
                    f"Experiment '{experiment.name}' targets GPU families {experiment.target_gpu_families}, "
                    f"but '{resolved_gpu_name}' resolves to '{traits.family.value}'."
                ),
                component="gpu",
            )
        if not _matches_model_class(experiment.target_model_classes, model_variant):
            _append_issue(
                issues,
                severity="error",
                code="experiment_model_class_mismatch",
                message=(
                    f"Experiment '{experiment.name}' targets model classes {experiment.target_model_classes}, "
                    f"but '{model_name}' resolves to "
                    f"'{model_variant.model_class.value if model_variant else 'unknown'}'."
                ),
                component="model",
            )
        if experiment.topology.mode != "single_endpoint" and num_gpus is not None and num_gpus < 2:
            _append_issue(
                issues,
                severity="error",
                code="insufficient_gpus_for_topology",
                message=(f"Topology '{experiment.topology.mode}' requires at least 2 GPUs; got {num_gpus}."),
                component="topology",
            )
        if experiment.cache.strategy == "offloading_connector":
            if selected_engine and selected_engine != "vllm":
                _append_issue(
                    issues,
                    severity="error",
                    code="offloading_connector_engine_mismatch",
                    message="OffloadingConnector benchmark lanes require the vLLM engine.",
                    component="cache",
                )
            if experiment.topology.mode != "single_endpoint":
                _append_issue(
                    issues,
                    severity="error",
                    code="offloading_connector_topology_mismatch",
                    message="OffloadingConnector benchmark lanes require single-endpoint topology.",
                    component="topology",
                )
        if (
            experiment.cache.strategy == "lmcache"
            and experiment.topology.mode == "single_endpoint"
            and selected_engine != "dynamo"
        ):
            _append_issue(
                issues,
                severity="error",
                code="lmcache_requires_split_topology",
                message=(
                    "LMCache benchmark lanes require split prefill/decode topology "
                    "unless Dynamo manages the worker topology."
                ),
                component="cache",
            )
        if experiment.cache.strategy == "hicache" and selected_engine and selected_engine != "sglang":
            _append_issue(
                issues,
                severity="error",
                code="hicache_engine_mismatch",
                message="HiCache benchmark lanes require the SGLang engine.",
                component="cache",
            )
        if experiment.cache.strategy == "nixl":
            transport_ok = bool(has_rdma) or traits.has_high_speed_interconnect
            if not transport_ok:
                _append_issue(
                    issues,
                    severity="warning",
                    code="nixl_transport_degraded",
                    message=(
                        "NIXL was selected without RDMA or a high-speed interconnect; "
                        "benchmark results will be transport-degraded."
                    ),
                    component="transport",
                )
        if "grace_coherent" in experiment.cache.tiers and not traits.is_grace:
            _append_issue(
                issues,
                severity="error",
                code="grace_tier_requires_grace",
                message="Grace-coherent cache tiers require GH200/GB200/GB300-class Grace systems.",
                component="cache",
            )

    if workload is not None and traits is not None and not _matches_gpu_family(workload.target_gpu_families, traits):
        resolved_gpu_name = gpu_profile.name if gpu_profile is not None else (gpu_name or "unknown")
        _append_issue(
            issues,
            severity="warning",
            code="workload_gpu_family_mismatch",
            message=(
                f"Workload '{workload.name}' targets GPU families {workload.target_gpu_families}, "
                f"but '{resolved_gpu_name}' resolves to '{traits.family.value}'."
            ),
            component="gpu",
        )
    if workload is not None and not _matches_model_class(workload.target_model_classes, model_variant):
        _append_issue(
            issues,
            severity="warning",
            code="workload_model_class_mismatch",
            message=(
                f"Workload '{workload.name}' targets model classes {workload.target_model_classes}, "
                f"but '{model_name}' resolves to '{model_variant.model_class.value if model_variant else 'unknown'}'."
            ),
            component="model",
        )

    resolved_prompt_tokens = _resolve_context_tokens(prompt_tokens, workload)
    if (
        model_variant is not None
        and resolved_prompt_tokens is not None
        and resolved_prompt_tokens > model_variant.context_length
    ):
        _append_issue(
            issues,
            severity="error",
            code="context_length_exceeded",
            message=(
                f"Benchmark prompt length {resolved_prompt_tokens} exceeds model context length "
                f"{model_variant.context_length} for '{model_variant.name}'."
            ),
            component="context",
        )

    if num_gpus is not None and num_gpus < 1:
        _append_issue(
            issues,
            severity="error",
            code="invalid_gpu_count",
            message="num_gpus must be at least 1.",
            component="gpu",
        )

    if any(issue.severity == "error" for issue in issues):
        status: SupportStatus = "unsupported"
    elif any(issue.code.startswith("missing_") for issue in issues):
        status = "unknown"
    elif any(issue.severity == "warning" for issue in issues):
        status = "degraded"
    else:
        status = "supported"

    return BenchmarkSupportProfile(
        status=status,
        gpu=gpu_profile.name if gpu_profile is not None else (gpu_name or None),
        gpu_isa=(gpu_profile.compute_capability if gpu_profile is not None else None),
        gpu_architecture=(gpu_profile.architecture if gpu_profile is not None else None),
        gpu_memory_gb=(gpu_profile.memory_gb if gpu_profile is not None else None),
        num_gpus=num_gpus,
        model=(model_variant.name if model_variant is not None else (model_name or None)),
        model_class=(model_variant.model_class.value if model_variant is not None else None),
        model_context_length=(model_variant.context_length if model_variant is not None else None),
        engine=(selected_engine or None),
        engine_support_tier=(
            resolve_engine_support(selected_engine, gpu_profile).tier.value
            if selected_engine and gpu_profile is not None
            else None
        ),
        platform_family=(traits.family.value if traits is not None else None),
        fp8_support=(gpu_profile.fp8_support if gpu_profile is not None else None),
        fp4_support=(gpu_profile.fp4_support if gpu_profile is not None else None),
        issues=issues,
    )
