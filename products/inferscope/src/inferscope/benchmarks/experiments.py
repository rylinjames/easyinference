"""Benchmark experiment specs and resolved run plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from inferscope.benchmarks.models import WorkloadPack
from inferscope.benchmarks.support import BenchmarkSupportProfile

MetricTargetRole = Literal["primary", "router", "prefill", "decode", "cache", "other"]
TopologyMode = Literal["single_endpoint", "prefill_decode_split", "router_prefill_decode"]
SessionRoutingMode = Literal["unknown", "none", "sticky", "hash"]
CacheStrategy = Literal["unknown", "none", "prefix_only", "lmcache", "hicache", "offloading_connector", "nixl"]
CacheTier = Literal["gpu_hbm", "grace_coherent", "cpu_dram", "local_ssd", "remote_cache"]


class MetricCaptureTargetSpec(BaseModel):
    """Declarative metrics capture target in an experiment spec."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: MetricTargetRole = "primary"
    endpoint_source: Literal["request_endpoint", "metrics_endpoint", "named_override", "explicit"] = "metrics_endpoint"
    override_key: str | None = None
    explicit_endpoint: str | None = None
    expected_engine: str | None = None
    required: bool = True

    @model_validator(mode="after")
    def validate_source(self) -> MetricCaptureTargetSpec:
        if self.endpoint_source == "named_override" and not (self.override_key or self.name):
            raise ValueError("named_override targets require override_key or name")
        if self.endpoint_source == "explicit" and not self.explicit_endpoint:
            raise ValueError("explicit targets require explicit_endpoint")
        return self


class ResolvedMetricCaptureTarget(BaseModel):
    """Concrete metrics capture target used for one benchmark run."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: MetricTargetRole = "primary"
    endpoint: str
    expected_engine: str | None = None
    required: bool = True


class BenchmarkTopologyMetadata(BaseModel):
    """Deployment topology metadata for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    mode: TopologyMode = "single_endpoint"
    session_routing: SessionRoutingMode = "unknown"
    session_header_name: str = "X-Session-ID"
    request_target_name: str = "primary"
    kv_connector: str | None = None
    notes: list[str] = Field(default_factory=list)


class BenchmarkCacheMetadata(BaseModel):
    """Cache/offload metadata for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    strategy: CacheStrategy = "unknown"
    tiers: list[CacheTier] = Field(default_factory=list)
    connector: str | None = None
    session_affinity: bool | None = None
    prefix_caching: bool | None = None
    prefix_cache_expected: bool | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_prefix_caching(self) -> BenchmarkCacheMetadata:
        if self.prefix_caching is False and self.prefix_cache_expected is True:
            raise ValueError("prefix_caching=False cannot be combined with prefix_cache_expected=True")
        return self


class BenchmarkGoodputSLO(BaseModel):
    """Optional SLO thresholds used to derive goodput during replay."""

    model_config = ConfigDict(extra="forbid")

    ttft_p95_ms: float | dict[str, float] | None = None
    ttft_p99_ms: float | dict[str, float] | None = None
    tpot_p95_ms: float | None = None


class BenchmarkExecutionProfile(BaseModel):
    """Execution mechanics for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["auto", "openai-chat", "openai-completions", "trtllm-generate-stream"] = "auto"
    request_rate_rps: float | None = None
    arrival_model: Literal["immediate", "poisson", "gamma"] = "immediate"
    arrival_shape: float | None = None
    warmup_requests: int = Field(default=0, ge=0, le=10_000)
    request_timeout_seconds: int = Field(default=600, ge=1, le=86_400)
    total_timeout_seconds: int = Field(default=7_200, ge=1, le=86_400)
    capture_outputs: bool = False
    goodput_slo: BenchmarkGoodputSLO = Field(default_factory=BenchmarkGoodputSLO)

    @model_validator(mode="after")
    def validate_execution_profile(self) -> BenchmarkExecutionProfile:
        if self.arrival_model == "gamma" and self.arrival_shape is not None and self.arrival_shape <= 0:
            raise ValueError("arrival_shape must be > 0 for gamma arrival")
        return self


class BenchmarkExperimentSpec(BaseModel):
    """Reusable benchmark experiment template."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    name: str
    description: str = ""
    engine: Literal["vllm", "sglang", "trtllm", "dynamo", "atom"]
    workload: str
    benchmark_role: str = "operator_extension"
    target_gpu_families: list[str] = Field(default_factory=list)
    target_model_classes: list[str] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)
    model: str | None = None
    concurrency: int | None = Field(default=None, ge=1, le=1024)
    topology: BenchmarkTopologyMetadata = Field(default_factory=BenchmarkTopologyMetadata)
    cache: BenchmarkCacheMetadata = Field(default_factory=BenchmarkCacheMetadata)
    metrics_targets: list[MetricCaptureTargetSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_metric_target_names(self) -> BenchmarkExperimentSpec:
        names = [target.name for target in self.metrics_targets]
        if len(names) != len(set(names)):
            raise ValueError("metric target names must be unique")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> BenchmarkExperimentSpec:
        file_path = Path(path)
        raw = file_path.read_text()
        suffix = file_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(raw)
        elif suffix == ".json":
            data = json.loads(raw)
        else:
            raise ValueError(f"Unsupported experiment file type: {suffix}")
        if not isinstance(data, dict):
            raise ValueError("Experiment file must contain a mapping/object at the top level")
        return cls.model_validate(data)


class BenchmarkRunPlan(BaseModel):
    """Resolved plan for executing and observing one benchmark run."""

    model_config = ConfigDict(extra="forbid")

    source_experiment: str | None = None
    workload_ref: str
    request_endpoint: str
    model: str
    concurrency: int = Field(ge=1, le=1024)
    engine: str | None = None
    topology: BenchmarkTopologyMetadata = Field(default_factory=BenchmarkTopologyMetadata)
    cache: BenchmarkCacheMetadata = Field(default_factory=BenchmarkCacheMetadata)
    execution: BenchmarkExecutionProfile = Field(default_factory=BenchmarkExecutionProfile)
    support: BenchmarkSupportProfile | None = None
    metrics_targets: list[ResolvedMetricCaptureTarget] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_targets(self) -> BenchmarkRunPlan:
        names = [target.name for target in self.metrics_targets]
        if len(names) != len(set(names)):
            raise ValueError("resolved metric target names must be unique")
        role_set = {target.role for target in self.metrics_targets}
        if "primary" not in role_set:
            raise ValueError("run plan requires a primary metrics target")
        if self.topology.mode == "prefill_decode_split" and not {"prefill", "decode"}.issubset(role_set):
            raise ValueError("prefill_decode_split requires prefill and decode targets")
        if self.topology.mode == "router_prefill_decode" and not {"router", "prefill", "decode"}.issubset(role_set):
            raise ValueError("router_prefill_decode requires router, prefill, and decode targets")
        return self


def parse_metrics_target_overrides(values: list[str] | None) -> dict[str, str]:
    """Parse repeated CLI values in the form name=url."""
    overrides: dict[str, str] = {}
    for value in values or []:
        name, sep, endpoint = value.partition("=")
        if not sep or not name.strip() or not endpoint.strip():
            raise ValueError(f"Invalid metrics target override '{value}'. Use name=url")
        overrides[name.strip()] = endpoint.strip()
    return overrides


def _infer_role(name: str) -> MetricTargetRole:
    normalized = name.strip().lower()
    if normalized in {"primary", "router", "prefill", "decode", "cache"}:
        return normalized  # type: ignore[return-value]
    return "other"


def _validate_engine_cache(engine: str | None, cache: BenchmarkCacheMetadata) -> None:
    if engine == "vllm" and cache.strategy == "hicache":
        raise ValueError("vLLM does not support HiCache")
    if engine == "sglang" and cache.strategy == "offloading_connector":
        raise ValueError("SGLang does not use vLLM OffloadingConnector")


def _resolve_targets(
    request_endpoint: str,
    metrics_endpoint: str | None,
    overrides: dict[str, str],
    target_specs: list[MetricCaptureTargetSpec],
) -> list[ResolvedMetricCaptureTarget]:
    if not target_specs:
        targets = [
            ResolvedMetricCaptureTarget(
                name="primary",
                role="primary",
                endpoint=metrics_endpoint or request_endpoint,
            )
        ]
        for name, endpoint in overrides.items():
            if name == "primary":
                targets[0] = ResolvedMetricCaptureTarget(name="primary", role="primary", endpoint=endpoint)
                continue
            targets.append(
                ResolvedMetricCaptureTarget(
                    name=name,
                    role=_infer_role(name),
                    endpoint=endpoint,
                )
            )
        return targets

    resolved: list[ResolvedMetricCaptureTarget] = []
    for spec in target_specs:
        if spec.endpoint_source == "request_endpoint":
            endpoint = request_endpoint
        elif spec.endpoint_source == "metrics_endpoint":
            endpoint = metrics_endpoint or request_endpoint
        elif spec.endpoint_source == "named_override":
            key = spec.override_key or spec.name
            endpoint = overrides.get(key, "")
            if not endpoint and spec.required:
                raise ValueError(f"Missing required metrics target override '{key}' for target '{spec.name}'")
            if not endpoint:
                continue
        else:
            endpoint = spec.explicit_endpoint or ""

        resolved.append(
            ResolvedMetricCaptureTarget(
                name=spec.name,
                role=spec.role,
                endpoint=endpoint,
                expected_engine=spec.expected_engine,
                required=spec.required,
            )
        )
    return resolved


def build_run_plan(
    workload: WorkloadPack,
    request_endpoint: str,
    *,
    workload_ref: str,
    experiment: BenchmarkExperimentSpec | None = None,
    model: str | None = None,
    concurrency: int | None = None,
    metrics_endpoint: str | None = None,
    metrics_target_overrides: dict[str, str] | None = None,
    topology_mode: str | None = None,
    session_routing: str | None = None,
    session_header_name: str | None = None,
    cache_strategy: str | None = None,
    cache_tiers: list[str] | None = None,
    cache_connector: str | None = None,
    prefix_caching: bool | None = None,
    session_affinity: bool | None = None,
    metrics_targets: list[ResolvedMetricCaptureTarget] | None = None,
    execution: BenchmarkExecutionProfile | None = None,
    support: BenchmarkSupportProfile | None = None,
) -> BenchmarkRunPlan:
    """Resolve workload, experiment defaults, and runtime overrides into one run plan."""
    selected_model = model or (experiment.model if experiment else None) or workload.model
    if not selected_model:
        raise ValueError("A model must be provided either in the workload pack, experiment, or as an override")

    if concurrency is not None and concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    effective_concurrency = (
        concurrency
        if concurrency is not None
        else ((experiment.concurrency if experiment else None) or workload.concurrency)
    )
    if effective_concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    topology_payload = (
        experiment.topology.model_dump(mode="python")
        if experiment
        else BenchmarkTopologyMetadata().model_dump(mode="python")
    )
    if topology_mode is not None:
        topology_payload["mode"] = topology_mode
    if session_routing is not None:
        topology_payload["session_routing"] = session_routing
    if session_header_name is not None:
        topology_payload["session_header_name"] = session_header_name
    topology = BenchmarkTopologyMetadata.model_validate(topology_payload)

    cache_payload = (
        experiment.cache.model_dump(mode="python") if experiment else BenchmarkCacheMetadata().model_dump(mode="python")
    )
    if cache_strategy is not None:
        cache_payload["strategy"] = cache_strategy
    if cache_tiers is not None:
        cache_payload["tiers"] = cache_tiers
    if cache_connector is not None:
        cache_payload["connector"] = cache_connector
    if prefix_caching is not None:
        cache_payload["prefix_caching"] = prefix_caching
    if session_affinity is not None:
        cache_payload["session_affinity"] = session_affinity
    cache = BenchmarkCacheMetadata.model_validate(cache_payload)

    engine = experiment.engine if experiment else None
    _validate_engine_cache(engine, cache)

    resolved_targets = metrics_targets or _resolve_targets(
        request_endpoint,
        metrics_endpoint,
        metrics_target_overrides or {},
        experiment.metrics_targets if experiment else [],
    )

    if topology.session_routing in {"sticky", "hash"} and any(
        request.session_id is None for request in workload.requests
    ):
        raise ValueError(
            "All workload requests must include session_id when session routing is enabled for the run plan"
        )

    return BenchmarkRunPlan(
        source_experiment=experiment.name if experiment else None,
        workload_ref=workload_ref,
        request_endpoint=request_endpoint,
        model=selected_model,
        concurrency=effective_concurrency,
        engine=engine,
        topology=topology,
        cache=cache,
        execution=execution or BenchmarkExecutionProfile(),
        support=support,
        metrics_targets=resolved_targets,
        tags=list(experiment.tags) if experiment else [],
    )
