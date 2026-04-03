"""Public models for runtime profiling."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from inferscope.optimization.serving_profile import BottleneckType
from inferscope.telemetry.models import MetricSnapshot


class ProfileSourceKind(StrEnum):
    """Where a profile report came from."""

    PROMETHEUS_RUNTIME = "prometheus_runtime"
    NSYS_TRACE = "nsys_trace"
    ROCPROFV3_TRACE = "rocprofv3_trace"
    KERNEL_TRACE = "kernel_trace"


class RuntimeIdentity(BaseModel):
    """Best-effort runtime identity and config enrichment."""

    model_config = ConfigDict(extra="forbid")

    engine: str
    engine_source: Literal["metrics", "adapter", "unknown"]
    adapter_name: str | None = None
    served_models: list[str] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    config_error: str = ""
    notes: list[str] = Field(default_factory=list)


class RuntimeBottleneck(BaseModel):
    """Grouped bottleneck view derived from lower-level audit findings."""

    model_config = ConfigDict(extra="forbid")

    kind: BottleneckType
    severity: Literal["critical", "warning", "info"]
    confidence: float
    summary: str
    trigger_check_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    supporting_metrics: dict[str, float | None] = Field(default_factory=dict)


class TuningAdjustment(BaseModel):
    """A single recommended parameter change."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str
    confidence: float
    trigger: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
            "trigger": self.trigger,
        }


class TuningPreview(BaseModel):
    """Preview of scheduler/cache changes derived from runtime findings."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["preview"] = "preview"
    adjustments: list[TuningAdjustment] = Field(default_factory=list)
    updated_scheduler: dict[str, Any] = Field(default_factory=dict)
    updated_cache: dict[str, Any] = Field(default_factory=dict)
    summary: str
    error: str = ""


class RuntimeContextHints(BaseModel):
    """Optional deployment hints that enrich runtime analysis."""

    model_config = ConfigDict(extra="forbid")

    engine: str = ""
    gpu_arch: str = ""
    gpu_name: str = ""
    model_name: str = ""
    model_type: str = ""
    attention_type: str = ""
    experts_total: int = 0
    tp: int = 1
    ep: int = 0
    quantization: str = ""
    kv_cache_dtype: str = ""
    gpu_memory_utilization: float = 0.0
    block_size: int = 0
    has_rdma: bool = False
    split_prefill_decode: bool = False


class RuntimeProfileReport(BaseModel):
    """Unified MCP/CLI runtime profiling report."""

    model_config = ConfigDict(extra="forbid")

    profile_version: str = "1"
    source_kind: ProfileSourceKind = ProfileSourceKind.PROMETHEUS_RUNTIME
    endpoint: str
    metrics_snapshot: MetricSnapshot
    metrics: dict[str, Any]
    health: dict[str, Any]
    memory_pressure: dict[str, Any]
    cache_effectiveness: dict[str, Any]
    reliability: dict[str, Any]
    workload: dict[str, Any]
    identity: RuntimeIdentity | None = None
    audit: dict[str, Any]
    bottlenecks: list[RuntimeBottleneck] = Field(default_factory=list)
    tuning_preview: TuningPreview | None = None
    profiling_intent: dict[str, Any] | None = None
    reasoning: list[str] = Field(default_factory=list)
    summary: str
    confidence: float
    evidence: str = "runtime_profile"
