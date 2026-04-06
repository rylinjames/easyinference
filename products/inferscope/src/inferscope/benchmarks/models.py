"""Typed workload packs and benchmark artifacts."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from inferscope.models.registry import get_model_variant
from inferscope.optimization.serving_profile import ModelClass
from inferscope.telemetry.models import MetricSampleRecord, MetricSnapshot

__all__ = [
    "BenchmarkArtifact",
    "BenchmarkArtifactProvenance",
    "BenchmarkLaneReference",
    "BenchmarkRequestResult",
    "BenchmarkSourceReference",
    "BenchmarkSummary",
    "ChatMessage",
    "MetricSampleRecord",
    "MetricSnapshot",
    "WorkloadPack",
    "WorkloadRequest",
]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def slugify(value: str) -> str:
    """Create a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "benchmark"


def sanitize_for_json(value: Any) -> Any:
    """Convert non-finite floats into JSON-safe values."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def _known_model_classes() -> set[str]:
    return {model_class.value for model_class in ModelClass}


class ChatMessage(BaseModel):
    """Minimal OpenAI-compatible chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class WorkloadRequest(BaseModel):
    """One request in a replay pack."""

    model_config = ConfigDict(extra="forbid")

    name: str
    messages: list[ChatMessage]
    session_id: str | None = None
    max_tokens: int = Field(default=512, ge=1, le=131072)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    think_time_ms: int = Field(default=0, ge=0, le=3_600_000)

    @model_validator(mode="after")
    def validate_messages(self) -> WorkloadRequest:
        if not self.messages:
            raise ValueError("WorkloadRequest.messages must contain at least one message")
        return self


class WorkloadPack(BaseModel):
    """A replayable workload pack."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    name: str
    description: str = ""
    workload_class: str
    benchmark_role: str = "operator_extension"
    target_gpu_families: list[str] = Field(default_factory=list)
    target_model_classes: list[str] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)
    model: str | None = None
    endpoint_path: str = "/v1/chat/completions"
    concurrency: int = Field(default=1, ge=1, le=1024)
    stream: bool = True
    tags: list[str] = Field(default_factory=list)
    requests: list[WorkloadRequest]

    @model_validator(mode="after")
    def validate_requests(self) -> WorkloadPack:
        if not self.requests:
            raise ValueError("WorkloadPack.requests must contain at least one request")
        if not self.endpoint_path.startswith("/"):
            raise ValueError("WorkloadPack.endpoint_path must start with '/'")
        invalid_model_classes = sorted(set(self.target_model_classes) - _known_model_classes())
        if invalid_model_classes:
            raise ValueError(
                "WorkloadPack.target_model_classes contains unknown classes: "
                f"{', '.join(invalid_model_classes)}"
            )
        if self.model:
            variant = get_model_variant(self.model)
            if variant is None:
                raise ValueError(
                    "WorkloadPack.model "
                    f"'{self.model}' does not resolve to a known model variant. "
                    "Register the model first or fix the workload metadata."
                )
            if self.target_model_classes and variant.model_class.value not in self.target_model_classes:
                raise ValueError(
                    "WorkloadPack.model "
                    f"'{variant.name}' resolves to class '{variant.model_class.value}', "
                    "which is not allowed by target_model_classes."
                )
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> WorkloadPack:
        """Load a workload pack from YAML or JSON."""
        file_path = Path(path)
        raw = file_path.read_text()
        suffix = file_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(raw)
        elif suffix == ".json":
            data = json.loads(raw)
        else:
            raise ValueError(f"Unsupported workload file type: {suffix}")
        if not isinstance(data, dict):
            raise ValueError("Workload pack file must contain a mapping/object at the top level")
        return cls.model_validate(data)

    def save(self, path: str | Path) -> Path:
        """Write the workload pack back to disk."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude_none=True)
        suffix = file_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            file_path.write_text(yaml.safe_dump(sanitize_for_json(payload), sort_keys=False))
        elif suffix == ".json":
            file_path.write_text(json.dumps(sanitize_for_json(payload), indent=2, allow_nan=False))
        else:
            raise ValueError(f"Unsupported workload file type: {suffix}")
        return file_path


class BenchmarkRequestResult(BaseModel):
    """Observed result for one replayed request."""

    model_config = ConfigDict(extra="forbid")

    name: str
    session_id: str | None = None
    status: Literal["ok", "error"]
    started_at: str
    completed_at: str
    elapsed_ms: float
    ttft_ms: float | None = None
    status_code: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: str = ""


class BenchmarkSummary(BaseModel):
    """Rollup metrics for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    total_requests: int
    succeeded: int
    failed: int
    concurrency: int
    wall_time_ms: float
    latency_avg_ms: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    ttft_avg_ms: float | None = None
    ttft_p90_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metrics_targets_total: int = 0
    metrics_targets_with_errors: int = 0
    metrics_capture_complete: bool = True


class BenchmarkSourceReference(BaseModel):
    """How a workload or experiment spec was resolved for this run."""

    model_config = ConfigDict(extra="forbid")

    reference: str
    resolved_path: str
    source_kind: Literal["builtin", "file"]


class BenchmarkLaneReference(BaseModel):
    """Product-facing lane metadata attached to a resolved benchmark run."""

    model_config = ConfigDict(extra="forbid")

    class_name: Literal[
        "production_validated",
        "benchmark_supported",
        "preview_smoke",
    ]
    claim_scope: Literal[
        "production_comparable",
        "benchmark_comparison_only",
        "toolchain_validation_only",
    ]
    model_support_tier: str
    production_target_name: str | None = None
    workload_pack: str
    experiment: str | None = None
    summary: str
    warnings: list[str] = Field(default_factory=list)


class BenchmarkArtifactProvenance(BaseModel):
    """Stable provenance bundle for artifact interpretation and export."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    workload: BenchmarkSourceReference
    experiment: BenchmarkSourceReference | None = None
    lane: BenchmarkLaneReference | None = None


class BenchmarkArtifact(BaseModel):
    """Serializable artifact for one benchmark run."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: str = "2"
    benchmark_id: str = Field(default_factory=lambda: uuid4().hex)
    pack_name: str
    workload_class: str
    endpoint: str
    metrics_endpoint: str | None = None
    model: str
    concurrency: int
    started_at: str
    completed_at: str
    run_plan: dict[str, Any] | None = None
    provenance: BenchmarkArtifactProvenance | None = None
    metrics_before: MetricSnapshot | None = None
    metrics_after: MetricSnapshot | None = None
    metrics_before_targets: list[MetricSnapshot] = Field(default_factory=list)
    metrics_after_targets: list[MetricSnapshot] = Field(default_factory=list)
    results: list[BenchmarkRequestResult]
    summary: BenchmarkSummary

    def save_json(self, path: str | Path) -> Path:
        """Write the artifact to disk."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = sanitize_for_json(self.model_dump(mode="json"))
        file_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
        return file_path

    @property
    def default_filename(self) -> str:
        """Stable artifact filename."""
        return f"{self.benchmark_id}-{slugify(self.pack_name)}.json"
