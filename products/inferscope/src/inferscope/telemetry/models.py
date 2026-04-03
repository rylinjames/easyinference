"""Shared telemetry persistence models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


class MetricSampleRecord(BaseModel):
    """A persisted Prometheus sample for artifact inspection."""

    model_config = ConfigDict(extra="forbid")

    name: str
    labels: dict[str, str] = Field(default_factory=dict)
    value: float | None = None


class MetricSnapshot(BaseModel):
    """Captured metrics around a benchmark run or runtime profile."""

    model_config = ConfigDict(extra="forbid")

    captured_at: str = Field(default_factory=utc_now_iso)
    endpoint: str
    engine: str
    target_name: str = "primary"
    target_role: str = "primary"
    expected_engine: str | None = None
    scrape_time_ms: float = 0.0
    error: str = ""
    raw_metrics: dict[str, float] = Field(default_factory=dict)
    normalized_metrics: dict[str, Any] = Field(default_factory=dict)
    samples: list[MetricSampleRecord] = Field(default_factory=list)
