"""Prometheus capture helpers for benchmark runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from inferscope.benchmarks.experiments import ResolvedMetricCaptureTarget
from inferscope.benchmarks.models import MetricSnapshot
from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.telemetry.capture import (
    MetricCaptureTarget,
)
from inferscope.telemetry.capture import (
    capture_endpoint_snapshot as _capture_endpoint_snapshot,
)
from inferscope.telemetry.capture import (
    capture_metrics_targets as _capture_metrics_targets,
)


async def capture_endpoint_snapshot(
    endpoint: str,
    allow_private: bool = True,
    *,
    target_name: str = "primary",
    target_role: str = "primary",
    expected_engine: str | None = None,
    metrics_auth: EndpointAuthConfig | None = None,
) -> MetricSnapshot:
    """Capture raw and normalized metrics for one endpoint."""
    return await _capture_endpoint_snapshot(
        endpoint,
        allow_private=allow_private,
        target_name=target_name,
        target_role=target_role,
        expected_engine=expected_engine,
        metrics_auth=metrics_auth,
    )


async def capture_metrics_targets(
    targets: Sequence[ResolvedMetricCaptureTarget],
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
    metrics_auth_overrides: dict[str, EndpointAuthConfig] | None = None,
) -> list[MetricSnapshot]:
    """Capture all metrics targets concurrently while preserving declared order."""
    return await _capture_metrics_targets(
        cast(Sequence[MetricCaptureTarget], targets),
        allow_private=allow_private,
        metrics_auth=metrics_auth,
        metrics_auth_overrides=metrics_auth_overrides,
    )
