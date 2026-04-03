"""Shared Prometheus capture helpers for benchmarks and runtime profiling."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.telemetry.models import MetricSampleRecord, MetricSnapshot
from inferscope.telemetry.normalizer import NormalizedMetrics, normalize
from inferscope.telemetry.prometheus import MetricSample, scrape_metrics


class MetricCaptureTarget(Protocol):
    """Structural typing for benchmark/runtime metric targets."""

    name: str
    role: str
    endpoint: str
    expected_engine: str | None


@dataclass(frozen=True)
class CapturedTelemetry:
    """A raw snapshot plus the typed normalized metrics used by analysis."""

    snapshot: MetricSnapshot
    normalized: NormalizedMetrics


def _persistable_samples(samples: list[MetricSample]) -> list[MetricSampleRecord]:
    persisted: list[MetricSampleRecord] = []
    for sample in samples:
        if sample.name.endswith("_bucket"):
            continue
        persisted.append(
            MetricSampleRecord(
                name=sample.name,
                labels=dict(sample.labels),
                value=sample.value,
            )
        )
    return persisted


async def capture_endpoint_telemetry(
    endpoint: str,
    allow_private: bool = True,
    *,
    target_name: str = "primary",
    target_role: str = "primary",
    expected_engine: str | None = None,
    metrics_auth: EndpointAuthConfig | None = None,
    include_samples: bool = True,
) -> CapturedTelemetry:
    """Capture raw and normalized metrics for one endpoint."""
    scrape = await scrape_metrics(endpoint, allow_private=allow_private, auth=metrics_auth)
    normalized = normalize(scrape)
    error = scrape.error
    if expected_engine and scrape.engine not in {"unknown", expected_engine}:
        mismatch = f"expected engine '{expected_engine}' but scraped '{scrape.engine}'"
        error = f"{error} | {mismatch}" if error else mismatch
    snapshot = MetricSnapshot(
        endpoint=scrape.endpoint,
        engine=scrape.engine,
        target_name=target_name,
        target_role=target_role,
        expected_engine=expected_engine,
        scrape_time_ms=scrape.scrape_time_ms,
        error=error,
        raw_metrics=dict(scrape.raw_metrics),
        normalized_metrics=normalized.to_dict(),
        samples=_persistable_samples(scrape.samples) if include_samples else [],
    )
    return CapturedTelemetry(snapshot=snapshot, normalized=normalized)


async def capture_endpoint_snapshot(
    endpoint: str,
    allow_private: bool = True,
    *,
    target_name: str = "primary",
    target_role: str = "primary",
    expected_engine: str | None = None,
    metrics_auth: EndpointAuthConfig | None = None,
    include_samples: bool = True,
) -> MetricSnapshot:
    """Compatibility wrapper returning just the snapshot."""
    captured = await capture_endpoint_telemetry(
        endpoint,
        allow_private=allow_private,
        target_name=target_name,
        target_role=target_role,
        expected_engine=expected_engine,
        metrics_auth=metrics_auth,
        include_samples=include_samples,
    )
    return captured.snapshot


async def capture_metrics_targets(
    targets: Sequence[MetricCaptureTarget],
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
    metrics_auth_overrides: dict[str, EndpointAuthConfig] | None = None,
    include_samples: bool = True,
) -> list[MetricSnapshot]:
    """Capture all metrics targets concurrently while preserving declared order."""
    tasks = [
        capture_endpoint_snapshot(
            target.endpoint,
            allow_private=allow_private,
            target_name=target.name,
            target_role=target.role,
            expected_engine=target.expected_engine,
            metrics_auth=(metrics_auth_overrides or {}).get(target.name, metrics_auth),
            include_samples=include_samples,
        )
        for target in targets
    ]
    return list(await asyncio.gather(*tasks))
