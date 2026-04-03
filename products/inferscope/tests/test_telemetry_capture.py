"""Tests for shared telemetry capture helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inferscope.telemetry.capture import capture_endpoint_snapshot, capture_metrics_targets
from inferscope.telemetry.prometheus import MetricSample, ScrapeResult


def _make_scrape(*, engine: str = "vllm") -> ScrapeResult:
    return ScrapeResult(
        endpoint="http://localhost:8000/metrics",
        engine=engine,
        raw_metrics={
            "vllm:num_requests_running": 2.0,
            "vllm:time_to_first_token_seconds_sum": 0.8,
            "vllm:time_to_first_token_seconds_count": 2.0,
        },
        samples=[
            MetricSample(name="vllm:num_requests_running", labels={}, value=2.0),
            MetricSample(name="vllm:time_to_first_token_seconds_bucket", labels={"le": "1"}, value=1.0),
        ],
        scrape_time_ms=12.5,
    )


def _make_dynamo_scrape() -> ScrapeResult:
    return ScrapeResult(
        endpoint="http://localhost:8000/metrics",
        engine="dynamo",
        raw_metrics={
            "dynamo_frontend_inflight_requests": 3.0,
            "dynamo_frontend_queued_requests": 2.0,
            "dynamo_frontend_model_migration_total": 1.0,
            "dynamo_frontend_time_to_first_token_seconds_sum": 1.2,
            "dynamo_frontend_time_to_first_token_seconds_count": 3.0,
            "dynamo_component_kvstats_gpu_cache_usage_percent": 0.82,
            "dynamo_component_kvstats_gpu_prefix_cache_hit_rate": 0.71,
            "dynamo_component_kvstats_active_blocks": 820.0,
            "dynamo_component_kvstats_total_blocks": 1000.0,
        },
        samples=[
            MetricSample(name="dynamo_frontend_inflight_requests", labels={}, value=3.0),
            MetricSample(name="dynamo_component_kvstats_gpu_cache_usage_percent", labels={}, value=0.82),
        ],
        scrape_time_ms=9.5,
    )


@pytest.mark.asyncio
async def test_capture_endpoint_snapshot_preserves_benchmark_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_scrape_metrics(endpoint: str, allow_private: bool = True, auth=None) -> ScrapeResult:
        del endpoint, allow_private, auth
        return _make_scrape()

    monkeypatch.setattr("inferscope.telemetry.capture.scrape_metrics", fake_scrape_metrics)

    snapshot = await capture_endpoint_snapshot("http://localhost:8000")

    assert snapshot.endpoint == "http://localhost:8000/metrics"
    assert snapshot.engine == "vllm"
    assert snapshot.target_name == "primary"
    assert snapshot.normalized_metrics["request_state"]["running"] == 2.0
    assert len(snapshot.samples) == 1
    assert snapshot.samples[0].name == "vllm:num_requests_running"


@pytest.mark.asyncio
async def test_capture_endpoint_snapshot_appends_expected_engine_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_scrape_metrics(endpoint: str, allow_private: bool = True, auth=None) -> ScrapeResult:
        del endpoint, allow_private, auth
        return _make_scrape(engine="vllm")

    monkeypatch.setattr("inferscope.telemetry.capture.scrape_metrics", fake_scrape_metrics)

    snapshot = await capture_endpoint_snapshot(
        "http://localhost:8000",
        expected_engine="sglang",
    )

    assert "expected engine 'sglang' but scraped 'vllm'" in snapshot.error


@pytest.mark.asyncio
async def test_capture_metrics_targets_preserves_declared_order(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_capture_endpoint_snapshot(
        endpoint: str,
        allow_private: bool = True,
        *,
        target_name: str = "primary",
        target_role: str = "primary",
        expected_engine: str | None = None,
        metrics_auth=None,
        include_samples: bool = True,
    ):
        del endpoint, allow_private, expected_engine, metrics_auth, include_samples
        return SimpleNamespace(target_name=target_name, target_role=target_role)

    monkeypatch.setattr("inferscope.telemetry.capture.capture_endpoint_snapshot", fake_capture_endpoint_snapshot)

    targets = [
        SimpleNamespace(name="router", role="router", endpoint="http://router", expected_engine="sglang"),
        SimpleNamespace(name="primary", role="primary", endpoint="http://primary", expected_engine="vllm"),
    ]

    snapshots = await capture_metrics_targets(targets)

    assert [snapshot.target_name for snapshot in snapshots] == ["router", "primary"]


@pytest.mark.asyncio
async def test_capture_endpoint_snapshot_normalizes_dynamo_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_scrape_metrics(endpoint: str, allow_private: bool = True, auth=None) -> ScrapeResult:
        del endpoint, allow_private, auth
        return _make_dynamo_scrape()

    monkeypatch.setattr("inferscope.telemetry.capture.scrape_metrics", fake_scrape_metrics)

    snapshot = await capture_endpoint_snapshot("http://localhost:8000")

    assert snapshot.engine == "dynamo"
    assert snapshot.normalized_metrics["request_state"]["running"] == 3.0
    assert snapshot.normalized_metrics["cache"]["kv_usage"] == 0.82
    assert snapshot.normalized_metrics["reliability"]["request_migrations_total"] == 1.0
