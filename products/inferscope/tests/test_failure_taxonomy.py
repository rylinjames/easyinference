"""Tests for failure-mode taxonomy helpers."""

from __future__ import annotations

from inferscope.telemetry.failure_taxonomy import FailureMode, classify_failure_modes, dominant_failure_mode
from inferscope.telemetry.normalizer import NormalizedMetrics


def test_classify_failure_modes_detects_multiple_modes() -> None:
    metrics = NormalizedMetrics(
        engine="dynamo",
        endpoint="http://localhost:8000",
        requests_waiting=30,
        queue_time_avg_s=2.5,
        nixl_transfer_failures=2,
        lmcache_hit_rate=0.1,
    )

    findings = classify_failure_modes(metrics)

    assert [finding.mode for finding in findings][:2] == [
        FailureMode.NIXL_FAILURE,
        FailureMode.DECODE_QUEUE_BACKUP,
    ]


def test_dominant_failure_mode_returns_highest_priority() -> None:
    metrics = NormalizedMetrics(
        engine="dynamo",
        endpoint="http://localhost:8000",
        scrape_error="connection refused",
        disconnected_clients=3,
    )

    finding = dominant_failure_mode(metrics)

    assert finding is not None
    assert finding.mode == FailureMode.WORKER_CRASH
