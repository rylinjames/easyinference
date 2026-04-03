"""Tests for shared runtime profiling orchestration."""

from __future__ import annotations

import pytest

from inferscope.optimization.checks import AuditFinding
from inferscope.optimization.serving_profile import BottleneckType, WorkloadMode
from inferscope.optimization.workload_classifier import WorkloadClassification
from inferscope.profiling.models import RuntimeContextHints, RuntimeIdentity
from inferscope.profiling.runtime import build_runtime_profile, derive_bottlenecks
from inferscope.telemetry.capture import CapturedTelemetry
from inferscope.telemetry.models import MetricSnapshot
from inferscope.telemetry.normalizer import NormalizedMetrics


def _normalized_metrics() -> NormalizedMetrics:
    return NormalizedMetrics(
        engine="vllm",
        endpoint="http://localhost:8000/metrics",
        requests_running=8,
        requests_waiting=24,
        kv_cache_usage=0.94,
        prefix_cache_hit_rate=0.12,
        cpu_cache_usage=0.42,
        preemptions_total=18,
        gen_throughput_tps=1800,
        ttft_avg_s=6.0,
        itl_avg_s=0.12,
    )


def _captured_telemetry() -> CapturedTelemetry:
    metrics = _normalized_metrics()
    snapshot = MetricSnapshot(
        endpoint=metrics.endpoint,
        engine=metrics.engine,
        raw_metrics={"vllm:num_requests_running": metrics.requests_running},
        normalized_metrics=metrics.to_dict(),
    )
    return CapturedTelemetry(snapshot=snapshot, normalized=metrics)


def _finding(check_id: str, severity: str = "warning", title: str = "Title") -> AuditFinding:
    return AuditFinding(
        check_id=check_id,
        severity=severity,
        title=title,
        description="desc",
        current_value="current",
        recommended_value="recommended",
        fix_command="fix",
        confidence=0.8,
        evidence="threshold_rule",
    )


@pytest.mark.asyncio
async def test_build_runtime_profile_returns_unified_report(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_capture_endpoint_telemetry(*args, **kwargs) -> CapturedTelemetry:
        del args, kwargs
        return _captured_telemetry()

    async def fake_enrich_runtime_identity(*args, **kwargs) -> RuntimeIdentity:
        del args, kwargs
        return RuntimeIdentity(
            engine="vllm",
            engine_source="metrics",
            adapter_name="vllm",
            served_models=["DeepSeek-R1"],
            config_snapshot={"data": [{"id": "DeepSeek-R1"}]},
        )

    monkeypatch.setattr("inferscope.profiling.runtime.capture_endpoint_telemetry", fake_capture_endpoint_telemetry)
    monkeypatch.setattr("inferscope.profiling.runtime.enrich_runtime_identity", fake_enrich_runtime_identity)
    monkeypatch.setattr(
        "inferscope.profiling.runtime.classify_workload",
        lambda metrics: WorkloadClassification(
            mode=WorkloadMode.CODING,
            confidence=0.8,
            signals={"prefix_reuse": "high"},
        ),
    )
    monkeypatch.setattr(
        "inferscope.profiling.runtime.run_all_checks",
        lambda metrics, ctx: [
            _finding("HIGH_TTFT", severity="critical", title="High TTFT"),
            _finding("KV_CACHE_CRITICAL", severity="warning", title="KV cache critical"),
        ],
    )

    report = await build_runtime_profile(
        "http://localhost:8000",
        context_hints=RuntimeContextHints(gpu_arch="sm_90a"),
    )

    assert report.endpoint == "http://localhost:8000"
    assert report.metrics_snapshot.endpoint == "http://localhost:8000/metrics"
    assert report.metrics["engine"] == "vllm"
    assert report.workload["mode"] == "coding"
    assert report.audit["total"] == 2
    assert len(report.bottlenecks) == 2
    assert report.identity is not None
    assert report.identity.served_models == ["DeepSeek-R1"]
    assert report.profiling_intent is not None
    assert report.profiling_intent["tool"] == "nsys"


def test_derive_bottlenecks_uses_stable_enum() -> None:
    metrics = _normalized_metrics()
    bottlenecks = derive_bottlenecks(
        [
            _finding("HIGH_TTFT", severity="critical", title="High TTFT"),
            _finding("KV_CACHE_CRITICAL", severity="warning", title="KV cache critical"),
        ],
        metrics,
    )

    assert [bottleneck.kind for bottleneck in bottlenecks] == [
        BottleneckType.PREFILL_COMPUTE,
        BottleneckType.CACHE_BOUND,
    ]
    assert all(isinstance(bottleneck.kind, BottleneckType) for bottleneck in bottlenecks)


@pytest.mark.asyncio
async def test_runtime_profile_degrades_when_identity_enrichment_has_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_capture_endpoint_telemetry(*args, **kwargs) -> CapturedTelemetry:
        del args, kwargs
        return _captured_telemetry()

    async def degraded_identity(*args, **kwargs) -> RuntimeIdentity:
        del args, kwargs
        return RuntimeIdentity(
            engine="vllm",
            engine_source="metrics",
            adapter_name="vllm",
            config_error="adapter lookup failed",
        )

    monkeypatch.setattr("inferscope.profiling.runtime.capture_endpoint_telemetry", fake_capture_endpoint_telemetry)
    monkeypatch.setattr("inferscope.profiling.runtime.enrich_runtime_identity", degraded_identity)
    monkeypatch.setattr(
        "inferscope.profiling.runtime.classify_workload",
        lambda metrics: WorkloadClassification(
            mode=WorkloadMode.CHAT,
            confidence=0.7,
            signals={"queue": "moderate"},
        ),
    )
    monkeypatch.setattr("inferscope.profiling.runtime.run_all_checks", lambda metrics, ctx: [])

    report = await build_runtime_profile(
        "http://localhost:8000",
        context_hints=RuntimeContextHints(gpu_arch="sm_90a"),
    )

    assert report.identity is not None
    assert report.identity.config_error == "adapter lookup failed"
    assert report.confidence < 0.95


@pytest.mark.asyncio
async def test_runtime_profile_respects_tuning_and_raw_metrics_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_capture_endpoint_telemetry(*args, **kwargs) -> CapturedTelemetry:
        del args, kwargs
        return _captured_telemetry()

    monkeypatch.setattr("inferscope.profiling.runtime.capture_endpoint_telemetry", fake_capture_endpoint_telemetry)
    monkeypatch.setattr("inferscope.profiling.runtime.run_all_checks", lambda metrics, ctx: [_finding("HIGH_TTFT")])
    monkeypatch.setattr(
        "inferscope.profiling.runtime.classify_workload",
        lambda metrics: WorkloadClassification(mode=WorkloadMode.CHAT, confidence=0.6, signals={}),
    )

    report = await build_runtime_profile(
        "http://localhost:8000",
        include_identity=False,
        include_tuning_preview=False,
        include_raw_metrics=False,
    )

    assert report.identity is None
    assert report.tuning_preview is None
    assert report.metrics_snapshot.raw_metrics == {}
    assert report.metrics_snapshot.normalized_metrics == report.metrics
