"""Regression tests for legacy runtime analysis surfaces after refactoring."""

from __future__ import annotations

import pytest

from inferscope.optimization.checks import AuditFinding
from inferscope.optimization.serving_profile import WorkloadMode
from inferscope.optimization.workload_classifier import WorkloadClassification
from inferscope.profiling.models import TuningAdjustment, TuningPreview
from inferscope.profiling.runtime import RuntimeAnalysisBundle
from inferscope.telemetry.models import MetricSnapshot
from inferscope.telemetry.normalizer import NormalizedMetrics
from inferscope.tools.audit import audit_deployment
from inferscope.tools.diagnose import (
    check_deployment,
    check_memory_pressure,
    get_cache_effectiveness,
)
from inferscope.tools.live_tuner import auto_tune_deployment


def _finding(check_id: str, severity: str = "warning") -> AuditFinding:
    return AuditFinding(
        check_id=check_id,
        severity=severity,
        title=f"{check_id} title",
        description="desc",
        current_value="current",
        recommended_value="recommended",
        fix_command="fix",
        confidence=0.8,
        evidence="threshold_rule",
    )


def _bundle() -> RuntimeAnalysisBundle:
    metrics = NormalizedMetrics(
        engine="vllm",
        endpoint="http://localhost:8000/metrics",
        requests_running=4,
        requests_waiting=2,
        kv_cache_usage=0.72,
        prefix_cache_hit_rate=0.44,
    )
    snapshot = MetricSnapshot(endpoint=metrics.endpoint, engine=metrics.engine, normalized_metrics=metrics.to_dict())
    findings = [_finding("HIGH_TTFT", severity="critical")]
    preview = TuningPreview(
        adjustments=[
            TuningAdjustment(
                parameter="scheduler.chunked_prefill",
                current_value=True,
                recommended_value=False,
                reason="reason",
                confidence=0.7,
                trigger="HIGH_TTFT",
            )
        ],
        updated_scheduler={"chunked_prefill": False},
        updated_cache={"gpu_memory_utilization": 0.92},
        summary="1 config adjustment(s) suggested",
    )
    return RuntimeAnalysisBundle(
        snapshot=snapshot,
        normalized=metrics,
        health={"status": "healthy", "issues": [], "issue_count": 0},
        memory_pressure={
            "level": "moderate",
            "kv_cache_usage": 0.72,
            "prefix_cache_hit_rate": 0.44,
            "preemptions_total": 0,
            "cpu_cache_usage": 0.0,
            "findings": ["KV cache healthy"],
            "summary": "Memory pressure: moderate",
        },
        cache_effectiveness={
            "effectiveness": "good",
            "prefix_hit_rate": 0.44,
            "kv_cache_usage": 0.72,
            "recommendations": ["Canonicalize prompts"],
            "summary": "Cache effectiveness: good",
        },
        workload=WorkloadClassification(mode=WorkloadMode.CODING, confidence=0.8, signals={"prefix": "high"}),
        findings=findings,
        tuning_preview=preview,
    )


@pytest.mark.asyncio
async def test_audit_deployment_response_shape_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_analyze_runtime(*args, **kwargs) -> RuntimeAnalysisBundle:
        del args, kwargs
        return _bundle()

    monkeypatch.setattr("inferscope.tools.audit.analyze_runtime", fake_analyze_runtime)

    result = await audit_deployment("http://localhost:8000", gpu_arch="sm_90a")

    assert set(result) == {"audit", "workload", "metrics", "engine", "endpoint", "summary", "confidence", "evidence"}
    assert result["audit"]["total"] == 1
    assert result["workload"]["mode"] == "coding"


@pytest.mark.asyncio
async def test_auto_tune_deployment_response_shape_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_analyze_runtime(*args, **kwargs) -> RuntimeAnalysisBundle:
        del args, kwargs
        return _bundle()

    monkeypatch.setattr("inferscope.tools.live_tuner.analyze_runtime", fake_analyze_runtime)

    result = await auto_tune_deployment("http://localhost:8000")

    assert set(result) == {
        "detections",
        "adjustments",
        "updated_scheduler",
        "updated_cache",
        "reasoning",
        "metrics_snapshot",
        "summary",
        "confidence",
        "evidence",
    }
    assert len(result["detections"]) == 1
    assert len(result["adjustments"]) == 1


@pytest.mark.asyncio
async def test_diagnose_surfaces_keep_existing_response_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_analyze_runtime(*args, **kwargs) -> RuntimeAnalysisBundle:
        del args, kwargs
        return _bundle()

    monkeypatch.setattr("inferscope.tools.diagnose.analyze_runtime", fake_analyze_runtime)

    health = await check_deployment("http://localhost:8000")
    memory = await check_memory_pressure("http://localhost:8000")
    cache = await get_cache_effectiveness("http://localhost:8000")

    assert set(health) == {"metrics", "health", "summary", "confidence", "evidence"}
    assert set(memory) == {"memory_pressure", "findings", "engine", "summary", "confidence", "evidence"}
    assert set(cache) == {"cache", "recommendations", "engine", "summary", "confidence", "evidence"}
