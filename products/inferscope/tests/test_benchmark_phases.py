"""Tests for live benchmark phase execution helpers."""

from __future__ import annotations

import pytest

from inferscope.benchmarks.kv_cache_behavior import run_kv_cache_behavior
from inferscope.benchmarks.kv_capacity_probe import run_kv_capacity_probe
from inferscope.benchmarks.kv_disagg_transfer import run_kv_disagg_transfer
from inferscope.benchmarks.kv_pressure_ramp import run_kv_pressure_ramp
from inferscope.benchmarks.models import BenchmarkArtifact, BenchmarkRequestResult, BenchmarkSummary, MetricSnapshot


def _artifact(
    *,
    pack_name: str,
    ttft_avg_ms: float = 100.0,
    ttft_p99_ms: float = 120.0,
    failed: int = 0,
    total_requests: int = 1,
    normalized_metrics: dict | None = None,
) -> BenchmarkArtifact:
    return BenchmarkArtifact(
        pack_name=pack_name,
        workload_class=pack_name,
        endpoint="http://localhost:8000",
        model="Kimi-K2.5",
        concurrency=total_requests,
        started_at="2026-04-04T00:00:00Z",
        completed_at="2026-04-04T00:00:01Z",
        metrics_after=MetricSnapshot(
            endpoint="http://localhost:8000/metrics",
            engine="dynamo",
            normalized_metrics=normalized_metrics or {},
        ),
        results=[
            BenchmarkRequestResult(
                name="req-1",
                status="ok" if failed == 0 else "error",
                started_at="2026-04-04T00:00:00Z",
                completed_at="2026-04-04T00:00:01Z",
                elapsed_ms=100.0,
                ttft_ms=ttft_avg_ms,
                status_code=200 if failed == 0 else 500,
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                error="" if failed == 0 else "boom",
            )
        ],
        summary=BenchmarkSummary(
            total_requests=total_requests,
            succeeded=total_requests - failed,
            failed=failed,
            concurrency=total_requests,
            wall_time_ms=1000.0,
            ttft_avg_ms=ttft_avg_ms,
            ttft_p99_ms=ttft_p99_ms,
            metrics_capture_complete=True,
        ),
    )


@pytest.mark.asyncio
async def test_run_kv_capacity_probe_produces_capacity_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_openai_replay(workload, *args, **kwargs):
        del args, kwargs
        concurrency = workload.concurrency
        failed = 1 if concurrency > 4 else 0
        return _artifact(
            pack_name=workload.name,
            failed=failed,
            total_requests=concurrency,
            normalized_metrics={"cache": {"kv_usage": 0.75}},
        )

    monkeypatch.setattr("inferscope.benchmarks.kv_capacity_probe.run_openai_replay", fake_run_openai_replay)

    result = await run_kv_capacity_probe(
        "http://localhost:8000",
        "Kimi-K2.5",
        isl_list=[1024, 2048],
        concurrency_candidates=[1, 2, 4, 8],
    )

    assert [point.max_concurrent for point in result.capacity_curve] == [4, 4]


@pytest.mark.asyncio
async def test_run_kv_pressure_ramp_returns_points(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_openai_replay(workload, *args, **kwargs):
        del args, kwargs
        return _artifact(
            pack_name=workload.name,
            total_requests=workload.concurrency,
            normalized_metrics={
                "cache": {"kv_usage": 0.8},
                "latency": {"itl_avg_ms": 22.0},
                "request_state": {"waiting": 3},
                "reliability": {"kv_active_blocks": 128},
                "throughput": {"gen_throughput_tps": 250.0},
            },
        )

    monkeypatch.setattr("inferscope.benchmarks.kv_pressure_ramp.run_openai_replay", fake_run_openai_replay)

    result = await run_kv_pressure_ramp(
        "http://localhost:8000",
        "Kimi-K2.5",
        isl=4096,
        base_capacity=8,
        pressure_levels=[0.5, 1.0],
    )

    assert len(result.pressure_points) == 2
    assert result.pressure_points[0].throughput_tps == 250.0


@pytest.mark.asyncio
async def test_run_kv_cache_behavior_returns_all_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = {"count": 0}

    async def fake_run_openai_replay(workload, *args, **kwargs):
        del args, kwargs
        counter["count"] += 1
        return _artifact(
            pack_name=workload.name,
            ttft_avg_ms=100.0 + counter["count"],
            normalized_metrics={"cache": {"prefix_hit_rate": 0.6}},
        )

    monkeypatch.setattr("inferscope.benchmarks.kv_cache_behavior.run_openai_replay", fake_run_openai_replay)

    result = await run_kv_cache_behavior("http://localhost:8000", "Kimi-K2.5")

    assert result.cold_start is not None
    assert len(result.prefix_reuse) == 3
    assert len(result.session_growth) == 5


@pytest.mark.asyncio
async def test_run_kv_disagg_transfer_returns_transfer_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_openai_replay(workload, *args, **kwargs):
        del args, kwargs
        isl = workload.requests[0].metadata["isl"]
        return _artifact(
            pack_name=workload.name,
            normalized_metrics={
                "disaggregation": {
                    "nixl_transfer_latency_ms": isl / 100.0,
                    "transfer_bandwidth_gbps": 120.0,
                    "decode_idle_fraction": 0.2,
                    "nixl_transfer_failures": 0.0,
                    "kvbm_offload_d2h": 1.0,
                    "kvbm_onboard_h2d": 2.0,
                }
            },
        )

    monkeypatch.setattr("inferscope.benchmarks.kv_disagg_transfer.run_openai_replay", fake_run_openai_replay)

    result = await run_kv_disagg_transfer(
        "http://localhost:8000",
        "Kimi-K2.5",
        topology="prefill-decode-split",
        isl_list=[4096, 8192],
    )

    assert len(result.transfer_curve) == 2
    assert result.avg_decode_idle_fraction == 0.2
