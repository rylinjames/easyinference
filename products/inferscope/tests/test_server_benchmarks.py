"""Tests for narrowed MCP-side benchmark probe helpers."""

from __future__ import annotations

import pytest
from fastmcp import FastMCP

from inferscope.benchmarks import BenchmarkArtifact, BenchmarkSummary
from inferscope.server_benchmarks import _resolve_benchmark_plan, register_benchmark_tools


def test_resolve_benchmark_plan_supports_procedural_workloads() -> None:
    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "kimi-k2-long-context-coding",
        "http://localhost:8000",
        synthetic_requests=4,
        synthetic_input_tokens=2048,
        synthetic_output_tokens=256,
    )
    assert error is None
    assert workload_reference == "kimi-k2-long-context-coding"
    assert workload_pack is not None
    assert run_plan is not None
    assert support is not None
    assert len(workload_pack.requests) == 4
    assert run_plan.workload_ref == "kimi-k2-long-context-coding"
    assert run_plan.source_experiment == "dynamo-aggregated-lmcache-kimi-k2"
    assert support.status == "unknown"


def test_resolve_benchmark_plan_rejects_context_file_for_mcp() -> None:
    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "kimi-k2-long-context-coding",
        "http://localhost:8000",
        synthetic_requests=2,
        context_file="repo_context.txt",
    )
    assert error is not None
    assert "context_file is not supported" in error["error"]
    assert workload_reference is None
    assert workload_pack is None
    assert run_plan is None
    assert support is None


@pytest.mark.asyncio
async def test_tool_get_production_contract_returns_default_experiment() -> None:
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool("tool_get_production_contract")
    payload = result.structured_content

    assert payload["evidence"] == "production_target_contract"
    assert payload["production_target"]["default_experiment"] == "dynamo-aggregated-lmcache-kimi-k2"


@pytest.mark.asyncio
async def test_tool_resolve_benchmark_plan_rejects_unsupported_grace_gpu() -> None:
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool(
        "tool_resolve_benchmark_plan",
        {
            "workload": "kimi-k2-long-context-coding",
            "endpoint": "http://localhost:8000",
            "gpu": "gb200",
            "num_gpus": 4,
        },
    )
    payload = result.structured_content

    assert "error" in payload
    assert payload["evidence"] == "production_target_validation"
    assert "H100, H200, B200, B300" in payload["error"]


@pytest.mark.asyncio
async def test_tool_run_benchmark_returns_observed_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_openai_replay(*args, **kwargs):
        del args, kwargs
        return BenchmarkArtifact(
            pack_name="kimi-k2-long-context-coding",
            workload_class="coding",
            endpoint="http://localhost:8000",
            model="Kimi-K2.5",
            concurrency=2,
            started_at="2026-03-25T00:00:00Z",
            completed_at="2026-03-25T00:00:01Z",
            run_plan={"observed_runtime": {"request_throughput_rps": 2.5}},
            results=[],
            summary=BenchmarkSummary(
                total_requests=2,
                succeeded=2,
                failed=0,
                concurrency=2,
                wall_time_ms=1000.0,
                metrics_capture_complete=True,
            ),
        )

    monkeypatch.setattr("inferscope.server_benchmarks.run_openai_replay", fake_run_openai_replay)
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool(
        "tool_run_benchmark",
        {
            "workload": "kimi-k2-long-context-coding",
            "endpoint": "http://localhost:8000",
            "synthetic_requests": 2,
            "synthetic_input_tokens": 2048,
            "synthetic_output_tokens": 256,
            "gpu": "b200",
            "num_gpus": 4,
        },
    )
    payload = result.structured_content

    assert payload["observed_runtime"]["request_throughput_rps"] == 2.5
    assert payload["support"]["gpu_isa"] == "sm_100"
    assert payload["production_readiness"]["ready"] is True
