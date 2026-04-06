"""Tests for narrowed MCP-side benchmark probe helpers."""

from __future__ import annotations

import pytest
from fastmcp import FastMCP

from inferscope.benchmarks import (
    BenchmarkArtifact,
    BenchmarkArtifactProvenance,
    BenchmarkLaneReference,
    BenchmarkSourceReference,
    BenchmarkSummary,
)
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
    assert run_plan.workload_source is not None
    assert run_plan.workload_source.source_kind == "builtin"
    assert run_plan.experiment_source is not None
    assert run_plan.reference_lane is not None
    assert run_plan.reference_lane.class_name == "production_validated"
    assert support.status == "unknown"


def test_resolve_benchmark_plan_defaults_public_single_endpoint_lane() -> None:
    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "coding-long-context",
        "http://localhost:8000",
        gpu="h100",
        num_gpus=1,
    )
    assert error is None
    assert workload_reference == "coding-long-context"
    assert workload_pack is not None
    assert run_plan is not None
    assert support is not None
    assert run_plan.source_experiment == "vllm-single-endpoint-baseline"
    assert run_plan.model == "Qwen3.5-32B"
    assert run_plan.reference_lane is not None
    assert run_plan.reference_lane.class_name == "benchmark_supported"
    assert support.status == "degraded"


def test_resolve_benchmark_plan_defaults_budget_smoke_lane() -> None:
    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "coding-smoke",
        "http://localhost:8000",
        gpu="a10g",
        num_gpus=1,
    )
    assert error is None
    assert workload_reference == "coding-smoke"
    assert workload_pack is not None
    assert run_plan is not None
    assert support is not None
    assert run_plan.source_experiment == "vllm-single-endpoint-smoke"
    assert run_plan.model == "Qwen2.5-7B-Instruct"
    assert run_plan.reference_lane is not None
    assert run_plan.reference_lane.class_name == "preview_smoke"
    assert support.status == "degraded"


def test_resolve_benchmark_plan_no_strict_support_keeps_preview_gpu_plan() -> None:
    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "coding-long-context",
        "http://localhost:8000",
        gpu="a10g",
        num_gpus=1,
        strict_support=False,
    )
    assert error is None
    assert workload_reference == "coding-long-context"
    assert workload_pack is not None
    assert run_plan is not None
    assert support is not None
    assert run_plan.source_experiment == "vllm-single-endpoint-baseline"
    assert support.status == "unsupported"
    assert any(issue.code == "experiment_gpu_family_mismatch" for issue in support.issues)


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
    assert "preview_smoke" in payload["production_target"]["lane_classes"]
    assert payload["production_target"]["lane_classes"]["preview_smoke"]["gpu_aliases"] == ["a10g"]


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
    assert "Allowed GPU aliases for this lane: h100, h200, b200, b300" in payload["error"]
    assert "next_steps" in payload
    assert "QUICKSTART.md" in payload["next_steps"][1]


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
    assert payload["lane"]["class"] == "production_validated"
    assert payload["support"]["gpu_isa"] == "sm_100"
    assert payload["production_readiness"]["ready"] is True


@pytest.mark.asyncio
async def test_tool_validate_production_lane_rejects_preview_smoke_artifact(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The MCP artifact sandbox rejects paths outside settings.benchmark_dir
    # (SSRF/path-traversal defense). Point the sandbox at tmp_path so the
    # test artifact is reachable through the MCP tool.
    from inferscope.config import settings
    monkeypatch.setattr(settings, "benchmark_dir", tmp_path)

    artifact = BenchmarkArtifact(
        pack_name="coding-smoke",
        workload_class="coding",
        endpoint="http://localhost:8000",
        model="Qwen2.5-7B-Instruct",
        concurrency=1,
        started_at="2026-04-05T00:00:00Z",
        completed_at="2026-04-05T00:00:01Z",
        run_plan={
            "preflight_validation": {"valid": True, "errors": [], "warnings": [], "info": []},
            "observed_runtime": {"reliability": {}, "observability": {}},
        },
        provenance=BenchmarkArtifactProvenance(
            workload=BenchmarkSourceReference(
                reference="coding-smoke",
                resolved_path=str(tmp_path / "workload.yaml"),
                source_kind="builtin",
            ),
            experiment=BenchmarkSourceReference(
                reference="vllm-single-endpoint-smoke",
                resolved_path=str(tmp_path / "experiment.yaml"),
                source_kind="builtin",
            ),
            lane=BenchmarkLaneReference(
                class_name="preview_smoke",
                claim_scope="toolchain_validation_only",
                model_support_tier="benchmark_supported",
                production_target_name=None,
                workload_pack="coding-smoke",
                experiment="vllm-single-endpoint-smoke",
                summary="preview smoke",
                warnings=[],
            ),
        ),
        results=[],
        summary=BenchmarkSummary(
            total_requests=1,
            succeeded=1,
            failed=0,
            concurrency=1,
            wall_time_ms=1000.0,
            metrics_capture_complete=True,
        ),
    )
    artifact_path = artifact.save_json(tmp_path / "smoke-artifact.json")

    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)
    result = await mcp.call_tool("tool_validate_production_lane", {"candidate_artifact": str(artifact_path)})
    payload = result.structured_content

    assert payload["valid"] is False
    assert payload["evidence"] == "production_lane_validation"
    assert "preview_smoke" in payload["issues"][0]
