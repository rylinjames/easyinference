"""End-to-end MCP benchmark pipeline integration coverage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastmcp import FastMCP

from inferscope.benchmarks.openai_replay import run_openai_replay as real_run_openai_replay
from inferscope.config import settings
from inferscope.server_benchmarks import register_benchmark_tools


class DelayedSSE(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, delay_seconds: float) -> None:
        self._chunks = chunks
        self._delay_seconds = delay_seconds

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
            await asyncio.sleep(self._delay_seconds)

    async def aclose(self) -> None:
        return None


def _sse_response(*events: object, delay_seconds: float) -> httpx.Response:
    chunks = [f"data: {json.dumps(event)}\n\n".encode() for event in events]
    chunks.append(b"data: [DONE]\n\n")
    return httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=DelayedSSE(chunks, delay_seconds=delay_seconds),
    )


@pytest.mark.asyncio
async def test_mcp_pipeline_runs_benchmarks_and_compares_saved_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "benchmark_dir", tmp_path)

    async def patched_run_openai_replay(workload, endpoint, **kwargs):
        delay_seconds = 0.001 if "baseline" in endpoint else 0.01

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path != "/v1/chat/completions":
                raise AssertionError(f"Unexpected benchmark request path: {request.url.path}")
            return _sse_response(
                {"choices": [{"delta": {"content": "patch"}}]},
                {"choices": [{"delta": {"content": " plan"}}]},
                {"usage": {"prompt_tokens": 512, "completion_tokens": 2, "total_tokens": 514}},
                delay_seconds=delay_seconds,
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
        try:
            return await real_run_openai_replay(
                workload,
                endpoint,
                client=client,
                **kwargs,
            )
        finally:
            await client.aclose()

    monkeypatch.setattr("inferscope.server_benchmarks.run_openai_replay", patched_run_openai_replay)

    mcp = FastMCP("test-benchmark-pipeline")
    register_benchmark_tools(mcp)

    common_args = {
        "workload": "coding-smoke",
        "gpu": "h100",
        "num_gpus": 1,
        "capture_metrics": False,
        "save_artifact": True,
        "synthetic_requests": 2,
        "synthetic_input_tokens": 512,
        "synthetic_output_tokens": 64,
    }

    baseline_result = await mcp.call_tool(
        "tool_run_benchmark",
        {"endpoint": "https://baseline.example", **common_args},
    )
    candidate_result = await mcp.call_tool(
        "tool_run_benchmark",
        {"endpoint": "https://candidate.example", **common_args},
    )

    baseline_payload = baseline_result.structured_content
    candidate_payload = candidate_result.structured_content
    baseline_path = Path(baseline_payload["artifact_path"])
    candidate_path = Path(candidate_payload["artifact_path"])

    assert baseline_path.exists()
    assert candidate_path.exists()
    assert baseline_payload["benchmark_summary"]["succeeded"] == 2
    assert candidate_payload["benchmark_summary"]["succeeded"] == 2
    assert baseline_payload["lane"]["class"] == "preview_smoke"
    assert candidate_payload["lane"]["class"] == "preview_smoke"

    compare_result = await mcp.call_tool(
        "tool_compare_benchmarks",
        {
            "baseline_artifact": baseline_path.name,
            "candidate_artifact": candidate_path.name,
        },
    )
    compare_payload = compare_result.structured_content

    assert compare_payload["evidence"] == "benchmark_artifact_comparison"
    assert compare_payload["compatibility"]["comparable"] is True
    assert compare_payload["baseline"]["lane"]["class"] == "preview_smoke"
    assert compare_payload["candidate"]["lane"]["class"] == "preview_smoke"
    assert compare_payload["deltas"]["latency_p95_ms"] is not None
    assert compare_payload["deltas"]["latency_p95_ms"] > 0
