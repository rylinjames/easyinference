"""Tests for the packaged benchmark runtime and observed-runtime metrics."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from inferscope.benchmarks import (
    BenchmarkExecutionProfile,
    BenchmarkGoodputSLO,
    WorkloadPack,
    WorkloadRequest,
    build_run_plan,
    run_openai_replay,
)
from inferscope.benchmarks.models import ChatMessage


class DelayedSSE(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay_seconds: float = 0.001) -> None:
        self._chunks = chunks
        self._delay_seconds = delay_seconds

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
            await asyncio.sleep(self._delay_seconds)

    async def aclose(self) -> None:
        return None


def _sse_response(*events: object, status_code: int = 200, delay_seconds: float = 0.001) -> httpx.Response:
    chunks = [f"data: {json.dumps(event)}\n\n".encode() for event in events]
    chunks.append(b"data: [DONE]\n\n")
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "text/event-stream"},
        stream=DelayedSSE(chunks, delay_seconds=delay_seconds),
    )


def _tool_agent_workload() -> WorkloadPack:
    return WorkloadPack(
        name="tool-agent",
        description="tool benchmark",
        workload_class="tool_agent",
        model="Qwen3.5-32B",
        concurrency=2,
        stream=True,
        requests=[
            WorkloadRequest(
                name="tool-call-1",
                messages=[ChatMessage(role="user", content="Call a tool to inspect the repo")],
                metadata={"bridge_source": "mcp_tool_call", "approx_context_tokens": 8192},
                max_tokens=128,
            ),
            WorkloadRequest(
                name="tool-call-2",
                messages=[ChatMessage(role="user", content="Call a tool again")],
                metadata={"bridge_source": "mcp_tool_call", "approx_context_tokens": 8192},
                max_tokens=128,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_run_openai_replay_records_observed_runtime_metrics() -> None:
    request_counter = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        request_counter["count"] += 1
        return _sse_response(
            {"choices": [{"delta": {"content": '{"name":"read_file"'}}]},
            {"choices": [{"delta": {"content": ',"arguments":{'}}]},
            {"choices": [{"delta": {"content": '"path":"README.md"}}'}}]},
            {"usage": {"prompt_tokens": 1024, "completion_tokens": 3, "total_tokens": 1027}},
        )

    workload = _tool_agent_workload()
    run_plan = build_run_plan(
        workload,
        "http://benchmark.local",
        workload_ref=workload.name,
        execution=BenchmarkExecutionProfile(
            request_rate_rps=5.0,
            arrival_model="poisson",
            warmup_requests=1,
            goodput_slo=BenchmarkGoodputSLO(ttft_p95_ms=5_000, tpot_p95_ms=5_000),
        ),
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
    try:
        artifact = await run_openai_replay(
            workload,
            "http://benchmark.local",
            run_plan=run_plan,
            capture_metrics=False,
            client=client,
        )
    finally:
        await client.aclose()

    observed = (artifact.run_plan or {}).get("observed_runtime", {})
    assert artifact.summary.succeeded == 2
    assert request_counter["count"] == 3  # one warmup + two measured requests
    assert observed["request_throughput_rps"] > 0
    assert observed["tool_parse_success_rate"] == 1.0
    assert observed["ttft_ms"]["p95"] is not None
    assert observed["tpot_ms"]["p95"] is not None
    assert observed["itl_ms"]["p95"] is not None


@pytest.mark.asyncio
async def test_run_openai_replay_skips_later_session_requests_after_failure() -> None:
    workload = WorkloadPack(
        name="coding-long-context",
        description="session benchmark",
        workload_class="coding",
        model="Qwen3.5-32B",
        concurrency=2,
        stream=True,
        requests=[
            WorkloadRequest(
                name="first-fails",
                session_id="session-a",
                messages=[ChatMessage(role="user", content="fail this session")],
                metadata={"approx_context_tokens": 4096},
                max_tokens=64,
            ),
            WorkloadRequest(
                name="second-skipped",
                session_id="session-a",
                messages=[ChatMessage(role="user", content="should be skipped")],
                metadata={"approx_context_tokens": 4096},
                max_tokens=64,
            ),
            WorkloadRequest(
                name="independent-success",
                session_id="session-b",
                messages=[ChatMessage(role="user", content="succeed here")],
                metadata={"approx_context_tokens": 4096},
                max_tokens=64,
            ),
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        body = json.dumps(payload)
        if "fail this session" in body:
            return httpx.Response(status_code=500, text="boom")
        return _sse_response(
            {"choices": [{"delta": {"content": "ok"}}]},
            {"usage": {"prompt_tokens": 256, "completion_tokens": 2, "total_tokens": 258}},
        )

    run_plan = build_run_plan(workload, "http://benchmark.local", workload_ref=workload.name)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
    try:
        artifact = await run_openai_replay(
            workload,
            "http://benchmark.local",
            run_plan=run_plan,
            capture_metrics=False,
            client=client,
        )
    finally:
        await client.aclose()

    assert artifact.summary.succeeded == 1
    assert artifact.summary.failed == 2
    assert artifact.results[1].status == "error"
    assert "Skipped because a prior request in the same session failed" in artifact.results[1].error
