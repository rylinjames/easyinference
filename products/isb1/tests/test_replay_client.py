"""Tests for the internal OpenAI-compatible replay client."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web

from harness.client import BenchmarkClient
from harness.replay_client import ReplayRunResult, run_rate
from workloads.base import Request


async def _streaming_chat_handler(request: web.Request) -> web.StreamResponse:
    payload = await request.json()
    assert payload["stream"] is True

    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream"},
    )
    await response.prepare(request)

    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "{"}}]},
        {"choices": [{"delta": {"content": '"name":"read_file"}'}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 128,
                "completion_tokens": 2,
                "total_tokens": 130,
            },
        },
    ]

    for event in events:
        await response.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        await asyncio.sleep(0.01)
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


def test_run_rate_captures_streaming_metrics(unused_tcp_port: int) -> None:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _streaming_chat_handler)

    async def _exercise() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        port = unused_tcp_port
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()

        try:
            request_pool = [
                Request(
                    request_id="req-1",
                    messages=[
                        {"role": "system", "content": "You are a test assistant."},
                        {"role": "user", "content": "Call the read_file tool."},
                    ],
                    expected_output_tokens=32,
                    session_id="session-1",
                    metadata={"approx_context_tokens": 50_000},
                )
            ]
            result = await run_rate(
                base_url=f"http://127.0.0.1:{port}",
                model="test-model",
                request_pool=request_pool,
                request_count=1,
                request_rate=float("inf"),
                arrival_model="poisson",
                arrival_shape=None,
                seed=42,
                goodput_slo={
                    "ttft_p95_ms": {"32k": 6000, "96k": 20000},
                    "tpot_p95_ms": 100,
                },
            )
        finally:
            await runner.cleanup()

        assert result.completed == 1
        assert result.failed == 0
        assert result.total_output_tokens == 2
        assert result.per_request[0].ttft is not None
        assert len(result.per_request[0].token_timestamps) == 2
        assert all(t > 0 for t in result.per_request[0].token_timestamps)
        assert result.per_request[0].ttft_slo_seconds == 20.0
        assert result.per_request[0].tpot_slo_seconds == 0.1

    asyncio.run(_exercise())


async def _failing_run_rate(**_: object) -> ReplayRunResult:
    return ReplayRunResult(
        completed=0,
        failed=1,
        duration=1.0,
        request_throughput=0.0,
        output_throughput=0.0,
        generation_throughput=0.0,
        total_input_tokens=0,
        total_output_tokens=0,
        error_rate=1.0,
        goodput=0.0,
        slo_attainment=0.0,
        per_request=[],
        runner_metadata={"mode": "test"},
    )


def test_benchmark_client_raises_when_rate_point_has_no_successes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr("harness.client.run_rate", _failing_run_rate)
    client = BenchmarkClient(
        base_url="http://localhost:8000",
        model="test-model",
        result_dir=tmp_path,
        requests=[
            Request(
                request_id="req-1",
                messages=[{"role": "user", "content": "hello"}],
                expected_output_tokens=16,
            )
        ],
        arrival_model="poisson",
        arrival_shape=None,
        goodput_slo=None,
    )

    with pytest.raises(RuntimeError, match="completed zero requests"):
        client.run(
            request_rate=1.0,
            request_pool_size=1,
            measurement_duration_seconds=30.0,
            rate_index=0,
        )

    assert (tmp_path / "000-1rps.json").exists()
