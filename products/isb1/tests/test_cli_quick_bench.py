"""Tests for ISB-1 quick-bench smoke behavior."""

from __future__ import annotations

from click.testing import CliRunner
import requests

from harness.cli import main
from harness.replay_client import ReplayRequestResult, ReplayRunResult


class _FakeResponse:
    def __init__(
        self,
        *,
        json_data: dict | None = None,
        text: str = "",
        ok: bool = True,
        status_code: int = 200,
    ) -> None:
        self._json_data = json_data or {}
        self.text = text
        self.ok = ok
        self.status_code = status_code

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:
        if not self.ok or self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _successful_run_result() -> ReplayRunResult:
    return ReplayRunResult(
        completed=1,
        failed=0,
        duration=1.0,
        request_throughput=1.0,
        output_throughput=8.0,
        generation_throughput=8.0,
        total_input_tokens=32,
        total_output_tokens=8,
        error_rate=0.0,
        goodput=1.0,
        slo_attainment=1.0,
        per_request=[
            ReplayRequestResult(
                request_id="req-1",
                session_id=None,
                status="ok",
                timestamp=0.0,
                ttft=0.5,
                e2e_latency=1.0,
                output_tokens=8,
                prompt_tokens=32,
                total_tokens=40,
                token_timestamps=[0.5, 0.6, 0.7],
                error=False,
            )
        ],
        runner_metadata={"runner": "test"},
    )


def test_quick_bench_retries_model_detection_and_warms_serverless_endpoint(
    monkeypatch,
) -> None:
    runner = CliRunner()
    detect_attempts = {"count": 0}
    warmup_calls = {"count": 0}
    metrics_calls = {"count": 0}
    captured: dict[str, object] = {}

    def fake_get(url: str, headers=None, timeout=None):  # noqa: ANN001
        del headers
        if url.endswith("/v1/models"):
            detect_attempts["count"] += 1
            if detect_attempts["count"] == 1:
                raise requests.Timeout("cold start")
            assert timeout == 30
            return _FakeResponse(json_data={"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]})
        if url.endswith("/metrics"):
            metrics_calls["count"] += 1
            return _FakeResponse(
                text=(
                    "# HELP vllm:kv_cache_usage_perc cache usage\n"
                    "vllm:kv_cache_usage_perc 0.25\n"
                )
            )
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(url: str, json=None, headers=None, timeout=None):  # noqa: ANN001
        del json, headers
        warmup_calls["count"] += 1
        assert url == "https://demo.modal.run/v1/chat/completions"
        assert timeout == 180
        return _FakeResponse(json_data={"choices": [{"message": {"content": "OK"}}]})

    async def fake_run_rate(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return _successful_run_result()

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("harness.replay_client.run_rate", fake_run_rate)

    result = runner.invoke(
        main,
        [
            "quick-bench",
            "https://demo.modal.run",
            "--requests",
            "1",
            "--duration",
            "30",
        ],
    )

    assert result.exit_code == 0, result.output
    assert detect_attempts["count"] == 2
    assert warmup_calls["count"] == 1
    assert metrics_calls["count"] == 1
    assert captured["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert captured["request_timeout_seconds"] == 180
    assert captured["total_timeout_seconds"] == 240
    assert "Detected model: Qwen/Qwen2.5-7B-Instruct" in result.output
    assert "Warmup completed in" in result.output
    assert "Profile: serverless endpoint heuristics enabled" in result.output


def test_quick_bench_respects_explicit_model_and_skips_warmup(monkeypatch) -> None:
    runner = CliRunner()
    metrics_calls = {"count": 0}
    captured: dict[str, object] = {}

    def fake_get(url: str, headers=None, timeout=None):  # noqa: ANN001
        del headers, timeout
        if url.endswith("/metrics"):
            metrics_calls["count"] += 1
            return _FakeResponse(
                text=(
                    "# HELP vllm:gpu_prefix_cache_hit_rate hit rate\n"
                    "vllm:gpu_prefix_cache_hit_rate 0.5\n"
                )
            )
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(url: str, json=None, headers=None, timeout=None):  # noqa: ANN001
        del url, json, headers, timeout
        raise AssertionError("warmup POST should not be called")

    async def fake_run_rate(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return _successful_run_result()

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("harness.replay_client.run_rate", fake_run_rate)

    result = runner.invoke(
        main,
        [
            "quick-bench",
            "http://localhost:8000/v1",
            "--model-id",
            "Qwen/Qwen2.5-7B-Instruct",
            "--workload",
            "coding",
            "--requests",
            "1",
            "--duration",
            "30",
            "--no-warmup",
        ],
    )

    assert result.exit_code == 0, result.output
    assert metrics_calls["count"] == 1
    assert captured["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert captured["request_timeout_seconds"] == 120
    assert captured["total_timeout_seconds"] == 180
    assert "Detected model:" not in result.output
    assert "Warmup completed in" not in result.output
    assert "Profile: serverless endpoint heuristics enabled" not in result.output
