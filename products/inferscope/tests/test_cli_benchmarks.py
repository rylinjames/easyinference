"""CLI tests for narrowed benchmark probe commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from inferscope.benchmarks import BenchmarkArtifact, BenchmarkSummary
from inferscope.cli import app

runner = CliRunner()


def test_cli_benchmark_plan_defaults_to_supported_probe() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark-plan",
            "kimi-k2-long-context-coding",
            "http://localhost:8000",
            "--gpu",
            "b200",
            "--num-gpus",
            "4",
            "--metrics-target",
            "frontend=http://localhost:9100",
            "--metrics-target",
            "worker=http://localhost:9200",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Resolved probe plan for kimi-k2-long-context-coding" in result.stdout
    assert '"source_experiment": "dynamo-aggregated-lmcache-kimi-k2"' in result.stdout
    assert '"support"' in result.stdout
    assert '"gpu_isa": "sm_100"' in result.stdout


def test_cli_removed_benchmark_matrix_command_is_absent() -> None:
    result = runner.invoke(app, ["benchmark-matrix"])

    assert result.exit_code != 0
    assert "No such command 'benchmark-matrix'" in result.output


def test_cli_benchmark_command_reports_production_readiness(
    monkeypatch,
    tmp_path: Path,
) -> None:
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

    monkeypatch.setattr("inferscope.cli_benchmarks.run_openai_replay", fake_run_openai_replay)

    output_path = tmp_path / "probe-artifact.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "kimi-k2-long-context-coding",
            "http://localhost:8000",
            "--gpu",
            "b200",
            "--num-gpus",
            "4",
            "--output",
            str(output_path),
            "--synthetic-requests",
            "2",
            "--synthetic-input-tokens",
            "2048",
            "--synthetic-output-tokens",
            "256",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Probe completed: 2/2 requests succeeded" in result.stdout
    assert '"request_throughput_rps": 2.5' in result.stdout
    assert '"production_readiness"' in result.stdout
    assert '"ready": true' in result.stdout
