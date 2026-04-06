"""CLI tests for narrowed benchmark probe commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from inferscope.benchmarks import (
    BenchmarkArtifact,
    BenchmarkArtifactProvenance,
    BenchmarkLaneReference,
    BenchmarkSourceReference,
    BenchmarkSummary,
)
from inferscope.cli import app

runner = CliRunner()


def _artifact_with_lane(*, lane_class: str, claim_scope: str, production_target_name: str | None) -> BenchmarkArtifact:
    return BenchmarkArtifact(
        pack_name="coding-smoke" if lane_class == "preview_smoke" else "kimi-k2-long-context-coding",
        workload_class="coding",
        endpoint="http://localhost:8000",
        model="Qwen2.5-7B-Instruct" if lane_class == "preview_smoke" else "Kimi-K2.5",
        concurrency=1,
        started_at="2026-04-05T00:00:00Z",
        completed_at="2026-04-05T00:00:01Z",
        run_plan={
            "preflight_validation": {
                "valid": True,
                "errors": [],
                "warnings": [],
                "info": ["Memory fit OK"],
            },
            "observed_runtime": {"reliability": {}, "observability": {}},
        },
        provenance=BenchmarkArtifactProvenance(
            workload=BenchmarkSourceReference(
                reference="coding-smoke" if lane_class == "preview_smoke" else "kimi-k2-long-context-coding",
                resolved_path="/tmp/workload.yaml",
                source_kind="builtin",
            ),
            experiment=BenchmarkSourceReference(
                reference="vllm-single-endpoint-smoke"
                if lane_class == "preview_smoke"
                else "dynamo-aggregated-lmcache-kimi-k2",
                resolved_path="/tmp/experiment.yaml",
                source_kind="builtin",
            ),
            lane=BenchmarkLaneReference(
                class_name=lane_class,
                claim_scope=claim_scope,
                model_support_tier="benchmark_supported" if lane_class == "preview_smoke" else "production_validated",
                production_target_name=production_target_name,
                workload_pack="coding-smoke" if lane_class == "preview_smoke" else "kimi-k2-long-context-coding",
                experiment="vllm-single-endpoint-smoke"
                if lane_class == "preview_smoke"
                else "dynamo-aggregated-lmcache-kimi-k2",
                summary="preview smoke" if lane_class == "preview_smoke" else "production lane",
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


def test_cli_benchmark_plan_defaults_to_public_single_endpoint_lane() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark-plan",
            "coding-long-context",
            "http://localhost:8000",
            "--gpu",
            "h100",
            "--num-gpus",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Resolved probe plan for coding-long-context" in result.stdout
    assert '"class": "benchmark_supported"' in result.stdout
    assert '"source_experiment": "vllm-single-endpoint-baseline"' in result.stdout
    assert '"model": "Qwen3.5-32B"' in result.stdout
    assert '"gpu_isa": "sm_90a"' in result.stdout


def test_cli_benchmark_plan_supports_budget_smoke_lane() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark-plan",
            "coding-smoke",
            "http://localhost:8000",
            "--gpu",
            "a10g",
            "--num-gpus",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert '"class": "preview_smoke"' in result.stdout
    assert '"source_experiment": "vllm-single-endpoint-smoke"' in result.stdout
    assert '"model": "Qwen2.5-7B-Instruct"' in result.stdout
    assert '"gpu_isa": "sm_86"' in result.stdout


def test_cli_benchmark_plan_no_strict_support_allows_preview_gpu_exploration() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark-plan",
            "coding-long-context",
            "http://localhost:8000",
            "--gpu",
            "a10g",
            "--num-gpus",
            "1",
            "--no-strict-support",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert '"source_experiment": "vllm-single-endpoint-baseline"' in result.stdout
    assert '"status": "unsupported"' in result.stdout
    assert '"experiment_gpu_family_mismatch"' in result.stdout


def test_cli_removed_benchmark_matrix_command_is_absent() -> None:
    result = runner.invoke(app, ["benchmark-matrix"])

    assert result.exit_code != 0
    assert "No such command 'benchmark-matrix'" in result.output


def test_cli_benchmark_plan_rejection_is_actionable() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark-plan",
            "kimi-k2-long-context-coding",
            "http://localhost:8000",
            "--gpu",
            "gb200",
            "--num-gpus",
            "4",
        ],
    )

    assert result.exit_code != 0
    assert "not supported for the current InferScope lane" in result.output
    assert "Workaround:" in result.output
    assert "QUICKSTART.md" in result.output


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


def test_cli_validate_production_lane_rejects_preview_smoke_artifact(tmp_path: Path) -> None:
    smoke_artifact = _artifact_with_lane(
        lane_class="preview_smoke",
        claim_scope="toolchain_validation_only",
        production_target_name=None,
    )
    smoke_path = smoke_artifact.save_json(tmp_path / "smoke-artifact.json")

    result = runner.invoke(app, ["validate-production-lane", str(smoke_path)])

    assert result.exit_code == 0, result.stdout
    assert '"valid": false' in result.stdout
    assert "preview_smoke" in result.stdout
    assert "does not satisfy the canonical production-lane contract" in result.stdout


def test_cli_validate_production_lane_accepts_matching_production_pair(tmp_path: Path) -> None:
    baseline = _artifact_with_lane(
        lane_class="production_validated",
        claim_scope="production_comparable",
        production_target_name="dynamo_long_context_coding",
    )
    candidate = _artifact_with_lane(
        lane_class="production_validated",
        claim_scope="production_comparable",
        production_target_name="dynamo_long_context_coding",
    )
    baseline_path = baseline.save_json(tmp_path / "baseline.json")
    candidate_path = candidate.save_json(tmp_path / "candidate.json")

    result = runner.invoke(
        app,
        [
            "validate-production-lane",
            str(candidate_path),
            "--baseline",
            str(baseline_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert '"valid": true' in result.stdout
    assert '"comparison"' in result.stdout
    assert "matches the canonical production lane" in result.stdout
