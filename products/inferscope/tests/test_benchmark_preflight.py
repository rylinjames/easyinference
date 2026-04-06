"""Preflight validation tests for local model artifacts and runtime fit."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from inferscope.benchmarks.preflight import validate_benchmark_preflight
from inferscope.server_benchmarks import _resolve_benchmark_plan


def _write(path: Path, content: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_validate_benchmark_preflight_rejects_incomplete_weight_directory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "qwen-weights"
    _write(artifact_dir / "config.json")

    result = validate_benchmark_preflight(
        model_name="Qwen3.5-32B",
        gpu_name="h100",
        num_gpus=1,
        engine_name="vllm",
        model_artifact_path=str(artifact_dir),
    )

    assert result.valid is False
    assert any("tokenizer metadata" in error for error in result.errors)
    assert any("missing weights" in error for error in result.errors)


def test_validate_benchmark_preflight_rejects_compiled_engine_tp_mismatch(tmp_path: Path) -> None:
    engine_dir = tmp_path / "trtllm-engine"
    _write(engine_dir / "config.json")
    _write(engine_dir / "rank0.engine", "engine-bytes")
    manifest_path = _write(
        tmp_path / "artifact-manifest.yaml",
        dedent(
            """
            schema_version: "1"
            artifact_kind: compiled_engine
            model: Qwen3.5-32B
            engine: trtllm
            tensor_parallel_size: 4
            gpu_family: hopper
            """
        ).strip()
        + "\n",
    )

    result = validate_benchmark_preflight(
        model_name="Qwen3.5-32B",
        gpu_name="h100",
        num_gpus=1,
        engine_name="trtllm",
        model_artifact_path=str(engine_dir),
        artifact_manifest=str(manifest_path),
    )

    assert result.valid is False
    assert any("expects TP=4" in error for error in result.errors)


def test_validate_benchmark_preflight_rejects_memory_oversubscription() -> None:
    result = validate_benchmark_preflight(
        model_name="Kimi-K2.5",
        gpu_name="h100",
        num_gpus=1,
        engine_name="dynamo",
    )

    assert result.valid is False
    assert any("does not fit in GPU memory" in error for error in result.errors)


def test_resolve_benchmark_plan_rejects_invalid_model_artifact_directory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bad-model-dir"
    _write(artifact_dir / "config.json")

    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "coding-long-context",
        "https://benchmark.example",
        gpu="h100",
        num_gpus=1,
        model_artifact_path=str(artifact_dir),
    )

    assert error is not None
    assert "tokenizer metadata" in error["error"]
    assert workload_reference is None
    assert workload_pack is None
    assert run_plan is None
    assert support is None


def test_resolve_benchmark_plan_attaches_preflight_validation_for_valid_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "good-model-dir"
    _write(artifact_dir / "config.json")
    _write(artifact_dir / "tokenizer_config.json")
    _write(artifact_dir / "model.safetensors", "weights")

    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "coding-smoke",
        "https://benchmark.example",
        gpu="a10g",
        num_gpus=1,
        model_artifact_path=str(artifact_dir),
    )

    assert error is None
    assert workload_reference == "coding-smoke"
    assert workload_pack is not None
    assert run_plan is not None
    assert support is not None
    assert run_plan.preflight_validation is not None
    assert run_plan.preflight_validation.valid is True
    assert run_plan.preflight_validation.model_artifact_path == str(artifact_dir)
