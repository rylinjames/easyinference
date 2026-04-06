"""Validation tests for workload and experiment contract loading."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from inferscope.server_benchmarks import _resolve_benchmark_plan


def _write_yaml(path: Path, payload: str) -> Path:
    path.write_text(dedent(payload).strip() + "\n")
    return path


def test_resolve_benchmark_plan_rejects_unknown_model_in_custom_workload(tmp_path: Path) -> None:
    workload_path = _write_yaml(
        tmp_path / "coding-long-context.yaml",
        """
        version: "1"
        name: coding-long-context
        description: Custom invalid workload
        workload_class: coding
        model: TotallyUnknown-9000
        target_model_classes: [qwen35_hybrid]
        concurrency: 1
        stream: true
        requests:
          - name: repo-review
            session_id: custom-session-1
            max_tokens: 64
            messages:
              - role: user
                content: Review the codebase.
        """,
    )

    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        str(workload_path),
        "https://benchmark.example",
        gpu="h100",
        num_gpus=1,
    )

    assert error is not None
    assert "does not resolve to a known model variant" in error["error"]
    assert workload_reference is None
    assert workload_pack is None
    assert run_plan is None
    assert support is None


def test_resolve_benchmark_plan_preserves_custom_workload_file_source(tmp_path: Path) -> None:
    workload_path = _write_yaml(
        tmp_path / "coding-long-context.yaml",
        """
        version: "1"
        name: coding-long-context
        description: Custom long-context coding pack for validation
        workload_class: coding
        model: Qwen3.5-32B
        target_model_classes: [qwen35_hybrid]
        concurrency: 1
        stream: true
        requests:
          - name: repo-review
            session_id: custom-session-1
            max_tokens: 64
            metadata:
              approx_context_tokens: 16384
            messages:
              - role: user
                content: Review the codebase.
        """,
    )

    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        str(workload_path),
        "https://benchmark.example",
        gpu="h100",
        num_gpus=1,
    )

    assert error is None
    assert workload_pack is not None
    assert run_plan is not None
    assert support is not None
    assert workload_reference == str(workload_path)
    assert workload_pack.description == "Custom long-context coding pack for validation"
    assert run_plan.workload_ref == str(workload_path)
    assert run_plan.workload_source is not None
    assert run_plan.workload_source.source_kind == "file"
    assert run_plan.workload_source.resolved_path == str(workload_path.resolve())


def test_resolve_benchmark_plan_rejects_model_class_mismatch_in_custom_experiment(tmp_path: Path) -> None:
    experiment_path = _write_yaml(
        tmp_path / "vllm-single-endpoint-baseline.yaml",
        """
        version: "1"
        name: vllm-single-endpoint-baseline
        description: Invalid experiment for validation
        engine: vllm
        workload: coding-long-context
        model: Qwen2.5-7B-Instruct
        target_model_classes: [frontier_mla_moe]
        topology:
          mode: single_endpoint
          session_routing: sticky
          session_header_name: X-Session-ID
          request_target_name: primary
        cache:
          strategy: prefix_only
          tiers: [gpu_hbm]
          session_affinity: true
          prefix_cache_expected: true
        metrics_targets:
          - name: primary
            role: primary
            endpoint_source: metrics_endpoint
            expected_engine: vllm
        tags: [vllm, baseline]
        """,
    )

    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "coding-long-context",
        "https://benchmark.example",
        experiment=str(experiment_path),
        gpu="h100",
        num_gpus=1,
    )

    assert error is not None
    assert "resolves to class 'dense_gqa'" in error["error"]
    assert workload_reference is None
    assert workload_pack is None
    assert run_plan is None
    assert support is None
