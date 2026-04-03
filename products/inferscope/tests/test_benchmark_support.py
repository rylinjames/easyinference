"""Unit tests for benchmark support assessment."""

from __future__ import annotations

from inferscope.benchmarks import assess_benchmark_support, load_experiment, load_workload


def test_assess_benchmark_support_rejects_grace_lane_on_h100() -> None:
    experiment = load_experiment("vllm-disagg-prefill-lmcache-grace")
    workload = load_workload(experiment.workload)

    support = assess_benchmark_support(
        model_name="Qwen3.5-72B",
        gpu_name="h100",
        num_gpus=4,
        engine_name="vllm",
        workload=workload,
        experiment=experiment,
        prompt_tokens=96_000,
    )

    assert support.status == "unsupported"
    assert support.gpu_isa == "sm_90a"
    assert any(issue.code == "grace_tier_requires_grace" for issue in support.issues)


def test_assess_benchmark_support_marks_nixl_transport_degraded_without_rdma() -> None:
    experiment = load_experiment("vllm-disagg-prefill-nixl")
    workload = load_workload(experiment.workload)

    support = assess_benchmark_support(
        model_name="Qwen3.5-32B",
        gpu_name="h100_pcie",
        num_gpus=2,
        engine_name="vllm",
        workload=workload,
        experiment=experiment,
        prompt_tokens=16_384,
        has_rdma=False,
    )

    assert support.status == "degraded"
    assert any(issue.code == "nixl_transport_degraded" for issue in support.issues)


def test_assess_benchmark_support_marks_preview_engine_degraded() -> None:
    support = assess_benchmark_support(
        model_name="Kimi-K2.5",
        gpu_name="b200",
        num_gpus=1,
        engine_name="vllm",
        prompt_tokens=4_096,
    )

    assert support.status == "degraded"
    assert support.engine_support_tier == "preview"
    assert any(issue.code == "preview_engine" for issue in support.issues)
