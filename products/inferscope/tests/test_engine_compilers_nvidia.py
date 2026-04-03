"""Regression tests for NVIDIA compiler-specific behavior."""

from __future__ import annotations

from inferscope.engines.base import DeploymentInventory
from inferscope.engines.dynamo import DynamoCompiler
from inferscope.engines.trtllm import TRTLLMCompiler
from inferscope.engines.vllm import VLLMCompiler
from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.optimization.platform_policy import resolve_platform_traits
from inferscope.optimization.serving_profile import (
    CacheSpec,
    EngineType,
    ModelClass,
    PrecisionSpec,
    SchedulerSpec,
    ServingProfile,
    TopologySpec,
    WorkloadMode,
)


def _inventory(gpu_name: str) -> DeploymentInventory:
    gpu = get_gpu_profile(gpu_name)
    assert gpu is not None
    traits = resolve_platform_traits(gpu)
    return DeploymentInventory(
        gpu_type=gpu.name,
        gpu_arch=gpu.compute_capability,
        gpu_count=4,
        gpu_memory_gb=gpu.memory_gb,
        gpu_memory_bandwidth_tb_s=gpu.memory_bandwidth_tb_s,
        interconnect=f"nvlink{gpu.nvlink_version}" if gpu.nvlink_version else gpu.pcie,
        interconnect_bandwidth_gb_s=gpu.nvlink_bandwidth_gb_s or gpu.if_bandwidth_gb_s,
        fp8_support=gpu.fp8_support,
        fp4_support=gpu.fp4_support,
        fp8_format=gpu.fp8_format,
        platform_family=traits.family.value,
        has_grace=traits.is_grace,
        grace_memory_gb=traits.grace_memory_gb,
        grace_memory_bandwidth_gb_s=traits.grace_memory_bandwidth_gb_s,
        c2c_bandwidth_gb_s=traits.c2c_bandwidth_gb_s,
        has_decompression_engine=traits.has_decompression_engine,
        has_helix_parallelism=traits.has_helix_parallelism,
        has_accelerated_softmax=traits.has_accelerated_softmax,
        platform_features=traits.to_dict(),
    )


def _profile() -> ServingProfile:
    return ServingProfile(
        model="DeepSeek-V3",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        engine=EngineType.VLLM,
        gpu_type="B200",
        num_gpus=4,
        workload_mode=WorkloadMode.CHAT,
        topology=TopologySpec(tp=4, dp=1, ep=1),
        scheduler=SchedulerSpec(batched_token_budget=16384, prefill_chunk_tokens=16384, max_num_seqs=64),
        cache=CacheSpec(gpu_memory_utilization=0.93),
        precision=PrecisionSpec(weights="fp4", activations="fp8", kv_cache="fp8_e4m3"),
    )


def _kimi_dynamo_profile() -> ServingProfile:
    return ServingProfile(
        model="Kimi-K2.5",
        model_class=ModelClass.CLASSICAL_MOE,
        engine=EngineType.DYNAMO,
        gpu_type="B200",
        num_gpus=4,
        workload_mode=WorkloadMode.CODING,
        topology=TopologySpec(tp=2, dp=2, ep=1),
        scheduler=SchedulerSpec(
            batched_token_budget=32768,
            prefill_chunk_tokens=32768,
            max_num_seqs=64,
        ),
        cache=CacheSpec(
            cache_backend="lmcache",
            lmcache_mode="local",
            session_affinity=True,
            gpu_memory_utilization=0.94,
        ),
        precision=PrecisionSpec(weights="fp4", activations="fp8", kv_cache="fp8_e4m3"),
    )


def test_vllm_compiler_differentiates_b200_from_gb200() -> None:
    compiler = VLLMCompiler()
    profile = _profile()

    b200_cfg = compiler.compile(profile, _inventory("b200"))
    gb200_cfg = compiler.compile(profile, _inventory("gb200"))

    assert all("Grace" not in note for note in b200_cfg.notes)
    assert any("Grace" in note for note in gb200_cfg.notes)


def test_vllm_compiler_labels_gb300_without_gb200_mislabel() -> None:
    compiler = VLLMCompiler()
    profile = _profile()

    cfg = compiler.compile(profile, _inventory("gb300"))

    assert any("GB300" in note for note in cfg.notes)
    assert all("GB200" not in note for note in cfg.notes)


def test_trtllm_compiler_uses_batched_token_budget_and_marks_preview() -> None:
    compiler = TRTLLMCompiler()
    profile = _profile()

    cfg = compiler.compile(profile, _inventory("b200"))

    assert cfg.cli_flags["max_num_tokens"] == 16384
    assert cfg.support_tier == "preview"
    assert any("Preview engine" in warning for warning in cfg.warnings)


def test_dynamo_compiler_marks_supported() -> None:
    compiler = DynamoCompiler()
    profile = _kimi_dynamo_profile()

    cfg = compiler.compile(profile, _inventory("b200"))

    assert cfg.support_tier == "supported"
    assert "Dynamo + LMCache" in cfg.support_reason
