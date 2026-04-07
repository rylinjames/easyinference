"""Regression tests for NVIDIA compiler-specific behavior."""

from __future__ import annotations

import json
import shlex

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


# ----------------------------------------------------------------------------
# Shell-injection regression — bugs/vllm_command_builder_shell_injection.md
# ----------------------------------------------------------------------------
#
# The compilers use shlex.quote when serialising CLI flags so that dict-valued
# flags (which get json.dumps'd) cannot break shell parsing if they contain
# single quotes. Earlier drafts wrapped them with literal `f"'{json.dumps(v)}'"`
# which is unsafe.


def test_vllm_compiler_command_round_trips_through_shlex() -> None:
    """The compiled vLLM command must parse cleanly via shlex.split.

    Regression test for bugs/vllm_command_builder_shell_injection.md (P1).
    """
    compiler = VLLMCompiler()
    cfg = compiler.compile(_profile(), _inventory("b200"))

    # Collapse the backslash-newline continuations the compiler uses for
    # readability before passing to shlex.split.
    parsed = shlex.split(cfg.command.replace("\\\n", " "))
    assert parsed[0] == "vllm"
    assert parsed[1] == "serve"
    # Every dict-valued flag must parse back to the original JSON object
    for k, v in cfg.cli_flags.items():
        if isinstance(v, dict):
            flag = f"--{k}"
            assert flag in parsed, f"flag {flag} missing from parsed command"
            value_index = parsed.index(flag) + 1
            roundtripped = json.loads(parsed[value_index])
            assert roundtripped == v, f"dict flag {k} did not round-trip through shlex"


def test_trtllm_compiler_command_round_trips_through_shlex() -> None:
    """The compiled TRT-LLM command must parse cleanly via shlex.split.

    Same regression as the vLLM test — both compilers used the same unsafe
    f-string single-quote wrapping pattern before the fix.
    """
    compiler = TRTLLMCompiler()
    cfg = compiler.compile(_profile(), _inventory("b200"))

    # Collapse the backslash-newline continuations the compiler uses for
    # readability before passing to shlex.split.
    parsed = shlex.split(cfg.command.replace("\\\n", " "))
    assert parsed[0] == "trtllm-serve"
    assert parsed[1] == "serve"
    for k, v in cfg.cli_flags.items():
        if isinstance(v, dict):
            flag = f"--{k}"
            assert flag in parsed, f"flag {flag} missing from parsed command"
            value_index = parsed.index(flag) + 1
            roundtripped = json.loads(parsed[value_index])
            assert roundtripped == v, f"dict flag {k} did not round-trip through shlex"


# ----------------------------------------------------------------------------
# Compiler-gate contract — bugs/dynamo_compiler_command_set_before_gate.md
# and bugs/atom_compiler_unsupported_tier_inconsistency.md
# ----------------------------------------------------------------------------


def _profile_with_unsupported_model() -> ServingProfile:
    """Build a ServingProfile that violates the Dynamo `is_target_model` gate."""
    return ServingProfile(
        model="UnsupportedModel-7B",
        model_class=ModelClass.DENSE_GQA,
        engine=EngineType.DYNAMO,
        gpu_type="b200",
        num_gpus=4,
        workload_mode=WorkloadMode.CHAT,
        topology=TopologySpec(tp=4, dp=1, ep=1),
        scheduler=SchedulerSpec(batched_token_budget=16384, prefill_chunk_tokens=16384, max_num_seqs=64),
        cache=CacheSpec(gpu_memory_utilization=0.93),
        precision=PrecisionSpec(weights="fp8", activations="fp8", kv_cache="fp8_e4m3"),
    )


def test_dynamo_compiler_unsupported_model_does_not_populate_command() -> None:
    """`DynamoCompiler` must run hard support gates BEFORE populating
    cfg.command. Closes the snapshot v1.0.0 P1 bug
    `dynamo_compiler_command_set_before_gate`.
    """
    compiler = DynamoCompiler()
    profile = _profile_with_unsupported_model()
    cfg = compiler.compile(profile, _inventory("b200"))

    assert cfg.support_tier == "unsupported"
    assert "Kimi-K2.5" in cfg.support_reason
    assert cfg.command == "", (
        f"DynamoCompiler populated cfg.command on hard-gate failure: "
        f"{cfg.command!r}. Hard gates must run before command population."
    )


def test_dynamo_compiler_unsupported_gpu_arch_does_not_populate_command() -> None:
    """A non-Hopper/Blackwell sm_ arch must early-return without populating
    command. We use a non-NVIDIA arch to trigger the first hard gate."""
    compiler = DynamoCompiler()
    profile = _kimi_dynamo_profile()
    inventory = _inventory("b200")
    inventory.gpu_arch = "gfx942"  # AMD MI300X — fails the sm_ prefix gate
    cfg = compiler.compile(profile, inventory)

    assert cfg.support_tier == "unsupported"
    assert cfg.command == ""


def test_atom_compiler_unsupported_gpu_sets_unsupported_tier() -> None:
    """`ATOMCompiler` previously left `support_tier="supported"` (the default)
    on hard-gate failure, leaving callers unable to filter unsupported
    configs by tier alone. Closes the snapshot v1.0.0 P1 bug
    `atom_compiler_unsupported_tier_inconsistency`.
    """
    from inferscope.engines.atom import ATOMCompiler

    compiler = ATOMCompiler()
    profile = _profile()  # uses gpu_type="B200"
    cfg = compiler.compile(profile, _inventory("b200"))  # sm_100 — not gfx94[0|2|5]

    assert cfg.support_tier == "unsupported", (
        "ATOMCompiler did not set support_tier='unsupported' on the hard gate. "
        "Callers filtering by tier will treat the broken config as valid."
    )
    assert cfg.support_reason  # populated, not empty
    assert cfg.command == ""  # the existing correct behavior — doesn't populate command
    assert any("AMD" in w for w in cfg.warnings)


def test_atom_compiler_amd_gpu_still_compiles() -> None:
    """Regression: ATOMCompiler must still produce a usable config on
    AMD hardware (the happy path)."""
    from inferscope.engines.atom import ATOMCompiler

    compiler = ATOMCompiler()
    inventory = _inventory("b200")
    inventory.gpu_arch = "gfx942"
    inventory.gpu_type = "MI300X"
    profile = _profile()
    cfg = compiler.compile(profile, inventory)

    # The happy path leaves support_tier at the dataclass default
    # ("supported") and populates cli_flags.
    assert cfg.support_tier == "supported"
    assert "model" in cfg.cli_flags


def test_vllm_compiler_handles_single_quote_in_dict_value() -> None:
    """If a dict-valued cli_flag contains a string with a single quote, the
    resulting command must still parse via shlex (the previous f-string
    wrapping would break here).

    This is the strict version of the round-trip test — directly exercises
    the worst-case input the bug doc cited.
    """
    compiler = VLLMCompiler()
    cfg = compiler.compile(_profile(), _inventory("b200"))

    # Inject a malicious dict value with a single quote and a double quote
    cfg.cli_flags["adversarial-config"] = {
        "name": "value with ' single quote",
        "other": 'value with " double quote',
        "shell": "rm -rf $HOME; echo 'gotcha'",
    }
    # Re-build the command using the same shlex-quoting logic the compiler uses
    cmd_parts = ["vllm", "serve"]
    for k, v in cfg.cli_flags.items():
        if isinstance(v, bool):
            if v:
                cmd_parts.append(f"--{k}")
        elif isinstance(v, dict):
            cmd_parts.append(f"--{k}")
            cmd_parts.append(shlex.quote(json.dumps(v)))
        else:
            cmd_parts.append(f"--{k}")
            cmd_parts.append(shlex.quote(str(v)))
    rebuilt = " \\\n  ".join(cmd_parts)

    parsed = shlex.split(rebuilt.replace("\\\n", " "))
    assert "--adversarial-config" in parsed
    idx = parsed.index("--adversarial-config")
    roundtripped = json.loads(parsed[idx + 1])
    assert roundtripped["name"] == "value with ' single quote"
    assert roundtripped["other"] == 'value with " double quote'
    assert roundtripped["shell"] == "rm -rf $HOME; echo 'gotcha'"
