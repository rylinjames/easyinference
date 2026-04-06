"""Broad matrix coverage for the recommender.

The existing test_recommend_nvidia_platforms.py covers a small set of
concrete Kimi scenarios. This file locks down:

  * the full (model × GPU × workload) supported matrix
  * memory-plan internal accounting consistency
  * determinism of recommendations
  * shape of error payloads on unsupported inputs
  * consistency between recommend_config and recommend_engine

All tests are pure-Python and do not need GPU hardware.
"""

from __future__ import annotations

import pytest

from inferscope.production_target import (
    SUPPORTED_WORKLOAD_MODES,
    supported_gpu_aliases,
    supported_model_names,
)
from inferscope.tools.recommend import recommend_config, recommend_engine

# ----------------------------------------------------------------------------
# Matrix configuration
# ----------------------------------------------------------------------------
#
# These counts were chosen so every (model, gpu) combination is close to the
# most sensible single-node deployment for that model class. They are not the
# only valid counts — the recommender handles many — but using one-per-model
# keeps the parametrized test matrix small (48 cells instead of hundreds).

_DEFAULT_NUM_GPUS = {
    "Kimi-K2.5": 8,
    "Qwen3-Coder-480B-A35B-Instruct": 8,
    "Qwen3.5-32B": 2,
    "Qwen2.5-7B-Instruct": 1,
    "Qwen3-Coder-30B-A3B-Instruct": 2,
    "Qwen3-Coder-Next": 4,
}


def _matrix_cells() -> list[tuple[str, str, str, int]]:
    """All (model, gpu, workload, num_gpus) combinations in the supported matrix."""
    cells: list[tuple[str, str, str, int]] = []
    for model in supported_model_names():
        for gpu in supported_gpu_aliases():
            for workload in SUPPORTED_WORKLOAD_MODES:
                cells.append((model, gpu, workload, _DEFAULT_NUM_GPUS[model]))
    return cells


# ----------------------------------------------------------------------------
# Matrix: every cell produces a well-shaped payload
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("model,gpu,workload,num_gpus", _matrix_cells())
def test_recommend_config_matrix_returns_well_shaped_payload(
    model: str, gpu: str, workload: str, num_gpus: int
) -> None:
    """Every supported matrix cell must return a dict with either a
    valid recommendation or a clean, actionable error — never raise."""
    result = recommend_config(model, gpu, workload=workload, num_gpus=num_gpus)

    assert isinstance(result, dict), f"{model}/{gpu}/{workload}: not a dict"
    # Every payload has a summary the operator can read.
    assert "summary" in result, f"{model}/{gpu}/{workload}: missing summary"
    # Confidence always present and in [0, 1].
    assert 0.0 <= result.get("confidence", -1) <= 1.0

    if "error" in result:
        # Error-path payloads must at least say what went wrong.
        assert isinstance(result["error"], str) and result["error"]
    else:
        # Success-path payloads must carry all three building blocks.
        assert "serving_profile" in result
        assert "engine_config" in result
        assert "memory_plan" in result
        assert "launch_command" in result
        # Engine must be the production-lane engine for every supported cell.
        assert result["engine_config"]["engine"] == "dynamo"


# ----------------------------------------------------------------------------
# Memory plan consistency — weights + kv + activations must fit in usable
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("model,gpu,workload,num_gpus", _matrix_cells())
def test_memory_plan_accounting_is_internally_consistent(
    model: str, gpu: str, workload: str, num_gpus: int
) -> None:
    """If the memory planner says the config fits, the declared components
    must actually add up to less than the declared usable budget. Catches
    refactors that drop a term from the planner or change a unit."""
    result = recommend_config(model, gpu, workload=workload, num_gpus=num_gpus)
    if "error" in result:
        return  # planner said no — nothing to verify about a non-plan
    plan = result["memory_plan"]

    # Usable must be less than (or equal to) total.
    assert plan["usable_memory_gb"] <= plan["total_gpu_memory_gb"]

    # fits=True implies components sum to <= usable. Allow a tiny epsilon
    # to cover floating-point rounding in the planner.
    if plan["fits"]:
        components = (
            plan["weight_gb"]
            + plan["kv_cache_budget_gb"]
            + plan["activation_overhead_gb"]
        )
        assert components <= plan["usable_memory_gb"] + 1e-3, (
            f"{model}/{gpu}/{workload}: {components:.1f} GB claimed to fit in "
            f"{plan['usable_memory_gb']:.1f} GB usable budget"
        )


# ----------------------------------------------------------------------------
# Determinism — same inputs, identical output
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,gpu,num_gpus",
    [
        ("Kimi-K2.5", "h100", 8),
        ("Kimi-K2.5", "b200", 4),
        ("Qwen3.5-32B", "h200", 2),
    ],
)
def test_recommend_config_is_deterministic(model: str, gpu: str, num_gpus: int) -> None:
    """Two back-to-back calls with identical inputs must return identical
    output. Catches bugs where dict ordering, random seeding, or time-based
    fields sneak into the recommendation."""
    first = recommend_config(model, gpu, workload="coding", num_gpus=num_gpus)
    second = recommend_config(model, gpu, workload="coding", num_gpus=num_gpus)
    # The reasoning_trace inside engine_config is a list of strings built by
    # the DAG and should be identical. The memory_plan and serving_profile
    # are derived purely from inputs.
    assert first["serving_profile"] == second["serving_profile"]
    assert first["memory_plan"] == second["memory_plan"]
    assert first["launch_command"] == second["launch_command"]


# ----------------------------------------------------------------------------
# B300 — previously only covered by a consistency check, not a shape check
# ----------------------------------------------------------------------------


def test_recommend_b300_kimi_coding_produces_blackwell_ultra_config() -> None:
    result = recommend_config("Kimi-K2.5", "b300", workload="coding", num_gpus=4)
    profile = result["serving_profile"]
    assert profile["engine"] == "dynamo"
    # B300 is Blackwell Ultra, so FP4 should be preferred like B200.
    assert profile["precision"]["weights"] in {"fp4", "fp8"}
    assert result["memory_plan"]["fits"] is True


# ----------------------------------------------------------------------------
# Chat workload — previously only coding was exercised
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("gpu,num_gpus", [("h100", 8), ("h200", 4), ("b200", 4), ("b300", 4)])
def test_recommend_kimi_chat_workload_produces_valid_plan(gpu: str, num_gpus: int) -> None:
    result = recommend_config("Kimi-K2.5", gpu, workload="chat", num_gpus=num_gpus)
    assert "error" not in result, result.get("error")
    assert result["serving_profile"]["engine"] == "dynamo"
    assert result["memory_plan"]["fits"] is True


# ----------------------------------------------------------------------------
# Documented rejection: 480B model on H100 does not fit
# ----------------------------------------------------------------------------


def test_480b_qwen_on_h100_returns_clean_kv_headroom_error() -> None:
    """Qwen3-Coder-480B doesn't fit on 8× H100 with long-context KV headroom.
    The recommender must return a clean error rather than producing a config
    that claims to work (or raising). Locking down the current shape so a
    refactor that silently swallows this case fails the test."""
    result = recommend_config("Qwen3-Coder-480B-A35B-Instruct", "h100", workload="coding", num_gpus=8)
    assert "error" in result
    # Error must be actionable — mention what failed.
    assert "tensor-parallel" in result["error"] or "KV" in result["error"]


# ----------------------------------------------------------------------------
# Error-shape tests for unsupported inputs
# ----------------------------------------------------------------------------


def test_recommend_config_rejects_unsupported_gpu() -> None:
    result = recommend_config("Kimi-K2.5", "a10g", workload="coding", num_gpus=1)
    assert "error" in result
    assert "a10g" in result["error"].lower()
    assert "available_gpus" in result


def test_recommend_config_rejects_unsupported_model() -> None:
    result = recommend_config("Llama-3.1-8B", "h100", workload="coding", num_gpus=1)
    assert "error" in result
    assert "available_models" in result


def test_recommend_config_rejects_unsupported_workload() -> None:
    # Only 'coding' and 'chat' are supported.
    result = recommend_config("Kimi-K2.5", "h100", workload="agent", num_gpus=8)
    assert "error" in result


@pytest.mark.parametrize("num_gpus", [0, -1, -100])
def test_recommend_config_rejects_non_positive_num_gpus(num_gpus: int) -> None:
    result = recommend_config("Kimi-K2.5", "h100", workload="coding", num_gpus=num_gpus)
    assert "error" in result


# ----------------------------------------------------------------------------
# recommend_config vs recommend_engine agreement across the matrix
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("model,gpu,num_gpus", [
    (model, gpu, _DEFAULT_NUM_GPUS[model])
    for model in supported_model_names()
    for gpu in supported_gpu_aliases()
])
def test_recommend_engine_agrees_with_recommend_config(
    model: str, gpu: str, num_gpus: int
) -> None:
    """Every cell where recommend_config returns a valid config should also
    have recommend_engine pick the same engine (dynamo). Catches drift
    between the two code paths."""
    cfg = recommend_config(model, gpu, workload="coding", num_gpus=num_gpus)
    if "error" in cfg:
        return  # skip cells where the config was rejected
    eng = recommend_engine(model, gpu, workload="coding", num_gpus=num_gpus)
    assert "error" not in eng, f"{model}/{gpu}: recommend_engine errored but recommend_config did not"
    assert eng["rankings"][0]["engine"] == cfg["engine_config"]["engine"]
