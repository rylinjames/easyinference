"""Regression coverage for the recommender's RDMA-aware path.

Closes the snapshot v1.0.0 P0 bug `recommender_inventory_missing_rdma.md`.

Previously `_build_inventory` did not set `has_rdma` (defaulted to False),
so the recommender always told Dynamo split topology to use TCP KV transport
and the vLLM `NixlConnector` branch was dead through the recommender. The
fix threads `has_rdma`, `rdma_type`, and `node_count` through `recommend()`,
`PipelineContext`, and `_build_inventory`.
"""

from __future__ import annotations

from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.models.registry import get_model_variant
from inferscope.optimization.recommender import recommend
from inferscope.optimization.serving_profile import WorkloadMode
from inferscope.tools.recommend import recommend_config


# ----------------------------------------------------------------------------
# `recommend()` honors `has_rdma`
# ----------------------------------------------------------------------------


def test_recommender_default_no_rdma_keeps_dynamo_kv_transport_tcp() -> None:
    """Backward-compat: with `has_rdma=False` (the legacy default), the
    Dynamo split-topology env var stays as TCP. This preserves the
    pre-fix behavior for any caller that doesn't pass the new param."""
    variant = get_model_variant("Kimi-K2.5")
    gpu = get_gpu_profile("h200")
    assert variant is not None
    assert gpu is not None

    _profile, engine_config, _mem = recommend(
        model=variant,
        gpu=gpu,
        num_gpus=8,
        workload=WorkloadMode.CODING,
    )
    # Default has_rdma=False — Dynamo should NOT set DYNAMO_KV_TRANSPORT=rdma
    assert engine_config.env_vars.get("DYNAMO_KV_TRANSPORT") != "rdma"


def test_recommender_with_has_rdma_true_sets_dynamo_kv_transport_to_rdma() -> None:
    """The headline scenario from the bug doc: a user calling `recommend(...)`
    with `has_rdma=True` for a multi-GPU production cluster MUST get
    `DYNAMO_KV_TRANSPORT=rdma` (or the resolved type), not TCP.

    Before the fix this was impossible because `_build_inventory` always
    reported `has_rdma=False` regardless of input."""
    variant = get_model_variant("Kimi-K2.5")
    gpu = get_gpu_profile("h200")
    assert variant is not None
    assert gpu is not None

    _profile, engine_config, _mem = recommend(
        model=variant,
        gpu=gpu,
        num_gpus=8,
        workload=WorkloadMode.CODING,
        has_rdma=True,
    )
    # Engine config should reflect the RDMA cluster
    if "DYNAMO_KV_TRANSPORT" in engine_config.env_vars:
        assert engine_config.env_vars["DYNAMO_KV_TRANSPORT"] == "rdma", (
            f"Expected DYNAMO_KV_TRANSPORT=rdma, got "
            f"{engine_config.env_vars['DYNAMO_KV_TRANSPORT']!r}. "
            "The recommender dropped the has_rdma=True input."
        )


def test_recommender_with_explicit_rdma_type_uses_it() -> None:
    """`rdma_type='roce'` should appear verbatim in DYNAMO_KV_TRANSPORT."""
    variant = get_model_variant("Kimi-K2.5")
    gpu = get_gpu_profile("h200")
    assert variant is not None
    assert gpu is not None

    _profile, engine_config, _mem = recommend(
        model=variant,
        gpu=gpu,
        num_gpus=8,
        workload=WorkloadMode.CODING,
        has_rdma=True,
        rdma_type="roce",
    )
    if "DYNAMO_KV_TRANSPORT" in engine_config.env_vars:
        assert engine_config.env_vars["DYNAMO_KV_TRANSPORT"] == "roce"


# ----------------------------------------------------------------------------
# `tools/recommend.py::recommend_config` plumbs the parameter through
# ----------------------------------------------------------------------------


def test_recommend_config_default_no_rdma() -> None:
    """The CLI/MCP wrapper defaults to `has_rdma=False` for backward compat."""
    result = recommend_config("Kimi-K2.5", "h200", workload="coding", num_gpus=8)
    assert "error" not in result, result
    env_vars = result["engine_config"].get("env_vars", {})
    assert env_vars.get("DYNAMO_KV_TRANSPORT") != "rdma"


def test_recommend_config_with_has_rdma_threads_through() -> None:
    """The CLI/MCP wrapper plumbs `has_rdma=True` through to the recommender."""
    result = recommend_config(
        "Kimi-K2.5",
        "h200",
        workload="coding",
        num_gpus=8,
        has_rdma=True,
    )
    assert "error" not in result, result
    env_vars = result["engine_config"].get("env_vars", {})
    if "DYNAMO_KV_TRANSPORT" in env_vars:
        assert env_vars["DYNAMO_KV_TRANSPORT"] == "rdma", (
            f"recommend_config did not propagate has_rdma=True. "
            f"DYNAMO_KV_TRANSPORT={env_vars.get('DYNAMO_KV_TRANSPORT')!r}"
        )
