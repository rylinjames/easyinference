"""Regression tests for the narrowed Kimi production recommendation policy."""

from __future__ import annotations

from inferscope.tools.recommend import recommend_config, recommend_engine


def test_recommend_h100_kimi_coding_uses_dynamo_fp8_tp8() -> None:
    result = recommend_config("Kimi-K2.5", "h100", workload="coding", num_gpus=8)

    profile = result["serving_profile"]
    assert profile["engine"] == "dynamo"
    assert profile["precision"]["weights"] == "fp8"
    assert profile["topology"]["tp"] == 8
    assert profile["topology"]["dp"] == 1
    assert result["memory_plan"]["fits"] is True


def test_recommend_h200_kimi_coding_uses_dynamo_fp8_tp4() -> None:
    result = recommend_config("Kimi-K2.5", "h200", workload="coding", num_gpus=4)

    profile = result["serving_profile"]
    assert profile["engine"] == "dynamo"
    assert profile["precision"]["weights"] == "fp8"
    assert profile["topology"]["tp"] == 4
    assert profile["topology"]["dp"] == 1
    assert result["memory_plan"]["fits"] is True


def test_recommend_b200_kimi_coding_uses_blackwell_fp4_tp2() -> None:
    result = recommend_config("Kimi-K2.5", "b200", workload="coding", num_gpus=4)

    profile = result["serving_profile"]
    engine_config = result["engine_config"]
    assert profile["engine"] == "dynamo"
    assert profile["precision"]["weights"] == "fp4"
    assert profile["topology"]["tp"] == 2
    assert profile["topology"]["dp"] == 2
    assert result["memory_plan"]["fits"] is True
    assert engine_config["env_vars"]["DYNAMO_ENABLE_NVCOMP"] == "1"


def test_recommend_gb200_is_rejected_outside_target_scope() -> None:
    result = recommend_config("Kimi-K2.5", "gb200", workload="coding", num_gpus=4)

    assert result["error"] == "Unsupported GPU: 'gb200'"
    assert "H100/H200/B200/B300" in result["summary"]


def test_recommend_engine_matches_recommend_config_for_supported_kimi_workloads() -> None:
    for gpu, num_gpus in (("h100", 8), ("h200", 4), ("b200", 4), ("b300", 1)):
        config = recommend_config("Kimi-K2.5", gpu, workload="coding", num_gpus=num_gpus)
        engines = recommend_engine("Kimi-K2.5", gpu, workload="coding", num_gpus=num_gpus)

        assert engines["rankings"][0]["engine"] == config["engine_config"]["engine"]
