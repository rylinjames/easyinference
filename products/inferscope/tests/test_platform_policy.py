"""Regression tests for engine support policy control flow."""

from __future__ import annotations

from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.optimization.platform_policy import resolve_engine_support


def test_unknown_engine_returns_unsupported() -> None:
    gpu = get_gpu_profile("h200")
    assert gpu is not None

    support = resolve_engine_support("madeup", gpu)

    assert support.engine == "madeup"
    assert support.tier.value == "unsupported"


def test_dynamo_h200_returns_recommended() -> None:
    gpu = get_gpu_profile("h200")
    assert gpu is not None

    support = resolve_engine_support("dynamo", gpu)

    assert support.engine == "dynamo"
    assert support.tier.value == "recommended"
