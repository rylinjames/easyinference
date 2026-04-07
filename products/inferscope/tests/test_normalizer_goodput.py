"""Regression coverage for `_compute_goodput` across all 4 engine branches.

Closes the snapshot v1.0.0 P1 bug `goodput_dead_for_three_engines.md`.

The previous implementation read `gen_throughput_tps`, which is only set
by the SGLang branch in `normalize()`. vLLM, Dynamo, and ATOM never set
the field, so `_compute_goodput` short-circuited and `goodput_tps` /
`goodput_ratio` stayed at 0 for those engines. The fix adds a fallback
that derives throughput from `requests_running / itl_avg_s` (both fields
ARE populated by all 4 engines).
"""

from __future__ import annotations

import pytest

from inferscope.telemetry.normalizer import NormalizedMetrics, _compute_goodput


def _baseline_metrics(engine: str) -> NormalizedMetrics:
    """Build a healthy-baseline `NormalizedMetrics` for the given engine.

    Both `requests_running` and `itl_avg_s` are populated; `gen_throughput_tps`
    is left at 0 for non-SGLang engines (matches the production normalize()
    behavior).
    """
    m = NormalizedMetrics(
        engine=engine,
        endpoint=f"http://localhost:8000/metrics",
        requests_running=8.0,
        requests_waiting=0.0,
        itl_avg_s=0.025,  # 25 ms ITL — typical
        request_success_total=1000.0,
        preemptions_total=0.0,
    )
    if engine == "sglang":
        m.gen_throughput_tps = 320.0  # SGLang exposes this directly
    return m


# ----------------------------------------------------------------------------
# SGLang — direct gauge path (already worked before the fix, regression cover)
# ----------------------------------------------------------------------------


def test_compute_goodput_uses_direct_gauge_for_sglang() -> None:
    """SGLang exposes `sglang:gen_throughput` directly. Goodput should match
    raw throughput when there are no preemptions."""
    m = _baseline_metrics("sglang")
    _compute_goodput(m)
    assert m.goodput_tps == pytest.approx(320.0)
    assert m.goodput_ratio == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# vLLM / Dynamo / ATOM — fallback path (the bug fix)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["vllm", "dynamo", "atom"])
def test_compute_goodput_falls_back_to_itl_for_non_sglang_engines(engine: str) -> None:
    """For engines that don't populate `gen_throughput_tps`, goodput must
    derive throughput from requests_running / itl_avg_s.

    With 8 concurrent requests at 25ms ITL: 8 / 0.025 = 320 tokens/sec.
    """
    m = _baseline_metrics(engine)
    assert m.gen_throughput_tps == 0.0  # confirms the engine didn't set it

    _compute_goodput(m)

    assert m.goodput_tps > 0, (
        f"goodput_tps remained 0 for {engine} — fallback path not exercised"
    )
    expected = 8.0 / 0.025  # 320 tps
    assert m.goodput_tps == pytest.approx(expected)
    assert m.goodput_ratio == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# Preemption waste still applies in the fallback path
# ----------------------------------------------------------------------------


def test_compute_goodput_fallback_still_accounts_for_preemption_waste() -> None:
    """The fallback throughput must still be reduced by the preemption-waste
    fraction the original code computed."""
    m = _baseline_metrics("vllm")
    m.preemptions_total = 100.0  # 10% preemption rate over 1000 successes
    # waste_fraction = min(0.10 * 0.5, 0.3) = 0.05

    _compute_goodput(m)

    raw = 8.0 / 0.025  # 320 tps
    expected_goodput = raw * (1.0 - 0.05)
    assert m.goodput_tps == pytest.approx(expected_goodput)
    assert m.goodput_ratio == pytest.approx(0.95)


# ----------------------------------------------------------------------------
# Edge cases — fallback gracefully no-ops when fields are missing
# ----------------------------------------------------------------------------


def test_compute_goodput_noops_when_no_signal_available() -> None:
    """If neither `gen_throughput_tps` nor (requests_running, itl_avg_s) are
    available, the function must no-op (leave goodput at 0)."""
    m = NormalizedMetrics(engine="vllm", endpoint="http://localhost:8000/metrics")
    # All fields at their dataclass defaults
    _compute_goodput(m)
    assert m.goodput_tps == 0.0
    assert m.goodput_ratio == 0.0


def test_compute_goodput_noops_when_itl_is_zero() -> None:
    """Avoid divide-by-zero when itl_avg_s is exactly 0."""
    m = NormalizedMetrics(
        engine="dynamo",
        endpoint="http://localhost:8000/metrics",
        requests_running=8.0,
        itl_avg_s=0.0,
    )
    _compute_goodput(m)
    assert m.goodput_tps == 0.0
    assert m.goodput_ratio == 0.0


def test_compute_goodput_noops_when_requests_running_is_zero() -> None:
    """A deployment with zero in-flight requests has no throughput to compute."""
    m = NormalizedMetrics(
        engine="vllm",
        endpoint="http://localhost:8000/metrics",
        requests_running=0.0,
        itl_avg_s=0.025,
    )
    _compute_goodput(m)
    assert m.goodput_tps == 0.0
