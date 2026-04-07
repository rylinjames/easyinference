"""Regression coverage for the cpu_cache_usage Dynamo fallback.

Closes the snapshot v1.0.0 P1 bug `cpu_cache_usage_dynamo_zero.md`.

Previously the Dynamo branch in `normalize()` did NOT set
`cpu_cache_usage`, even though Dynamo's vLLM workers expose
`vllm:cpu_cache_usage_perc` whenever an operator scrapes a vLLM worker
endpoint alongside the Dynamo frontend. The field stayed at 0 for
Dynamo deployments, which silently disabled the
`_check_pcie_offload_thrash` audit check (it gates on
`m.cpu_cache_usage > 0.1`).
"""

from __future__ import annotations

from inferscope.telemetry.normalizer import normalize
from inferscope.telemetry.prometheus import ScrapeResult


def _dynamo_scrape(extra: dict[str, float] | None = None) -> ScrapeResult:
    """Build a synthetic Dynamo scrape with the minimum metrics needed."""
    raw = {
        # Minimum Dynamo identity metric (the engine detector keys on a
        # `dynamo_*` prefix being present, but normalize() takes the
        # engine field directly so we don't need to fake the detector).
        "dynamo_frontend_inflight_requests": 8.0,
        "dynamo_frontend_queued_requests": 0.0,
    }
    if extra:
        raw.update(extra)
    return ScrapeResult(
        endpoint="http://test.local/metrics",
        engine="dynamo",
        raw_metrics=raw,
    )


# ----------------------------------------------------------------------------
# The fix: vllm:cpu_cache_usage_perc must populate cpu_cache_usage on Dynamo
# ----------------------------------------------------------------------------


def test_dynamo_normalize_reads_vllm_cpu_cache_usage_perc() -> None:
    """When Dynamo's vLLM worker exposes `vllm:cpu_cache_usage_perc`, the
    Dynamo branch in normalize() must populate `cpu_cache_usage` from it.

    This is the one-line fix in the snapshot v1.0.0 P1 bug
    `cpu_cache_usage_dynamo_zero.md`.
    """
    scrape = _dynamo_scrape({"vllm:cpu_cache_usage_perc": 0.15})
    m = normalize(scrape)
    assert m.cpu_cache_usage == 0.15


def test_dynamo_normalize_handles_missing_vllm_cpu_cache_metric() -> None:
    """If Dynamo is scraped without a vLLM worker endpoint, cpu_cache_usage
    should stay at 0 (the dataclass default), not crash."""
    scrape = _dynamo_scrape()  # no vllm:cpu_cache_usage_perc key
    m = normalize(scrape)
    assert m.cpu_cache_usage == 0.0


def test_dynamo_cpu_cache_usage_above_threshold_can_now_fire_pcie_check() -> None:
    """A Dynamo deployment with high cpu_cache_usage must produce a metric
    that the `_check_pcie_offload_thrash` audit check can read. This is
    the downstream value of the fix — the check gate is `> 0.1`.

    We don't run the check itself here (that's covered in test_checks.py);
    we only verify that the field is populated above the threshold."""
    scrape = _dynamo_scrape({"vllm:cpu_cache_usage_perc": 0.42})
    m = normalize(scrape)
    assert m.cpu_cache_usage > 0.1, (
        "fix did not populate cpu_cache_usage above the PCIE_OFFLOAD_THRASH "
        "check threshold (0.1) — `_check_pcie_offload_thrash` will silently "
        "skip even when KV is thrashing"
    )


# ----------------------------------------------------------------------------
# Regression: vLLM branch still works (this was always working)
# ----------------------------------------------------------------------------


def test_vllm_normalize_still_reads_cpu_cache_usage_perc() -> None:
    """The original vLLM branch path must continue to work."""
    scrape = ScrapeResult(
        endpoint="http://test.local/metrics",
        engine="vllm",
        raw_metrics={
            "vllm:cpu_cache_usage_perc": 0.27,
        },
    )
    m = normalize(scrape)
    assert m.cpu_cache_usage == 0.27
