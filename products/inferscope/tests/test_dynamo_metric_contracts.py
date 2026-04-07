"""Regression coverage: every Dynamo metric prefix / full name declared by
upstream config files must exist in `telemetry/prometheus.py::DYNAMO_METRICS`.

Closes the snapshot v1.0.0 P0 bug
`bugs/dynamo_required_metric_prefixes_fictional.md` and provides a forward
guard against re-introducing fictional names.
"""

from __future__ import annotations

from inferscope.engines import dynamo as dynamo_engine
from inferscope.production_target import (
    BACKEND_METRICS,
    FRONTEND_METRICS,
    KV_METRICS,
)
from inferscope.telemetry.prometheus import DYNAMO_METRICS, LMCACHE_METRICS


def _all_real_metric_names() -> set[str]:
    """The full set of real metric names InferScope knows about."""
    return set(DYNAMO_METRICS.keys()) | set(LMCACHE_METRICS.keys())


# ----------------------------------------------------------------------------
# engines/dynamo.py prefix declarations
# ----------------------------------------------------------------------------


def test_required_primary_prefixes_match_real_metric_names() -> None:
    """Every prefix in `_REQUIRED_PRIMARY_PREFIXES` must match at least one
    real metric name in DYNAMO_METRICS or LMCACHE_METRICS."""
    real_names = _all_real_metric_names()
    for prefix in dynamo_engine._REQUIRED_PRIMARY_PREFIXES:
        matches = [name for name in real_names if name.startswith(prefix)]
        assert matches, (
            f"Prefix '{prefix}' from _REQUIRED_PRIMARY_PREFIXES does not match "
            f"any real metric name in DYNAMO_METRICS or LMCACHE_METRICS"
        )


def test_required_cache_prefixes_match_real_metric_names() -> None:
    """Every prefix in `_REQUIRED_CACHE_PREFIXES` must match a real metric name."""
    real_names = _all_real_metric_names()
    for prefix in dynamo_engine._REQUIRED_CACHE_PREFIXES:
        matches = [name for name in real_names if name.startswith(prefix)]
        assert matches, (
            f"Prefix '{prefix}' from _REQUIRED_CACHE_PREFIXES does not match "
            f"any real metric name. The kvstats namespace was retired in the "
            f"Grove → KVBM correction; cache metrics fall under "
            f"'dynamo_component_*' (covered by _REQUIRED_PRIMARY_PREFIXES)."
        )


def test_required_prefill_prefixes_match_real_metric_names() -> None:
    real_names = _all_real_metric_names()
    for prefix in dynamo_engine._REQUIRED_PREFILL_PREFIXES:
        matches = [name for name in real_names if name.startswith(prefix)]
        assert matches, f"Prefix '{prefix}' from _REQUIRED_PREFILL_PREFIXES is fictional"


def test_required_decode_prefixes_match_real_metric_names() -> None:
    real_names = _all_real_metric_names()
    for prefix in dynamo_engine._REQUIRED_DECODE_PREFIXES:
        matches = [name for name in real_names if name.startswith(prefix)]
        assert matches, f"Prefix '{prefix}' from _REQUIRED_DECODE_PREFIXES is fictional"


def test_no_fictional_kvstats_or_grove_or_slo_prefixes() -> None:
    """Defense-in-depth: assert that none of the historically-fictional
    prefixes have been re-introduced."""
    fictional_substrings = (
        "kvstats",
        "dynamo_grove",
        "dynamo_slo_",
        "dynamo_lmcache",
        "dynamo_router_",  # the engine should use dynamo_router_overhead_, not bare dynamo_router_
        "dynamo_request_",
        "dynamo_scheduler_",
        "dynamo_prefill_",
        "dynamo_decode_",
    )
    all_prefixes: list[str] = (
        dynamo_engine._REQUIRED_PRIMARY_PREFIXES
        + dynamo_engine._REQUIRED_CACHE_PREFIXES
        + dynamo_engine._REQUIRED_PREFILL_PREFIXES
        + dynamo_engine._REQUIRED_DECODE_PREFIXES
    )
    for prefix in all_prefixes:
        for fictional in fictional_substrings:
            # `dynamo_router_overhead_` is NOT fictional and should not match
            # the bare `dynamo_router_` substring check.
            if fictional == "dynamo_router_" and prefix == "dynamo_router_overhead_":
                continue
            assert fictional not in prefix, (
                f"Fictional substring '{fictional}' re-introduced in prefix "
                f"'{prefix}'. See bugs/dynamo_required_metric_prefixes_fictional.md."
            )


# ----------------------------------------------------------------------------
# production_target.py full-name declarations
# ----------------------------------------------------------------------------


def test_frontend_metrics_are_all_real() -> None:
    """Every entry in production_target.FRONTEND_METRICS must exist in DYNAMO_METRICS."""
    real = set(DYNAMO_METRICS.keys())
    for name in FRONTEND_METRICS:
        assert name in real, f"FRONTEND_METRICS contains fictional name '{name}'"


def test_backend_metrics_are_all_real() -> None:
    real = set(DYNAMO_METRICS.keys())
    for name in BACKEND_METRICS:
        assert name in real, f"BACKEND_METRICS contains fictional name '{name}'"


def test_kv_metrics_are_all_real() -> None:
    """KV_METRICS previously contained 4 fictional `dynamo_component_kvstats_*`
    names. The fix replaced them with the 2 real names. This test guards
    against re-introduction."""
    real = set(DYNAMO_METRICS.keys())
    for name in KV_METRICS:
        assert name in real, (
            f"KV_METRICS contains fictional name '{name}'. The kvstats namespace "
            f"does not exist in current Dynamo source — see "
            f"telemetry/prometheus.py:134-148 for the historical correction."
        )


def test_kv_metrics_includes_the_two_real_names() -> None:
    """KV_METRICS must contain at least the two real KV-stats metric names."""
    assert "dynamo_component_total_blocks" in KV_METRICS
    assert "dynamo_component_gpu_cache_usage_percent" in KV_METRICS
