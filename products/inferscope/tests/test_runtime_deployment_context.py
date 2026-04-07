"""Regression coverage for `build_deployment_context` env_vars and prefix_caching plumbing.

Closes the snapshot v1.0.0 P0 bug `runtime_deployment_context_dead_branches.md`.

Previously `build_deployment_context` hardcoded `env_vars={}` and
`prefix_caching=True`, silently disabling 5 audit checks:
- `_check_prefix_cache_disabled` (silenced)
- `_check_aiter_disabled` (false positive on AMD)
- `_check_wrong_attention_backend` (silenced)
- `_check_fp8bmm_crash_risk` (silenced — the most dangerous one,
  it gates a "WILL CRASH" warning on MI300X)

The fix adds `env_vars: dict[str, str]` and `prefix_caching: bool`
fields to `RuntimeContextHints`, threads them through
`build_deployment_context`, and exposes them as parameters on
`audit_deployment`, `profile_runtime`, and `auto_tune_deployment`.
"""

from __future__ import annotations

from inferscope.optimization.checks import (
    DeploymentContext,
    _check_fp8bmm_crash_risk,
    _check_prefix_cache_disabled,
    run_all_checks,
)
from inferscope.profiling.models import RuntimeContextHints
from inferscope.profiling.runtime import build_deployment_context
from inferscope.telemetry.normalizer import NormalizedMetrics


# ----------------------------------------------------------------------------
# Schema: RuntimeContextHints carries the new fields
# ----------------------------------------------------------------------------


def test_runtime_context_hints_has_env_vars_and_prefix_caching() -> None:
    """The fix adds 2 fields to RuntimeContextHints."""
    hints = RuntimeContextHints(
        env_vars={"VLLM_ROCM_USE_AITER_FP8BMM": "1"},
        prefix_caching=False,
    )
    assert hints.env_vars == {"VLLM_ROCM_USE_AITER_FP8BMM": "1"}
    assert hints.prefix_caching is False


def test_runtime_context_hints_defaults_are_safe() -> None:
    """Default `env_vars={}` and `prefix_caching=True` preserves the legacy behavior
    for callers that don't supply the new fields."""
    hints = RuntimeContextHints()
    assert hints.env_vars == {}
    assert hints.prefix_caching is True


# ----------------------------------------------------------------------------
# build_deployment_context honors the new fields
# ----------------------------------------------------------------------------


def test_build_deployment_context_propagates_env_vars() -> None:
    """The previously-hardcoded `env_vars={}` must now reflect the hint."""
    metrics = NormalizedMetrics(engine="vllm", endpoint="http://test/metrics")
    hints = RuntimeContextHints(
        gpu_arch="gfx942",
        env_vars={"VLLM_ROCM_USE_AITER_FP8BMM": "1"},
    )
    ctx = build_deployment_context(metrics, hints)
    assert ctx.env_vars == {"VLLM_ROCM_USE_AITER_FP8BMM": "1"}


def test_build_deployment_context_propagates_prefix_caching() -> None:
    """The previously-hardcoded `prefix_caching=True` must now reflect the hint."""
    metrics = NormalizedMetrics(engine="vllm", endpoint="http://test/metrics")
    hints = RuntimeContextHints(prefix_caching=False)
    ctx = build_deployment_context(metrics, hints)
    assert ctx.prefix_caching is False


def test_build_deployment_context_defaults_when_hint_omits_fields() -> None:
    """Backward compatibility — old call sites that don't set the new
    fields get the legacy defaults (empty dict + True)."""
    metrics = NormalizedMetrics(engine="vllm", endpoint="http://test/metrics")
    hints = RuntimeContextHints()  # all defaults
    ctx = build_deployment_context(metrics, hints)
    assert ctx.env_vars == {}
    assert ctx.prefix_caching is True


# ----------------------------------------------------------------------------
# The 4 affected audit checks now actually fire
# ----------------------------------------------------------------------------


def test_fp8bmm_crash_risk_check_now_fires_via_build_deployment_context() -> None:
    """The end-to-end test: the bug doc's headline scenario.

    A user calling audit_deployment with `gpu_arch="gfx942"` and
    `env_vars={"VLLM_ROCM_USE_AITER_FP8BMM": "1"}` MUST see the
    FP8BMM_CRASH_RISK finding. Before the fix, the env_vars dict
    was dropped at build_deployment_context, so this critical
    "WILL CRASH" warning never reached the operator.
    """
    metrics = NormalizedMetrics(engine="vllm", endpoint="http://test/metrics")
    hints = RuntimeContextHints(
        gpu_arch="gfx942",
        env_vars={"VLLM_ROCM_USE_AITER_FP8BMM": "1"},
    )
    ctx = build_deployment_context(metrics, hints)

    # Direct check call
    finding = _check_fp8bmm_crash_risk(metrics, ctx)
    assert finding is not None, "FP8BMM_CRASH_RISK check did not fire — env_vars dropped"
    assert finding.check_id == "FP8BMM_CRASH_RISK"
    assert finding.severity == "critical"


def test_fp8bmm_crash_risk_does_not_fire_without_the_env_var() -> None:
    """Inverse: a gfx942 deployment WITHOUT the dangerous env var should NOT fire."""
    metrics = NormalizedMetrics(engine="vllm", endpoint="http://test/metrics")
    hints = RuntimeContextHints(gpu_arch="gfx942", env_vars={})
    ctx = build_deployment_context(metrics, hints)
    assert _check_fp8bmm_crash_risk(metrics, ctx) is None


def test_prefix_cache_disabled_check_now_fires() -> None:
    """An operator who passes `prefix_caching=False` and has 0% hit rate
    must see the PREFIX_CACHE_DISABLED finding."""
    metrics = NormalizedMetrics(
        engine="vllm",
        endpoint="http://test/metrics",
        prefix_cache_hit_rate=0.0,
    )
    hints = RuntimeContextHints(prefix_caching=False)
    ctx = build_deployment_context(metrics, hints)

    finding = _check_prefix_cache_disabled(metrics, ctx)
    assert finding is not None
    assert finding.check_id == "PREFIX_CACHE_DISABLED"


def test_prefix_cache_disabled_does_not_fire_when_caching_is_on() -> None:
    """Default `prefix_caching=True` keeps the check silent for the common case."""
    metrics = NormalizedMetrics(
        engine="vllm",
        endpoint="http://test/metrics",
        prefix_cache_hit_rate=0.0,
    )
    hints = RuntimeContextHints()  # prefix_caching defaults to True
    ctx = build_deployment_context(metrics, hints)
    assert _check_prefix_cache_disabled(metrics, ctx) is None


# ----------------------------------------------------------------------------
# End-to-end: run_all_checks surfaces the FP8BMM finding
# ----------------------------------------------------------------------------


def test_run_all_checks_surfaces_fp8bmm_crash_risk_when_env_var_is_set() -> None:
    """The full audit pipeline must surface the FP8BMM finding when run
    against a gfx942 deployment with the dangerous env var."""
    metrics = NormalizedMetrics(engine="vllm", endpoint="http://test/metrics")
    hints = RuntimeContextHints(
        gpu_arch="gfx942",
        env_vars={"VLLM_ROCM_USE_AITER_FP8BMM": "1"},
    )
    ctx = build_deployment_context(metrics, hints)

    findings = run_all_checks(metrics, ctx)
    finding_ids = [f.check_id for f in findings]
    assert "FP8BMM_CRASH_RISK" in finding_ids, (
        f"FP8BMM_CRASH_RISK not in findings: {finding_ids}. "
        "build_deployment_context dropped the env_vars again."
    )
