"""Unit tests for the 31 ISA-grounded audit checks.

Each check in `inferscope.optimization.checks` is a pure function that takes
`NormalizedMetrics` + `DeploymentContext` and returns `AuditFinding | None`.

Tests here encode the intended fire/no-fire behavior check-by-check so that:
  * regressions in threshold math show up as failing tests
  * the check contract is documented in executable form
  * counter-vs-rate bugs (where a raw Prometheus counter is compared against
    a static threshold) are caught and locked down

Organized in the same tier order as the coverage survey: Tier 1 high-risk
checks first, then hardware-gated guards, then bulk thresholds.
"""

from __future__ import annotations

from inferscope.optimization.checks import (
    DeploymentContext,
    _check_gpu_underutilization,
    _check_grove_eviction_storm,
    _check_grove_tier_imbalance,
    _check_high_itl,
    _check_high_ttft,
    _check_kv_cache_critical,
    _check_kv_preemption_storm,
    _check_missing_quantization,
    _check_nixl_transfer_dominates,
    _check_oom_despite_free,
    _check_pcie_offload_thrash,
    _check_prefix_cache_disabled,
    run_all_checks,
)
from inferscope.telemetry.normalizer import NormalizedMetrics


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _metrics(**overrides) -> NormalizedMetrics:
    """Build a healthy-baseline NormalizedMetrics and apply overrides.

    The baseline represents a deployment that is serving traffic successfully
    with no bottlenecks — all checks should no-fire against it.
    """
    base = NormalizedMetrics(
        engine="vllm",
        endpoint="http://localhost:8000/metrics",
        requests_running=8.0,
        requests_waiting=0.0,
        kv_cache_usage=0.45,
        prefix_cache_hit_rate=0.70,
        request_success_total=1000.0,
        preemptions_total=0.0,
        ttft_avg_s=0.3,
        itl_avg_s=0.02,
        gen_throughput_tps=1500.0,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _ctx(**overrides) -> DeploymentContext:
    """Build a healthy-baseline DeploymentContext and apply overrides.

    The baseline represents an H100 Dynamo/vLLM deployment configured sensibly:
    FP8 enabled on an FP8-capable GPU, prefix caching on, no AMD env vars.
    """
    base = DeploymentContext(
        engine="vllm",
        gpu_arch="sm_90a",
        gpu_name="H100",
        gpu_memory_gb=80.0,
        gpu_vendor="nvidia",
        model_name="Kimi-K2.5",
        model_type="dense",
        attention_type="GQA",
        tp=1,
        fp8_support=True,
        fp8_format="OCP",
        gpu_memory_utilization=0.93,
        kv_cache_dtype="fp8_e4m3",
        quantization="fp8",
        block_size=16,
        prefix_caching=True,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


# ----------------------------------------------------------------------------
# Baseline sanity: a healthy deployment fires no checks
# ----------------------------------------------------------------------------


def test_healthy_baseline_fires_no_checks() -> None:
    findings = run_all_checks(_metrics(), _ctx())
    assert findings == [], f"Healthy baseline should be silent, got: {[f.check_id for f in findings]}"


# ----------------------------------------------------------------------------
# Tier 1: KV_PREEMPTION_STORM — counter-vs-rate bug
# ----------------------------------------------------------------------------


def test_kv_preemption_storm_fires_on_active_storm() -> None:
    """Clear current storm: many preemptions relative to successes should fire."""
    m = _metrics(preemptions_total=50.0, request_success_total=200.0)  # 25% rate
    assert _check_kv_preemption_storm(m, _ctx()) is not None


def test_kv_preemption_storm_silent_on_accumulated_counter() -> None:
    """A long-running deployment with a small accumulated preemption count
    relative to total successful requests should NOT fire.

    This test encodes the correct behavior: the check must look at
    preemptions as a rate (relative to successes), not as a raw counter.
    `normalizer.py:181-183` already treats preemptions_total this way
    for the goodput calculation.

    Under the original buggy implementation this test FAILS because the
    check uses a raw counter threshold (`preemptions_total > 10`), which
    false-positives on any deployment that has been serving long enough
    to accumulate >10 preemptions in its lifetime — even if current
    health is fine.
    """
    # Deployment has served 10,000 requests successfully; 11 preemptions
    # over that lifetime is ~0.1% — noise, not a storm.
    m = _metrics(preemptions_total=11.0, request_success_total=10_000.0)
    assert _check_kv_preemption_storm(m, _ctx()) is None


def test_kv_preemption_storm_silent_without_successes_baseline() -> None:
    """Absent a success baseline we cannot compute a rate, so the check
    should stay silent rather than false-positive on raw counter value."""
    m = _metrics(preemptions_total=20.0, request_success_total=0.0)
    assert _check_kv_preemption_storm(m, _ctx()) is None


# ----------------------------------------------------------------------------
# Tier 1: OOM_DESPITE_FREE — same counter bug
# ----------------------------------------------------------------------------


def test_oom_despite_free_fires_on_fragmentation() -> None:
    """Active fragmentation: preemption rate is high while KV has headroom."""
    m = _metrics(
        preemptions_total=30.0,
        request_success_total=200.0,  # 15% rate
        kv_cache_usage=0.5,  # plenty of free memory — must be fragmentation
    )
    assert _check_oom_despite_free(m, _ctx()) is not None


def test_oom_despite_free_silent_on_accumulated_counter() -> None:
    """Same counter-vs-rate bug as KV_PREEMPTION_STORM. A long-running
    deployment with 6 lifetime preemptions out of 20,000 successes
    (~0.03%) is healthy — it must not false-positive."""
    m = _metrics(
        preemptions_total=6.0,
        request_success_total=20_000.0,
        kv_cache_usage=0.5,
    )
    assert _check_oom_despite_free(m, _ctx()) is None


# ----------------------------------------------------------------------------
# Tier 1: PCIE_OFFLOAD_THRASH — same counter bug
# ----------------------------------------------------------------------------


def test_pcie_offload_thrash_fires_on_active_thrash() -> None:
    """CPU offload active + high preemption rate + elevated ITL = thrash."""
    m = _metrics(
        cpu_cache_usage=0.3,
        preemptions_total=50.0,
        request_success_total=200.0,  # 25% rate
        requests_running=20.0,
        itl_avg_s=0.12,
    )
    assert _check_pcie_offload_thrash(m, _ctx()) is not None


def test_pcie_offload_thrash_silent_on_accumulated_counter() -> None:
    """Lifetime preemption count of 6 across 20,000 successes is noise —
    CPU offload being active for cold sessions is normal, not a crisis."""
    m = _metrics(
        cpu_cache_usage=0.3,
        preemptions_total=6.0,
        request_success_total=20_000.0,
        requests_running=20.0,
        itl_avg_s=0.12,
    )
    assert _check_pcie_offload_thrash(m, _ctx()) is None


# ----------------------------------------------------------------------------
# Tier 1: GROVE_EVICTION_STORM — same counter bug, different metric
# ----------------------------------------------------------------------------


def test_grove_eviction_storm_fires_on_active_churn() -> None:
    """Many evictions relative to successes while GPU tier is full = real churn."""
    m = _metrics(
        grove_evictions=500.0,
        request_success_total=2_000.0,  # 25% eviction rate
        grove_tier_gpu_pct=0.92,
    )
    assert _check_grove_eviction_storm(m, _ctx()) is not None


def test_grove_eviction_storm_silent_on_accumulated_counter() -> None:
    """101 lifetime evictions across 50,000 successful requests (~0.2%)
    with a warm GPU tier is normal steady-state behavior, not a storm."""
    m = _metrics(
        grove_evictions=101.0,
        request_success_total=50_000.0,
        grove_tier_gpu_pct=0.86,
    )
    assert _check_grove_eviction_storm(m, _ctx()) is None


# ----------------------------------------------------------------------------
# Tier 1: MISSING_QUANTIZATION — hardware-constraint check
# ----------------------------------------------------------------------------


def test_missing_quantization_fires_on_bf16_with_fp8_gpu() -> None:
    finding = _check_missing_quantization(_metrics(), _ctx(quantization="bf16"))
    assert finding is not None
    assert finding.check_id == "MISSING_QUANTIZATION"


def test_missing_quantization_silent_when_fp8_configured() -> None:
    # healthy baseline already has quantization="fp8"
    assert _check_missing_quantization(_metrics(), _ctx()) is None


def test_missing_quantization_silent_on_non_fp8_gpu() -> None:
    # An A10G (Ampere, no FP8) running bf16 is correct, not wrong.
    finding = _check_missing_quantization(
        _metrics(),
        _ctx(gpu_name="A10G", gpu_arch="sm_86", fp8_support=False, quantization="bf16"),
    )
    assert finding is None


def test_missing_quantization_fires_on_empty_quant_string() -> None:
    # Empty quantization is equivalent to 'auto' — should still flag on FP8 GPUs.
    finding = _check_missing_quantization(_metrics(), _ctx(quantization=""))
    assert finding is not None


# ----------------------------------------------------------------------------
# Tier 1: KV_CACHE_CRITICAL — threshold boundary
# ----------------------------------------------------------------------------


def test_kv_cache_critical_fires_above_95() -> None:
    assert _check_kv_cache_critical(_metrics(kv_cache_usage=0.96), _ctx()) is not None


def test_kv_cache_critical_silent_at_threshold() -> None:
    # Strict > not >=, so 0.95 exactly should not fire.
    assert _check_kv_cache_critical(_metrics(kv_cache_usage=0.95), _ctx()) is None


def test_kv_cache_critical_silent_in_healthy_range() -> None:
    assert _check_kv_cache_critical(_metrics(kv_cache_usage=0.70), _ctx()) is None


# ----------------------------------------------------------------------------
# Tier 1: HIGH_TTFT / HIGH_ITL — unit consistency (seconds, not ms)
# ----------------------------------------------------------------------------


def test_high_ttft_fires_above_5_seconds() -> None:
    finding = _check_high_ttft(_metrics(ttft_avg_s=6.0), _ctx())
    assert finding is not None
    # Description must render seconds→ms correctly (6.0s = 6000ms).
    assert "6000ms" in finding.title


def test_high_ttft_silent_when_unset() -> None:
    assert _check_high_ttft(_metrics(ttft_avg_s=None), _ctx()) is None


def test_high_ttft_silent_at_threshold() -> None:
    # Strict > threshold; 5.0 exactly should not fire.
    assert _check_high_ttft(_metrics(ttft_avg_s=5.0), _ctx()) is None


def test_high_itl_fires_above_100ms() -> None:
    finding = _check_high_itl(_metrics(itl_avg_s=0.15), _ctx())
    assert finding is not None
    assert "150ms" in finding.title


def test_high_itl_silent_when_unset() -> None:
    assert _check_high_itl(_metrics(itl_avg_s=None), _ctx()) is None


# ----------------------------------------------------------------------------
# Tier 1: GPU_UNDERUTILIZATION — paired signal
# ----------------------------------------------------------------------------


def test_gpu_underutilization_fires_when_queue_despite_headroom() -> None:
    m = _metrics(kv_cache_usage=0.15, requests_waiting=20.0)
    assert _check_gpu_underutilization(m, _ctx()) is not None


def test_gpu_underutilization_silent_when_queue_matches_load() -> None:
    # High queue is expected when KV is also full.
    m = _metrics(kv_cache_usage=0.90, requests_waiting=20.0)
    assert _check_gpu_underutilization(m, _ctx()) is None


def test_gpu_underutilization_silent_when_idle() -> None:
    # No queue is not a problem regardless of KV state.
    m = _metrics(kv_cache_usage=0.15, requests_waiting=0.0)
    assert _check_gpu_underutilization(m, _ctx()) is None


# ----------------------------------------------------------------------------
# Tier 1: PREFIX_CACHE_DISABLED — narrow intent check
# ----------------------------------------------------------------------------


def test_prefix_cache_disabled_fires_when_declared_off_with_no_hits() -> None:
    m = _metrics(prefix_cache_hit_rate=0.0)
    assert _check_prefix_cache_disabled(m, _ctx(prefix_caching=False)) is not None


def test_prefix_cache_disabled_silent_when_declared_on() -> None:
    # Intentionally narrow: the check only catches the declared-off case,
    # not "declared on but silently broken". Document the narrow intent.
    m = _metrics(prefix_cache_hit_rate=0.0)
    assert _check_prefix_cache_disabled(m, _ctx(prefix_caching=True)) is None


# ----------------------------------------------------------------------------
# Tier 1: NIXL_TRANSFER_DOMINATES — disagg-only ratio check
# ----------------------------------------------------------------------------


def test_nixl_transfer_dominates_fires_in_disagg_with_high_latency() -> None:
    m = _metrics(nixl_transfer_latency_s=0.8, ttft_avg_s=4.0)
    assert _check_nixl_transfer_dominates(m, _ctx(split_prefill_decode=True)) is not None


def test_nixl_transfer_dominates_silent_in_aggregated_topology() -> None:
    # NIXL metrics don't apply to single-endpoint deployments.
    m = _metrics(nixl_transfer_latency_s=0.8, ttft_avg_s=4.0)
    assert _check_nixl_transfer_dominates(m, _ctx(split_prefill_decode=False)) is None


def test_nixl_transfer_dominates_silent_when_metric_missing() -> None:
    m = _metrics(nixl_transfer_latency_s=None, ttft_avg_s=4.0)
    assert _check_nixl_transfer_dominates(m, _ctx(split_prefill_decode=True)) is None


# ----------------------------------------------------------------------------
# Tier 1: GROVE_TIER_IMBALANCE — multi-tier ratio
# ----------------------------------------------------------------------------


def test_grove_tier_imbalance_fires_when_gpu_saturated_but_lower_tiers_empty() -> None:
    m = _metrics(grove_tier_gpu_pct=0.95, grove_tier_cpu_pct=0.05, grove_tier_ssd_pct=0.0)
    assert _check_grove_tier_imbalance(m, _ctx()) is not None


def test_grove_tier_imbalance_silent_when_tiers_balanced() -> None:
    m = _metrics(grove_tier_gpu_pct=0.80, grove_tier_cpu_pct=0.40, grove_tier_ssd_pct=0.15)
    assert _check_grove_tier_imbalance(m, _ctx()) is None


# ----------------------------------------------------------------------------
# Integration: run_all_checks ordering contract
# ----------------------------------------------------------------------------


def test_run_all_checks_orders_by_severity() -> None:
    """Critical findings must come before warnings, warnings before info."""
    # Construct a scenario that fires at least one of each severity class:
    #   critical: KV_CACHE_CRITICAL (kv > 0.95)
    #   warning:  HIGH_ITL          (itl > 0.1)
    #   info:     MEMORY_UTIL_LOW   (0 < gpu_memory_util < 0.9)
    m = _metrics(kv_cache_usage=0.98, itl_avg_s=0.15)
    c = _ctx(gpu_memory_utilization=0.80)
    findings = run_all_checks(m, c)

    severities = [f.severity for f in findings]
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    ranks = [severity_rank.get(s, 99) for s in severities]
    assert ranks == sorted(ranks), f"Findings not severity-ordered: {severities}"

    ids = {f.check_id for f in findings}
    assert "KV_CACHE_CRITICAL" in ids
    assert "HIGH_ITL" in ids
    assert "MEMORY_UTIL_LOW" in ids
