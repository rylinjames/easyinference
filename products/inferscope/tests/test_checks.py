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

import pytest

from inferscope.optimization.checks import (
    DeploymentContext,
    _check_aiter_disabled,
    _check_atom_not_used,
    _check_batch_itl_tradeoff,
    _check_batch_size_mismatch,
    _check_block_size_wrong,
    _check_decode_starvation,
    _check_disagg_without_rdma,
    _check_fp8bmm_crash_risk,
    _check_gpu_underutilization,
    _check_grove_eviction_storm,
    _check_grove_tier_imbalance,
    _check_high_itl,
    _check_high_queue_depth,
    _check_high_ttft,
    _check_kv_cache_critical,
    _check_kv_dtype_suboptimal,
    _check_kv_fragmentation_high,
    _check_kv_preemption_storm,
    _check_lmcache_cold_start,
    _check_low_prefix_hit_coding,
    _check_memory_util_low,
    _check_missing_quantization,
    _check_moe_ep_missing,
    _check_nixl_transfer_dominates,
    _check_oom_despite_free,
    _check_pcie_offload_thrash,
    _check_prefill_starvation,
    _check_prefix_cache_disabled,
    _check_slo_violation_rate,
    _check_speculative_overhead,
    _check_wrong_attention_backend,
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


# ============================================================================
# TIER 2: Hardware-gated guards
# ============================================================================


# ---------- AITER_DISABLED ----------


def test_aiter_disabled_fires_on_amd_without_env() -> None:
    ctx = _ctx(gpu_vendor="amd", gpu_arch="gfx942", gpu_name="MI300X", env_vars={})
    assert _check_aiter_disabled(_metrics(), ctx) is not None


def test_aiter_disabled_silent_when_aiter_enabled() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        gpu_arch="gfx942",
        gpu_name="MI300X",
        env_vars={"VLLM_ROCM_USE_AITER": "1"},
    )
    assert _check_aiter_disabled(_metrics(), ctx) is None


def test_aiter_disabled_silent_on_nvidia() -> None:
    # Baseline ctx is NVIDIA; env is empty. Must not fire.
    assert _check_aiter_disabled(_metrics(), _ctx()) is None


# ---------- BLOCK_SIZE_WRONG ----------


def test_block_size_wrong_fires_for_mla_on_rocm_without_block_1() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        gpu_arch="gfx942",
        engine="vllm",
        attention_type="MLA",
        block_size=16,
    )
    assert _check_block_size_wrong(_metrics(), ctx) is not None


def test_block_size_wrong_silent_when_block_size_1() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        gpu_arch="gfx942",
        engine="vllm",
        attention_type="MLA",
        block_size=1,
    )
    assert _check_block_size_wrong(_metrics(), ctx) is None


def test_block_size_wrong_silent_on_nvidia_mla() -> None:
    # MLA on NVIDIA vLLM doesn't need block-size 1 — the ROCm kernel quirk
    # is specifically the AMD path.
    ctx = _ctx(attention_type="MLA", block_size=16)
    assert _check_block_size_wrong(_metrics(), ctx) is None


def test_block_size_wrong_silent_for_gqa_on_rocm() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        gpu_arch="gfx942",
        engine="vllm",
        attention_type="GQA",
        block_size=16,
    )
    assert _check_block_size_wrong(_metrics(), ctx) is None


# ---------- KV_DTYPE_SUBOPTIMAL ----------


def test_kv_dtype_suboptimal_fires_with_auto_on_fp8_gpu() -> None:
    ctx = _ctx(kv_cache_dtype="auto")
    assert _check_kv_dtype_suboptimal(_metrics(), ctx) is not None


def test_kv_dtype_suboptimal_fires_with_empty_string() -> None:
    ctx = _ctx(kv_cache_dtype="")
    assert _check_kv_dtype_suboptimal(_metrics(), ctx) is not None


def test_kv_dtype_suboptimal_silent_when_fp8_configured() -> None:
    # Baseline has kv_cache_dtype="fp8_e4m3" already
    assert _check_kv_dtype_suboptimal(_metrics(), _ctx()) is None


def test_kv_dtype_suboptimal_silent_on_non_fp8_gpu() -> None:
    ctx = _ctx(fp8_support=False, kv_cache_dtype="auto")
    assert _check_kv_dtype_suboptimal(_metrics(), ctx) is None


# ---------- MOE_EP_MISSING ----------


def test_moe_ep_missing_fires_for_large_moe_without_ep() -> None:
    ctx = _ctx(model_type="moe", experts_total=128, tp=8, ep=0)
    assert _check_moe_ep_missing(_metrics(), ctx) is not None


def test_moe_ep_missing_silent_when_ep_enabled() -> None:
    ctx = _ctx(model_type="moe", experts_total=128, tp=4, ep=2)
    assert _check_moe_ep_missing(_metrics(), ctx) is None


def test_moe_ep_missing_silent_for_dense_model() -> None:
    ctx = _ctx(model_type="dense", tp=8)
    assert _check_moe_ep_missing(_metrics(), ctx) is None


def test_moe_ep_missing_silent_for_small_moe() -> None:
    # 32 experts — not big enough to benefit from EP.
    ctx = _ctx(model_type="moe", experts_total=32, tp=8, ep=0)
    assert _check_moe_ep_missing(_metrics(), ctx) is None


# ---------- ATOM_NOT_USED ----------


def test_atom_not_used_fires_for_mla_on_mi355x_with_vllm() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        gpu_arch="gfx950",
        gpu_name="MI355X",
        engine="vllm",
        attention_type="MLA",
    )
    assert _check_atom_not_used(_metrics(), ctx) is not None


def test_atom_not_used_silent_when_using_atom() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        gpu_arch="gfx950",
        gpu_name="MI355X",
        engine="atom",
        attention_type="MLA",
    )
    assert _check_atom_not_used(_metrics(), ctx) is None


def test_atom_not_used_silent_on_mi300x() -> None:
    # gfx942 is MI300X, not MI355X — this check is specifically for the
    # CDNA4 MI355X that ATOM is tuned for.
    ctx = _ctx(
        gpu_vendor="amd",
        gpu_arch="gfx942",
        gpu_name="MI300X",
        engine="vllm",
        attention_type="MLA",
    )
    assert _check_atom_not_used(_metrics(), ctx) is None


# ---------- WRONG_ATTENTION_BACKEND (two sub-cases) ----------


def test_wrong_attention_backend_fires_mla_model_on_fa_backend() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        attention_type="MLA",
        env_vars={"ROCM_AITER_FA": "1"},
    )
    finding = _check_wrong_attention_backend(_metrics(), ctx)
    assert finding is not None
    assert finding.severity == "critical"  # MLA-on-FA is worse than FA-on-MLA


def test_wrong_attention_backend_fires_gqa_model_on_mla_backend() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        attention_type="GQA",
        env_vars={"ROCM_AITER_MLA": "1"},
    )
    finding = _check_wrong_attention_backend(_metrics(), ctx)
    assert finding is not None
    assert finding.severity == "warning"  # GQA-on-MLA just suboptimal, not broken


def test_wrong_attention_backend_silent_when_matched() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        attention_type="MLA",
        env_vars={"ROCM_AITER_MLA": "1"},
    )
    assert _check_wrong_attention_backend(_metrics(), ctx) is None


def test_wrong_attention_backend_silent_on_nvidia() -> None:
    # AMD-specific check — NVIDIA never triggers.
    ctx = _ctx(attention_type="MLA", env_vars={"ROCM_AITER_FA": "1"})
    assert _check_wrong_attention_backend(_metrics(), ctx) is None


# ---------- FP8BMM_CRASH_RISK ----------


def test_fp8bmm_crash_risk_fires_on_mi300x_with_flag() -> None:
    ctx = _ctx(
        gpu_vendor="amd",
        gpu_arch="gfx942",
        env_vars={"VLLM_ROCM_USE_AITER_FP8BMM": "1"},
    )
    finding = _check_fp8bmm_crash_risk(_metrics(), ctx)
    assert finding is not None
    assert finding.severity == "critical"


def test_fp8bmm_crash_risk_silent_on_mi355x() -> None:
    # gfx950 (MI355X) is where FP8BMM actually works.
    ctx = _ctx(
        gpu_vendor="amd",
        gpu_arch="gfx950",
        env_vars={"VLLM_ROCM_USE_AITER_FP8BMM": "1"},
    )
    assert _check_fp8bmm_crash_risk(_metrics(), ctx) is None


def test_fp8bmm_crash_risk_silent_without_flag() -> None:
    ctx = _ctx(gpu_vendor="amd", gpu_arch="gfx942", env_vars={})
    assert _check_fp8bmm_crash_risk(_metrics(), ctx) is None


# ---------- DISAGG_WITHOUT_RDMA ----------


def test_disagg_without_rdma_fires_when_disagg_but_no_rdma() -> None:
    ctx = _ctx(split_prefill_decode=True, has_rdma=False)
    assert _check_disagg_without_rdma(_metrics(), ctx) is not None


def test_disagg_without_rdma_silent_with_rdma() -> None:
    ctx = _ctx(split_prefill_decode=True, has_rdma=True)
    assert _check_disagg_without_rdma(_metrics(), ctx) is None


def test_disagg_without_rdma_silent_for_aggregated() -> None:
    ctx = _ctx(split_prefill_decode=False, has_rdma=False)
    assert _check_disagg_without_rdma(_metrics(), ctx) is None


# ============================================================================
# TIER 3: Bulk threshold checks
# ============================================================================


# ---------- MEMORY_UTIL_LOW ----------


@pytest.mark.parametrize(
    "util,should_fire",
    [
        (0.80, True),   # clearly conservative
        (0.85, True),   # still below threshold
        (0.90, False),  # at threshold — check uses strict `< 0.90`
        (0.92, False),  # in recommended range
        (0.95, False),  # aggressive but fine
        (0.0, False),   # not set / not configured — must stay silent
    ],
)
def test_memory_util_low(util: float, should_fire: bool) -> None:
    ctx = _ctx(gpu_memory_utilization=util)
    fired = _check_memory_util_low(_metrics(), ctx) is not None
    assert fired is should_fire


# ---------- HIGH_QUEUE_DEPTH ----------


@pytest.mark.parametrize(
    "waiting,should_fire",
    [
        (0.0, False),
        (10.0, False),
        (50.0, False),   # strict `> 50`
        (51.0, True),
        (200.0, True),
    ],
)
def test_high_queue_depth(waiting: float, should_fire: bool) -> None:
    fired = _check_high_queue_depth(_metrics(requests_waiting=waiting), _ctx()) is not None
    assert fired is should_fire


# ---------- LOW_PREFIX_HIT_RATE ----------


def test_low_prefix_hit_fires_when_hit_rate_low_and_kv_hot() -> None:
    m = _metrics(prefix_cache_hit_rate=0.10, kv_cache_usage=0.80)
    assert _check_low_prefix_hit_coding(m, _ctx()) is not None


def test_low_prefix_hit_silent_when_hit_rate_is_exactly_zero() -> None:
    # The `0 <` lower bound intentionally treats "exactly 0" as "metric
    # missing / not reported" rather than a true zero, to avoid false-
    # positives on engines that don't expose prefix hit rate.
    m = _metrics(prefix_cache_hit_rate=0.0, kv_cache_usage=0.80)
    assert _check_low_prefix_hit_coding(m, _ctx()) is None


def test_low_prefix_hit_silent_when_kv_is_cold() -> None:
    m = _metrics(prefix_cache_hit_rate=0.10, kv_cache_usage=0.20)
    assert _check_low_prefix_hit_coding(m, _ctx()) is None


def test_low_prefix_hit_silent_when_hit_rate_healthy() -> None:
    m = _metrics(prefix_cache_hit_rate=0.70, kv_cache_usage=0.80)
    assert _check_low_prefix_hit_coding(m, _ctx()) is None


# ---------- KV_FRAGMENTATION_HIGH ----------


def test_kv_fragmentation_fires_when_high_kv_but_few_sequences() -> None:
    m = _metrics(kv_cache_usage=0.85, requests_running=5.0, requests_waiting=0.0)
    assert _check_kv_fragmentation_high(m, _ctx()) is not None


def test_kv_fragmentation_silent_when_queue_present() -> None:
    # If there's a queue, high KV usage is normal — the allocator is busy.
    m = _metrics(kv_cache_usage=0.85, requests_running=5.0, requests_waiting=10.0)
    assert _check_kv_fragmentation_high(m, _ctx()) is None


def test_kv_fragmentation_silent_when_many_sequences() -> None:
    m = _metrics(kv_cache_usage=0.85, requests_running=40.0, requests_waiting=0.0)
    assert _check_kv_fragmentation_high(m, _ctx()) is None


def test_kv_fragmentation_silent_when_kv_cold() -> None:
    m = _metrics(kv_cache_usage=0.50, requests_running=5.0, requests_waiting=0.0)
    assert _check_kv_fragmentation_high(m, _ctx()) is None


# ---------- SPECULATIVE_OVERHEAD ----------


def test_speculative_overhead_fires_on_low_accept_high_concurrency() -> None:
    m = _metrics(spec_acceptance_rate=0.40, requests_running=50.0)
    assert _check_speculative_overhead(m, _ctx()) is not None


def test_speculative_overhead_silent_when_spec_disabled() -> None:
    # Lower bound `> 0` means "spec decoding is on"; 0 = off = no complaint.
    m = _metrics(spec_acceptance_rate=0.0, requests_running=50.0)
    assert _check_speculative_overhead(m, _ctx()) is None


def test_speculative_overhead_silent_on_low_concurrency() -> None:
    # Spec decoding is legitimately useful at low concurrency.
    m = _metrics(spec_acceptance_rate=0.40, requests_running=5.0)
    assert _check_speculative_overhead(m, _ctx()) is None


def test_speculative_overhead_silent_when_accept_healthy() -> None:
    m = _metrics(spec_acceptance_rate=0.75, requests_running=50.0)
    assert _check_speculative_overhead(m, _ctx()) is None


# ---------- BATCH_SIZE_MISMATCH ----------


def test_batch_size_mismatch_fires_when_tiny_budget_with_queue() -> None:
    m = _metrics(requests_waiting=30.0)
    ctx = _ctx(max_num_batched_tokens=2048)
    assert _check_batch_size_mismatch(m, ctx) is not None


def test_batch_size_mismatch_silent_when_budget_unconfigured() -> None:
    # 0 = not declared, so the check can't meaningfully compare.
    m = _metrics(requests_waiting=30.0)
    ctx = _ctx(max_num_batched_tokens=0)
    assert _check_batch_size_mismatch(m, ctx) is None


def test_batch_size_mismatch_silent_when_budget_adequate() -> None:
    m = _metrics(requests_waiting=30.0)
    ctx = _ctx(max_num_batched_tokens=8192)
    assert _check_batch_size_mismatch(m, ctx) is None


def test_batch_size_mismatch_silent_when_no_queue() -> None:
    m = _metrics(requests_waiting=0.0)
    ctx = _ctx(max_num_batched_tokens=2048)
    assert _check_batch_size_mismatch(m, ctx) is None


# ---------- LMCACHE_COLD_START ----------


def test_lmcache_cold_start_fires_on_near_zero_hit_rate() -> None:
    m = _metrics(lmcache_hit_rate=0.02, requests_running=20.0)
    assert _check_lmcache_cold_start(m, _ctx()) is not None


def test_lmcache_cold_start_silent_on_exact_zero() -> None:
    # Same `0 <` pattern as LOW_PREFIX_HIT_RATE — exact 0 means "not
    # reported", not "cold start in progress".
    m = _metrics(lmcache_hit_rate=0.0, requests_running=20.0)
    assert _check_lmcache_cold_start(m, _ctx()) is None


def test_lmcache_cold_start_silent_when_warmed() -> None:
    m = _metrics(lmcache_hit_rate=0.60, requests_running=20.0)
    assert _check_lmcache_cold_start(m, _ctx()) is None


# ---------- BATCH_ITL_TRADEOFF ----------


def test_batch_itl_tradeoff_fires_on_high_batch_with_elevated_itl() -> None:
    m = _metrics(itl_avg_s=0.08, requests_running=80.0, gen_throughput_tps=3000.0)
    assert _check_batch_itl_tradeoff(m, _ctx()) is not None


def test_batch_itl_tradeoff_silent_at_low_concurrency() -> None:
    m = _metrics(itl_avg_s=0.08, requests_running=20.0, gen_throughput_tps=500.0)
    assert _check_batch_itl_tradeoff(m, _ctx()) is None


def test_batch_itl_tradeoff_silent_when_throughput_missing() -> None:
    m = _metrics(itl_avg_s=0.08, requests_running=80.0, gen_throughput_tps=0.0)
    assert _check_batch_itl_tradeoff(m, _ctx()) is None


# ---------- SLO_VIOLATION_RATE ----------


def test_slo_violation_rate_critical_above_15_percent() -> None:
    m = _metrics(
        slo_ttft_violations=120.0,
        slo_itl_violations=80.0,
        request_success_total=1000.0,  # 20% rate
    )
    finding = _check_slo_violation_rate(m, _ctx())
    assert finding is not None
    assert finding.severity == "critical"


def test_slo_violation_rate_warning_in_5_to_15_range() -> None:
    m = _metrics(
        slo_ttft_violations=60.0,
        slo_itl_violations=20.0,
        request_success_total=1000.0,  # 8% rate
    )
    finding = _check_slo_violation_rate(m, _ctx())
    assert finding is not None
    assert finding.severity == "warning"


def test_slo_violation_rate_silent_below_5_percent() -> None:
    m = _metrics(
        slo_ttft_violations=30.0,
        slo_itl_violations=10.0,
        request_success_total=2000.0,  # 2% rate — healthy
    )
    assert _check_slo_violation_rate(m, _ctx()) is None


def test_slo_violation_rate_silent_with_no_violations() -> None:
    m = _metrics(request_success_total=5000.0)
    assert _check_slo_violation_rate(m, _ctx()) is None


# ---------- DECODE_STARVATION / PREFILL_STARVATION (paired) ----------


def test_decode_starvation_fires_when_itl_high_but_ttft_ok() -> None:
    m = _metrics(itl_avg_s=0.15, ttft_avg_s=0.5)
    assert _check_decode_starvation(m, _ctx()) is not None


def test_decode_starvation_silent_when_both_high() -> None:
    # Both high = general overload, not specifically decode starvation.
    m = _metrics(itl_avg_s=0.15, ttft_avg_s=2.0)
    assert _check_decode_starvation(m, _ctx()) is None


def test_decode_starvation_silent_when_ttft_unset() -> None:
    m = _metrics(itl_avg_s=0.15, ttft_avg_s=None)
    assert _check_decode_starvation(m, _ctx()) is None


def test_prefill_starvation_fires_when_ttft_high_but_itl_ok() -> None:
    m = _metrics(ttft_avg_s=8.0, itl_avg_s=0.02)
    assert _check_prefill_starvation(m, _ctx()) is not None


def test_prefill_starvation_silent_when_both_high() -> None:
    m = _metrics(ttft_avg_s=8.0, itl_avg_s=0.15)
    assert _check_prefill_starvation(m, _ctx()) is None


def test_prefill_starvation_silent_when_itl_unset() -> None:
    m = _metrics(ttft_avg_s=8.0, itl_avg_s=None)
    assert _check_prefill_starvation(m, _ctx()) is None


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
