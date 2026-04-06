"""End-to-end parser-fixture tests for Dynamo Prometheus metrics.

These tests feed a realistic Dynamo /metrics text fixture through the
full pipeline:

    text → parse_prometheus_text → ScrapeResult → normalize → run_all_checks

The fixture lives in tests/fixtures/dynamo_metrics_healthy.txt. It was
built against the authoritative Dynamo metric schema
(github.com/ai-dynamo/dynamo docs/observability/metrics.md plus
lib/runtime/src/metrics/prometheus_names.rs).

What this file protects against:

  1. Schema drift. If a future refactor renames a field on
     NormalizedMetrics, removes a metric from DYNAMO_METRICS, or
     changes the parse layer in a way that breaks the healthy
     baseline, these tests fail loudly at the same layer the break
     happens in.

  2. False positives on healthy deployments. The "no findings" test
     against the healthy fixture guarantees that a realistic
     production scrape doesn't trigger any audit check. If a check
     regresses to the raw-counter pattern again (the
     KV_PREEMPTION_STORM bug), the healthy baseline will start
     producing findings and this test will catch it.

  3. False negatives on pathological deployments. The scenario tests
     overlay specific metric changes onto the healthy fixture and
     assert that the expected checks fire. If a check silently stops
     firing because its input field was renamed or its data source
     went away, these tests catch it.
"""

from __future__ import annotations

from pathlib import Path

from inferscope.optimization.checks import DeploymentContext, run_all_checks
from inferscope.telemetry.normalizer import normalize
from inferscope.telemetry.prometheus import (
    DYNAMO_METRICS,
    ScrapeResult,
    detect_engine_from_metrics,
    parse_prometheus_text,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _scrape_from_text(text: str) -> ScrapeResult:
    """Simulate the parse half of scrape_metrics() from a raw text body.

    This mirrors the post-HTTP path inside scrape_metrics() exactly so
    the test covers the same code paths a live scrape would hit.
    """
    result = ScrapeResult(
        endpoint="http://test.local/metrics",
        engine=detect_engine_from_metrics(text),
    )
    result.samples = parse_prometheus_text(text)
    for sample in result.samples:
        # Mirrors the _bucket{ filter in scrape_metrics: keep _sum and
        # _count for histograms, skip individual bucket entries.
        if "_bucket{" in f"{sample.name}{{":
            continue
        result.raw_metrics[sample.name] = sample.value
    return result


def _load_healthy_fixture() -> ScrapeResult:
    """Load the main Dynamo frontend + backend fixture.

    This represents a scrape of the normal Dynamo /metrics endpoint on
    port 8000. KVBM metrics are on a separate endpoint (port 6880) and
    have their own fixture — see _load_kvbm_fixture below.
    """
    text = (_FIXTURE_DIR / "dynamo_metrics_healthy.txt").read_text()
    return _scrape_from_text(text)


def _load_kvbm_fixture() -> ScrapeResult:
    """Load the KVBM /metrics fixture.

    KVBM is exposed on a separate Prometheus endpoint (default port
    6880 via DYN_KVBM_METRICS_PORT) and only when the operator
    launches Dynamo with DYN_KVBM_METRICS=true. In a real deployment
    an operator would add this as a second metrics_target on the
    scrape; here we load it as an independent ScrapeResult and merge
    it into the main scrape's raw_metrics dict in tests that need the
    full picture.
    """
    text = (_FIXTURE_DIR / "dynamo_metrics_kvbm_healthy.txt").read_text()
    return _scrape_from_text(text)


def _load_combined_fixture() -> ScrapeResult:
    """Merge the main + KVBM fixtures into a single ScrapeResult.

    Mirrors what would happen if an operator configured InferScope
    with both the frontend and the KVBM endpoint as metrics targets:
    the normalizer sees all metrics in a single scrape regardless of
    which port they came from, because the Prometheus text format is
    stateless.
    """
    main = _load_healthy_fixture()
    kvbm = _load_kvbm_fixture()
    main.raw_metrics.update(kvbm.raw_metrics)
    main.samples = list(main.samples) + list(kvbm.samples)
    return main


def _healthy_ctx() -> DeploymentContext:
    """DeploymentContext matching the fixture's declared Kimi deployment."""
    return DeploymentContext(
        engine="dynamo",
        gpu_arch="sm_90a",
        gpu_name="H100 SXM",
        gpu_memory_gb=80.0,
        gpu_vendor="nvidia",
        model_name="Kimi-K2.5",
        model_type="moe",
        attention_type="MLA",
        experts_total=128,
        # tp=4, ep=2 across 8 GPUs — a realistic Kimi-K2.5 split that
        # satisfies the MOE_EP_MISSING check (which fires when a large
        # MoE model has ep<=1). A "healthy" fixture must reflect a
        # correctly-configured deployment, so expert parallelism is on.
        tp=4,
        ep=2,
        fp8_support=True,
        fp8_format="OCP",
        gpu_memory_utilization=0.92,
        kv_cache_dtype="fp8_e4m3",
        quantization="fp8",
        block_size=16,
        prefix_caching=True,
        max_num_batched_tokens=16384,
    )


# ============================================================================
# Engine detection + basic parse sanity
# ============================================================================


def test_fixture_detects_dynamo_engine() -> None:
    scrape = _load_healthy_fixture()
    assert scrape.engine == "dynamo"


#: Metric names in DYNAMO_METRICS that are optional object-storage tier
#: KVBM telemetry — only emitted when the operator configures an S3/GCS
#: tier. A deployment with only GPU + CPU + SSD tiers won't emit these,
#: so the schema canary doesn't require them in the KVBM fixture.
_OPTIONAL_METRICS: frozenset[str] = frozenset(
    {
        "kvbm_offload_blocks_d2o",
        "kvbm_onboard_blocks_o2d",
    }
)


def test_fixture_covers_all_dynamo_metrics() -> None:
    """Schema-drift canary.

    For every metric name in DYNAMO_METRICS, assert that the combined
    (main frontend + KVBM) fixture contains either the bare metric
    name (for gauges/counters) OR the `_sum` / `_count` pair (for
    histograms). If a future refactor adds a new metric to
    DYNAMO_METRICS and forgets to update the fixture, this test fails
    on the first CI run.

    Uses the combined fixture so the canary covers every scrape
    endpoint InferScope expects to talk to — main Dynamo on :8000
    plus KVBM on :6880. Operators who only configure the main endpoint
    will never see KVBM metrics, which is correct no-data behavior;
    the canary's job is to catch schema drift, not to enforce a
    particular deployment topology.
    """
    scrape = _load_combined_fixture()
    raw = scrape.raw_metrics
    missing: list[str] = []
    for name in DYNAMO_METRICS:
        if name in _OPTIONAL_METRICS:
            continue
        bare_present = name in raw
        histogram_present = (f"{name}_sum" in raw) and (f"{name}_count" in raw)
        if not (bare_present or histogram_present):
            missing.append(name)
    assert not missing, (
        "Fixture is missing these metrics from DYNAMO_METRICS. Either "
        "add them to the appropriate fixture file (dynamo_metrics_healthy.txt "
        "for main scrape, dynamo_metrics_kvbm_healthy.txt for KVBM) with "
        "realistic healthy values, or add the name to _OPTIONAL_METRICS "
        "if it only appears under specific deployment configurations. "
        f"Missing: {sorted(missing)}"
    )


def test_fixture_parses_all_expected_metric_families() -> None:
    scrape = _load_healthy_fixture()
    # Spot-check each metric family so schema drift (a future rename of
    # any of these) trips immediately, not silently.
    expected = [
        "dynamo_frontend_inflight_requests",
        "dynamo_frontend_queued_requests",
        "dynamo_frontend_requests_total",
        "dynamo_frontend_time_to_first_token_seconds_sum",
        "dynamo_frontend_time_to_first_token_seconds_count",
        "dynamo_frontend_cached_tokens_sum",
        "dynamo_frontend_router_queue_pending_requests",
        "dynamo_frontend_model_total_kv_blocks",
        "dynamo_frontend_model_max_num_seqs",
        "dynamo_frontend_model_context_length",
        "dynamo_component_total_blocks",
        "dynamo_component_gpu_cache_usage_percent",
        "dynamo_component_errors_total",
        "dynamo_component_kv_cache_events_applied",
        "dynamo_router_overhead_total_ms_sum",
        "dynamo_router_overhead_block_hashing_ms_sum",
        "dynamo_router_overhead_indexer_find_matches_ms_sum",
        "dynamo_router_overhead_scheduling_ms_sum",
        "dynamo_component_router_kv_hit_rate_sum",
        "lmcache:num_hit_tokens_total",
        "lmcache:num_requested_tokens_total",
    ]
    missing = [name for name in expected if name not in scrape.raw_metrics]
    assert not missing, f"Fixture is missing expected metrics: {missing}"


# ============================================================================
# Normalization: fixture → NormalizedMetrics field population
# ============================================================================


def test_healthy_fixture_normalizes_frontend_latency_fields() -> None:
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    # TTFT: 3200 sum / 10000 count = 0.32 s
    assert m.ttft_avg_s is not None
    assert abs(m.ttft_avg_s - 0.32) < 1e-6
    # ITL: 200 / 10000 = 0.02 s = 20ms
    assert m.itl_avg_s is not None
    assert abs(m.itl_avg_s - 0.02) < 1e-6
    # e2e: 12000 / 10000 = 1.2s
    assert m.e2e_avg_s is not None
    assert abs(m.e2e_avg_s - 1.2) < 1e-6


def test_healthy_fixture_normalizes_kv_cache_signals() -> None:
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    # kv_cache_usage from dynamo_component_gpu_cache_usage_percent
    assert abs(m.kv_cache_usage - 0.50) < 1e-6
    # prefix_cache_hit_rate is now sourced from
    # dynamo_component_router_kv_hit_rate (router-side approximation),
    # not the fictional dynamo_component_gpu_prefix_cache_hit_rate.
    # The fixture's router_kv_hit_rate has sum=7100, count=10000 -> 0.71
    assert abs(m.prefix_cache_hit_rate - 0.71) < 1e-6
    # Total blocks from dynamo_component_total_blocks
    assert m.kv_total_blocks == 16384.0


def test_healthy_fixture_normalizes_request_counters() -> None:
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    assert m.request_success_total == 10000.0
    assert m.generation_tokens_total == 4500000.0
    assert m.request_migrations_total == 3.0
    assert m.disconnected_clients == 0.0


def test_healthy_fixture_normalizes_router_overhead() -> None:
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    # total: 35000 / 10000 = 3.5 ms avg
    assert m.router_overhead_total_ms is not None
    assert abs(m.router_overhead_total_ms - 3.5) < 1e-6
    # block hashing: 8000/10000 = 0.8 ms
    assert m.router_overhead_block_hashing_ms is not None
    assert abs(m.router_overhead_block_hashing_ms - 0.8) < 1e-6
    # indexer: 12000/10000 = 1.2 ms
    assert abs(m.router_overhead_indexer_ms - 1.2) < 1e-6
    # scheduling: 10000/10000 = 1.0 ms
    assert abs(m.router_overhead_scheduling_ms - 1.0) < 1e-6


def test_healthy_fixture_normalizes_router_kv_hit_rate() -> None:
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    # 7100 / 10000 = 0.71 (histogram of 0-1 values → average)
    assert m.router_kv_hit_rate is not None
    assert abs(m.router_kv_hit_rate - 0.71) < 1e-6


def test_healthy_fixture_normalizes_lmcache_from_upstream_prefix() -> None:
    """Guards against a regression back to the invented
    `dynamo_lmcache_hit_rate` name."""
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    # 45M hits / 60M requested = 0.75
    assert abs(m.lmcache_hit_rate - 0.75) < 1e-6


def test_healthy_fixture_normalizes_model_config_gauges() -> None:
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    assert m.model_total_kv_blocks == 16384.0
    assert m.model_max_num_seqs == 64.0
    assert m.model_max_num_batched_tokens == 16384.0
    assert m.model_context_length == 131072.0
    assert m.model_kv_cache_block_size == 16.0


def test_healthy_fixture_has_no_grove_or_slo_fields() -> None:
    """Regression guard: the NormalizedMetrics dataclass must not
    resurrect the deleted grove_tier_* / slo_*_violations fields."""
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    # hasattr returns False because the fields are gone; if a future
    # commit re-introduces them, this test fails.
    assert not hasattr(m, "grove_tier_gpu_pct")
    assert not hasattr(m, "grove_tier_cpu_pct")
    assert not hasattr(m, "grove_tier_ssd_pct")
    assert not hasattr(m, "grove_evictions")
    assert not hasattr(m, "slo_ttft_violations")
    assert not hasattr(m, "slo_itl_violations")


# ============================================================================
# Full pipeline: text → findings. Healthy fixture must produce no findings.
# ============================================================================


def test_healthy_fixture_produces_no_audit_findings() -> None:
    """A realistic healthy deployment must pass all 30 audit checks
    clean. If any check ever false-positives on this fixture it's a
    product bug — fix the check, not the fixture."""
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    findings = run_all_checks(m, _healthy_ctx())
    assert findings == [], (
        "Healthy Dynamo fixture should fire no audit findings, got: "
        f"{[f.check_id for f in findings]}"
    )


# ============================================================================
# Pathological scenarios: mutate the fixture and verify the right checks fire.
# ============================================================================


def test_high_router_overhead_fires_router_overhead_dominates() -> None:
    """Overlay: router overhead jumps to 150ms total, TTFT becomes 300ms.
    → ROUTER_OVERHEAD_DOMINATES should fire (50% of TTFT, above 50ms floor)."""
    scrape = _load_healthy_fixture()
    # Replace the histogram sum: 150ms * 10000 = 1_500_000
    scrape.raw_metrics["dynamo_router_overhead_total_ms_sum"] = 1_500_000
    # And make TTFT 300ms = 0.3s → sum 0.3 * 10000 = 3000
    scrape.raw_metrics["dynamo_frontend_time_to_first_token_seconds_sum"] = 3000
    m = normalize(scrape)
    findings = run_all_checks(m, _healthy_ctx())
    ids = {f.check_id for f in findings}
    assert "ROUTER_OVERHEAD_DOMINATES" in ids


def test_preemption_rate_spike_fires_kv_preemption_storm() -> None:
    """Overlay: preemptions counter spikes to 5% of request_success_total.
    → KV_PREEMPTION_STORM should fire (rate-based, not raw counter)."""
    scrape = _load_healthy_fixture()
    # vllm:num_preemptions_total isn't in the fixture; the dynamo branch
    # of normalize doesn't currently read it. So we set the normalized
    # field directly via a post-normalize mutation to exercise the check
    # logic on this scenario. A future commit that wires a
    # dynamo-native preemption metric would replace this.
    m = normalize(scrape)
    m.preemptions_total = 500.0  # 500 / 10000 = 5%
    findings = run_all_checks(m, _healthy_ctx())
    ids = {f.check_id for f in findings}
    assert "KV_PREEMPTION_STORM" in ids


def test_kv_cache_critical_fires_when_cache_hot() -> None:
    """Overlay: GPU KV cache jumps to 97%. → KV_CACHE_CRITICAL fires."""
    scrape = _load_healthy_fixture()
    scrape.raw_metrics["dynamo_component_gpu_cache_usage_percent"] = 0.97
    m = normalize(scrape)
    findings = run_all_checks(m, _healthy_ctx())
    ids = {f.check_id for f in findings}
    assert "KV_CACHE_CRITICAL" in ids


def test_lmcache_cold_start_fires_when_hit_rate_crashes() -> None:
    """Overlay: LMCache hit rate drops to ~2% (cold cache after restart).
    → LMCACHE_COLD_START fires."""
    scrape = _load_healthy_fixture()
    # hits / requested = 0.02 → set hits to 2% of requested
    scrape.raw_metrics["lmcache:num_hit_tokens_total"] = 1_200_000  # 2% of 60M
    m = normalize(scrape)
    # Also needs `requests_running > 5` per the check's condition; the
    # fixture has 12 running, so this is already satisfied.
    findings = run_all_checks(m, _healthy_ctx())
    ids = {f.check_id for f in findings}
    assert "LMCACHE_COLD_START" in ids


def test_high_ttft_fires_at_10_seconds() -> None:
    """Overlay: TTFT sum becomes 100000 (10s average). → HIGH_TTFT fires."""
    scrape = _load_healthy_fixture()
    scrape.raw_metrics["dynamo_frontend_time_to_first_token_seconds_sum"] = 100000
    m = normalize(scrape)
    findings = run_all_checks(m, _healthy_ctx())
    ids = {f.check_id for f in findings}
    assert "HIGH_TTFT" in ids


def test_high_itl_fires_at_150ms() -> None:
    """Overlay: ITL sum becomes 1500 (150ms average). → HIGH_ITL fires."""
    scrape = _load_healthy_fixture()
    scrape.raw_metrics["dynamo_frontend_inter_token_latency_seconds_sum"] = 1500
    m = normalize(scrape)
    findings = run_all_checks(m, _healthy_ctx())
    ids = {f.check_id for f in findings}
    assert "HIGH_ITL" in ids


# ============================================================================
# KVBM fixture — separate scrape endpoint, parsed independently
# ============================================================================


def test_kvbm_fixture_parses_all_expected_kvbm_metrics() -> None:
    """Every kvbm_* name in DYNAMO_METRICS that's in the KVBM fixture
    must parse. Catches a rename or typo at the parse layer before the
    normalizer even runs."""
    scrape = _load_kvbm_fixture()
    expected_kvbm = [name for name in DYNAMO_METRICS if name.startswith("kvbm_")]
    for name in expected_kvbm:
        if name in _OPTIONAL_METRICS:
            continue
        assert name in scrape.raw_metrics, (
            f"KVBM fixture missing `{name}` — either add it to "
            f"dynamo_metrics_kvbm_healthy.txt or mark it as optional."
        )


def test_kvbm_fixture_normalizes_into_tiering_fields() -> None:
    """When a KVBM scrape is merged into the main scrape, the normalizer
    populates the kvbm_* fields on NormalizedMetrics. This is the code
    path an operator hits when they configure both :8000 and :6880 as
    scrape targets."""
    combined = _load_combined_fixture()
    m = normalize(combined)
    # Host tier hit rate from the KVBM fixture is 0.75
    assert abs(m.kvbm_host_hit_rate - 0.75) < 1e-6
    # Disk tier hit rate is 0.18
    assert abs(m.kvbm_disk_hit_rate - 0.18) < 1e-6
    # Object tier hit rate is 0.0 (no S3/GCS tier configured)
    assert m.kvbm_object_hit_rate == 0.0
    # Block counter populated
    assert m.kvbm_offload_d2h == 12500.0
    assert m.kvbm_onboard_h2d == 9400.0


def test_kvbm_fixture_healthy_combined_scrape_produces_no_findings() -> None:
    """A healthy deployment with working tiered caching (high host hit
    rate, low eviction signal) must pass all audit checks clean —
    including KVBM_TIERING_INEFFECTIVE, which should silently no-fire
    because the data shows tiering is working."""
    combined = _load_combined_fixture()
    m = normalize(combined)
    findings = run_all_checks(m, _healthy_ctx())
    assert findings == [], (
        "Combined healthy fixture (main + KVBM) should fire no findings, "
        f"got: {[f.check_id for f in findings]}"
    )


def test_kvbm_tiering_ineffective_fires_from_fixture_overlay() -> None:
    """End-to-end: overlay a cold host tier on the KVBM fixture and
    saturate the GPU cache, then confirm the KVBM_TIERING_INEFFECTIVE
    check fires through the full pipeline."""
    combined = _load_combined_fixture()
    # Push GPU cache into the danger zone
    combined.raw_metrics["dynamo_component_gpu_cache_usage_percent"] = 0.92
    # Crash the host tier hit rate to near zero — demotions happening
    # but no re-use = tier capacity exceeded
    combined.raw_metrics["kvbm_host_cache_hit_rate"] = 0.01
    m = normalize(combined)
    findings = run_all_checks(m, _healthy_ctx())
    ids = {f.check_id for f in findings}
    assert "KVBM_TIERING_INEFFECTIVE" in ids


# ============================================================================
# Regression guard: grove/slo checks must NOT fire (they should be deleted)
# ============================================================================


def test_healthy_fixture_never_triggers_deleted_check_ids() -> None:
    """Make sure nobody accidentally re-adds the deleted Grove/SLO
    checks to _ALL_CHECKS by asserting their IDs never appear in the
    findings for any fixture-based scenario."""
    scrape = _load_healthy_fixture()
    m = normalize(scrape)
    findings = run_all_checks(m, _healthy_ctx())
    ids = {f.check_id for f in findings}
    deleted_ids = {
        "GROVE_TIER_IMBALANCE",
        "GROVE_EVICTION_STORM",
        "SLO_VIOLATION_RATE",
    }
    assert not (ids & deleted_ids), f"Deleted check re-appeared: {ids & deleted_ids}"
