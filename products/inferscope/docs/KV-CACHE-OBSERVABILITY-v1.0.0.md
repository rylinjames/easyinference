# InferScope KV-Cache Observability Upgrade — Implementation Spec v1.0

**Version:** 1.0.0
**Date:** 2026-03-27
**Scope:** Disaggregated KV cache metrics for Kimi-K2.5 on Hopper/Blackwell via Dynamo + LMCache
**Status:** Draft — **partially retracted, see CORRECTIONS below**

---

## CORRECTIONS (added 2026-04-06)

Parts of the Prometheus metric schema specified below were derived from an incorrect reading of the Dynamo observability surface. The following metric names **do not exist** in NVIDIA Dynamo and have been removed from the InferScope codebase:

- `dynamo_grove_evictions_total` / `dynamo_grove_tier_{gpu,cpu,ssd}_usage_percent` — **"Grove" in Dynamo is a Kubernetes gang-scheduling component** ([docs](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/)), not a KV cache tiering system. It has no Prometheus metrics of its own. This spec confused Grove with KVBM (KV Block Manager), which is the real tiering feature.
- `dynamo_lmcache_hit_rate` / `dynamo_lmcache_miss_rate` — LMCache is an upstream project ([github.com/LMCache/LMCache](https://github.com/LMCache/LMCache)) that exposes its own metrics under the `lmcache:` prefix. Dynamo does not re-export them under a `dynamo_lmcache_*` prefix. InferScope now reads `lmcache:num_hit_tokens_total` / `lmcache:num_requested_tokens_total` and computes the hit rate client-side.
- `dynamo_slo_ttft_violations_total` / `dynamo_slo_itl_violations_total` — Dynamo does not expose server-side SLO violation counters. SLO accounting is a client-side concern that must be computed from the TTFT/ITL histograms. The existing `HIGH_TTFT` and `HIGH_ITL` checks cover the same diagnostic surface.
- `dynamo_nixl_transfer_*` — NIXL is an upstream project whose metrics live on a **separate** Prometheus endpoint (`NIXL_TELEMETRY_PROMETHEUS_PORT`). The exact metric names are not documented in the current Dynamo repo. These field names are kept in the codebase as provisional so the dormant `NIXL_TRANSFER_DOMINATES` check can become live once a real NIXL scrape is captured and the schema confirmed.

The real Dynamo metric surface is documented at [github.com/ai-dynamo/dynamo/blob/main/docs/observability/metrics.md](https://github.com/ai-dynamo/dynamo/blob/main/docs/observability/metrics.md) and the authoritative base names live in `lib/runtime/src/metrics/prometheus_names.rs`. The features this spec proposes (eviction detection, disaggregation health, etc.) are valid and worth implementing — but must be grounded in those real names, not the ones in the tables below.

---

## 1. Motivation

InferScope's MCP tools currently capture basic throughput and cache utilization but lack granular observability for disaggregated Prefill/Decode serving. The `CLAUDE.md` Dynamo Observability Contract defines NIXL, Grove, LMCache, and SLO Prometheus metrics that are **not yet implemented** in the scraper or normalizer. Additionally, the benchmark comparison tool has no strict validation — it reports deltas but never fails on pathological behavior like eviction cascades or TTFT tail latency blowups.

This spec adds 4 features to close that gap and a 5th cross-cutting validation layer.

---

## 2. Features

### Feature 1 — KV Cache Eviction & Fragmentation Metrics

**Problem:** Standard tools show "80% KV Cache full." They don't show fragmentation. Bursty long-context sessions cause memory fragmentation, triggering eviction cascades that silently ruin goodput.

**Solution:** Scrape Grove eviction counters and LMCache hit/miss rates (already defined in the observability contract but not implemented). Derive fragmentation ratio from active/total KV block counts.

**Files:**

| File | Change |
|------|--------|
| `telemetry/prometheus.py` | Add 6 metric keys to `DYNAMO_METRICS`: `dynamo_grove_evictions_total`, `dynamo_grove_tier_{gpu,cpu,ssd}_usage_percent`, `dynamo_lmcache_{hit,miss}_rate` |
| `telemetry/normalizer.py` | Add 8 fields to `NormalizedMetrics`: `kv_eviction_rate`, `kv_fragmentation_ratio`, `kv_compaction_events`, `grove_{gpu,cpu,ssd}_usage`, `grove_evictions_total`, `lmcache_{hit,miss}_rate`. Wire Dynamo `normalize()` branch. Derive fragmentation: `max(0, kv_cache_usage - active_blocks/total_blocks)`. Update `to_dict()` cache section. |

**Metric-to-field mapping:**

| Prometheus Metric | NormalizedMetrics Field | Type |
|-------------------|------------------------|------|
| `dynamo_grove_evictions_total` | `grove_evictions_total` | counter |
| `dynamo_grove_tier_gpu_usage_percent` | `grove_gpu_usage` | gauge 0-1 |
| `dynamo_grove_tier_cpu_usage_percent` | `grove_cpu_usage` | gauge 0-1 |
| `dynamo_grove_tier_ssd_usage_percent` | `grove_ssd_usage` | gauge 0-1 |
| `dynamo_lmcache_hit_rate` | `lmcache_hit_rate` | gauge 0-1 |
| `dynamo_lmcache_miss_rate` | `lmcache_miss_rate` | gauge 0-1 |
| *(derived)* `kv_cache_usage - active_blocks/total_blocks` | `kv_fragmentation_ratio` | derived 0-1 |

---

### Feature 2 — Inter-Node RDMA Transfer Latency (Prefill -> Decode)

**Problem:** No measurement of the milliseconds it takes to migrate KV-cache blocks over RDMA/NVLink from Prefill to Decode nodes. If the network stutters, the Decode GPU starves at 0% utilization while waiting.

**Solution:** Scrape NIXL transfer latency histogram and DCGM GPU compute utilization. Cross-correlate to detect decode starvation.

**Files:**

| File | Change |
|------|--------|
| `telemetry/prometheus.py` | Add 6 metric keys to `DYNAMO_METRICS`: `dynamo_nixl_transfer_{latency_seconds,bytes_total,failures_total}`, `dynamo_slo_{ttft,itl}_violations_total`, `DCGM_FI_PROF_GR_ENGINE_ACTIVE` |
| `telemetry/normalizer.py` | Add 6 fields: `kv_transfer_latency_ms` (NIXL histogram avg * 1000), `kv_transfer_{bytes,failures}_total`, `gpu_compute_utilization_pct` (DCGM 0-1 -> 0-100), `slo_{ttft,itl}_violations_total`. Wire Dynamo branch. Add `"disaggregation"` sub-dict to `to_dict()`. |

**Metric-to-field mapping:**

| Prometheus Metric | NormalizedMetrics Field | Conversion |
|-------------------|------------------------|------------|
| `dynamo_nixl_transfer_latency_seconds` | `kv_transfer_latency_ms` | histogram avg * 1000 |
| `dynamo_nixl_transfer_bytes_total` | `kv_transfer_bytes_total` | counter, direct |
| `dynamo_nixl_transfer_failures_total` | `kv_transfer_failures_total` | counter, direct |
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | `gpu_compute_utilization_pct` | gauge 0-1 -> 0-100 |
| `dynamo_slo_ttft_violations_total` | `slo_ttft_violations_total` | counter, direct |
| `dynamo_slo_itl_violations_total` | `slo_itl_violations_total` | counter, direct |

**Starvation detection logic:**
```
IF kv_transfer_latency_ms > 50ms AND gpu_compute_utilization_pct < 10%
THEN flag DECODE_GPU_STARVATION (severity: critical)
```

---

### Feature 3 — P90/P99 TTFT Distribution + Strict Goodput SLO

**Problem:** `BenchmarkSummary` only has `ttft_avg_ms` and `ttft_p95_ms`. Inference engineers care about tail latency (P99), not averages. No SLO gating exists — a benchmark can "pass" comparison with catastrophic P99.

**Solution:** Add P90/P99 to summary (reusing existing `_percentile()` helper at `runtime.py:81`). Add P99 threshold to `BenchmarkGoodputSLO`. Gate comparison on P99 SLO breach.

**Files:**

| File | Change |
|------|--------|
| `benchmarks/models.py` | Add `ttft_p90_ms: float | None = None` and `ttft_p99_ms: float | None = None` to `BenchmarkSummary` (after `ttft_p95_ms`) |
| `benchmarks/runtime.py` | Populate both in `_build_summary()`: `ttft_p90_ms=_percentile(ttfts, 0.90)`, `ttft_p99_ms=_percentile(ttfts, 0.99)` |
| `benchmarks/experiments.py` | Add `ttft_p99_ms: float | dict[str, float] | None = None` to `BenchmarkGoodputSLO` (after `tpot_p95_ms`) |
| `benchmarks/catalog.py` | Add P90/P99 deltas and ratios to comparison output |

**New comparison fields:**

| Deltas | Ratios |
|--------|--------|
| `ttft_p90_ms` | `ttft_p90` |
| `ttft_p99_ms` | `ttft_p99` |

---

### Feature 4 — Idle HBM "Cost per Session"

**Problem:** A 100K-token Kimi context sits in HBM while the user writes their next prompt. No cost modeling exists — `recommend_kv_strategy()` only checks if KV fits, not whether holding it in HBM is cost-efficient vs offloading to DRAM.

**Solution:** Model idle HBM holding cost per session based on GPU pricing and KV footprint fraction. Calculate DRAM reload latency penalty. Recommend `cpu_dram` offload when savings exceed latency cost.

**Files:**

| File | Change |
|------|--------|
| `tools/kv_cache.py` | Add `_GPU_COST_PER_HOUR` dict, `_CPU_DRAM_BANDWIDTH_GB_S`, `_compute_idle_hbm_cost()` helper, wire into `recommend_kv_strategy()` |

**GPU pricing table (approximate on-demand rates):**

| GPU | $/hr |
|-----|------|
| H100 SXM | 2.00 |
| H100 PCIe | 1.80 |
| H100 NVL | 1.90 |
| H200 SXM | 3.50 |
| H200 NVL | 3.50 |
| B200 | 5.00 |
| B300 | 6.50 |

**Cost model:**
```
hbm_fraction_per_session = kv_per_session_gb / gpu_memory_gb
idle_cost_per_session_per_min = hbm_fraction * gpu_cost_per_hour / 60
dram_reload_latency_ms = (kv_per_session_gb / 300 GB/s) * 1000
offload_recommended = (savings_per_5min_cycle > $0.001) AND (reload_latency < 50ms)
```

**New return key in `recommend_kv_strategy()`:**
```json
{
  "idle_hbm_cost": {
    "gpu_cost_per_hour_usd": 2.00,
    "kv_per_session_gb": 0.125,
    "hbm_fraction_per_session": 0.001563,
    "idle_cost_per_session_per_minute_usd": 0.000052,
    "total_idle_cost_per_hour_usd": 0.0936,
    "dram_reload_latency_ms": 0.417,
    "breakeven_analysis": {
      "typical_idle_minutes": 5.0,
      "savings_per_offload_cycle_usd": 0.00026,
      "offload_recommended": true
    },
    "offload_recommendation": "cpu_dram"
  }
}
```

---

### Feature 5 — Strict Benchmark Comparison Validation

**Problem:** `compare_benchmark_artifacts()` computes deltas but never fails. A candidate with 5x eviction rate and P99 TTFT blowup gets a clean comparison.

**Solution:** Add a validation layer with 4 checks that runs after deltas/ratios are computed. Validation failures are separate from compatibility warnings.

**Files:**

| File | Change |
|------|--------|
| `benchmarks/catalog.py` | Add `ComparisonValidationThresholds` frozen dataclass, `_normalized_metric()` helper, `_validate_comparison()` with 4 checks, wire into `compare_benchmark_artifacts()` |
| `server_benchmarks.py` | Add optional `validation_thresholds: dict` param to `tool_compare_benchmarks()` |

**Validation checks:**

| Check ID | Trigger | Severity | Confidence |
|----------|---------|----------|------------|
| `EVICTION_RATE_SPIKE` | Candidate eviction rate > 2x baseline OR > 0.1/sec absolute | failure | 0.85 |
| `P99_TTFT_SLO_BREACH` | Candidate `ttft_p99_ms` > experiment spec `goodput_slo.ttft_p99_ms` | failure | 0.90 |
| `DECODE_GPU_STARVATION` | NIXL latency > 50ms AND decode GPU utilization < 10% | failure | 0.85 |
| `CACHE_FRAGMENTATION` | Fragmentation > 0.5 + high eviction (failure) or > 0.3 (warning) | failure/warning | 0.75 |

**Default thresholds (`ComparisonValidationThresholds`):**
```python
eviction_rate_spike_ratio: float = 2.0
eviction_rate_absolute_max: float = 0.1       # evictions/sec
kv_transfer_latency_max_ms: float = 50.0
gpu_starvation_floor_pct: float = 10.0        # percent
kv_fragmentation_warning: float = 0.3
kv_fragmentation_failure: float = 0.5
```

**Output format:**
```json
{
  "validation": {
    "passed": false,
    "failures": [
      {
        "check_id": "EVICTION_RATE_SPIKE",
        "severity": "failure",
        "title": "KV eviction rate spike in candidate",
        "description": "Candidate eviction rate (0.25/s) is 5.0x baseline (0.05/s)...",
        "metric_values": {
          "baseline_eviction_rate": 0.05,
          "candidate_eviction_rate": 0.25,
          "ratio": 5.0
        },
        "threshold": {
          "relative_max": 2.0,
          "absolute_max": 0.1
        },
        "confidence": 0.85,
        "evidence": "metric_comparison"
      }
    ],
    "warnings": []
  }
}
```

---

## 3. Dependency Order

```
Phase 1 (telemetry — no cross-deps):
  1. telemetry/prometheus.py     — metric key documentation
  2. telemetry/normalizer.py     — fields + normalize() + to_dict()

Phase 2 (benchmark models — no cross-deps):
  3. benchmarks/models.py        — BenchmarkSummary P90/P99
  4. benchmarks/experiments.py   — BenchmarkGoodputSLO P99

Phase 3 (wiring — depends on Phase 1+2):
  5. benchmarks/runtime.py       — populate P90/P99 in _build_summary()
  6. benchmarks/catalog.py       — deltas, ratios, validation layer
  7. tools/kv_cache.py           — idle cost helper + integration
  8. server_benchmarks.py        — MCP tool validation_thresholds param

Phase 4 (tests):
  9. tests/
```

Phases 1 and 2 have no cross-dependencies and can be implemented in parallel.

---

## 4. Reused Existing Code

| Function | Location | Usage |
|----------|----------|-------|
| `_percentile(values, p)` | `benchmarks/runtime.py:81` | P90/P99 computation |
| `_delta(new, base)` | `benchmarks/catalog.py` | New delta fields |
| `_ratio(new, base)` | `benchmarks/catalog.py` | New ratio fields |
| `_cache_effectiveness_metric()` | `benchmarks/catalog.py` | Nested cache metric access |
| `scrape.get()` | `telemetry/prometheus.py` | Gauge/counter extraction |
| `scrape.get_histogram_avg()` | `telemetry/prometheus.py` | Histogram _sum/_count avg |
| `resolve_platform_traits()` | `optimization/platform_policy.py` | Platform capability resolution |
| `plan_memory()` | `optimization/memory_planner.py` | GPU memory budget calculation |

---

## 5. Backward Compatibility

All changes are additive. No existing signatures, return shapes, or field names change.

| Component | Compatibility |
|-----------|--------------|
| `NormalizedMetrics` new fields | Default to `0.0` or `None` — existing scrapes unaffected |
| `BenchmarkSummary` new fields | Default to `None` — old JSON artifacts deserialize correctly |
| `compare_benchmark_artifacts()` | New `thresholds` kwarg defaults to `None` — existing callers unchanged |
| `tool_compare_benchmarks()` | New `validation_thresholds` param optional — existing MCP calls unchanged |
| `recommend_kv_strategy()` return | New `"idle_hbm_cost"` key — purely additive |
| `BenchmarkGoodputSLO` new field | `ttft_p99_ms` optional — existing YAML specs unaffected |

---

## 6. Verification

```bash
# Unit tests (all existing + new)
cd products/inferscope && uv run pytest tests/ -v

# Type check
cd products/inferscope && uv run mypy src/inferscope/

# Lint
cd products/inferscope && uv run ruff check src/
```

**New test coverage needed:**

| Area | Test |
|------|------|
| Normalizer | Dynamo scrape with Grove/NIXL/DCGM metrics maps correctly |
| Normalizer | SGLang/ATOM scrape leaves new fields as None/0.0 |
| Normalizer | `to_dict()` includes `"disaggregation"` sub-dict |
| Normalizer | Fragmentation derived from active/total blocks |
| Summary | `_build_summary()` populates `ttft_p90_ms` and `ttft_p99_ms` |
| Catalog | `EVICTION_RATE_SPIKE` check triggers with synthetic artifacts |
| Catalog | `P99_TTFT_SLO_BREACH` check triggers when P99 exceeds threshold |
| Catalog | `DECODE_GPU_STARVATION` check triggers on cross-metric correlation |
| Catalog | `CACHE_FRAGMENTATION` check emits warning vs failure correctly |
| Catalog | P90/P99 deltas and ratios appear in comparison output |
| KV Cache | `_compute_idle_hbm_cost()` math for H100/B200 is consistent |
| KV Cache | `recommend_kv_strategy()` return includes `idle_hbm_cost` key |
| KV Cache | Offload recommendation triggers correctly based on breakeven |
