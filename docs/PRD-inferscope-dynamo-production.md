# InferScope KV-Cache Observability & Optimization — Implementation Spec v2.0

**Version:** 2.0.0
**Date:** 2026-03-27
**Scope:** Disaggregated KV cache metrics, cost modeling, and strict validation for Kimi-K2.5 long-context coding on Hopper/Blackwell via vLLM → Dynamo + LMCache
**Status:** Draft
**Target audience:** Any engineer with Python + systems background; no prior knowledge of Kimi, Dynamo, or GPU inference required

---

## 0. Architecture Primer — Read This First

This section gives you the minimum mental model to implement every feature in this spec. If you already know disaggregated serving, skip to §1.

### 0.1 What We're Optimizing

**Kimi-K2.5** is a Mixture-of-Experts (MoE) large language model built for long-context coding tasks (64K–128K token contexts). MoE means the model has many "expert" sub-networks but only activates a few per token. The critical insight for KV cache work: **KV cache is per-attention-layer, not per-expert.** Cache size scales linearly with context length and number of attention heads, regardless of how many experts fire. For Kimi-K2.5, a 64K-token session uses ~0.125 GB of KV cache on H100 (FP8 KV, GQA heads).

**Why this matters for you:** When a user runs a long coding session with Kimi, the KV cache holds the entire conversation history. If that cache gets evicted, fragmented, or stuck in expensive GPU memory while the user is typing, performance and cost suffer. That's what this spec fixes.

### 0.2 Aggregated vs. Disaggregated Serving

**Aggregated serving (vLLM baseline):** One GPU does everything — reads the prompt (prefill), generates tokens (decode), and holds the KV cache. Simple, but the prefill phase (compute-heavy) and decode phase (memory-bandwidth-heavy) fight over the same GPU resources.

```
┌─────────────────────────────────┐
│          Single GPU             │
│  Prefill ──► KV Cache ──► Decode│
│  (compute)   (HBM)    (bandwidth)│
└─────────────────────────────────┘
```

**Disaggregated serving (Dynamo):** Prefill and decode run on separate GPU nodes. The prefill node computes the KV cache, then ships it over RDMA/NVLink to the decode node. This lets each node specialize, but introduces a new bottleneck: **KV transfer latency**.

```
┌──────────────┐   RDMA/NVLink   ┌──────────────┐
│  Prefill GPU │ ──── KV ────►   │  Decode GPU  │
│  (compute)   │   blocks        │  (bandwidth) │
│  Grove tier: │                  │  Grove tier: │
│  GPU HBM     │                  │  GPU HBM     │
│  CPU DRAM    │                  │  CPU DRAM    │
│  SSD         │                  │  SSD         │
└──────────────┘                  └──────────────┘
```

### 0.3 The Serving Stack

| Component | Role | Why you care |
|-----------|------|-------------|
| **vLLM** | Aggregated inference engine. Our benchmark control lane. | Baseline for comparison. Uses prefix caching but no disaggregation. |
| **Dynamo** | NVIDIA's disaggregated serving framework. Our production lane. | Manages prefill/decode split, KV transfer, request routing. |
| **Grove** | Dynamo's tiered KV cache manager. GPU HBM → CPU DRAM → SSD. | Eviction decisions here directly affect latency. When Grove evicts from GPU, the next request pays a reload penalty. |
| **LMCache** | Prefix-aware KV cache layer. Deduplicates shared prefixes across requests. | For coding sessions, many requests share the same system prompt + file context. LMCache avoids recomputing those shared prefixes. Hit rate is the key metric. |
| **NIXL** | Dynamo's RDMA/NVLink transfer layer for moving KV blocks between nodes. | Transfer latency here is the disaggregation tax. If NIXL stalls, the decode GPU starves. |
| **DCGM** | NVIDIA's GPU metrics exporter. | Gives us `GR_ENGINE_ACTIVE` — how busy the GPU compute units actually are. Cross-correlated with NIXL latency to detect decode starvation. |

### 0.4 The Four Benchmark Lanes

Every optimization you implement gets validated against these lanes:

| Lane | Engine | Mode | Cache | Purpose |
|------|--------|------|-------|---------|
| `vllm-aggregated-prefix-cache-kimi-k2` | vLLM | Aggregated | Prefix cache | **Control baseline.** Single-GPU reference. |
| `dynamo-aggregated-lmcache-kimi-k2` | Dynamo | Aggregated | LMCache | Aggregated Dynamo control. |
| `vllm-disagg-prefill-lmcache` | vLLM | Disaggregated | LMCache | Comparison split lane. |
| `dynamo-disagg-lmcache-kimi-k2` | Dynamo | Disaggregated | LMCache | **Production target.** What we ship. |

### 0.5 What the MCP Tools Do

InferScope exposes optimization capabilities as MCP tools that plug into coding agents (Claude Code, Cursor). A new engineer doesn't need to understand the internals — they call tools:

| MCP Tool | What it does | When to call it |
|----------|-------------|-----------------|
| `tool_profile_inference()` | Scrapes live Prometheus metrics from a running deployment. Returns normalized telemetry including cache state, transfer latency, GPU utilization. | "How is my deployment performing right now?" |
| `tool_recommend_kv_strategy()` | Given a model, GPU, and workload profile, recommends KV cache configuration (quantization, offloading tier, LMCache settings). | "What KV cache config should I use for 64K coding sessions on H100?" |
| `tool_run_benchmark()` | Runs a packaged benchmark experiment against a live endpoint. | "Run the Kimi coding benchmark against my deployment." |
| `tool_compare_benchmarks()` | Compares two benchmark artifacts (baseline vs. candidate). Computes deltas, ratios, and now validation checks. | "Is my Dynamo deployment better than the vLLM baseline?" |
| `tool_list_experiments()` | Lists available packaged benchmark experiments. | "What benchmarks can I run?" |

---

## 1. Motivation

InferScope's MCP tools currently capture basic throughput and cache utilization but lack granular observability for disaggregated prefill/decode serving. The `CLAUDE.md` Dynamo Observability Contract defines Grove, LMCache, NIXL, and SLO Prometheus metrics that are **not yet implemented** in the scraper or normalizer. The benchmark comparison tool computes deltas but never fails — a candidate with 5× eviction rate and catastrophic P99 TTFT gets a clean comparison.

Additionally, no cost modeling exists for idle KV cache. A 100K-token Kimi context sitting in HBM while the user types their next prompt costs real money, and `recommend_kv_strategy()` doesn't account for it.

This spec closes those gaps with 5 features, a cross-cutting validation layer, and explicit MCP tool surface for each.

---

## 2. Features

### Feature 1 — KV Cache Eviction & Fragmentation Metrics

**Problem:** Standard tools show "80% KV cache full." They don't show whether that 80% is contiguous (healthy) or fragmented (eviction cascade imminent). Bursty long-context coding sessions cause memory fragmentation in Grove, triggering eviction cascades that silently destroy goodput.

**Solution:** Scrape Grove eviction counters and LMCache hit/miss rates (defined in the observability contract, not yet implemented). Derive a `fragmentation_pressure` heuristic from eviction rate × cache usage level.

> **Design decision:** The original spec derived fragmentation from `active_blocks/total_blocks`, but those counters may not exist in every Dynamo deployment. `fragmentation_pressure` is a heuristic (`eviction_rate * kv_cache_usage`) that correlates with real fragmentation without depending on metrics that might be absent. This is documented honestly as a heuristic, not a physical measurement.

**Files:**

| File | Change |
|------|--------|
| `telemetry/prometheus.py` | Add 6 metric keys to `DYNAMO_METRICS`: `dynamo_grove_evictions_total`, `dynamo_grove_tier_{gpu,cpu,ssd}_usage_percent`, `dynamo_lmcache_{hit,miss}_rate` |
| `telemetry/normalizer.py` | Add 8 fields to `NormalizedMetrics`: `kv_eviction_rate`, `kv_fragmentation_pressure`, `kv_compaction_events`, `grove_{gpu,cpu,ssd}_usage`, `grove_evictions_total`, `lmcache_{hit,miss}_rate`. Wire Dynamo `normalize()` branch. Derive `fragmentation_pressure = grove_eviction_rate * kv_cache_usage`. Update `to_dict()` cache section. |

**Metric-to-field mapping:**

| Prometheus Metric | NormalizedMetrics Field | Type | Notes |
|-------------------|------------------------|------|-------|
| `dynamo_grove_evictions_total` | `grove_evictions_total` | counter | Raw counter; rate derived in normalizer |
| `dynamo_grove_evictions_total` (rate) | `kv_eviction_rate` | derived (evictions/sec) | Counter delta / scrape interval |
| `dynamo_grove_tier_gpu_usage_percent` | `grove_gpu_usage` | gauge 0–1 | |
| `dynamo_grove_tier_cpu_usage_percent` | `grove_cpu_usage` | gauge 0–1 | |
| `dynamo_grove_tier_ssd_usage_percent` | `grove_ssd_usage` | gauge 0–1 | |
| `dynamo_lmcache_hit_rate` | `lmcache_hit_rate` | gauge 0–1 | |
| `dynamo_lmcache_miss_rate` | `lmcache_miss_rate` | gauge 0–1 | |
| *(derived)* `eviction_rate × kv_cache_usage` | `kv_fragmentation_pressure` | heuristic 0–∞ | NOT a physical fragmentation ratio. A pressure signal. |

**MCP surface:** These fields appear in `tool_profile_inference()` return under the `cache` sub-dict.

---

### Feature 2 — Inter-Node RDMA Transfer Latency (Prefill → Decode)

**Problem:** No measurement of the milliseconds it takes to migrate KV-cache blocks over RDMA/NVLink from prefill to decode nodes. If the network stutters, the decode GPU sits at 0% utilization waiting for data — and no one knows.

**Solution:** Scrape NIXL transfer latency histogram and (optionally) DCGM GPU compute utilization. Cross-correlate to detect decode starvation. Degrade gracefully when DCGM is unavailable.

> **Design decision: GPU-exporter data is optional.** DCGM requires the `dcgm-exporter` sidecar, which not all deployments run. The starvation check operates in two modes:
> - **With DCGM:** Full cross-correlation. NIXL latency > 50ms AND GPU util < 10% → `DECODE_GPU_STARVATION` (severity: failure, confidence: 0.85).
> - **Without DCGM:** NIXL-only. NIXL latency > 50ms → `POSSIBLE_DECODE_STARVATION` (severity: warning, confidence: 0.60). The comparison output includes a `data_completeness` field so consumers know GPU utilization couldn't be verified.

**Threshold justification:**
- **50ms NIXL latency:** At 20 tokens/sec decode throughput, each token has a ~50ms budget. NIXL latency exceeding this means the KV fetch alone consumes the full token generation budget.
- **10% GPU utilization:** Below this, the SMs are effectively idle — the GPU is waiting for data, not computing.

**Files:**

| File | Change |
|------|--------|
| `telemetry/prometheus.py` | Add 6 metric keys to `DYNAMO_METRICS`: `dynamo_nixl_transfer_{latency_seconds,bytes_total,failures_total}`, `dynamo_slo_{ttft,itl}_violations_total`, `DCGM_FI_PROF_GR_ENGINE_ACTIVE` |
| `telemetry/normalizer.py` | Add 7 fields: `kv_transfer_latency_ms`, `kv_transfer_{bytes,failures}_total`, `gpu_compute_utilization_pct`, `slo_{ttft,itl}_violations_total`, `gpu_exporter_available: bool`. Wire Dynamo branch. Add `"disaggregation"` sub-dict to `to_dict()`. |

**Metric-to-field mapping:**

| Prometheus Metric | NormalizedMetrics Field | Conversion |
|-------------------|------------------------|------------|
| `dynamo_nixl_transfer_latency_seconds` | `kv_transfer_latency_ms` | histogram avg × 1000 |
| `dynamo_nixl_transfer_bytes_total` | `kv_transfer_bytes_total` | counter, direct |
| `dynamo_nixl_transfer_failures_total` | `kv_transfer_failures_total` | counter, direct |
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | `gpu_compute_utilization_pct` | gauge 0–1 → 0–100 |
| `dynamo_slo_ttft_violations_total` | `slo_ttft_violations_total` | counter, direct |
| `dynamo_slo_itl_violations_total` | `slo_itl_violations_total` | counter, direct |
| *(derived)* | `gpu_exporter_available` | `True` if DCGM metric returned non-None |

**MCP surface:** These fields appear in `tool_profile_inference()` return under the `disaggregation` sub-dict.

---

### Feature 3 — P90/P99 TTFT Distribution + Strict Goodput SLO

**Problem:** `BenchmarkSummary` only has `ttft_avg_ms` and `ttft_p95_ms`. Inference engineers care about tail latency (P99), not averages. No SLO gating exists — a benchmark can "pass" comparison with catastrophic P99.

**Solution:** Add P90/P99 to summary (reusing existing `_percentile()` helper at `runtime.py:81`). Add P99 threshold to `BenchmarkGoodputSLO`. Gate comparison on P99 SLO breach.

> **Data source:** `run_plan.observed_runtime` is the source of truth for TTFT tail metrics. `ttft_p90_ms` and `ttft_p99_ms` are mirrored into `BenchmarkSummary` for artifact readability and simpler MCP output, but they are computed from the observed runtime distribution, not re-derived.

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

**MCP surface:** These fields appear in `tool_compare_benchmarks()` return under `deltas` and `ratios`.

---

### Feature 4 — Idle HBM "Cost per Session"

**Problem:** A 100K-token Kimi context sits in HBM while the user writes their next prompt. No cost modeling exists — `recommend_kv_strategy()` only checks if KV fits, not whether holding it in HBM is cost-efficient vs. offloading to DRAM.

**Solution:** Model idle HBM holding cost per session based on GPU pricing and KV footprint fraction. Calculate DRAM reload latency penalty. Recommend `cpu_dram` offload when savings exceed latency cost.

> **Design decision: GPU pricing is configurable, not hardcoded.** On-demand GPU prices change monthly. The `_compute_idle_hbm_cost()` function takes `gpu_cost_per_hour_usd` as a parameter with sensible defaults. Users can override via the MCP tool's `gpu_cost_override_usd` param or environment config. The defaults below are approximate March 2026 on-demand rates and are documented as such.

**Files:**

| File | Change |
|------|--------|
| `tools/kv_cache.py` | Add `_GPU_COST_PER_HOUR_DEFAULTS` dict (used only when no override provided), `_CPU_DRAM_BANDWIDTH_GB_S = 300`, `_compute_idle_hbm_cost(kv_per_session_gb, gpu_memory_gb, gpu_cost_per_hour_usd, idle_minutes)` helper. Wire into `recommend_kv_strategy()`. |

**Default GPU pricing table (approximate on-demand, March 2026):**

| GPU | Default $/hr | GPU Memory (GB) |
|-----|-------------|-----------------|
| H100 SXM | 2.00 | 80 |
| H100 PCIe | 1.80 | 80 |
| H100 NVL | 1.90 | 94 |
| H200 SXM | 3.50 | 141 |
| H200 NVL | 3.50 | 141 |
| B200 | 5.00 | 192 |
| B300 | 6.50 | 288 |

**Cost model:**
```
hbm_fraction_per_session = kv_per_session_gb / gpu_memory_gb
idle_cost_per_session_per_min = hbm_fraction * gpu_cost_per_hour / 60
dram_reload_latency_ms = (kv_per_session_gb / cpu_dram_bandwidth_gb_s) * 1000
offload_recommended = (savings_per_idle_cycle > $0.001) AND (reload_latency < 50ms)
```

**MCP surface:** New `idle_hbm_cost` key in `tool_recommend_kv_strategy()` return:
```json
{
  "idle_hbm_cost": {
    "gpu_cost_per_hour_usd": 2.00,
    "gpu_cost_source": "default",
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

New optional parameter on `tool_recommend_kv_strategy()`: `gpu_cost_override_usd: float | None = None`.

---

### Feature 5 — Strict Benchmark Comparison Validation

**Problem:** `compare_benchmark_artifacts()` computes deltas but never fails. A candidate with 5× eviction rate and P99 TTFT blowup gets a clean comparison.

**Solution:** Add a validation layer with 4 checks that runs after deltas/ratios are computed. Validation failures are separate from compatibility warnings. GPU-exporter-dependent checks degrade to warnings when data is absent.

**Files:**

| File | Change |
|------|--------|
| `benchmarks/catalog.py` | Add `ComparisonValidationThresholds` frozen dataclass, `_normalized_metric()` helper, `_validate_comparison()` function, wire into `compare_benchmark_artifacts()` |
| `server_benchmarks.py` | Add optional `validation_thresholds: dict` param to `tool_compare_benchmarks()` |

**Validation checks:**

| Check ID | Trigger | Severity | Confidence | Requires GPU exporter? |
|----------|---------|----------|------------|----------------------|
| `EVICTION_RATE_SPIKE` | Candidate eviction rate > 2× baseline OR > 0.1/sec absolute | failure | 0.85 | No |
| `P99_TTFT_SLO_BREACH` | Candidate `ttft_p99_ms` > experiment spec `goodput_slo.ttft_p99_ms` | failure | 0.90 | No |
| `DECODE_GPU_STARVATION` | NIXL latency > 50ms AND decode GPU util < 10% | failure | 0.85 | **Yes** — degrades to `POSSIBLE_DECODE_STARVATION` (warning, 0.60) without DCGM |
| `CACHE_FRAGMENTATION` | `fragmentation_pressure` > 0.5 + high eviction = failure; > 0.3 = warning | failure/warning | 0.75 | No |

**Default thresholds (`ComparisonValidationThresholds`):**
```python
@dataclass(frozen=True)
class ComparisonValidationThresholds:
    eviction_rate_spike_ratio: float = 2.0
    eviction_rate_absolute_max: float = 0.1       # evictions/sec
    kv_transfer_latency_max_ms: float = 50.0
    gpu_starvation_floor_pct: float = 10.0        # percent
    kv_fragmentation_warning: float = 0.3         # pressure heuristic
    kv_fragmentation_failure: float = 0.5         # pressure heuristic
```

**Output format:**
```json
{
  "validation": {
    "passed": false,
    "data_completeness": {
      "gpu_exporter_available": false,
      "metrics_missing": ["DCGM_FI_PROF_GR_ENGINE_ACTIVE"],
      "checks_degraded": ["DECODE_GPU_STARVATION → POSSIBLE_DECODE_STARVATION"]
    },
    "failures": [
      {
        "check_id": "EVICTION_RATE_SPIKE",
        "severity": "failure",
        "title": "KV eviction rate spike in candidate",
        "description": "Candidate eviction rate (0.25/s) is 5.0x baseline (0.05/s), exceeding 2.0x threshold.",
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
    "warnings": [
      {
        "check_id": "POSSIBLE_DECODE_STARVATION",
        "severity": "warning",
        "title": "Possible decode GPU starvation (GPU exporter unavailable)",
        "description": "NIXL transfer latency (62ms) exceeds 50ms threshold. GPU utilization data unavailable — cannot confirm starvation. Deploy dcgm-exporter for full validation.",
        "confidence": 0.60,
        "evidence": "nixl_latency_only"
      }
    ]
  }
}
```

**MCP surface:** `tool_compare_benchmarks()` return always includes the `validation` block (in addition to existing `deltas`, `ratios`, `summary`). New optional parameter: `validation_thresholds: dict | None = None`.

---

## 3. Dependency Graph & Implementation Phases

```
Phase 1 — Telemetry Foundation (no cross-deps, parallelize freely)
  ├─ 1a. telemetry/prometheus.py     — metric key registration
  └─ 1b. telemetry/normalizer.py     — new fields + normalize() + to_dict()

Phase 2 — Benchmark Models (no cross-deps, parallelize with Phase 1)
  ├─ 2a. benchmarks/models.py        — BenchmarkSummary P90/P99
  └─ 2b. benchmarks/experiments.py   — BenchmarkGoodputSLO P99

Phase 3 — Wiring (depends on Phase 1 + 2)
  ├─ 3a. benchmarks/runtime.py       — populate P90/P99 in _build_summary()
  ├─ 3b. benchmarks/catalog.py       — deltas, ratios, validation layer
  ├─ 3c. tools/kv_cache.py           — idle cost helper + integration
  └─ 3d. server_benchmarks.py        — MCP tool validation_thresholds param

Phase 4 — Tests (depends on Phase 3)
  └─ 4a. tests/                      — unit + integration tests

Phase 5 — Live Validation (depends on Phase 4 green)
  └─ 5a. Modal H100/B200 end-to-end  — see §7
```

Phases 1 and 2 have zero cross-dependencies. A single engineer can implement them in sequence; two engineers can parallelize.

---

## 4. Task Breakdown

### Phase 1 — Telemetry Foundation

#### Task 1.1: Register Dynamo Metric Keys
**File:** `telemetry/prometheus.py`
**Estimated effort:** 30 min
**Prereqs:** None

**Subtasks:**
1. Add 6 Grove/LMCache keys to `DYNAMO_METRICS` dict:
   - `dynamo_grove_evictions_total` (counter)
   - `dynamo_grove_tier_gpu_usage_percent` (gauge)
   - `dynamo_grove_tier_cpu_usage_percent` (gauge)
   - `dynamo_grove_tier_ssd_usage_percent` (gauge)
   - `dynamo_lmcache_hit_rate` (gauge)
   - `dynamo_lmcache_miss_rate` (gauge)
2. Add 6 NIXL/SLO/DCGM keys to `DYNAMO_METRICS` dict:
   - `dynamo_nixl_transfer_latency_seconds` (histogram)
   - `dynamo_nixl_transfer_bytes_total` (counter)
   - `dynamo_nixl_transfer_failures_total` (counter)
   - `dynamo_slo_ttft_violations_total` (counter)
   - `dynamo_slo_itl_violations_total` (counter)
   - `DCGM_FI_PROF_GR_ENGINE_ACTIVE` (gauge)
3. Document each metric's type (counter/gauge/histogram) and expected label set in code comments.
4. Verify `scrape.get()` and `scrape.get_histogram_avg()` work for the expected metric types — no new scraper code needed, just key registration.

**Acceptance:** All 12 metrics are in `DYNAMO_METRICS`. `ruff check` and `mypy` pass.

---

#### Task 1.2: Extend NormalizedMetrics with Cache & Disaggregation Fields
**File:** `telemetry/normalizer.py`
**Estimated effort:** 2 hr
**Prereqs:** Task 1.1

**Subtasks:**
1. Add 8 cache fields to `NormalizedMetrics` class:
   - `grove_evictions_total: float = 0.0`
   - `kv_eviction_rate: float = 0.0` (evictions/sec, derived)
   - `kv_fragmentation_pressure: float = 0.0` (heuristic, derived)
   - `kv_compaction_events: int = 0`
   - `grove_gpu_usage: float = 0.0`
   - `grove_cpu_usage: float = 0.0`
   - `grove_ssd_usage: float = 0.0`
   - `lmcache_hit_rate: float = 0.0`
   - `lmcache_miss_rate: float = 0.0`
2. Add 7 disaggregation fields:
   - `kv_transfer_latency_ms: float = 0.0`
   - `kv_transfer_bytes_total: float = 0.0`
   - `kv_transfer_failures_total: float = 0.0`
   - `gpu_compute_utilization_pct: float = 0.0`
   - `gpu_exporter_available: bool = False`
   - `slo_ttft_violations_total: float = 0.0`
   - `slo_itl_violations_total: float = 0.0`
3. Wire the Dynamo `normalize()` branch:
   - Extract Grove gauges via `scrape.get()` and scale to 0–1.
   - Extract NIXL histogram avg via `scrape.get_histogram_avg()`, multiply by 1000 for ms.
   - Extract DCGM gauge, set `gpu_exporter_available = True` if non-None, scale 0–1 → 0–100.
   - Derive `kv_eviction_rate` from counter delta / scrape interval.
   - Derive `kv_fragmentation_pressure = kv_eviction_rate * kv_cache_usage`.
4. Update `to_dict()`:
   - Add Grove/LMCache fields to existing `"cache"` sub-dict.
   - Add new `"disaggregation"` sub-dict with NIXL, DCGM, SLO fields and `gpu_exporter_available`.
5. Ensure SGLang/ATOM/vLLM normalize branches leave all new fields at their defaults (0.0 / None / False). **Do not break existing engines.**

**Acceptance:** Dynamo scrape with synthetic Grove/NIXL/DCGM data maps correctly to all new fields. SGLang scrape leaves new fields at defaults. `to_dict()` output includes both sub-dicts. Types pass `mypy`.

---

### Phase 2 — Benchmark Models

#### Task 2.1: Add P90/P99 to BenchmarkSummary
**File:** `benchmarks/models.py`
**Estimated effort:** 15 min
**Prereqs:** None

**Subtasks:**
1. Add `ttft_p90_ms: float | None = None` after `ttft_p95_ms`.
2. Add `ttft_p99_ms: float | None = None` after `ttft_p90_ms`.
3. Verify Pydantic/dataclass serialization — old JSON artifacts without these fields must deserialize without error (both default to `None`).

**Acceptance:** Existing artifact JSON files load without error. New fields serialize/deserialize correctly. `mypy` passes.

---

#### Task 2.2: Add P99 SLO to BenchmarkGoodputSLO
**File:** `benchmarks/experiments.py`
**Estimated effort:** 15 min
**Prereqs:** None

**Subtasks:**
1. Add `ttft_p99_ms: float | dict[str, float] | None = None` after `tpot_p95_ms`.
2. Verify existing experiment YAML specs load without error (field is optional, defaults to `None`).

**Acceptance:** All existing experiment specs load. New field accepted in YAML. `mypy` passes.

---

### Phase 3 — Wiring

#### Task 3.1: Populate P90/P99 in Runtime
**File:** `benchmarks/runtime.py`
**Estimated effort:** 15 min
**Prereqs:** Tasks 2.1, 2.2

**Subtasks:**
1. In `_build_summary()`, compute and assign:
   - `ttft_p90_ms = _percentile(ttfts, 0.90)`
   - `ttft_p99_ms = _percentile(ttfts, 0.99)`
   - Use the existing `_percentile()` helper at line ~81.
2. Source TTFT values from `run_plan.observed_runtime` (source of truth), not from re-derived data.

**Acceptance:** Running a benchmark populates both fields in the summary artifact. Values are consistent with the observed runtime distribution.

---

#### Task 3.2: Comparison Deltas, Ratios, and Validation Layer
**File:** `benchmarks/catalog.py`
**Estimated effort:** 4 hr (largest task)
**Prereqs:** Tasks 1.2, 2.1, 2.2, 3.1

**Subtasks:**
1. **Add P90/P99 deltas and ratios** — use existing `_delta()` and `_ratio()` helpers:
   - Add `ttft_p90_ms` and `ttft_p99_ms` to delta computation.
   - Add `ttft_p90` and `ttft_p99` to ratio computation.
2. **Add `ComparisonValidationThresholds`** frozen dataclass with 6 threshold fields (see §2 Feature 5).
3. **Add `_normalized_metric()` helper** — safely extracts a metric from a benchmark artifact's telemetry snapshot, returning `None` if absent.
4. **Implement `_validate_comparison()`** with 4 checks:
   - **`EVICTION_RATE_SPIKE`:** Extract `kv_eviction_rate` from both artifacts. Check ratio > threshold OR absolute > max. Does NOT require GPU exporter.
   - **`P99_TTFT_SLO_BREACH`:** Compare candidate `ttft_p99_ms` against experiment spec `goodput_slo.ttft_p99_ms`. Only fires if SLO is defined. Does NOT require GPU exporter.
   - **`DECODE_GPU_STARVATION` / `POSSIBLE_DECODE_STARVATION`:** Check NIXL latency > threshold. If `gpu_exporter_available`, cross-correlate with GPU util < floor → failure. If not available → warning with reduced confidence.
   - **`CACHE_FRAGMENTATION`:** Check `kv_fragmentation_pressure` > failure threshold (with high eviction) → failure. > warning threshold → warning.
5. **Build `data_completeness` dict:** Report `gpu_exporter_available`, list `metrics_missing`, list `checks_degraded`.
6. **Wire `_validate_comparison()` into `compare_benchmark_artifacts()`** — call after deltas/ratios are computed. Add `validation` key to return dict.
7. **Add `thresholds` kwarg to `compare_benchmark_artifacts()`** — defaults to `None` (uses `ComparisonValidationThresholds()` defaults). Accept a dict that overrides individual threshold fields.

**Acceptance:** Synthetic test artifacts with known bad metrics trigger the correct validation failures/warnings. Clean artifacts pass. Missing GPU exporter data produces degraded checks, not crashes. Existing callers without `thresholds` kwarg still work.

---

#### Task 3.3: Idle HBM Cost Helper
**File:** `tools/kv_cache.py`
**Estimated effort:** 2 hr
**Prereqs:** None (can parallel with 3.1/3.2)

**Subtasks:**
1. **Add `_GPU_COST_PER_HOUR_DEFAULTS` dict** — keyed by GPU name string (e.g., `"H100 SXM"`, `"B200"`). Values are approximate March 2026 on-demand rates. Documented with comment: `"Approximate on-demand rates as of March 2026. Override via gpu_cost_override_usd param."`
2. **Add `_CPU_DRAM_BANDWIDTH_GB_S = 300`** constant.
3. **Implement `_compute_idle_hbm_cost()`:**
   ```python
   def _compute_idle_hbm_cost(
       kv_per_session_gb: float,
       gpu_memory_gb: float,
       gpu_cost_per_hour_usd: float,
       idle_minutes: float = 5.0,
   ) -> dict:
   ```
   - Compute `hbm_fraction = kv_per_session_gb / gpu_memory_gb`.
   - Compute `idle_cost_per_min = hbm_fraction * gpu_cost_per_hour / 60`.
   - Compute `dram_reload_latency_ms = (kv_per_session_gb / _CPU_DRAM_BANDWIDTH_GB_S) * 1000`.
   - Compute `savings_per_cycle = idle_cost_per_min * idle_minutes`.
   - Determine `offload_recommended = savings_per_cycle > 0.001 AND reload_latency < 50`.
   - Return the full `idle_hbm_cost` dict (see §2 Feature 4 for shape).
4. **Wire into `recommend_kv_strategy()`:**
   - Resolve GPU cost: use `gpu_cost_override_usd` if provided, else look up `_GPU_COST_PER_HOUR_DEFAULTS`, else skip cost analysis.
   - Call `_compute_idle_hbm_cost()` with resolved cost and KV size from the existing memory plan.
   - Add `"idle_hbm_cost"` key to return dict.
   - Add `"gpu_cost_source": "override" | "default" | "unavailable"` to indicate provenance.

**Acceptance:** Math is consistent for H100 (80GB, $2/hr) and B200 (192GB, $5/hr) reference cases. `offload_recommended` triggers `True` for 5-min idle sessions with ≥0.5GB KV. Override param works. Missing GPU in defaults returns `"gpu_cost_source": "unavailable"` and skips cost section (no crash).

---

#### Task 3.4: MCP Tool Parameter Wiring
**File:** `server_benchmarks.py`
**Estimated effort:** 30 min
**Prereqs:** Tasks 3.2, 3.3

**Subtasks:**
1. Add `validation_thresholds: dict | None = None` param to `tool_compare_benchmarks()`.
2. Pass through to `compare_benchmark_artifacts(thresholds=validation_thresholds)`.
3. Add `gpu_cost_override_usd: float | None = None` param to `tool_recommend_kv_strategy()` (if not already exposed).
4. Update MCP tool descriptions to mention new return fields:
   - `tool_compare_benchmarks()`: note `validation` block in return.
   - `tool_recommend_kv_strategy()`: note `idle_hbm_cost` key in return.
   - `tool_profile_inference()`: note `cache` and `disaggregation` sub-dicts.

**Acceptance:** MCP tool calls with and without new optional params succeed. Return shapes include new fields.

---

### Phase 4 — Tests

#### Task 4.1: Telemetry Tests
**File:** `tests/test_telemetry.py` (new or extend)
**Estimated effort:** 2 hr
**Prereqs:** Tasks 1.1, 1.2

**Test cases:**
1. Dynamo scrape with synthetic Grove/NIXL/DCGM data → all fields populated correctly.
2. Dynamo scrape with DCGM absent → `gpu_exporter_available = False`, `gpu_compute_utilization_pct = 0.0`.
3. SGLang scrape → all new fields remain at default (0.0 / None / False).
4. ATOM scrape → all new fields remain at default.
5. `to_dict()` output includes `"cache"` sub-dict with Grove/LMCache fields.
6. `to_dict()` output includes `"disaggregation"` sub-dict with NIXL/DCGM/SLO fields.
7. `kv_fragmentation_pressure` derived correctly: `eviction_rate * cache_usage`.

---

#### Task 4.2: Benchmark Model & Runtime Tests
**File:** `tests/test_benchmarks.py` (extend)
**Estimated effort:** 1 hr
**Prereqs:** Tasks 2.1, 2.2, 3.1

**Test cases:**
1. `BenchmarkSummary` with `ttft_p90_ms` and `ttft_p99_ms` set → serializes correctly.
2. `BenchmarkSummary` JSON without P90/P99 fields → deserializes with `None` defaults.
3. `BenchmarkGoodputSLO` with `ttft_p99_ms` set → accepted.
4. `BenchmarkGoodputSLO` YAML without `ttft_p99_ms` → loads with `None` default.
5. `_build_summary()` with known TTFT distribution → P90 and P99 match expected percentiles.

---

#### Task 4.3: Comparison Validation Tests
**File:** `tests/test_comparison_validation.py` (new)
**Estimated effort:** 3 hr
**Prereqs:** Task 3.2

**Test cases:**
1. `EVICTION_RATE_SPIKE`: Candidate 5× baseline eviction rate → failure with correct metric values.
2. `EVICTION_RATE_SPIKE`: Candidate 1.5× baseline → passes (below 2.0× threshold).
3. `EVICTION_RATE_SPIKE`: Candidate below ratio but above 0.1/sec absolute → failure.
4. `P99_TTFT_SLO_BREACH`: Candidate P99 exceeds SLO → failure.
5. `P99_TTFT_SLO_BREACH`: No SLO defined → check skipped.
6. `DECODE_GPU_STARVATION`: NIXL > 50ms AND GPU util < 10% → failure.
7. `POSSIBLE_DECODE_STARVATION`: NIXL > 50ms, no DCGM → warning with confidence 0.60.
8. `DECODE_GPU_STARVATION`: NIXL > 50ms but GPU util > 10% → passes (GPU is working, latency is high but not starving).
9. `CACHE_FRAGMENTATION`: Pressure > 0.5 + high eviction → failure.
10. `CACHE_FRAGMENTATION`: Pressure 0.35 → warning.
11. `CACHE_FRAGMENTATION`: Pressure 0.2 → passes.
12. `data_completeness`: Missing DCGM → `gpu_exporter_available: false`, `checks_degraded` lists starvation downgrade.
13. P90/P99 deltas and ratios appear in comparison output.
14. Custom `validation_thresholds` override defaults correctly.
15. Clean artifacts → `"passed": true`, empty failures and warnings.

---

#### Task 4.4: KV Cache Cost Tests
**File:** `tests/test_kv_cache_cost.py` (new)
**Estimated effort:** 1.5 hr
**Prereqs:** Task 3.3

**Test cases:**
1. H100 SXM (80GB, $2/hr), 0.125GB KV, 5min idle → verify all computed values.
2. B200 (192GB, $5/hr), 0.5GB KV, 10min idle → verify offload_recommended = True.
3. Tiny KV (0.001GB), short idle (1min) → offload_recommended = False (savings below $0.001 threshold).
4. `gpu_cost_override_usd` provided → uses override, `gpu_cost_source: "override"`.
5. Unknown GPU not in defaults, no override → `gpu_cost_source: "unavailable"`, no `idle_hbm_cost` section (or graceful skip).
6. `dram_reload_latency_ms` computation matches manual calculation.

---

### Phase 5 — Live Validation

#### Task 5.1: End-to-End Kimi Validation
**Environment:** Modal H100 / B200
**Estimated effort:** 4 hr (includes deployment)
**Prereqs:** All Phase 4 tests green

**Scenario:**
1. Deploy Kimi-K2.5 on 2×H100 (1 prefill, 1 decode) with Dynamo + LMCache.
2. Run `kimi-k2-long-context-coding` benchmark: 64K context, 8 concurrent sessions, 30s idle between turns.
3. Verify via `tool_profile_inference()`:
   - Grove eviction counters are non-zero.
   - LMCache hit rate > 0 (prefix reuse is happening).
   - NIXL transfer latency histogram is populated.
   - `gpu_exporter_available` reflects DCGM sidecar presence.
4. Verify via `tool_run_benchmark()` + `tool_compare_benchmarks()`:
   - Baseline: `vllm-aggregated-prefix-cache-kimi-k2` on H100.
   - Candidate: `dynamo-disagg-lmcache-kimi-k2` on H100.
   - P99 TTFT is populated in both artifacts.
   - `validation` block is present with `passed` status.
   - Inject a synthetic degraded candidate → validation catches it.
5. Verify via `tool_recommend_kv_strategy()`:
   - `idle_hbm_cost` key present.
   - For 30s idle coding gap, recommendation should trigger `cpu_dram`.

---

## 5. Reused Existing Code

| Function | Location | Usage |
|----------|----------|-------|
| `_percentile(values, p)` | `benchmarks/runtime.py:81` | P90/P99 computation |
| `_delta(new, base)` | `benchmarks/catalog.py` | New delta fields |
| `_ratio(new, base)` | `benchmarks/catalog.py` | New ratio fields |
| `_cache_effectiveness_metric()` | `benchmarks/catalog.py` | Nested cache metric access |
| `scrape.get()` | `telemetry/prometheus.py` | Gauge/counter extraction |
| `scrape.get_histogram_avg()` | `telemetry/prometheus.py` | Histogram _sum/_count avg for NIXL |
| `resolve_platform_traits()` | `optimization/platform_policy.py` | GPU memory/capability resolution |
| `plan_memory()` | `optimization/memory_planner.py` | KV size calculation for idle cost |

---

## 6. Backward Compatibility

All changes are strictly additive. No existing signatures, return shapes, or field names change.

| Component | Compatibility |
|-----------|--------------|
| `NormalizedMetrics` new fields | Default to `0.0`, `None`, or `False` — existing scrapes unaffected |
| `BenchmarkSummary` new fields | Default to `None` — old JSON artifacts deserialize correctly |
| `BenchmarkGoodputSLO` new field | `ttft_p99_ms` optional — existing YAML specs unaffected |
| `compare_benchmark_artifacts()` | New `thresholds` kwarg defaults to `None` — existing callers unchanged |
| `tool_compare_benchmarks()` | New `validation_thresholds` param optional — existing MCP calls unchanged |
| `recommend_kv_strategy()` return | New `"idle_hbm_cost"` key — purely additive |
| `tool_recommend_kv_strategy()` | New `gpu_cost_override_usd` param optional — existing MCP calls unchanged |

---

## 7. Verification

```bash
# Unit + integration tests
cd products/inferscope && uv run pytest tests/ -v --tb=short

# Type check
cd products/inferscope && uv run mypy src/inferscope/

# Lint + format
cd products/inferscope && uv run ruff check src/ tests/
cd products/inferscope && uv run ruff format --check src/ tests/

# Security scan
cd products/inferscope && uv run bandit -r src/inferscope/ -c pyproject.toml -ll
```

---

## 8. What This Spec Does NOT Cover (Explicit Non-Goals)

These are real work items that should be separate specs:

| Non-goal | Why deferred |
|----------|-------------|
| README / docs rewrite | Ship code first, update docs to match. Coupling docs to code blocks both. |
| Modal deployment guide | Infrastructure concern, not observability. |
| New benchmark lane packaging | Existing lane definitions are sufficient for validation. |
| Sample-aware Prometheus extraction | Real need for multi-GPU label cardinality, but underspecified. Deserves its own design. |
| Coding-specific benchmark behavior (sticky sessions, prefix reuse testing) | Workload design, not observability. |
| `CLAUDE.md` updates | Follow-on after code ships. |

---

## Appendix A: Quick Reference — "I'm a New Engineer, What Do I Do?"

### "I want to profile a running Kimi deployment"
→ Call `tool_profile_inference()`. Look at the `cache` sub-dict for Grove tier usage and LMCache hit rates. Look at `disaggregation` for NIXL latency and GPU starvation signals.

### "I want to know if my Dynamo setup is better than vLLM"
→ Run `tool_run_benchmark()` with `vllm-aggregated-prefix-cache-kimi-k2` as baseline and `dynamo-disagg-lmcache-kimi-k2` as candidate. Then call `tool_compare_benchmarks()`. The `validation` block will tell you if anything is pathologically wrong, even if the average numbers look fine.

### "My P99 TTFT is terrible, what do I check?"
→ Look at `kv_transfer_latency_ms` in the disaggregation metrics. If > 50ms, the RDMA transfer is the bottleneck. Look at `kv_fragmentation_pressure` — if high, Grove is thrashing. Look at `lmcache_hit_rate` — if low, your prefix cache isn't helping and you may need to tune LMCache's dedup config.

### "Is it worth offloading KV cache to DRAM between user turns?"
→ Call `tool_recommend_kv_strategy()`. The `idle_hbm_cost` section shows the cost of holding KV in GPU memory during idle periods vs. the reload latency penalty of offloading to DRAM. For long-context coding sessions with multi-second idle gaps, offloading almost always wins.

### "I don't have DCGM deployed, will everything break?"
→ No. All validation checks degrade gracefully. Without DCGM, the `DECODE_GPU_STARVATION` check becomes `POSSIBLE_DECODE_STARVATION` (a warning instead of a failure) with lower confidence. The `data_completeness` block tells you exactly what metrics are missing and which checks were affected.