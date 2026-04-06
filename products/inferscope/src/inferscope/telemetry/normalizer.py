"""Cross-engine metric normalization.

Converts vLLM, SGLang, ATOM, and Dynamo metrics into a common InferScope format
so audit checks and diagnostics work regardless of engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferscope.telemetry.prometheus import ScrapeResult


@dataclass
class NormalizedMetrics:
    """Engine-agnostic metric snapshot for InferScope analysis."""

    engine: str
    endpoint: str

    # Request state
    requests_running: float = 0.0
    requests_waiting: float = 0.0
    requests_swapped: float = 0.0  # vLLM only

    # Cache
    kv_cache_usage: float = 0.0  # 0-1
    prefix_cache_hit_rate: float = 0.0  # 0-1
    cpu_cache_usage: float = 0.0  # 0-1

    # Throughput (counters — need rate() for per-second)
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    preemptions_total: float = 0.0
    request_success_total: float = 0.0
    errors_total: float = 0.0  # backend component errors (Dynamo)

    # Latency (histogram averages in seconds)
    ttft_avg_s: float | None = None  # Time to first token
    itl_avg_s: float | None = None  # Inter-token latency
    e2e_avg_s: float | None = None  # End-to-end latency
    queue_time_avg_s: float | None = None

    # Speculative decoding
    spec_acceptance_rate: float = 0.0

    # Generation throughput (gauge, tokens/sec — SGLang only)
    gen_throughput_tps: float = 0.0

    # Dynamo reliability/observability signals
    request_migrations_total: float = 0.0
    request_cancellations_total: float = 0.0  # dynamo_frontend_model_cancellation_total
    kv_publisher_dropped_events_total: float = 0.0  # stale router cache signal
    request_bytes_total: float = 0.0  # dynamo_component_request_bytes_total
    response_bytes_total: float = 0.0  # dynamo_component_response_bytes_total
    component_uptime_seconds: float = 0.0
    disconnected_clients: float = 0.0

    # KV block counters. NOTE on `kv_active_blocks`: there is NO
    # `dynamo_component_active_blocks` metric in Dynamo — verified
    # against lib/runtime/src/metrics/prometheus_names.rs (the kvstats
    # module defines only TOTAL_BLOCKS and GPU_CACHE_USAGE_PERCENT) and
    # against the Python emission site at
    # components/src/dynamo/common/utils/prometheus.py:322-335 which
    # registers exactly two KV-stats gauges. The field is kept on this
    # dataclass because downstream profiling code consumes it, but for
    # Dynamo deployments it stays at 0.0 unless an operator wires a
    # vLLM/SGLang worker scrape that emits it directly. Active blocks
    # can also be derived as `kv_total_blocks * kv_cache_usage` if the
    # absolute count is needed.
    kv_active_blocks: float = 0.0
    kv_total_blocks: float = 0.0

    # KV disaggregation metrics
    kvbm_offload_d2h: float = 0.0
    kvbm_onboard_h2d: float = 0.0
    nixl_transfer_latency_s: float | None = None
    nixl_transfer_bytes: float = 0.0
    nixl_transfer_failures: float = 0.0

    # KVBM tiering metrics — scraped from a SEPARATE endpoint (default port
    # 6880) when the user launches Dynamo with `DYN_KVBM_METRICS=true`.
    # These fields are populated only when the operator configures a KVBM
    # scrape target; otherwise they stay at 0.0 and the KVBM-aware checks
    # no-fire because there's no data to judge.
    kvbm_host_hit_rate: float = 0.0
    kvbm_disk_hit_rate: float = 0.0
    kvbm_object_hit_rate: float = 0.0

    # LMCache metrics — scraped via the `lmcache:` Prometheus prefix.
    # LMCache is an upstream project (github.com/LMCache/LMCache) and
    # emits its own metrics; the Dynamo /metrics endpoint does NOT expose
    # them under `dynamo_lmcache_*` despite what older design notes claimed.
    lmcache_hit_rate: float = 0.0
    lmcache_retrieve_speed_tps: float = 0.0

    # Router-side metrics (Dynamo frontend router layer). These separate
    # "routing overhead" from "backend compute" — when TTFT is high, these
    # tell you whether the router or the backend is responsible.
    router_overhead_total_ms: float | None = None  # histogram avg, milliseconds
    router_overhead_block_hashing_ms: float | None = None
    router_overhead_indexer_ms: float | None = None  # dynamo_router_overhead_indexer_find_matches_ms
    router_overhead_scheduling_ms: float | None = None
    router_kv_hit_rate: float | None = None  # dynamo_component_router_kv_hit_rate (histogram 0-1)
    router_queue_depth: float = 0.0  # dynamo_frontend_router_queue_pending_requests

    # Cached-tokens-per-request histogram average. Direct measure of how
    # many tokens were served from the prefix/KV cache on average. More
    # reliable than hit_rate for coding workloads.
    cached_tokens_avg: float | None = None

    # Frontend tokenizer latency histogram average (milliseconds).
    # Useful for TTFT decomposition — if tokenizer_latency is a large
    # fraction of TTFT, the bottleneck is on the CPU pre-processing
    # path, not the GPU prefill path.
    tokenizer_latency_ms: float | None = None

    # KV cache event counter (applied events for the kv-aware router).
    kv_cache_events_applied: float = 0.0

    # Model-config gauges reported by the Dynamo frontend. Useful for
    # validating that a live deployment's declared budget matches what
    # the recommender suggested, and for audit-check context.
    model_total_kv_blocks: float = 0.0
    model_max_num_seqs: float = 0.0
    model_max_num_batched_tokens: float = 0.0
    model_context_length: float = 0.0
    model_kv_cache_block_size: float = 0.0

    # Computed goodput metrics (derived from inferencebreakpoints/11-observability/metrics/goodput)
    # Goodput = useful output tokens/sec, excluding waste from preemptions, failures, and SLO violations
    goodput_tps: float = 0.0  # tokens/sec accounting for waste
    goodput_ratio: float = 0.0  # goodput / raw throughput (1.0 = no waste)

    # TTFT decomposition (derived from inferencebreakpoints/06-prefill/ttft-analysis)
    # When queue_time is available, prefill_time = ttft - queue_time
    prefill_compute_s: float | None = None  # estimated pure prefill compute time
    queue_fraction: float = 0.0  # fraction of TTFT spent in queue (0-1)

    # Scrape metadata
    scrape_time_ms: float = 0.0
    scrape_error: str = ""

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "endpoint": self.endpoint,
            "request_state": {
                "running": self.requests_running,
                "waiting": self.requests_waiting,
                "swapped": self.requests_swapped,
            },
            "cache": {
                "kv_usage": round(self.kv_cache_usage, 4),
                "prefix_hit_rate": round(self.prefix_cache_hit_rate, 4),
                "cpu_usage": round(self.cpu_cache_usage, 4),
            },
            "throughput": {
                "prompt_tokens_total": self.prompt_tokens_total,
                "generation_tokens_total": self.generation_tokens_total,
                "preemptions_total": self.preemptions_total,
                "request_success_total": self.request_success_total,
                "gen_throughput_tps": round(self.gen_throughput_tps, 1),
            },
            "reliability": {
                "request_migrations_total": self.request_migrations_total,
                "request_cancellations_total": self.request_cancellations_total,
                "kv_publisher_dropped_events_total": self.kv_publisher_dropped_events_total,
                "disconnected_clients": self.disconnected_clients,
                "kv_active_blocks": self.kv_active_blocks,
                "kv_total_blocks": self.kv_total_blocks,
                "errors_total": self.errors_total,
                "component_uptime_seconds": self.component_uptime_seconds,
                "request_bytes_total": self.request_bytes_total,
                "response_bytes_total": self.response_bytes_total,
            },
            "disaggregation": {
                "kvbm_offload_d2h": self.kvbm_offload_d2h,
                "kvbm_onboard_h2d": self.kvbm_onboard_h2d,
                "nixl_transfer_latency_ms": round(self.nixl_transfer_latency_s * 1000, 2) if self.nixl_transfer_latency_s else None,
                "nixl_transfer_bytes": self.nixl_transfer_bytes,
                "nixl_transfer_failures": self.nixl_transfer_failures,
            },
            "tiering": {
                "kvbm_host_hit_rate": self.kvbm_host_hit_rate,
                "kvbm_disk_hit_rate": self.kvbm_disk_hit_rate,
                "kvbm_object_hit_rate": self.kvbm_object_hit_rate,
            },
            "lmcache": {
                "hit_rate": self.lmcache_hit_rate,
                "retrieve_speed_tps": self.lmcache_retrieve_speed_tps,
            },
            "router": {
                "overhead_total_ms": (
                    round(self.router_overhead_total_ms, 2)
                    if self.router_overhead_total_ms is not None
                    else None
                ),
                "overhead_block_hashing_ms": (
                    round(self.router_overhead_block_hashing_ms, 2)
                    if self.router_overhead_block_hashing_ms is not None
                    else None
                ),
                "overhead_indexer_ms": (
                    round(self.router_overhead_indexer_ms, 2)
                    if self.router_overhead_indexer_ms is not None
                    else None
                ),
                "overhead_scheduling_ms": (
                    round(self.router_overhead_scheduling_ms, 2)
                    if self.router_overhead_scheduling_ms is not None
                    else None
                ),
                "kv_hit_rate": (
                    round(self.router_kv_hit_rate, 4)
                    if self.router_kv_hit_rate is not None
                    else None
                ),
                "queue_depth": self.router_queue_depth,
                "kv_cache_events_applied": self.kv_cache_events_applied,
                "cached_tokens_avg": (
                    round(self.cached_tokens_avg, 1)
                    if self.cached_tokens_avg is not None
                    else None
                ),
                "tokenizer_latency_ms": (
                    round(self.tokenizer_latency_ms, 2)
                    if self.tokenizer_latency_ms is not None
                    else None
                ),
            },
            "model_config": {
                "total_kv_blocks": self.model_total_kv_blocks,
                "max_num_seqs": self.model_max_num_seqs,
                "max_num_batched_tokens": self.model_max_num_batched_tokens,
                "context_length": self.model_context_length,
                "kv_cache_block_size": self.model_kv_cache_block_size,
            },
            "goodput": {
                "goodput_tps": round(self.goodput_tps, 1),
                "goodput_ratio": round(self.goodput_ratio, 3),
            },
            "ttft_decomposition": {
                "prefill_compute_ms": round(self.prefill_compute_s * 1000, 1) if self.prefill_compute_s else None,
                "queue_fraction": round(self.queue_fraction, 3),
            },
            "latency": {
                "ttft_avg_ms": round(self.ttft_avg_s * 1000, 1) if self.ttft_avg_s else None,
                "itl_avg_ms": round(self.itl_avg_s * 1000, 1) if self.itl_avg_s else None,
                "e2e_avg_ms": round(self.e2e_avg_s * 1000, 1) if self.e2e_avg_s else None,
                "queue_time_avg_ms": (round(self.queue_time_avg_s * 1000, 1) if self.queue_time_avg_s else None),
            },
            "speculation": {
                "acceptance_rate": round(self.spec_acceptance_rate, 3),
            },
            "scrape": {
                "time_ms": round(self.scrape_time_ms, 1),
                "error": self.scrape_error,
            },
        }


def _compute_goodput(m: NormalizedMetrics) -> None:
    """Compute goodput from raw throughput and preemption rate.

    Goodput accounts for wasted compute: tokens generated for preempted
    requests that have to be recomputed. Grounded in
    inferencebreakpoints/11-observability/metrics/goodput.

    Note: earlier versions of this function also added an SLO-violation
    waste term, but Dynamo does not expose server-side SLO violation
    counters (that concept belongs to the client / harness side). Any
    SLO-related waste is already captured by the preemption signal and
    by the HIGH_TTFT / HIGH_ITL checks.
    """
    raw_throughput = m.gen_throughput_tps
    if raw_throughput <= 0:
        return

    waste_fraction = 0.0

    # Preemption waste: each preemption wastes the partial decode of that request
    if m.preemptions_total > 0 and m.request_success_total > 0:
        preemption_rate = m.preemptions_total / max(m.request_success_total, 1)
        waste_fraction += min(preemption_rate * 0.5, 0.3)  # cap at 30% from preemptions

    waste_fraction = min(waste_fraction, 0.5)  # total waste capped at 50%
    m.goodput_tps = raw_throughput * (1.0 - waste_fraction)
    m.goodput_ratio = 1.0 - waste_fraction


def _compute_ttft_decomposition(m: NormalizedMetrics) -> None:
    """Decompose TTFT into queue wait time and prefill compute time.

    When queue_time_avg_s is available: prefill_compute = ttft - queue_time.
    This helps operators distinguish 'is TTFT high because of queueing or
    because prefill is slow?' — a key diagnostic from
    inferencebreakpoints/06-prefill/ttft-analysis.
    """
    if m.ttft_avg_s is None or m.ttft_avg_s <= 0:
        return

    if m.queue_time_avg_s is not None and m.queue_time_avg_s >= 0:
        prefill_time = max(0.0, m.ttft_avg_s - m.queue_time_avg_s)
        m.prefill_compute_s = prefill_time
        m.queue_fraction = m.queue_time_avg_s / m.ttft_avg_s if m.ttft_avg_s > 0 else 0.0
    else:
        # Without queue time, we can still estimate from requests_waiting
        # If there's a significant queue, TTFT is likely queue-dominated
        if m.requests_waiting > 20:
            m.queue_fraction = 0.7  # heuristic: heavy queueing
        elif m.requests_waiting > 5:
            m.queue_fraction = 0.3  # heuristic: moderate queueing
        else:
            m.queue_fraction = 0.05  # heuristic: minimal queueing
            m.prefill_compute_s = m.ttft_avg_s * 0.95


def normalize(scrape: ScrapeResult) -> NormalizedMetrics:
    """Convert engine-specific ScrapeResult into NormalizedMetrics."""
    m = NormalizedMetrics(
        engine=scrape.engine,
        endpoint=scrape.endpoint,
        scrape_time_ms=scrape.scrape_time_ms,
        scrape_error=scrape.error,
    )

    if scrape.error:
        return m

    if scrape.engine == "vllm":
        m.requests_running = scrape.get("vllm:num_requests_running")
        m.requests_waiting = scrape.get("vllm:num_requests_waiting")
        m.requests_swapped = scrape.get("vllm:num_requests_swapped")
        m.kv_cache_usage = scrape.get("vllm:gpu_cache_usage_perc")
        m.prefix_cache_hit_rate = scrape.get("vllm:gpu_prefix_cache_hit_rate")
        m.cpu_cache_usage = scrape.get("vllm:cpu_cache_usage_perc")
        m.prompt_tokens_total = scrape.get("vllm:prompt_tokens_total")
        m.generation_tokens_total = scrape.get("vllm:generation_tokens_total")
        m.preemptions_total = scrape.get("vllm:num_preemptions_total")
        m.request_success_total = scrape.get("vllm:request_success_total")
        m.spec_acceptance_rate = scrape.get("vllm:spec_decode_draft_acceptance_rate")
        m.ttft_avg_s = scrape.get_histogram_avg("vllm:time_to_first_token_seconds")
        m.itl_avg_s = scrape.get_histogram_avg("vllm:time_per_output_token_seconds")
        m.e2e_avg_s = scrape.get_histogram_avg("vllm:e2e_request_latency_seconds")
        m.queue_time_avg_s = scrape.get_histogram_avg("vllm:request_queue_time_seconds")

    elif scrape.engine == "sglang":
        m.requests_running = scrape.get("sglang:num_running_reqs")
        m.requests_waiting = scrape.get("sglang:num_queue_reqs")
        m.kv_cache_usage = scrape.get("sglang:token_usage")
        m.prefix_cache_hit_rate = scrape.get("sglang:cache_hit_rate")
        m.prompt_tokens_total = scrape.get("sglang:prompt_tokens_total")
        m.generation_tokens_total = scrape.get("sglang:generation_tokens_total")
        m.gen_throughput_tps = scrape.get("sglang:gen_throughput")
        m.ttft_avg_s = scrape.get_histogram_avg("sglang:time_to_first_token_seconds")
        m.itl_avg_s = scrape.get_histogram_avg("sglang:time_per_output_token_seconds")
        m.e2e_avg_s = scrape.get_histogram_avg("sglang:e2e_request_latency_seconds")

    elif scrape.engine == "atom":
        # ATOM follows vLLM schema with atom: prefix
        m.requests_running = scrape.get("atom:num_requests_running")
        m.requests_waiting = scrape.get("atom:num_requests_waiting")
        m.kv_cache_usage = scrape.get("atom:kv_cache_usage_perc")
        m.ttft_avg_s = scrape.get_histogram_avg("atom:time_to_first_token_seconds")
        m.itl_avg_s = scrape.get_histogram_avg("atom:inter_token_latency_seconds")

    elif scrape.engine == "dynamo":
        m.requests_running = scrape.get("dynamo_frontend_inflight_requests") or scrape.get(
            "dynamo_component_inflight_requests"
        )
        m.requests_waiting = scrape.get("dynamo_frontend_queued_requests")
        # KV cache usage: emitted by decode workers as
        # dynamo_component_gpu_cache_usage_percent. Note: per
        # deploy/observability/grafana_dashboards/DASHBOARD_METRICS.md,
        # in disaggregated mode prefill workers do NOT expose this
        # metric — only decode workers do. The fallback to
        # vllm:gpu_cache_usage_perc covers the case where an operator
        # also scrapes a vLLM worker endpoint directly.
        m.kv_cache_usage = scrape.get("dynamo_component_gpu_cache_usage_percent") or scrape.get(
            "vllm:gpu_cache_usage_perc"
        )
        # Prefix cache hit rate: Dynamo does NOT emit a worker-side
        # `dynamo_component_gpu_prefix_cache_hit_rate` metric (verified
        # by exhaustive grep across the entire ai-dynamo/dynamo repo).
        # The closest router-side equivalent is the histogram
        # dynamo_component_router_kv_hit_rate, which measures the
        # fraction of input tokens the router found in any worker's
        # cache before dispatching the request. We use it here as the
        # primary signal and fall back to the vLLM worker prefix cache
        # hit rate if a vLLM worker is also being scraped.
        m.prefix_cache_hit_rate = (
            scrape.get_histogram_avg("dynamo_component_router_kv_hit_rate")
            or scrape.get("vllm:gpu_prefix_cache_hit_rate")
            or 0.0
        )
        m.prompt_tokens_total = scrape.get("vllm:prompt_tokens_total")
        m.generation_tokens_total = scrape.get("dynamo_frontend_output_tokens_total") or scrape.get(
            "vllm:generation_tokens_total"
        )
        m.request_success_total = scrape.get("dynamo_frontend_requests_total") or scrape.get(
            "dynamo_component_requests_total"
        )
        m.errors_total = scrape.get("dynamo_component_errors_total")
        m.ttft_avg_s = scrape.get_histogram_avg(
            "dynamo_frontend_time_to_first_token_seconds"
        ) or scrape.get_histogram_avg("vllm:time_to_first_token_seconds")
        m.itl_avg_s = scrape.get_histogram_avg(
            "dynamo_frontend_inter_token_latency_seconds"
        ) or scrape.get_histogram_avg("vllm:time_per_output_token_seconds")
        m.e2e_avg_s = scrape.get_histogram_avg("dynamo_frontend_request_duration_seconds") or scrape.get_histogram_avg(
            "dynamo_component_request_duration_seconds"
        )
        m.request_migrations_total = scrape.get("dynamo_frontend_model_migration_total")
        m.request_cancellations_total = scrape.get("dynamo_frontend_model_cancellation_total")
        m.kv_publisher_dropped_events_total = scrape.get(
            "dynamo_component_kv_publisher_engines_dropped_events_total"
        )
        m.request_bytes_total = scrape.get("dynamo_component_request_bytes_total")
        m.response_bytes_total = scrape.get("dynamo_component_response_bytes_total")
        m.component_uptime_seconds = scrape.get("dynamo_component_uptime_seconds")
        m.disconnected_clients = scrape.get("dynamo_frontend_disconnected_clients")
        m.kv_total_blocks = scrape.get("dynamo_component_total_blocks")
        # KVBM tiering metrics — only present if the operator launched
        # Dynamo with DYN_KVBM_METRICS=true and added the KVBM /metrics
        # endpoint (port 6880 by default) to the scrape target list.
        # If KVBM is not enabled, these fields stay at 0.0 and downstream
        # tiering checks silently no-fire (correct behavior: no data = no
        # finding, not a phantom clean bill of health).
        m.kvbm_offload_d2h = scrape.get("kvbm_offload_blocks_d2h")
        m.kvbm_onboard_h2d = scrape.get("kvbm_onboard_blocks_h2d")
        m.kvbm_host_hit_rate = scrape.get("kvbm_host_cache_hit_rate")
        m.kvbm_disk_hit_rate = scrape.get("kvbm_disk_cache_hit_rate")
        m.kvbm_object_hit_rate = scrape.get("kvbm_object_cache_hit_rate")

        # NIXL transfer metrics — NIXL exposes metrics on a SEPARATE
        # endpoint via NIXL_TELEMETRY_PROMETHEUS_PORT. The exact metric
        # name schema is not documented in the current Dynamo repo, so
        # these fields only populate when an operator wires a NIXL scrape
        # target whose metric names happen to match. Left dormant by
        # default — the NIXL_TRANSFER_DOMINATES check will no-fire until
        # a real NIXL schema is pinned down from a captured scrape.
        m.nixl_transfer_latency_s = scrape.get_histogram_avg(
            "dynamo_nixl_transfer_latency_seconds"
        ) or None
        m.nixl_transfer_bytes = scrape.get("dynamo_nixl_transfer_bytes_total")
        m.nixl_transfer_failures = scrape.get("dynamo_nixl_transfer_failures_total")

        # LMCache hit rate — computed from the upstream `lmcache:` prefix
        # (LMCache is its own project and emits its own metrics). Older
        # versions of InferScope looked for `dynamo_lmcache_hit_rate`,
        # which is not a metric Dynamo actually emits.
        lmc_hit_tokens = scrape.get("lmcache:num_hit_tokens_total")
        lmc_req_tokens = scrape.get("lmcache:num_requested_tokens_total")
        if lmc_req_tokens > 0:
            m.lmcache_hit_rate = lmc_hit_tokens / lmc_req_tokens
        lmc_speed_sum = scrape.get("lmcache:retrieve_speed_sum")
        lmc_speed_count = scrape.get("lmcache:retrieve_speed_count")
        if lmc_speed_count > 0:
            m.lmcache_retrieve_speed_tps = lmc_speed_sum / lmc_speed_count

        # Router-overhead histograms (unit suffix is _ms; values are raw
        # milliseconds, not seconds, per the Dynamo metric convention).
        m.router_overhead_total_ms = scrape.get_histogram_avg("dynamo_router_overhead_total_ms")
        m.router_overhead_block_hashing_ms = scrape.get_histogram_avg(
            "dynamo_router_overhead_block_hashing_ms"
        )
        m.router_overhead_indexer_ms = scrape.get_histogram_avg(
            "dynamo_router_overhead_indexer_find_matches_ms"
        )
        m.router_overhead_scheduling_ms = scrape.get_histogram_avg(
            "dynamo_router_overhead_scheduling_ms"
        )
        # Router KV hit rate histogram — values are a 0-1 rate.
        m.router_kv_hit_rate = scrape.get_histogram_avg(
            "dynamo_component_router_kv_hit_rate"
        )
        m.router_queue_depth = scrape.get("dynamo_frontend_router_queue_pending_requests")

        # Tokens-served-from-cache histogram average (per request).
        m.cached_tokens_avg = scrape.get_histogram_avg("dynamo_frontend_cached_tokens")

        # Frontend tokenizer latency histogram (raw milliseconds).
        m.tokenizer_latency_ms = scrape.get_histogram_avg("dynamo_frontend_tokenizer_latency_ms")

        # KV cache event counter.
        m.kv_cache_events_applied = scrape.get("dynamo_component_kv_cache_events_applied")

        # Model-config gauges reported by the frontend.
        m.model_total_kv_blocks = scrape.get("dynamo_frontend_model_total_kv_blocks")
        m.model_max_num_seqs = scrape.get("dynamo_frontend_model_max_num_seqs")
        m.model_max_num_batched_tokens = scrape.get(
            "dynamo_frontend_model_max_num_batched_tokens"
        )
        m.model_context_length = scrape.get("dynamo_frontend_model_context_length")
        m.model_kv_cache_block_size = scrape.get(
            "dynamo_frontend_model_kv_cache_block_size"
        )

    # ── Compute derived metrics ────────────────────────────────────────
    _compute_goodput(m)
    _compute_ttft_decomposition(m)

    return m
