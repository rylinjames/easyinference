"""Prometheus metrics scraper for vLLM, SGLang, ATOM, and Dynamo endpoints.

Parses the standard Prometheus text exposition format from /metrics endpoints.
Metric names are sourced from official docs:
- vLLM:   https://docs.vllm.ai/en/v0.8.5/design/v1/metrics.html
- SGLang: https://docs.sglang.io/references/production_metrics.html
- Dynamo: https://github.com/ai-dynamo/dynamo/blob/main/docs/observability/metrics.md
          and lib/runtime/src/metrics/prometheus_names.rs (authoritative constants)

Historical note: earlier revisions of this file also listed fictional
`dynamo_grove_*`, `dynamo_lmcache_*`, and `dynamo_slo_*` metric names.
Those were invented — Dynamo does not emit them. Grove is a Kubernetes
gang-scheduling component with no Prometheus metrics; LMCache uses the
upstream `lmcache:` prefix; SLO violation counters don't exist
server-side (compute from histogram buckets client-side if needed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import httpx

from inferscope.endpoint_auth import EndpointAuthConfig, build_auth_headers
from inferscope.logging import get_logger, sanitize_log_url
from inferscope.security import validate_endpoint

log = get_logger(component="prometheus")

# Metric name → human-readable description for the metrics we care about
VLLM_METRICS = {
    # Gauges
    "vllm:num_requests_running": "Active requests being processed",
    "vllm:num_requests_waiting": "Requests queued waiting for processing",
    "vllm:num_requests_swapped": "Requests swapped to CPU",
    "vllm:gpu_cache_usage_perc": "GPU KV cache utilization (0-1)",
    "vllm:cpu_cache_usage_perc": "CPU KV cache utilization (0-1)",
    "vllm:gpu_prefix_cache_hit_rate": "GPU prefix cache hit rate (0-1)",
    "vllm:spec_decode_draft_acceptance_rate": "Speculative decode acceptance rate",
    # Counters
    "vllm:prompt_tokens_total": "Total prompt tokens processed",
    "vllm:generation_tokens_total": "Total generation tokens produced",
    "vllm:request_success_total": "Total successful requests",
    "vllm:num_preemptions_total": "Total KV cache preemptions",
    # Histograms (we extract _sum and _count for averages)
    "vllm:time_to_first_token_seconds": "Time to first token (TTFT)",
    "vllm:time_per_output_token_seconds": "Inter-token latency (ITL)",
    "vllm:e2e_request_latency_seconds": "End-to-end request latency",
    "vllm:request_queue_time_seconds": "Time spent in queue",
}

SGLANG_METRICS = {
    # Gauges
    "sglang:num_running_reqs": "Active requests being processed",
    "sglang:num_queue_reqs": "Requests queued waiting",
    "sglang:token_usage": "Current token usage ratio",
    "sglang:cache_hit_rate": "Prefix cache hit rate",
    "sglang:gen_throughput": "Generation throughput (tokens/sec)",
    "sglang:num_used_tokens": "Currently allocated KV tokens",
    # Counters
    "sglang:prompt_tokens_total": "Total prompt tokens processed",
    "sglang:generation_tokens_total": "Total generation tokens produced",
    # Histograms
    "sglang:time_to_first_token_seconds": "Time to first token (TTFT)",
    "sglang:time_per_output_token_seconds": "Inter-token latency (ITL)",
    "sglang:e2e_request_latency_seconds": "End-to-end request latency",
}

# ATOM uses vLLM-compatible metric names with atom: prefix
ATOM_METRICS = {
    "atom:num_requests_running": "Active requests being processed",
    "atom:num_requests_waiting": "Requests queued waiting",
    "atom:kv_cache_usage_perc": "KV cache utilization",
    "atom:time_to_first_token_seconds": "Time to first token (TTFT)",
    "atom:inter_token_latency_seconds": "Inter-token latency (ITL)",
}

DYNAMO_METRICS = {
    # --- Frontend request and latency metrics ---
    "dynamo_frontend_inflight_requests": "Inflight requests at the HTTP frontend",
    "dynamo_frontend_queued_requests": "Queued requests at the HTTP frontend",
    "dynamo_frontend_disconnected_clients": "Disconnected streaming clients",
    "dynamo_frontend_output_tokens_total": "Total generated output tokens",
    "dynamo_frontend_requests_total": "Total requests observed by the frontend",
    "dynamo_frontend_time_to_first_token_seconds": "Frontend time to first token (TTFT)",
    "dynamo_frontend_inter_token_latency_seconds": "Frontend inter-token latency (ITL)",
    "dynamo_frontend_request_duration_seconds": "Frontend end-to-end request latency",
    "dynamo_frontend_model_migration_total": "Total request migrations due to worker unavailability",
    "dynamo_frontend_model_cancellation_total": "Total requests cancelled by the client",
    "dynamo_frontend_input_sequence_tokens": "Input sequence token count distribution (histogram)",
    "dynamo_frontend_output_sequence_tokens": "Output sequence token count distribution (histogram)",
    "dynamo_frontend_cached_tokens": "Tokens served from cache per request (histogram)",
    # --- Frontend model-config gauges ---
    "dynamo_frontend_model_total_kv_blocks": "Total KV blocks configured for the served model",
    "dynamo_frontend_model_max_num_seqs": "Declared max concurrent sequences",
    "dynamo_frontend_model_max_num_batched_tokens": "Declared batched token budget",
    "dynamo_frontend_model_context_length": "Declared max context length",
    "dynamo_frontend_model_kv_cache_block_size": "KV cache block size (tokens per block)",
    "dynamo_frontend_model_migration_limit": "Per-request migration limit",
    # --- Frontend router gauges ---
    "dynamo_frontend_router_queue_pending_requests": "Router-side pending request queue depth",
    # --- Per-worker gauges (labeled with worker_id, dp_rank, worker_type) ---
    "dynamo_frontend_worker_active_decode_blocks": "Active decode KV blocks per worker",
    "dynamo_frontend_worker_active_prefill_tokens": "Active prefill tokens per worker",
    "dynamo_frontend_worker_last_time_to_first_token_seconds": "Last observed TTFT per worker",
    "dynamo_frontend_worker_last_input_sequence_tokens": "Last observed input sequence tokens per worker",
    "dynamo_frontend_worker_last_inter_token_latency_seconds": "Last observed ITL per worker",
    # --- Router-overhead histograms (values are raw ms, not seconds) ---
    # Names verified at the registration site in
    # lib/llm/src/kv_router/metrics.rs. (The `ROUTING_OVERHEAD` namespace
    # constant in lib/runtime/src/metrics/prometheus_names.rs is an alt
    # namespace that is not used at the kv_router emission site — the
    # emission uses `dynamo_router_overhead_` directly.)
    "dynamo_router_overhead_block_hashing_ms": "Block hashing overhead inside the router",
    "dynamo_router_overhead_indexer_find_matches_ms": "KV indexer match-finding overhead",
    "dynamo_router_overhead_seq_hashing_ms": "Sequence hashing overhead",
    "dynamo_router_overhead_scheduling_ms": "Router scheduling overhead",
    "dynamo_router_overhead_total_ms": "Total routing overhead per request",
    # Tokenizer latency histogram (raw ms, not seconds). Emitted by the
    # HTTP frontend — verified at the registration site in
    # lib/llm/src/http/service/metrics.rs.
    "dynamo_frontend_tokenizer_latency_ms": "Per-request tokenization overhead (histogram, ms)",
    # --- Backend component metrics ---
    "dynamo_component_inflight_requests": "Requests currently processed by a backend component",
    "dynamo_component_request_duration_seconds": "Backend component request duration",
    "dynamo_component_requests_total": "Total requests processed by a backend component",
    "dynamo_component_request_bytes_total": "Total request payload bytes",
    "dynamo_component_response_bytes_total": "Total response payload bytes",
    "dynamo_component_uptime_seconds": "Component uptime in seconds",
    "dynamo_component_kv_publisher_engines_dropped_events_total":
        "KV publisher dropped events — indicates the router's cache view is stale",
    # KV-stats metrics emitted by the worker/router. Note: there is NO
    # `kvstats` segment in the real metric names. The bare constants in
    # the Dynamo source (lib/runtime/src/metrics/prometheus_names.rs)
    # are `active_blocks`, `total_blocks`, `gpu_cache_usage_percent`,
    # and `gpu_prefix_cache_hit_rate`; they get the `dynamo_component_`
    # prefix at registration time. Verified against the live PromQL
    # queries in deploy/observability/grafana_dashboards/disagg-dashboard.json.
    "dynamo_component_active_blocks": "Active KV blocks on the worker",
    "dynamo_component_total_blocks": "Total KV blocks on the worker",
    "dynamo_component_gpu_cache_usage_percent": "Worker GPU KV cache utilization",
    "dynamo_component_gpu_prefix_cache_hit_rate": "Worker GPU prefix cache hit rate",
    "dynamo_component_errors_total": "Total backend component errors",
    "dynamo_component_kv_cache_events_applied": "Count of KV cache events applied by the router",
    # --- Backend component-router histograms (Dynamo mirrors router metrics ---
    # --- on the component side for per-worker attribution)                 ---
    "dynamo_component_router_requests_total": "Component-side router request counter",
    "dynamo_component_router_kv_hit_rate": "Component-side router KV hit rate (0-1 histogram)",
    "dynamo_component_router_time_to_first_token_seconds": "Component-side TTFT (histogram)",
    "dynamo_component_router_inter_token_latency_seconds": "Component-side ITL (histogram)",
    "dynamo_component_router_input_sequence_tokens": "Component-side input-token histogram",
    "dynamo_component_router_output_sequence_tokens": "Component-side output-token histogram",
    # KVBM offload/onboard block counters. KVBM exposes these on a SEPARATE
    # endpoint (default port 6880 via DYN_KVBM_METRICS_PORT) and only when
    # launched with DYN_KVBM_METRICS=true. An operator must add that port
    # as an additional metrics_target for these to appear in the scrape.
    "kvbm_offload_blocks_d2h": "GPU-to-CPU KV block offload operations",
    "kvbm_offload_blocks_h2d": "CPU-to-disk KV block offload operations",
    "kvbm_offload_blocks_d2d": "Device-to-device KV block offload operations",
    "kvbm_offload_blocks_d2o": "Device-to-object-storage KV block offload operations",
    "kvbm_onboard_blocks_h2d": "CPU-to-GPU KV block onboard operations",
    "kvbm_onboard_blocks_d2d": "Device-to-device KV block onboard operations",
    "kvbm_onboard_blocks_o2d": "Object-storage-to-device KV block onboard operations",
    "kvbm_host_cache_hit_rate": "CPU tier KV cache hit rate",
    "kvbm_disk_cache_hit_rate": "Disk tier KV cache hit rate",
    "kvbm_object_cache_hit_rate": "Object storage tier KV cache hit rate",
    "kvbm_matched_tokens": "Tokens reused from KVBM cache",
}

# KV TRANSFER LATENCY — two data sources, both in progress.
#
# Source 1 (preferred once merged): Dynamo's own KV router computes an
# upper-bound KV transfer latency for disaggregated serving and exposes
# it as `dynamo_component_router_kv_transfer_latency_seconds`
# (histogram, 15 log-scale buckets from 1ms to 10s). This is introduced
# in ai-dynamo/dynamo PR #7590 which is currently OPEN (not merged).
# Once merged, the dormant NIXL_TRANSFER_DOMINATES check can be rewired
# to read this field directly from the normal Dynamo frontend /metrics
# scrape, no separate NIXL scrape target required. Follow the PR at:
#   https://github.com/ai-dynamo/dynamo/pull/7590
#
# Source 2 (upstream NIXL library): NIXL is exposed on a SEPARATE
# Prometheus endpoint (port 19090 per Dynamo's own
# deploy/observability/prometheus.yml scrape config) and is emitted by
# the NIXL upstream library, NOT by Dynamo runtime. The metric name
# schema is not documented in the Dynamo repo — searching it for
# `dynamo_nixl_` returns zero results — so the earlier
# `dynamo_nixl_transfer_*` names this module carried were invented.
# If an operator specifically needs to monitor NIXL independently of
# Dynamo's router-computed upper bound, they'd need to capture a real
# NIXL /metrics scrape from the :19090 endpoint and add the real
# metric names here.
#
# Both sources feed the `nixl_transfer_latency_s` field on
# NormalizedMetrics. Until one of them is wired in, the dormant
# NIXL_TRANSFER_DOMINATES check stays silent on a clean deployment —
# the correct no-finding outcome when there's no data to judge.

LMCACHE_METRICS: dict[str, str] = {
    "lmcache:num_hit_tokens_total": "Total tokens found in LMCache",
    "lmcache:num_requested_tokens_total": "Total tokens requested from LMCache",
    "lmcache:retrieve_speed_sum": "Cumulative retrieve speed (for averaging)",
    "lmcache:retrieve_speed_count": "Retrieve speed sample count",
    "lmcache:store_speed_sum": "Cumulative store speed (for averaging)",
    "lmcache:store_speed_count": "Store speed sample count",
    "lmcache:local_cache_usage": "Local LMCache storage utilization",
}

# Regex for Prometheus text format: metric_name{labels} value [timestamp]
_METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"  # metric name
    r"(?:\{([^}]*)\})?"  # optional labels
    r"\s+"  # whitespace
    r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|NaN|[+-]Inf)"  # value
)


@dataclass
class MetricSample:
    """A single Prometheus metric sample."""

    name: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0


@dataclass
class ScrapeResult:
    """Result of scraping a Prometheus /metrics endpoint."""

    endpoint: str
    engine: str  # "vllm" | "sglang" | "atom" | "dynamo" | "unknown"
    raw_metrics: dict[str, float] = field(default_factory=dict)
    samples: list[MetricSample] = field(default_factory=list)
    error: str = ""
    scrape_time_ms: float = 0.0

    def get(self, name: str, default: float = 0.0) -> float:
        """Get a metric value by name."""
        return self.raw_metrics.get(name, default)

    def get_histogram_avg(self, base_name: str) -> float | None:
        """Calculate average from histogram _sum and _count."""
        total = self.raw_metrics.get(f"{base_name}_sum", 0.0)
        count = self.raw_metrics.get(f"{base_name}_count", 0.0)
        if count > 0:
            return total / count
        return None


def parse_prometheus_text(text: str) -> list[MetricSample]:
    """Parse Prometheus text exposition format into MetricSample list."""
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if match:
            name = match.group(1)
            labels_str = match.group(2) or ""
            value_str = match.group(3)

            # Parse labels
            labels: dict[str, str] = {}
            if labels_str:
                for pair in labels_str.split(","):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        labels[k.strip()] = v.strip().strip('"')

            # Parse value
            try:
                value = float(value_str)
            except ValueError:
                value = 0.0

            samples.append(MetricSample(name=name, labels=labels, value=value))
    return samples


def detect_engine_from_metrics(text: str) -> str:
    """Detect which engine is running from its /metrics output."""
    if "dynamo_" in text:
        return "dynamo"
    if "vllm:" in text:
        return "vllm"
    if "sglang:" in text:
        return "sglang"
    if "atom:" in text:
        return "atom"
    if "lmcache:" in text:
        return "lmcache"
    return "unknown"


def resolve_metrics_url(endpoint: str, metrics_path: str = "/metrics") -> str:
    """Resolve either a base URL or a full metrics URL into the final scrape URL."""
    if not metrics_path.startswith("/"):
        raise ValueError("metrics_path must start with '/'")

    parsed = urlsplit(endpoint)
    existing_path = parsed.path or ""
    if existing_path.rstrip("/").endswith(metrics_path.rstrip("/")):
        return endpoint.rstrip("/")

    resolved_path = metrics_path if existing_path in {"", "/"} else f"{existing_path.rstrip('/')}{metrics_path}"
    return urlunsplit((parsed.scheme, parsed.netloc, resolved_path, parsed.query, parsed.fragment)).rstrip("/")


def resolve_api_base_url(endpoint: str, metrics_path: str = "/metrics") -> str:
    """Resolve a base API URL from either a base URL or a full metrics URL."""
    if not metrics_path.startswith("/"):
        raise ValueError("metrics_path must start with '/'")

    parsed = urlsplit(endpoint)
    existing_path = parsed.path or ""
    metrics_suffix = metrics_path.rstrip("/")
    if existing_path.rstrip("/").endswith(metrics_suffix):
        base_path = existing_path[: -len(metrics_suffix)]
        base_path = base_path.rstrip("/") or "/"
        return urlunsplit((parsed.scheme, parsed.netloc, base_path, "", "")).rstrip("/")

    return urlunsplit((parsed.scheme, parsed.netloc, existing_path or "/", "", "")).rstrip("/")


async def scrape_metrics(
    endpoint: str,
    allow_private: bool = True,
    *,
    metrics_path: str = "/metrics",
    auth: EndpointAuthConfig | None = None,
    timeout_seconds: float = 30.0,
) -> ScrapeResult:
    """Scrape Prometheus metrics from an inference engine endpoint.

    Args:
        endpoint: Base URL of the engine or an explicit /metrics URL.
        allow_private: Whether to allow private/localhost IPs. True for CLI (local operator),
                       should be False for network-exposed MCP tools.

    Returns:
        ScrapeResult with parsed metrics, detected engine, and timing.
    """
    import time

    result = ScrapeResult(endpoint=endpoint, engine="unknown")

    start = time.monotonic()
    try:
        url = validate_endpoint(endpoint, allow_private=allow_private)
        metrics_url = resolve_metrics_url(url, metrics_path=metrics_path)
        result.endpoint = sanitize_log_url(metrics_url)
    except Exception as e:  # noqa: BLE001
        log.warning("endpoint_validation_failed", endpoint=endpoint, error=str(e))
        result.error = str(e)
        return result

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            resp = await client.get(metrics_url, headers=build_auth_headers(auth))
            resp.raise_for_status()
            text = resp.text
    except httpx.HTTPStatusError as e:
        log.warning("scrape_http_error", url=metrics_url, status=e.response.status_code)
        result.error = f"HTTP {e.response.status_code} scraping {sanitize_log_url(metrics_url)}"
        return result
    except httpx.ConnectError:
        log.warning("scrape_connection_refused", url=metrics_url)
        result.error = f"Connection refused: {sanitize_log_url(metrics_url)} — is the engine running?"
        return result
    except Exception as e:  # noqa: BLE001
        log.error("scrape_failed", url=metrics_url, error=str(e))
        result.error = f"Scrape failed: {e}"
        return result
    finally:
        result.scrape_time_ms = (time.monotonic() - start) * 1000

    # Parse
    result.engine = detect_engine_from_metrics(text)
    result.samples = parse_prometheus_text(text)

    # Build flat metric dict (latest value per metric name without labels)
    for sample in result.samples:
        # For histogram buckets, keep _sum and _count, skip _bucket
        if "_bucket{" in f"{sample.name}{{":
            continue
        result.raw_metrics[sample.name] = sample.value

    return result
