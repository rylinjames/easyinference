# Sample `/metrics` Response

This page shows a representative `/metrics` response from a healthy
Dynamo HTTP frontend so downstream observability tooling (custom
Grafana dashboards, Prometheus exporter shims, monitoring adapters,
etc.) can see exactly what the exposition looks like before they hit
a real deployment.

The example below was constructed against the authoritative metric
constants in `lib/runtime/src/metrics/prometheus_names.rs` and the
registration sites in `lib/llm/src/http/service/metrics.rs` and
`lib/llm/src/kv_router/metrics.rs`. It uses the same label schema
Dynamo actually emits (`model` on the frontend, `dynamo_namespace` /
`dynamo_component` / `dynamo_endpoint` on the backend, `worker_id` /
`dp_rank` / `worker_type` on per-worker gauges, `router_id` on
router-overhead histograms, `migration_type` on the migration
counter).

The deployment this example represents: a healthy Kimi-K2.5 long
context coding workload at ~50% GPU KV cache utilization, ~10,000
requests served, 2 decode workers, warm prefix caching, fast router
stages.

## What to notice

- **Label namespaces differ by surface.** The HTTP frontend uses
  `model`. The backend components use the standard runtime hierarchy
  labels (`dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`).
  Per-worker gauges add `worker_id`, `dp_rank`, `worker_type`.
  Router-overhead histograms use `router_id`.
- **Histograms come with `_sum` and `_count`** following the
  Prometheus convention. `_bucket{le="..."}` lines are also emitted
  but omitted below for brevity — downstream code should parse them
  as usual.
- **Counter values are monotonic lifetime totals.** Rate-based
  diagnostic thresholds should always normalize by
  `dynamo_frontend_requests_total` or a similar denominator, never
  compare raw counter values against a static threshold (that's a
  classic pitfall that bites both custom dashboards and downstream
  checks).
- **KV cache metrics use `dynamo_component_` directly**, not
  `dynamo_component_kvstats_`. There is no `kvstats` segment in real
  metric names; the bare constants in `prometheus_names.rs` are
  `active_blocks`, `total_blocks`, `gpu_cache_usage_percent`, and
  `gpu_prefix_cache_hit_rate`, prefixed with `dynamo_component_` at
  registration time.
- **KVBM metrics are emitted on a separate endpoint** (default port
  6880 via `DYN_KVBM_METRICS_PORT`, only when launched with
  `DYN_KVBM_METRICS=true`). They use a bare `kvbm_*` prefix, not
  `dynamo_*`. See the KVBM portion at the bottom of this file for
  the separate-endpoint exposition.

## Example frontend `/metrics` (port 8000)

```
# HELP dynamo_frontend_inflight_requests Inflight requests at the HTTP frontend
# TYPE dynamo_frontend_inflight_requests gauge
dynamo_frontend_inflight_requests{model="Kimi-K2.5"} 12

# HELP dynamo_frontend_queued_requests Queued requests at the HTTP frontend
# TYPE dynamo_frontend_queued_requests gauge
dynamo_frontend_queued_requests{model="Kimi-K2.5"} 0

# HELP dynamo_frontend_disconnected_clients Disconnected streaming clients
# TYPE dynamo_frontend_disconnected_clients gauge
dynamo_frontend_disconnected_clients{model="Kimi-K2.5"} 0

# HELP dynamo_frontend_requests_total Total requests observed by the frontend
# TYPE dynamo_frontend_requests_total counter
dynamo_frontend_requests_total{model="Kimi-K2.5"} 10000

# HELP dynamo_frontend_output_tokens_total Total output tokens generated
# TYPE dynamo_frontend_output_tokens_total counter
dynamo_frontend_output_tokens_total{model="Kimi-K2.5"} 4500000

# HELP dynamo_frontend_model_migration_total Total request migrations
# TYPE dynamo_frontend_model_migration_total counter
dynamo_frontend_model_migration_total{model="Kimi-K2.5",migration_type="worker_drain"} 3

# HELP dynamo_frontend_model_cancellation_total Total client cancellations
# TYPE dynamo_frontend_model_cancellation_total counter
dynamo_frontend_model_cancellation_total{model="Kimi-K2.5"} 15

# HELP dynamo_frontend_time_to_first_token_seconds Frontend TTFT histogram
# TYPE dynamo_frontend_time_to_first_token_seconds histogram
dynamo_frontend_time_to_first_token_seconds_sum{model="Kimi-K2.5"} 3200
dynamo_frontend_time_to_first_token_seconds_count{model="Kimi-K2.5"} 10000

# HELP dynamo_frontend_inter_token_latency_seconds Frontend ITL histogram
# TYPE dynamo_frontend_inter_token_latency_seconds histogram
dynamo_frontend_inter_token_latency_seconds_sum{model="Kimi-K2.5"} 200
dynamo_frontend_inter_token_latency_seconds_count{model="Kimi-K2.5"} 10000

# HELP dynamo_frontend_request_duration_seconds Frontend request duration histogram
# TYPE dynamo_frontend_request_duration_seconds histogram
dynamo_frontend_request_duration_seconds_sum{model="Kimi-K2.5"} 12000
dynamo_frontend_request_duration_seconds_count{model="Kimi-K2.5"} 10000

# HELP dynamo_frontend_cached_tokens Tokens served from cache per request
# TYPE dynamo_frontend_cached_tokens histogram
dynamo_frontend_cached_tokens_sum{model="Kimi-K2.5"} 30000000
dynamo_frontend_cached_tokens_count{model="Kimi-K2.5"} 10000

# HELP dynamo_frontend_input_sequence_tokens Input sequence tokens per request
# TYPE dynamo_frontend_input_sequence_tokens histogram
dynamo_frontend_input_sequence_tokens_sum{model="Kimi-K2.5"} 20000000
dynamo_frontend_input_sequence_tokens_count{model="Kimi-K2.5"} 10000

# HELP dynamo_frontend_output_sequence_tokens Output sequence tokens per request
# TYPE dynamo_frontend_output_sequence_tokens histogram
dynamo_frontend_output_sequence_tokens_sum{model="Kimi-K2.5"} 4500000
dynamo_frontend_output_sequence_tokens_count{model="Kimi-K2.5"} 10000

# HELP dynamo_frontend_tokenizer_latency_ms Per-request tokenizer latency (ms)
# TYPE dynamo_frontend_tokenizer_latency_ms histogram
dynamo_frontend_tokenizer_latency_ms_sum{model="Kimi-K2.5"} 50000
dynamo_frontend_tokenizer_latency_ms_count{model="Kimi-K2.5"} 10000

# HELP dynamo_frontend_router_queue_pending_requests Router-side pending queue
# TYPE dynamo_frontend_router_queue_pending_requests gauge
dynamo_frontend_router_queue_pending_requests{worker_type="decode"} 0

# HELP dynamo_frontend_model_total_kv_blocks Total KV blocks configured
# TYPE dynamo_frontend_model_total_kv_blocks gauge
dynamo_frontend_model_total_kv_blocks{model="Kimi-K2.5"} 16384

# HELP dynamo_frontend_model_max_num_seqs Declared max concurrent sequences
# TYPE dynamo_frontend_model_max_num_seqs gauge
dynamo_frontend_model_max_num_seqs{model="Kimi-K2.5"} 64

# HELP dynamo_frontend_model_max_num_batched_tokens Declared batched token budget
# TYPE dynamo_frontend_model_max_num_batched_tokens gauge
dynamo_frontend_model_max_num_batched_tokens{model="Kimi-K2.5"} 16384

# HELP dynamo_frontend_model_context_length Declared max context length
# TYPE dynamo_frontend_model_context_length gauge
dynamo_frontend_model_context_length{model="Kimi-K2.5"} 131072

# HELP dynamo_frontend_model_kv_cache_block_size KV block size in tokens
# TYPE dynamo_frontend_model_kv_cache_block_size gauge
dynamo_frontend_model_kv_cache_block_size{model="Kimi-K2.5"} 16

# HELP dynamo_frontend_model_migration_limit Per-request migration limit
# TYPE dynamo_frontend_model_migration_limit gauge
dynamo_frontend_model_migration_limit{model="Kimi-K2.5"} 2

# --- Per-worker last-value gauges ---

# HELP dynamo_frontend_worker_active_decode_blocks Active decode KV blocks per worker
# TYPE dynamo_frontend_worker_active_decode_blocks gauge
dynamo_frontend_worker_active_decode_blocks{worker_id="w0",dp_rank="0",worker_type="decode"} 180
dynamo_frontend_worker_active_decode_blocks{worker_id="w1",dp_rank="0",worker_type="decode"} 210

# HELP dynamo_frontend_worker_active_prefill_tokens Active prefill tokens per worker
# TYPE dynamo_frontend_worker_active_prefill_tokens gauge
dynamo_frontend_worker_active_prefill_tokens{worker_id="w0",dp_rank="0",worker_type="prefill"} 4096
dynamo_frontend_worker_active_prefill_tokens{worker_id="w1",dp_rank="0",worker_type="prefill"} 3072

# HELP dynamo_frontend_worker_last_time_to_first_token_seconds Last TTFT per worker
# TYPE dynamo_frontend_worker_last_time_to_first_token_seconds gauge
dynamo_frontend_worker_last_time_to_first_token_seconds{worker_id="w0",dp_rank="0",worker_type="decode"} 0.28
dynamo_frontend_worker_last_time_to_first_token_seconds{worker_id="w1",dp_rank="0",worker_type="decode"} 0.31

# HELP dynamo_frontend_worker_last_input_sequence_tokens Last input tokens per worker
# TYPE dynamo_frontend_worker_last_input_sequence_tokens gauge
dynamo_frontend_worker_last_input_sequence_tokens{worker_id="w0",dp_rank="0",worker_type="decode"} 2040
dynamo_frontend_worker_last_input_sequence_tokens{worker_id="w1",dp_rank="0",worker_type="decode"} 1960

# HELP dynamo_frontend_worker_last_inter_token_latency_seconds Last ITL per worker
# TYPE dynamo_frontend_worker_last_inter_token_latency_seconds gauge
dynamo_frontend_worker_last_inter_token_latency_seconds{worker_id="w0",dp_rank="0",worker_type="decode"} 0.019
dynamo_frontend_worker_last_inter_token_latency_seconds{worker_id="w1",dp_rank="0",worker_type="decode"} 0.021

# --- Backend component metrics ---

# HELP dynamo_component_inflight_requests Inflight requests at backend
# TYPE dynamo_component_inflight_requests gauge
dynamo_component_inflight_requests{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 12

# HELP dynamo_component_requests_total Total backend requests
# TYPE dynamo_component_requests_total counter
dynamo_component_requests_total{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 10000

# HELP dynamo_component_request_duration_seconds Backend request duration histogram
# TYPE dynamo_component_request_duration_seconds histogram
dynamo_component_request_duration_seconds_sum{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 11500
dynamo_component_request_duration_seconds_count{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 10000

# HELP dynamo_component_request_bytes_total Total request payload bytes
# TYPE dynamo_component_request_bytes_total counter
dynamo_component_request_bytes_total{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 41943040

# HELP dynamo_component_response_bytes_total Total response payload bytes
# TYPE dynamo_component_response_bytes_total counter
dynamo_component_response_bytes_total{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 104857600

# HELP dynamo_component_uptime_seconds Component uptime
# TYPE dynamo_component_uptime_seconds gauge
dynamo_component_uptime_seconds{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 3600

# HELP dynamo_component_errors_total Total backend errors
# TYPE dynamo_component_errors_total counter
dynamo_component_errors_total{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 5

# HELP dynamo_component_active_blocks Active KV blocks
# TYPE dynamo_component_active_blocks gauge
dynamo_component_active_blocks{dynamo_component="worker"} 8100

# HELP dynamo_component_total_blocks Total KV blocks
# TYPE dynamo_component_total_blocks gauge
dynamo_component_total_blocks{dynamo_component="worker"} 16384

# HELP dynamo_component_gpu_cache_usage_percent GPU KV cache usage
# TYPE dynamo_component_gpu_cache_usage_percent gauge
dynamo_component_gpu_cache_usage_percent{dynamo_component="worker"} 0.50

# HELP dynamo_component_gpu_prefix_cache_hit_rate GPU prefix cache hit rate
# TYPE dynamo_component_gpu_prefix_cache_hit_rate gauge
dynamo_component_gpu_prefix_cache_hit_rate{dynamo_component="worker"} 0.72

# HELP dynamo_component_kv_cache_events_applied KV cache event counter
# TYPE dynamo_component_kv_cache_events_applied counter
dynamo_component_kv_cache_events_applied{status="applied",event_type="store"} 45000

# HELP dynamo_component_kv_publisher_engines_dropped_events_total Dropped KV publisher events
# TYPE dynamo_component_kv_publisher_engines_dropped_events_total counter
dynamo_component_kv_publisher_engines_dropped_events_total{dynamo_namespace="default",dynamo_component="worker"} 0

# --- Router-overhead histograms (ms, not seconds) ---

# HELP dynamo_router_overhead_total_ms Total routing overhead per request
# TYPE dynamo_router_overhead_total_ms histogram
dynamo_router_overhead_total_ms_sum{router_id="r0"} 35000
dynamo_router_overhead_total_ms_count{router_id="r0"} 10000

# HELP dynamo_router_overhead_block_hashing_ms Block hashing overhead
# TYPE dynamo_router_overhead_block_hashing_ms histogram
dynamo_router_overhead_block_hashing_ms_sum{router_id="r0"} 8000
dynamo_router_overhead_block_hashing_ms_count{router_id="r0"} 10000

# HELP dynamo_router_overhead_indexer_find_matches_ms KV indexer match overhead
# TYPE dynamo_router_overhead_indexer_find_matches_ms histogram
dynamo_router_overhead_indexer_find_matches_ms_sum{router_id="r0"} 12000
dynamo_router_overhead_indexer_find_matches_ms_count{router_id="r0"} 10000

# HELP dynamo_router_overhead_scheduling_ms Router scheduling overhead
# TYPE dynamo_router_overhead_scheduling_ms histogram
dynamo_router_overhead_scheduling_ms_sum{router_id="r0"} 10000
dynamo_router_overhead_scheduling_ms_count{router_id="r0"} 10000

# HELP dynamo_router_overhead_seq_hashing_ms Sequence hashing overhead
# TYPE dynamo_router_overhead_seq_hashing_ms histogram
dynamo_router_overhead_seq_hashing_ms_sum{router_id="r0"} 5000
dynamo_router_overhead_seq_hashing_ms_count{router_id="r0"} 10000

# --- Component-side router histograms ---

# HELP dynamo_component_router_requests_total Component-side router request counter
# TYPE dynamo_component_router_requests_total counter
dynamo_component_router_requests_total{dynamo_component="router"} 10000

# HELP dynamo_component_router_kv_hit_rate Router KV hit rate
# TYPE dynamo_component_router_kv_hit_rate histogram
dynamo_component_router_kv_hit_rate_sum{dynamo_component="router"} 7100
dynamo_component_router_kv_hit_rate_count{dynamo_component="router"} 10000

# HELP dynamo_component_router_time_to_first_token_seconds Component-side TTFT
# TYPE dynamo_component_router_time_to_first_token_seconds histogram
dynamo_component_router_time_to_first_token_seconds_sum{dynamo_component="router"} 3150
dynamo_component_router_time_to_first_token_seconds_count{dynamo_component="router"} 10000

# HELP dynamo_component_router_inter_token_latency_seconds Component-side ITL
# TYPE dynamo_component_router_inter_token_latency_seconds histogram
dynamo_component_router_inter_token_latency_seconds_sum{dynamo_component="router"} 195
dynamo_component_router_inter_token_latency_seconds_count{dynamo_component="router"} 10000

# HELP dynamo_component_router_input_sequence_tokens Component-side input tokens histogram
# TYPE dynamo_component_router_input_sequence_tokens histogram
dynamo_component_router_input_sequence_tokens_sum{dynamo_component="router"} 20000000
dynamo_component_router_input_sequence_tokens_count{dynamo_component="router"} 10000

# HELP dynamo_component_router_output_sequence_tokens Component-side output tokens histogram
# TYPE dynamo_component_router_output_sequence_tokens histogram
dynamo_component_router_output_sequence_tokens_sum{dynamo_component="router"} 4500000
dynamo_component_router_output_sequence_tokens_count{dynamo_component="router"} 10000
```

## Example KVBM `/metrics` (separate endpoint, port 6880)

KVBM metrics are only emitted when Dynamo is launched with
`DYN_KVBM_METRICS=true` and are exposed on the port set by
`DYN_KVBM_METRICS_PORT` (default 6880). The names use a bare `kvbm_`
prefix, not `dynamo_`.

```
# HELP kvbm_offload_blocks_d2h GPU-to-host KV block offload operations
# TYPE kvbm_offload_blocks_d2h counter
kvbm_offload_blocks_d2h 12500

# HELP kvbm_offload_blocks_h2d Host-to-disk KV block offload operations
# TYPE kvbm_offload_blocks_h2d counter
kvbm_offload_blocks_h2d 3200

# HELP kvbm_offload_blocks_d2d Device-to-device KV block offload operations
# TYPE kvbm_offload_blocks_d2d counter
kvbm_offload_blocks_d2d 850

# HELP kvbm_onboard_blocks_h2d Host-to-device KV block onboard operations
# TYPE kvbm_onboard_blocks_h2d counter
kvbm_onboard_blocks_h2d 9400

# HELP kvbm_onboard_blocks_d2d Device-to-device KV block onboard operations
# TYPE kvbm_onboard_blocks_d2d counter
kvbm_onboard_blocks_d2d 820

# HELP kvbm_matched_tokens Tokens reused from KVBM cache
# TYPE kvbm_matched_tokens counter
kvbm_matched_tokens 88000000

# HELP kvbm_host_cache_hit_rate CPU tier KV cache hit rate
# TYPE kvbm_host_cache_hit_rate gauge
kvbm_host_cache_hit_rate 0.75

# HELP kvbm_disk_cache_hit_rate Disk tier KV cache hit rate
# TYPE kvbm_disk_cache_hit_rate gauge
kvbm_disk_cache_hit_rate 0.18

# HELP kvbm_object_cache_hit_rate Object storage tier KV cache hit rate
# TYPE kvbm_object_cache_hit_rate gauge
kvbm_object_cache_hit_rate 0.0
```

The object-storage tier metrics (`kvbm_offload_blocks_d2o`,
`kvbm_onboard_blocks_o2d`, `kvbm_offload_bytes_object`,
`kvbm_onboard_bytes_object`, `kvbm_object_read_failures`,
`kvbm_object_write_failures`) are only emitted when the operator
configures an S3/GCS tier, so they are omitted from this example.
