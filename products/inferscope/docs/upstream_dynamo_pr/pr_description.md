## Summary

Adds a new file `docs/observability/sample-metrics-response.md` that
shows a realistic `/metrics` response from both the Dynamo HTTP
frontend (port 8000) and the KVBM endpoint (port 6880). Also adds a
one-line cross-link from the existing `docs/observability/metrics.md`
reference so readers can find the example.

The example is built directly against the current metric constants
in `lib/runtime/src/metrics/prometheus_names.rs` and the registration
sites in `lib/llm/src/http/service/metrics.rs` and
`lib/llm/src/kv_router/metrics.rs`, so every name and label schema
is source-accurate.

## Motivation

While building an external observability tool that consumes Dynamo's
`/metrics`, I found that the metric names were hard to pin down from
the docs alone — several names I'd assumed based on natural naming
conventions turned out to not exist, and I eventually had to
reverse-engineer the real schema by reading `prometheus_names.rs`,
`metrics.rs` at the registration sites, and the Grafana dashboard
JSONs in `deploy/observability/grafana_dashboards/`.

A single annotated example block would have saved a lot of
archaeology. It would also serve as a test fixture for downstream
tools that parse Dynamo metrics, and as a regression reference if
anyone in the future renames a metric — the file makes it obvious at
a glance when the canonical schema drifts.

Things the example calls out in its prose header:

- Label namespaces differ by surface (frontend uses `model`, backend
  uses the `dynamo_namespace`/`dynamo_component`/`dynamo_endpoint`
  hierarchy, per-worker gauges add `worker_id`/`dp_rank`/`worker_type`,
  router-overhead histograms use `router_id`).
- Counter values are monotonic lifetime totals — downstream checks
  should rate-normalize, not compare raw counters against fixed
  thresholds.
- KV cache metrics use `dynamo_component_*` directly, not
  `dynamo_component_kvstats_*` (there's no `kvstats` segment in real
  metric names — easy thing to get wrong if you only read the metric
  list without a concrete example).
- KVBM metrics are on a separate endpoint (6880, behind
  `DYN_KVBM_METRICS=true`) and use a bare `kvbm_*` prefix.
- The object-storage tier KVBM metrics are only emitted when an
  S3/GCS tier is configured, so they're omitted from the example with
  a note.

## Changes

- Add `docs/observability/sample-metrics-response.md` (new file).
- Add one cross-link in `docs/observability/metrics.md` pointing at
  the new file.

## Test plan

This is a docs-only change. I verified the example is well-formed
Prometheus exposition by parsing it with the standard
`prometheus_client` Python parser and confirming every metric line
parses into a valid `Metric` object. The names and label schemas
were cross-checked against:

- `lib/runtime/src/metrics/prometheus_names.rs`
- `lib/llm/src/http/service/metrics.rs`
- `lib/llm/src/kv_router/metrics.rs`
- `deploy/observability/grafana_dashboards/dynamo.json`
- `deploy/observability/grafana_dashboards/disagg-dashboard.json`
- `deploy/observability/grafana_dashboards/kvbm.json`
- `deploy/observability/prometheus.yml` (for port assignments)

so the example reflects what `main` actually emits today.

## Notes for reviewers

- The example uses Kimi-K2.5 as the model name for concreteness. If
  you'd prefer a more generic or upstream-neutral placeholder I can
  change it — happy to match whatever the project's doc voice
  prefers.
- The counter values are "realistic healthy long-context deployment
  ~10k requests served" — not meant to represent any specific
  benchmark result. If the project has a preferred "example deployment"
  narrative I can adapt.
- The PR description mentions the `kvstats` non-segment as a
  motivating example. If that feels like a negative framing I can
  soften it — my intent is only to illustrate the kind of mistake the
  example prevents, not to criticize the existing docs.
