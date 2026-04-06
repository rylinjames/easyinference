# Draft upstream contribution for ai-dynamo/dynamo

This directory holds a draft contribution that should be filed upstream
against [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo). It is
**not** part of InferScope's shipped surface; it's staged here so a
maintainer can copy it into a fork and open the PR.

## Why this contribution exists

Over the course of auditing InferScope's Dynamo metric parser against
the real Dynamo schema, the single biggest gap we hit was **there is
no published example of a real `/metrics` response anywhere**. Every
InferScope drift bug we found (Grove, `dynamo_lmcache_*`, `dynamo_slo_*`,
`dynamo_component_kvstats_*`, `dynamo_nixl_*`) could have been caught
immediately if `docs/observability/metrics.md` had shown a
representative exposition example with actual values. Instead the
names had to be reverse-engineered from

  - `lib/runtime/src/metrics/prometheus_names.rs` (bare constants)
  - `lib/llm/src/http/service/metrics.rs` and `lib/llm/src/kv_router/metrics.rs`
    (emission sites)
  - `deploy/observability/grafana_dashboards/*.json` (runtime PromQL queries)

…and cross-checked against each other. That's a lot of work for any
downstream tool builder.

Adding a single annotated example block to the observability docs
would save the next person in the same position (observability tool,
Grafana dashboard author, Prometheus exporter shim, etc.) several
hours of archaeology.

## Files in this draft

1. **`sample_metrics_response.md`** — the file to add upstream, at the
   path `docs/observability/sample-metrics-response.md`. It contains a
   realistic synthetic `/metrics` scrape from a healthy Dynamo frontend
   deployment with labels and values, plus a short prose header
   explaining what a reader should look at.

2. **`suggested_metrics_md_crosslink.md`** — a suggested cross-link
   to add to `docs/observability/metrics.md` so operators looking at
   the metric reference find the example.

3. **`pr_description.md`** — a ready-to-paste PR description.

## How to file it

1. Fork [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) under
   your own GitHub identity.
2. Clone the fork locally.
3. Create a branch: `git checkout -b docs/sample-metrics-response`.
4. Copy the contents of `sample_metrics_response.md` to
   `docs/observability/sample-metrics-response.md` in the fork.
5. Apply the one-line cross-link from `suggested_metrics_md_crosslink.md`
   to `docs/observability/metrics.md`.
6. Commit and push to your fork.
7. Open a PR against `ai-dynamo/dynamo:main`, pasting the content of
   `pr_description.md` as the PR body.

The draft files are kept simple enough that you shouldn't need to
modify anything — they're source-verified against the same emission
sites InferScope's own `tests/fixtures/dynamo_metrics_healthy.txt`
uses, so they match the current Dynamo schema on `main`.

## If the PR is rejected or changes requested

If a maintainer asks for changes (e.g. wants specific labels added,
wants a different model name, wants more or fewer metrics), update
the draft files here first so InferScope's downstream fixture stays in
sync, then push the update to the fork and rebase the PR. Don't let
the upstream PR and the InferScope fixture drift out of alignment —
the whole point of the contribution is that both sides agree on
ground truth.
