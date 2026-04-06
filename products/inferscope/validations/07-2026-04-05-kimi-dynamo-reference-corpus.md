# Kimi Dynamo production reference corpus

Date: **April 5, 2026**

## Scope

This validation pins the canonical **production-validated** InferScope lane in version control using synthetic reference artifacts.

## Canonical lane

- production target: `dynamo_long_context_coding`
- model: `Kimi-K2.5`
- workload pack: `kimi-k2-long-context-coding`
- experiment: `dynamo-aggregated-lmcache-kimi-k2`
- claim scope: `production_comparable`

## Checked-in corpus

- `docs/examples/benchmark-artifact-baseline.json`
- `docs/examples/benchmark-artifact-candidate.json`
- `docs/examples/benchmark-comparison-example.json`
- `docs/examples/kimi-dynamo-production-reference-summary.json`

## Acceptance

- both artifacts include builtin workload and experiment provenance
- both artifacts declare the production-validated lane explicitly
- the comparison fixture stays comparable inside the same Kimi/Dynamo lane

## Limitations

- these files are synthetic fixtures, not live production exports
- they exist to keep the artifact schema, lane metadata, and comparison semantics stable as the product evolves
