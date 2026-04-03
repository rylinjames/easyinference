# InferScope Benchmark Plan

InferScope's benchmark plan is no longer "build benchmark infrastructure and hope it becomes useful."

It is this:

> keep a very small probe surface that helps operators explain KV-cache, offload, and disaggregated-serving behavior in real deployments.

## Product rule

InferScope should not become:

- a public benchmark clone
- a benchmark matrix browser
- a benchmark suite planner
- a stack-bundle generator
- a generic benchmark wrapper over MCP

Those jobs either belong to **InferenceX** or to **ISB-1**.

## What InferScope should do

InferScope should be the fastest path from a deployment question to concrete evidence:

1. profile a live endpoint
2. resolve a supported probe plan
3. run the probe against the endpoint
4. save an artifact
5. compare artifacts before and after a change
6. connect runtime evidence to the next remediation step

## Current implementation stance

The supported benchmark lane is deliberately narrow:

- model: `Kimi-K2.5`
- production engine: `dynamo`
- comparison engine: `vllm`
- workload pack: `kimi-k2-long-context-coding`
- experiments:
  - `dynamo-aggregated-lmcache-kimi-k2`
  - `vllm-disagg-prefill-lmcache`
  - `dynamo-disagg-lmcache-kimi-k2`

The default probe path is the aggregated Dynamo lane.

## What was removed

These surfaces were intentionally cut:

- benchmark workload catalog commands
- benchmark experiment catalog commands
- benchmark matrix surfaces
- benchmark strategy planning surfaces
- benchmark stack-plan generation
- benchmark stack materialization

They created generic benchmark sprawl without improving operator truth.

## What remains

Retained public surfaces:

- `benchmark-plan`
- `benchmark`
- `benchmark-compare`
- `tool_get_production_contract`
- `tool_resolve_benchmark_plan`
- `tool_run_benchmark`
- `tool_compare_benchmarks`
- `tool_get_benchmark_artifact`

## What to build next

The next useful benchmark work is not more framework.
It is more evidence.

Priority order:

1. richer KV/offload/disaggregation metrics in artifacts
2. stronger provenance and reproducibility data in artifact manifests
3. phase-aware telemetry for prefill, handoff, and decode
4. gap-analysis logic that explains why production misses frontier behavior
5. remediation logic that turns those gaps into concrete actions

## Design test

Any new benchmark feature should have to answer one question:

> does this help explain a real operator bottleneck, or is it just more benchmark-looking infrastructure?

If it is the second one, it should not be added here.
