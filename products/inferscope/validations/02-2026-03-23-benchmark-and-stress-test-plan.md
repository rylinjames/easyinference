# InferScope Benchmark and Stress-Test Build Plan

**Plan version:** 02  
**Date:** March 23, 2026  
**Target package line:** 0.1.x

## Objective

Build a benchmark and stress-test system that proves or falsifies InferScope's claims under real workloads, real inference engines, real Prometheus metrics, and production-style logs.

## Current architecture assumption

The benchmark system now lives inside the packaged evaluation subsystem under `src/inferscope/benchmarks/`. That subsystem is the current source of truth for:

- workload packs
- experiment specs
- replay orchestration
- artifact capture
- benchmark stack planning

This plan should no longer assume a second authoritative repo-root `benchmarks/` asset tree.

## Baseline quality gates

Required checks:

```bash
uv run ruff check src/
uv run bandit -r src/inferscope/ -c pyproject.toml -ll
uv run mypy src/inferscope/
```

Optional checks when the suite exists:

```bash
uv run pytest tests/ -q
```

## Product requirement

InferScope should make inference engineering simple for an engineer who does not specialize in LLM systems. The product should provide:

- recommended engine and topology
- exact engine config and launch surface
- KV/offload/disaggregation guidance
- live audit findings
- benchmark evidence for the recommendation

## Benchmark workstreams

1. **Workload replay** for coding, RAG, legal review, mixed traffic, and tool-agent flows
2. **Metrics capture** for TTFT, latency, throughput, KV pressure, queue depth, and cache effectiveness
3. **Artifact comparison** so rollout changes can be reviewed after remote runs
4. **Stack planning** for colocated, cache-aware, and disaggregated experiments
5. **Engine maturity tracking** so TRT-LLM and Dynamo claims stay conservative until live validation expands

## Delivery direction

- keep packaged benchmark assets authoritative
- keep runtime artifacts outside the repo tree
- keep benchmark code depending inward on the optimizer
- keep the public CLI and MCP benchmark surfaces stable

## Future extraction note

If benchmarking becomes its own repo later, extraction should happen from the current packaged subsystem rather than by reintroducing a second benchmark asset tree.
