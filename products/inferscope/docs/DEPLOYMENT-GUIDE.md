# InferScope Deployment Guide

InferScope is meant to sit close to the operator.

Its deployment value is not model hosting by itself. Its value is the ability to profile a real deployment, run a narrow probe, and compare evidence before and after a change.

## Recommended use

### Local or staging use

Use InferScope to:

- profile a local or remote endpoint through Prometheus metrics
- resolve the supported probe plan for the current deployment lane
- run a probe against the endpoint
- compare saved artifacts after a tuning or topology change

### Fleet-facing use

Use InferScope as a read-only diagnostics layer in front of a serving fleet.

Typical flow:

1. profile the live or staging endpoint
2. review bottlenecks and audit findings
3. run the supported probe before a change
4. apply the change
5. run the probe again
6. compare artifacts and decide whether the change was actually useful

## MCP-first workflow

The highest-value MCP flow is now narrower than before.

Retained benchmark/profiling surfaces are:

- `tool_profile_runtime`
- `tool_get_production_contract`
- `tool_resolve_benchmark_plan`
- `tool_run_benchmark`
- `tool_compare_benchmarks`
- `tool_get_benchmark_artifact`

InferScope is no longer positioned as a generic MCP toolkit for hardware catalogs or benchmark strategy planning.

## Current product lane

The public deployment contract is currently narrowed to:

- model: `Kimi-K2.5`
- production engine: `dynamo`
- comparison engine: `vllm`
- workload pack: `kimi-k2-long-context-coding`
- GPUs: `h100`, `h200`, `b200`, `b300`

This is the lane the benchmark and MCP docs assume.

## Runtime storage

Artifacts default to `~/.inferscope/benchmarks/`.
Treat that path as operational evidence, not disposable scratch space.

Runtime profiles are returned directly to the CLI or MCP caller in v1. They are not written to disk by default.

## Profiling boundary

`src/inferscope/profiling/` is the seam for runtime diagnostics today and deeper profiler/kernel work later.

- v1 ships Prometheus-based runtime profiling
- future work can add deeper trace and kernel-facing integrations there
- probe execution should keep consuming shared telemetry models rather than rebuilding profiling logic
