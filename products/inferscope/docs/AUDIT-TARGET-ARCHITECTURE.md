# InferScope Audit and Target Architecture

This document records the architectural cleanup direction behind the current narrowing work.

## Executive verdict

InferScope had drifted toward three bad shapes at once:

- generic benchmark infrastructure
- generic MCP tool wrapping
- broad planning abstractions that did not match the real product thesis

The correct product shape is narrower:

> InferScope should explain why a real deployment misses frontier performance on KV reuse, offload, and disaggregated serving, then point to the next recommended step.

## What is worth keeping

The following areas are aligned enough to keep and build on:

- live runtime profiling in `src/inferscope/profiling/`
- telemetry capture and normalization in `src/inferscope/telemetry/`
- artifact replay and comparison in `src/inferscope/benchmarks/`
- the product-scope contract in `src/inferscope/production_target.py`

## What was strategically wrong

The following benchmark/MCP shapes were wrong for InferScope:

- benchmark matrix catalogs
- benchmark strategy planners
- stack-plan generation and materialization
- duplicate scope authorities
- generic MCP hardware/model/recommendation wrappers as the public product face

These abstractions looked modular, but they were not tied to a real operator workflow.

## Architectural changes implemented in this cleanup

### 1. One scope authority

`src/inferscope/production_target.py` is now the single supported-contract source.

The duplicate target-profile module was removed.

### 2. Shared probe resolution

`src/inferscope/benchmarks/probe_resolution.py` now owns the shared logic for:

- procedural expansion
- experiment defaulting
- support assessment
- run-plan construction
- CLI vs MCP `context_file` policy

Both CLI and MCP benchmark surfaces route through this module.

### 3. Public benchmark surface narrowed

Retained CLI commands:

- `benchmark-plan`
- `benchmark`
- `benchmark-compare`

Retained MCP tools:

- `tool_get_production_contract`
- `tool_resolve_benchmark_plan`
- `tool_run_benchmark`
- `tool_compare_benchmarks`
- `tool_get_benchmark_artifact`

Removed public surfaces:

- workload catalog commands
- experiment catalog commands
- benchmark matrix commands/tools
- benchmark strategy commands/tools
- stack-plan generation/materialization commands/tools

### 4. MCP surface narrowed

The top-level MCP server no longer presents InferScope as a generic GPU/model/recommendation toolkit.

It now centers the product on:

- profiling
- production contract output
- narrow probe execution
- artifact comparison

### 5. Dead benchmark abstractions removed

Removed modules:

- `src/inferscope/benchmarks/launchers.py`
- `src/inferscope/benchmarks/strategy.py`
- `src/inferscope/optimization/target_profile.py`

Removed test files that only defended those abstractions:

- `tests/test_benchmark_catalog.py`
- `tests/test_benchmark_launchers_nvidia.py`
- `tests/test_benchmark_strategy.py`

## Current supported product lane

InferScope is intentionally narrowed to:

- model: `Kimi-K2.5`
- production engine: `dynamo`
- comparison engine: `vllm`
- workload pack: `kimi-k2-long-context-coding`
- GPUs: `h100`, `h200`, `b200`, `b300`
- topologies: `single_endpoint`, `prefill_decode_split`
- cache strategy: `lmcache`

Supported probe experiments:

- `dynamo-aggregated-lmcache-kimi-k2`
- `vllm-disagg-prefill-lmcache`
- `dynamo-disagg-lmcache-kimi-k2`

## What still needs to be built

Narrowing the surface was necessary, but it is not the end state.

High-value missing pieces remain:

- phase-aware telemetry for prefill, decode, and handoff
- stronger KV/offload/disaggregation metrics in artifacts
- reproducibility and provenance manifests with stronger config/trace/version data
- diagnostics that map runtime + probe evidence into explicit recommended next steps
- reporting that explains frontier gap, not just latency deltas

## Design rules going forward

1. Do not add generic abstractions unless they directly serve the operator workflow.
2. Do not add MCP tools unless they are tied to a real deployment question.
3. Do not expand benchmark breadth for its own sake.
4. Prefer a small number of strong probes over a broad catalog of weak ones.
5. Keep ISB-1 as the benchmark-standard owner and InferScope as the operator-diagnostics owner.
6. Keep automated optimization, rollout, and rollback outside InferScope.

## Target architecture

```text
src/inferscope/
├── production_target.py      # product contract and support boundary
├── telemetry/                # metric capture and normalization
├── profiling/                # live runtime diagnostics
├── benchmarks/               # narrow probe resolution, replay, artifacts, comparison
├── diagnostics/              # future gap-analysis and remediation logic
├── cli.py                    # CLI composition root
├── cli_profiling.py          # profiling CLI surface
├── cli_benchmarks.py         # probe CLI surface
├── server.py                 # MCP composition root
├── server_profiling.py       # profiling MCP surface
└── server_benchmarks.py      # probe MCP surface
```

## Bottom line

InferScope should keep moving away from benchmark theater and toward operator truth.

If a proposed feature does not help explain real KV, offload, queueing, or disaggregation behavior in production, it probably belongs in ISB-1 or nowhere.
