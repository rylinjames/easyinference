# ISB-1 Benchmark Methodology

This document defines the measurement contract for ISB-1: Inference Serving Benchmark Standard 1.

## Positioning

ISB-1 is the benchmark standard inside EasyInference.

- **InferenceX** is the external, continuously updated public reference.
- **ISB-1** is the controlled methodology for reproducible replay, publication, and internal review.
- **InferScope** is the operator product that packages benchmark assets for CLI and MCP use.

The methodology is intentionally complementary to InferenceX. ISB-1 emphasizes reproducibility, trace persistence, and workload families that are useful for operator tuning and workflow-specific validation.

## Design principles

1. **Production relevance** — workloads should resemble real serving behavior, not uniform random text.
2. **Stable taxonomy** — benchmark families stay broad and durable.
3. **Reproducibility** — every run persists the request trace, manifest, and lockfile.
4. **Reviewability** — a third party should be able to inspect what was actually replayed.
5. **Separability** — benchmark methodology stays distinct from the operator MCP product.

## Canonical workload families

### `chat`
Short-to-medium context conversational serving with throughput-oriented concurrency.

### `agent`
Structured tool-calling or MCP-style flows with growing context and latency sensitivity.

### `rag`
Prefill-heavy retrieval workloads with long retrieved context and TTFT sensitivity.

### `coding`
Repository-context assistance with long prefixes, reuse opportunities, and multi-turn code review or editing patterns.

These families are stable. Downstream products may define richer scenario names, but those names should map back to one of these families.

Examples:

- InferScope `tool-agent` → ISB-1 `agent`
- InferScope `coding-long-context` → ISB-1 `coding`

## Request generation

Workload configs under `configs/workloads/*.yaml` define:

- context-length targets
- output-length targets
- arrival model and rate sweep
- SLO thresholds
- trace materialization defaults such as `trace.num_requests`

`workloads.materialize` converts those configs into a deterministic JSONL request pool.

Methodology requirements:

- a fixed seed must produce the same request pool
- the request pool must be persisted with the run
- the trace hash must be recorded in the lockfile

## Replay methodology

The current ISB-1 implementation uses an internal replay client against an OpenAI-compatible endpoint.

For each rate point:

1. warm up the serving stack
2. expand the saved request pool to the target measurement window
3. submit requests according to the configured arrival model
4. capture TTFT, E2E latency, token timestamps, token counts, and errors
5. compute throughput and SLO-derived goodput from the captured results

This replay path replaces the older dependency on `vllm.benchmarks.benchmark_serving`.

## Metric definitions

### TTFT
Time from request submission to first output token.

### TPOT
Average decode latency per emitted token, excluding TTFT.

```text
TPOT = (e2e_latency - ttft) / (output_tokens - 1)
```

### ITL
Gap between consecutive output tokens, starting after the first token.

### E2E latency
Time from request submission to request completion.

### Request throughput
Completed requests divided by wall-clock measurement duration.

### Generation throughput
Output tokens divided by wall-clock measurement duration.

### Goodput
Requests per second that satisfy the workload SLO thresholds.

### SLO attainment
Fraction of successful requests that satisfy the workload SLO thresholds.

## SLO policy

SLOs are defined in workload configs and may be:

- fixed thresholds for the whole workload
- bucketed thresholds keyed by approximate context size

Bucketed SLOs are important for long-context workloads such as RAG, where a 32K request and a 96K request should not be judged by the same TTFT threshold.

## Modes

ISB-1 currently supports three benchmark modes:

- **Mode A** — default serving configuration
- **Mode B** — optimized configuration
- **Mode C** — operator-submitted configuration

Comparisons are always made within the same hardware, model, and workload context.

## Reproducibility requirements

Every benchmark run must persist:

- the trace JSONL used for replay
- a manifest describing the benchmark cell and outputs
- a lockfile containing software, hardware, config hashes, benchmark runner, and trace hash
- raw replay results for each measured rate point

A result should be reviewable without guessing which workload inputs were used.

## Statistical expectations

The benchmark still expects:

- warmup before measurement
- multi-trial evaluation for meaningful comparisons
- variance checks and confidence-aware interpretation

The exact number of trials and thresholds remain configurable in the sweep definitions and analysis logic.
