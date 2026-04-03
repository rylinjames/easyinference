# ISB-1 Technical Architecture

This document describes the current product architecture for ISB-1 as implemented in `products/isb1/`.

## Design intent

ISB-1 is the benchmark standard inside EasyInference.

- It is **not** a public dashboard product.
- It is **not** intended to replace InferenceX as the external market-wide reference.
- It **is** the reproducible benchmark layer that EasyInference uses for methodology, publication, and operator review.

The current harness implementation launches vLLM and measures it through an internal OpenAI-compatible replay path.

## Module layout

```text
products/isb1/
├── workloads/
│   ├── base.py
│   ├── chat.py
│   ├── agent.py
│   ├── rag.py
│   ├── coding.py
│   ├── arrivals.py
│   └── materialize.py
├── harness/
│   ├── server.py
│   ├── replay_client.py
│   ├── client.py
│   ├── runner.py
│   ├── sweep.py
│   ├── warmup.py
│   ├── telemetry.py
│   ├── engine_metrics.py
│   ├── manifest.py
│   ├── lockfile.py
│   └── config_validator.py
├── analysis/
├── quality/
├── configs/
├── publication/
├── scripts/
└── tests/
```

## Dependency direction

```text
configs ──┬──> workloads ───────┐
          ├──> harness ─────────┼──> analysis ──> publication
          └──> quality ─────────┘
```

Key rules:

- `workloads/` owns canonical request generation.
- `harness/` owns execution, trace persistence, replay, manifests, and lockfiles.
- `analysis/` consumes raw results; it does not launch workloads.
- `publication/` consumes aggregated outputs; it does not compute primary metrics.

## Execution lifecycle

A single benchmark cell runs through the following lifecycle.

### 1. Configuration resolution

`harness.runner.BenchmarkRunner` loads GPU, model, and workload config, checks memory fit, resolves topology defaults, and constructs a `CellConfig`.

### 2. Trace materialization

`workloads.materialize.materialize_requests()` builds a deterministic request pool from the workload config.

Important properties:

- trace size comes from `trace.num_requests` unless overridden
- request content is deterministic under the configured seed
- the request pool is saved as `trace.jsonl` in the run directory
- the trace SHA-256 is recorded in the manifest and lockfile

### 3. Server startup

`harness.server.VLLMServer` launches the serving stack for the cell and waits for a healthy endpoint.

### 4. Warmup

`harness.warmup.WarmupValidator` validates that the serving stack reaches a stable state before the benchmark sweep proceeds.

### 5. Replay execution

`harness.client.BenchmarkClient` wraps the internal replay runner in `harness.replay_client`.

The replay runner:

- sends OpenAI-compatible `/v1/chat/completions` requests
- supports Poisson and Gamma arrival models
- expands the saved request pool to cover the target measurement window
- records TTFT, token timestamps, E2E latency, errors, and token counts
- resolves request-specific SLO thresholds where needed

This is the main architectural change from the older design: ISB-1 no longer shells out to `vllm.benchmarks.benchmark_serving`.

### 6. Telemetry collection

During replay, ISB-1 can collect:

- GPU telemetry
- engine metrics from the Prometheus endpoint
- manifest metadata for the run

### 7. Aggregation and publication

`analysis.metrics.MetricComputer` converts raw replay output into benchmark metrics such as TTFT, TPOT, ITL, throughput, goodput, and SLO attainment. Publication templates then consume those aggregated outputs.

## Canonical workload families

ISB-1 intentionally keeps a stable public taxonomy:

- `chat`
- `agent`
- `rag`
- `coding`

These families are broad enough to absorb more specific scenarios without fragmenting the benchmark.

Examples:

- MCP and tool-calling scenarios belong to the **agent** family.
- Long-context repository review belongs to the **coding** family.

## Relationship to InferScope

InferScope is the operator-facing product that packages benchmark assets for CLI and MCP use.

The bridge works like this:

1. ISB-1 defines the neutral benchmark families and replay methodology.
2. InferScope packages practical built-ins such as `tool-agent` and `coding-long-context`.
3. Those built-ins map back to the canonical ISB-1 families.

This keeps the benchmark standard stable while letting the MCP surface evolve faster.
