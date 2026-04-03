# InferScope Architecture

InferScope is an operator product with two public surfaces:

- the `inferscope` CLI
- the `inferscope serve` MCP server

Those surfaces are intentionally narrower than the rest of the codebase history suggests.

InferScope is not a benchmark platform.
It is a runtime diagnostics and narrow probe product.

## Product boundary

InferScope sits beside two other benchmark realities:

- **InferenceX** is the public frontier benchmark reference outside this repo
- **ISB-1** is the reproducible benchmark standard inside this repo
- **InferScope** is the operator-facing deployment analysis layer

That means InferScope should answer questions like:

- why is this deployment missing frontier performance?
- is KV reuse actually working?
- did disaggregation help or just add handoff tax?
- what changed between probe A and probe B?

It should not answer those questions by building a generic matrix browser, suite planner, or launch-bundle framework.

## Current supported contract

InferScope has one authoritative scope file:

- `src/inferscope/production_target.py`

That file defines the supported product lane:

- model: `Kimi-K2.5`
- production engine: `dynamo`
- comparison engine: `vllm`
- workload pack: `kimi-k2-long-context-coding`
- GPUs: `h100`, `h200`, `b200`, `b300`
- topologies: `single_endpoint`, `prefill_decode_split`
- cache strategy: `lmcache`

Any public CLI or MCP surface that contradicts that contract is wrong.

## Repository layout

```text
src/inferscope/
├── production_target.py      # authoritative product contract
├── hardware/                 # GPU metadata and detection
├── models/                   # model metadata
├── optimization/             # checks and recommendation helpers
├── engines/                  # production-lane engine adapters
├── telemetry/                # Prometheus capture and metric normalization
├── profiling/                # live runtime profiling core
├── benchmarks/               # workload resolution, replay, artifacts, probe resolution
├── tools/                    # operator-facing wrappers around diagnostics/recommendations
├── cli.py                    # CLI composition root
├── cli_profiling.py          # profiling CLI surface
├── cli_benchmarks.py         # narrow probe CLI surface
├── server.py                 # MCP composition root
├── server_profiling.py       # profiling MCP surface
└── server_benchmarks.py      # narrow probe MCP surface
```

## Dependency direction

```text
hardware ─┐
models ───┤
           ├──→ optimization ──→ engines
           │          │
           │          ▼
           ├──→ telemetry ──→ profiling
           │
           └──→ benchmarks
                      │
                      ▼
               cli*.py / server*.py
```

Rules:

- `production_target.py` is the only public scope authority
- `optimization` does not depend on benchmark orchestration
- `telemetry` owns metric capture and normalization
- `profiling` owns live runtime analysis
- `benchmarks` owns workload resolution, replay, artifact persistence, and probe-plan resolution
- CLI and MCP files are leaf composition layers

## Runtime profiling subsystem

`src/inferscope/profiling/` is the strongest on-thesis subsystem in the product.

Current flow:

1. scrape Prometheus metrics from the live deployment
2. normalize engine-specific metrics into a shared runtime shape
3. classify workload and memory/cache pressure heuristically
4. run deployment checks
5. group findings into bottlenecks
6. preview tuning changes
7. optionally enrich runtime identity from `/v1/models`

This is the live evidence path that should eventually feed remediation logic.

## Benchmark subsystem

`src/inferscope/benchmarks/` is now a narrow probe package.

It owns:

- packaged workload resolution
- packaged experiment resolution
- procedural expansion for supported packaged probe workloads
- replay execution against OpenAI-compatible endpoints
- artifact persistence and comparison
- support assessment
- shared probe resolution in `probe_resolution.py`

It no longer owns:

- benchmark matrix discovery
- benchmark strategy planning
- benchmark stack-plan generation
- stack bundle materialization

Those abstractions were removed because they pushed InferScope toward generic benchmark infrastructure instead of operator value.

## Public surfaces

### CLI benchmark surface

The retained benchmark CLI surface is:

- `benchmark-plan`
- `benchmark`
- `benchmark-compare`

### MCP benchmark surface

The retained MCP benchmark surface is:

- `tool_get_production_contract`
- `tool_resolve_benchmark_plan`
- `tool_run_benchmark`
- `tool_compare_benchmarks`
- `tool_get_benchmark_artifact`

### MCP profiling surface

The profiling MCP surface stays because it is product-aligned:

- `tool_profile_runtime`
- related profiling/audit helpers from `server_profiling.py`

## Design rule

InferScope should keep moving toward one product thesis:

> explain why a real deployment is underperforming on KV reuse, offload, and disaggregated serving, then point to the next concrete remediation step.

If a new abstraction does not help that job, it should not exist here.
