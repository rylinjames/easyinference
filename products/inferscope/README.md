# InferScope

InferScope is the operator-facing product in EasyInference.

It is not a generic benchmark project, not a generic observability platform, and not a generic MCP wrapper.

Its job is narrower:

- profile live inference deployments
- run a small number of production-aligned probes
- compare probe artifacts before and after a change
- explain KV-cache, offload, and disaggregated-serving behavior in operator terms

## Product role inside EasyInference

EasyInference has two products with different jobs:

- **ISB-1** owns reproducible benchmark methodology, configs, analysis, and publication
- **InferScope** owns operator-facing diagnostics, runtime profiling, and narrow probe execution

That separation is intentional.

- **InferenceX** owns the public frontier benchmark problem outside this repo.
- **ISB-1** owns the local benchmark standard inside this repo.
- **InferScope** owns the deployment-specific gap-analysis loop.

## Current supported production contract

As of **March 28, 2026**, InferScope is intentionally narrowed to one product lane:

- **Model:** `Kimi-K2.5`
- **Production engine:** `dynamo`
- **Comparison engine:** `vllm`
- **Workload pack:** `kimi-k2-long-context-coding`
- **GPUs:** `h100`, `h200`, `b200`, `b300`
- **Topologies:** `single_endpoint`, `prefill_decode_split`
- **Cache strategy:** `lmcache`

Supported packaged probe experiments:

- `dynamo-aggregated-lmcache-kimi-k2`
- `vllm-disagg-prefill-lmcache`
- `dynamo-disagg-lmcache-kimi-k2`

This scope is enforced by the product contract in `src/inferscope/production_target.py`.

## What InferScope does now

### 1. Runtime profiling

Primary surfaces:

- CLI: `inferscope profile-runtime`
- MCP: `tool_profile_runtime`

This is the highest-value live workflow today.

It scrapes Prometheus metrics, normalizes runtime state, runs deployment checks, groups bottlenecks, and returns tuning-relevant evidence.

### 2. Narrow probe execution

Primary surfaces:

- CLI: `inferscope benchmark-plan`
- CLI: `inferscope benchmark`
- CLI: `inferscope benchmark-compare`
- MCP: `tool_get_production_contract`
- MCP: `tool_resolve_benchmark_plan`
- MCP: `tool_run_benchmark`
- MCP: `tool_compare_benchmarks`
- MCP: `tool_get_benchmark_artifact`

The benchmark package is no longer presented as a catalog, matrix, strategy planner, or stack-plan generator.

It is a probe runner around one supported operator lane.

### 3. Artifact comparison

InferScope persists `BenchmarkArtifact` JSON outputs under:

```text
~/.inferscope/benchmarks/
```

The artifact comparison flow is for operational questions such as:

- did LMCache help or hurt latency?
- did split prefill/decode improve throughput?
- did a config change improve goodput but break reliability?

## What InferScope is not

InferScope is not trying to compete with InferenceX on public benchmark breadth.

It does **not** aim to be:

- a cross-market leaderboard
- a benchmark matrix browser
- a benchmark strategy planner
- a benchmark stack materialization framework
- a generic GPU/model fact sheet over MCP

If you need benchmark breadth, use **ISB-1**.
If you need deployment diagnosis, use **InferScope**.

## Quick start

Use the dedicated quickstart for the fastest supported path:

- [docs/QUICKSTART.md](docs/QUICKSTART.md)
- [docs/MCP_QUICKSTART.md](docs/MCP_QUICKSTART.md)

The short version is:

```bash
git clone https://github.com/rylinjames/easyinference.git
cd easyinference/products/inferscope
uv sync --dev --no-editable

# inspect live runtime behavior first
uv run inferscope profile-runtime http://localhost:8000 --gpu-arch sm_100

# resolve the supported probe plan once profiling works
uv run inferscope benchmark-plan \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --gpu b200 \
  --num-gpus 8

# run a production-lane probe
uv run inferscope benchmark \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --experiment dynamo-disagg-lmcache-kimi-k2 \
  --gpu b200 \
  --num-gpus 8

# compare artifacts
uv run inferscope benchmark-compare before.json after.json

# run MCP server after the CLI path works
uv run inferscope serve
```

## Procedural expansion

InferScope still allows procedural expansion for packaged probe workloads.

Supported knobs:

- `--synthetic-requests`
- `--synthetic-input-tokens`
- `--synthetic-output-tokens`
- `--synthetic-seed`
- `--context-file` (CLI only)

Example:

```bash
uv run inferscope benchmark-plan \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --gpu h200 \
  --num-gpus 8 \
  --synthetic-requests 4 \
  --synthetic-input-tokens 32768 \
  --synthetic-output-tokens 768 \
  --context-file ./repo_context.txt
```

MCP intentionally rejects `context_file` because it is a local file expansion mechanism, not a remote tool contract.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [docs/QUICKSTART.md](docs/QUICKSTART.md)
- [docs/MCP_QUICKSTART.md](docs/MCP_QUICKSTART.md)
- [docs/PROFILING.md](docs/PROFILING.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/BENCHMARK-PLAN.md](docs/BENCHMARK-PLAN.md)
- [docs/AUDIT-TARGET-ARCHITECTURE.md](docs/AUDIT-TARGET-ARCHITECTURE.md)
- [docs/DEPLOYMENT-GUIDE.md](docs/DEPLOYMENT-GUIDE.md)
- [VALIDATION.md](VALIDATION.md)

## Validation

Run from `products/inferscope/`:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/inferscope/
uv run pytest tests/ -v --tb=short
```

## License

MIT
