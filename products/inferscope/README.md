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
- **InferScope** owns the deployment-specific gap-analysis and recommendation loop.

The broader product picture can still include a third lane outside EasyInference:

- **Axion Optimize** or a separate optimization repo can own automated tuning, optimization-session lifecycle, deployment execution, and rollback control.

EasyInference should provide the benchmark truth and deployment evidence that such a product depends on. It should not absorb that product wholesale.

## Boundary with optimization products

InferScope can:

- diagnose a deployment
- run a narrow probe
- compare artifacts
- recommend what to test next

InferScope should not own:

- automated tuning loops
- candidate search and keep/reject workflows
- deployment-changing automation
- versioned optimization sessions with promote/rollback control
- managed endpoint orchestration or billing surfaces

Those belong in Axion Optimize or a separate optimization repo, not in the shipped EasyInference product.

## Current interface contract

### What exists today

InferScope is currently used through:

- CLI commands such as `profile-runtime`, `benchmark-plan`, `benchmark`, and `benchmark-compare`
- MCP tools exposed through `inferscope serve`

That is the supported product interface today.

### What this is building toward

InferScope is not meant to remain terminal-first forever.

The long-term EasyInference product shape is:

- **Primary:** managed web product for human operators
- **Enterprise option:** self-hosted or local deployment of the same platform
- **Secondary:** CLI and MCP for automation and power-user workflows
- **Shared core:** one analysis engine and one API model underneath all interfaces

This README describes the currently shipped CLI and MCP surface, not a fully built web product.

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

## Supported today vs planned expansion

InferScope should be read as a deliberately narrow product right now.

- **Production-validated today:** the Kimi/Dynamo lane listed above
- **Benchmark-supported today:** public-model comparison lanes such as `coding-long-context`
- **Preview smoke today:** low-cost single-endpoint validation such as `coding-smoke` on `a10g`
- **Planned later:** broader model, engine, hardware, and interface expansion

If a workflow or deployment shape is not covered by the current production contract, treat it as planned or experimental rather than implicitly supported.

## Low-cost preview smoke lane

InferScope also ships a deliberately smaller validation path for low-cost hosted endpoints.

- **Model:** `Qwen2.5-7B-Instruct`
- **Engine:** `vllm`
- **Workload pack:** `coding-smoke`
- **GPU target:** `a10g`
- **Purpose:** endpoint health, metrics visibility, CLI/MCP plumbing

This lane is intentionally **preview-only**. Use it to prove that a hosted endpoint behaves like an InferScope target, not to claim production-comparable Kimi numbers.

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

Artifact comparison is the evidence layer for those questions. Automatic keep/revert loops or deployment execution are intentionally out of scope here.

## What InferScope is not

InferScope is not trying to compete with InferenceX on public benchmark breadth.

It does **not** aim to be:

- a cross-market leaderboard
- a benchmark matrix browser
- a benchmark strategy planner
- a benchmark stack materialization framework
- a generic GPU/model fact sheet over MCP
- a managed optimization and deployment platform

If you need benchmark breadth, use **ISB-1**.
If you need deployment diagnosis, use **InferScope**.
If you need closed-loop optimization, use a separate optimization product or repo.

## Quick start

Use the dedicated quickstart for the fastest supported path:

- [docs/QUICKSTART.md](docs/QUICKSTART.md)
- [docs/MCP_QUICKSTART.md](docs/MCP_QUICKSTART.md)
- [docs/EXAMPLE_RESULTS.md](docs/EXAMPLE_RESULTS.md)
- [docs/MODAL_DYNAMO_KIMI.md](docs/MODAL_DYNAMO_KIMI.md)

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

# validate whether an artifact belongs to the canonical production lane
uv run inferscope validate-production-lane after.json --baseline before.json

# run MCP server after the CLI path works
uv run inferscope serve
```

Low-cost smoke path:

```bash
cd easyinference/products/inferscope
uv sync --dev --no-editable

uv run inferscope benchmark-plan \
  coding-smoke \
  https://<endpoint> \
  --gpu a10g \
  --num-gpus 1

uv run inferscope profile-runtime \
  https://<endpoint> \
  --metrics-endpoint https://<endpoint> \
  --scrape-timeout-seconds 90
```

That path is intended for Modal-style smoke validation and checked-in example generation, not production readiness claims.

If you want the repo-root operator path, use:

```bash
cd easyinference
./demo/run_low_cost_smoke.sh --endpoint https://<endpoint>
```

For a true Modal production-lane scaffold instead of the low-cost smoke path,
see:

```text
demo/modal_dynamo_kimi.py
docs/MODAL_DYNAMO_KIMI.md
```

### Artifact preflight

If you are pointing InferScope at a local weight directory or compiled engine,
pass the artifact inputs up front so plan resolution fails before you burn GPU
time on a bad launch:

```bash
uv run inferscope benchmark-plan \
  coding-smoke \
  https://<endpoint> \
  --gpu a10g \
  --num-gpus 1 \
  --model-artifact-path /path/to/model-dir \
  --artifact-manifest ./docs/examples/artifact-manifest-example.yaml
```

The optional manifest lets you declare the intended model, engine,
quantization, tensor parallel size, and GPU family. InferScope validates that
contract before benchmark replay and carries the resulting
`preflight_validation` bundle into the resolved plan artifact.

## Lightning Experiments

If you want the Lightning Experiments page to track InferScope runs, install the
optional logger and use the built-in wrapper command:

```bash
uv sync --dev --extra experiments --no-editable

uv run inferscope experiment-run \
  http://localhost:8000 \
  --teamspace easyinference-evaluation-project \
  --gpu h100 \
  --num-gpus 8 \
  --model-name Kimi-K2.5 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --model-artifact-path /path/to/model-dir \
  --artifact-manifest ./docs/examples/artifact-manifest-example.yaml
```

That command:

- logs the runtime profile result to Lightning
- logs the resolved benchmark plan as an artifact
- logs the manifest-backed preflight validation when artifact inputs are supplied
- optionally runs the probe when you add `--benchmark`
- writes local JSON outputs under `lightning_logs/<experiment-name>/`

If the endpoint is not up yet, the experiment still records the profiling
failure so you can keep setup attempts and benchmark plans in one place.

## Production-lane acceptance

Once you have a saved benchmark artifact, validate whether it is actually
eligible to support production-lane claims:

```bash
uv run inferscope validate-production-lane candidate.json --baseline baseline.json
```

That command fails closed on the most important product boundaries:

- missing provenance
- non-production lane classes such as `benchmark_supported` or `preview_smoke`
- failed benchmark readiness checks
- failed preflight validation captured in the run plan
- baseline/candidate comparisons that cross lane boundaries

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
- [docs/EXAMPLE_RESULTS.md](docs/EXAMPLE_RESULTS.md)
- [../../demo/modal_vllm.py](../../demo/modal_vllm.py)
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
