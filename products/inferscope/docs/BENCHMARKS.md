# InferScope Benchmarks

InferScope does not ship a generic benchmark platform anymore.

Its benchmark package is a **narrow probe layer** for one operator-facing product lane.

If you want broad benchmark coverage, use **ISB-1**.
If you want deployment-specific KV/disaggregation diagnostics, use **InferScope**.
If you want closed-loop optimization or deployment-changing automation, use a separate optimization product or repo.

## Scope

InferScope benchmark support is intentionally split into three surfaces:

### Production-validated lane

- **model:** `Kimi-K2.5`
- **production engine:** `dynamo`
- **comparison engine:** `vllm`
- **workload pack:** `kimi-k2-long-context-coding`
- **GPUs:** `h100`, `h200`, `b200`, `b300`
- **topologies:** `single_endpoint`, `prefill_decode_split`
- **cache strategy:** `lmcache`

Production experiments:

- `dynamo-aggregated-lmcache-kimi-k2`
- `vllm-disagg-prefill-lmcache`
- `dynamo-disagg-lmcache-kimi-k2`

### Benchmark-supported public lanes

- `coding-long-context` on `Qwen3.5-32B`
- larger Qwen coder comparison lanes used for benchmark-supported replay work

### Preview smoke lane

- **model:** `Qwen2.5-7B-Instruct`
- **engine:** `vllm`
- **workload pack:** `coding-smoke`
- **GPU:** `a10g`
- **purpose:** endpoint health, observability, CLI/MCP validation

Smoke experiment:

- `vllm-single-endpoint-smoke`

The source of truth is `src/inferscope/production_target.py`.

## What this layer does

The benchmark layer now exists to do four things:

1. resolve the supported workload + experiment into a concrete run plan
2. replay that plan against a live OpenAI-compatible endpoint
3. persist a `BenchmarkArtifact`
4. compare two artifacts to quantify change

That is it.

It is the evidence layer for deployment changes, not the system that executes those changes.

It does **not** exist to:

- list benchmark catalogs
- build benchmark matrices
- plan benchmark suites
- generate benchmark stack plans
- materialize benchmark bundles

Those surfaces were removed because they turned InferScope into generic benchmark infrastructure instead of a useful operator product.

## Public surfaces

### CLI

- `inferscope benchmark-plan`
- `inferscope benchmark`
- `inferscope benchmark-compare`

### MCP

- `tool_get_production_contract`
- `tool_resolve_benchmark_plan`
- `tool_run_benchmark`
- `tool_compare_benchmarks`
- `tool_get_benchmark_artifact`

## Typical workflow

```bash
# inspect the resolved run plan
uv run inferscope benchmark-plan \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --gpu b200 \
  --num-gpus 8

# run the aggregated production lane
uv run inferscope benchmark \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --experiment dynamo-aggregated-lmcache-kimi-k2 \
  --gpu b200 \
  --num-gpus 8

# run the disaggregated production lane
uv run inferscope benchmark \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --experiment dynamo-disagg-lmcache-kimi-k2 \
  --gpu b200 \
  --num-gpus 8

# compare results
uv run inferscope benchmark-compare aggregated.json disagg.json
```

Preview smoke workflow:

```bash
uv run inferscope benchmark-plan \
  coding-smoke \
  https://<endpoint> \
  --gpu a10g \
  --num-gpus 1
```

## Probe resolution

Both CLI and MCP now route through:

- `src/inferscope/benchmarks/probe_resolution.py`

That module is the shared contract for:

- procedural expansion
- experiment defaulting
- support assessment
- run-plan construction
- MCP vs CLI context-file policy

Important rules:

- blank `experiment` defaults to the correct lane for the chosen workload pack
- unsupported workload packs are rejected
- unsupported experiments are rejected
- model/engine are derived from the supported experiment and workload, not public override knobs
- `context_file` is CLI-only and rejected from MCP

## Artifact model

The persisted artifact remains `BenchmarkArtifact`.

What matters operationally in current outputs:

- `run_plan.execution`
- `run_plan.support`
- `run_plan.observed_runtime`
- request success/failure summary
- saved metrics snapshots
- comparison deltas and ratios

Artifacts are written under:

```text
~/.inferscope/benchmarks/
```

## Procedural expansion

InferScope still supports procedural expansion for the supported packaged probe workload.

Available knobs:

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

## Relationship to profiling

The benchmark layer is not the product center.
The profiling layer is.

The intended operator loop is:

1. profile the live deployment
2. run a narrow probe
3. compare artifacts or compare runtime vs probe behavior
4. decide whether KV policy, cache routing, or topology changed anything meaningful

The next step after that may be a recommendation to test or promote a change, but keep/revert automation belongs outside InferScope.

## What is still missing

The benchmark layer is narrower now, but it is not finished.

High-value missing pieces are still:

- deeper KV-tier and offload metrics in artifacts
- phase-aware telemetry for prefill vs decode vs handoff
- stronger provenance on artifact manifests
- gap-analysis logic that turns runtime + probe evidence into concrete recommended next steps

That missing work is fine.
What is no longer fine is pretending InferScope needs a generic benchmark framework while those operator-grade pieces are still absent.
