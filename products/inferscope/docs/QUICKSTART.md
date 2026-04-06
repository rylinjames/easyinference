# InferScope Quickstart

This is the fastest supported path from clone to a real InferScope result.

Use this guide if you want to:

- install InferScope locally
- profile a live endpoint in one command
- confirm what successful output looks like
- connect InferScope over MCP after the first CLI run

This quickstart is intentionally narrow. It uses the strongest product path in InferScope today:

- `profile-runtime` against a live endpoint
- optional `benchmark-plan` once the runtime profile succeeds
- optional MCP connection after the CLI path works

## Supported product lane

This quickstart is written for the current production-validated InferScope lane, not the full long-term EasyInference platform vision.

- **Model lane:** `Kimi-K2.5`
- **Production engine:** `dynamo`
- **Comparison engine:** `vllm`
- **Workload pack:** `kimi-k2-long-context-coding`
- **GPU targets:** `h100`, `h200`, `b200`, `b300`

If you are outside that lane, use this guide as a source-checkout and workflow reference, not as a claim that your exact deployment shape is already supported.

## Preview smoke lane

InferScope also ships a cheaper smoke path for hosted endpoints and low-cost GPUs.

- **Model lane:** `Qwen2.5-7B-Instruct`
- **Engine:** `vllm`
- **Workload pack:** `coding-smoke`
- **GPU target:** `a10g`
- **Purpose:** endpoint health, observability, and CLI/MCP plumbing

Treat that lane as preview-only. It is useful for proving the toolchain path, not for making production-comparable Kimi claims.

## Before You Start

You need:

- Python 3.11+
- `uv`
- a running inference endpoint with a Prometheus-style `/metrics` surface

The examples below assume a local endpoint at `http://localhost:8000`.

## 1. Install

Clone the repo and sync the InferScope environment:

```bash
git clone https://github.com/rylinjames/easyinference.git
cd easyinference/products/inferscope
uv sync --dev --no-editable
```

Sanity-check that the CLI is available:

```bash
uv run inferscope --help
```

Success looks like a Typer help screen with commands such as:

- `profile-runtime`
- `benchmark-plan`
- `benchmark`
- `benchmark-compare`
- `serve`

## 2. Run Your First Probe

Profile a live runtime first. This is the highest-value path in InferScope today.

```bash
uv run inferscope profile-runtime http://localhost:8000
```

If you already know deployment details, add them for better recommendations:

```bash
uv run inferscope profile-runtime http://localhost:8000 \
  --gpu-arch sm_100 \
  --model-name Kimi-K2.5 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3
```

Success looks like:

- a bold summary line
- an optional confidence score
- a "Why this recommendation" section
- a JSON block with runtime metrics, bottlenecks, findings, and tuning-relevant evidence

You are looking for evidence such as:

- memory pressure
- cache effectiveness
- queue or scheduler bottlenecks
- deployment checks and audit findings

## 3. Resolve the Supported Probe Plan

Once runtime profiling works, resolve the supported packaged probe plan:

```bash
uv run inferscope benchmark-plan \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --gpu b200 \
  --num-gpus 8
```

Success looks like a JSON response describing the supported plan for the current product lane, including the workload pack, experiment options, and probe metadata.

If you want to run a probe after that:

```bash
uv run inferscope benchmark \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --experiment dynamo-disagg-lmcache-kimi-k2 \
  --gpu b200 \
  --num-gpus 8
```

Benchmark artifacts are persisted under:

```text
~/.inferscope/benchmarks/
```

For the canonical production lane, the checked-in acceptance corpus is:

- `docs/examples/kimi-dynamo-production-reference-summary.json`
- `docs/examples/benchmark-artifact-baseline.json`
- `docs/examples/benchmark-artifact-candidate.json`
- `docs/examples/benchmark-comparison-example.json`

Use those files to verify the expected JSON shape and provenance fields before you compare against live exports.

Validate whether a saved artifact is actually eligible for production-lane
claims:

```bash
uv run inferscope validate-production-lane after.json --baseline before.json
```

This is the acceptance gate for production-lane assertions. A Modal or A10G
smoke artifact should fail here even if the replay itself succeeded.

### Optional: preflight a local model artifact

If you are benchmarking from a local model or engine directory, validate that
directory before launch:

```bash
uv run inferscope benchmark-plan \
  coding-smoke \
  https://<endpoint> \
  --gpu a10g \
  --num-gpus 1 \
  --model-artifact-path /path/to/model-dir \
  --artifact-manifest ./docs/examples/artifact-manifest-example.yaml
```

That adds a `preflight_validation` section to the resolved plan and fails early
if the directory is incomplete, the manifest disagrees with the selected model
or engine, or the requested topology cannot fit in memory.

## 3a. Resolve the preview smoke plan

If you only need a low-cost validation run, use the smoke lane instead:

```bash
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

Success here means:

- the endpoint answered OpenAI-compatible requests
- the metrics surface was scrapeable
- InferScope produced a plan and runtime profile for the preview lane
- any supplied local artifact or manifest passed preflight validation first

It does **not** mean the deployment is equivalent to the production Kimi lane.

If you already have a hosted endpoint and want the whole low-cost path from repo root, use:

```bash
cd easyinference
./demo/run_low_cost_smoke.sh --endpoint https://<endpoint>
```

## 4. Connect InferScope Over MCP

After the CLI path works, start the MCP server:

```bash
uv run inferscope serve
```

Use this configuration in an MCP client such as Cursor or Claude Desktop:

```json
{
  "mcpServers": {
    "InferScope": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/EasyInference/products/inferscope",
        "inferscope",
        "serve"
      ]
    }
  }
}
```

Replace `/absolute/path/to/EasyInference/products/inferscope` with your local checkout path.

Once connected, the first tool to try is the profiling surface:

- `tool_profile_runtime`

Then use the benchmark tools if needed:

- `tool_resolve_benchmark_plan`
- `tool_run_benchmark`
- `tool_compare_benchmarks`

## 5. What To Do If It Fails

If `uv sync --dev` fails:

- confirm `uv` is installed
- confirm you are in `products/inferscope`
- if you are working from a source checkout, use `uv sync --dev --no-editable`

If `profile-runtime` fails immediately:

- confirm the endpoint is reachable
- confirm `/metrics` exists
- confirm any metrics auth headers are being passed correctly
- for serverless endpoints, pass `--scrape-timeout-seconds 90` or higher
- if `/metrics` lives on the same base URL but under a different frontend path, pass `--metrics-endpoint`

If the output is thin or generic:

- add known deployment metadata such as `--gpu-arch`, `--model-name`, `--quantization`, and `--kv-cache-dtype`

If MCP connects but tools do not appear:

- make sure the MCP command points at `products/inferscope`
- restart the MCP client after saving the config

## Next Docs

- [Example Results](./EXAMPLE_RESULTS.md)
- [MCP Quickstart](./MCP_QUICKSTART.md)
- [Runtime Profiling](./PROFILING.md)
- [Benchmark Plan](./BENCHMARK-PLAN.md)
- [Benchmarks](./BENCHMARKS.md)
- [Deployment Guide](./DEPLOYMENT-GUIDE.md)
