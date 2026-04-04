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

This quickstart is written for the current supported InferScope lane, not the full long-term EasyInference platform vision.

- **Model lane:** `Kimi-K2.5`
- **Production engine:** `dynamo`
- **Comparison engine:** `vllm`
- **Workload pack:** `kimi-k2-long-context-coding`
- **GPU targets:** `h100`, `h200`, `b200`, `b300`

If you are outside that lane, use this guide as a source-checkout and workflow reference, not as a claim that your exact deployment shape is already supported.

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

If the output is thin or generic:

- add known deployment metadata such as `--gpu-arch`, `--model-name`, `--quantization`, and `--kv-cache-dtype`

If MCP connects but tools do not appear:

- make sure the MCP command points at `products/inferscope`
- restart the MCP client after saving the config

## Next Docs

- [MCP Quickstart](./MCP_QUICKSTART.md)
- [Runtime Profiling](./PROFILING.md)
- [Benchmark Plan](./BENCHMARK-PLAN.md)
- [Benchmarks](./BENCHMARKS.md)
- [Deployment Guide](./DEPLOYMENT-GUIDE.md)
