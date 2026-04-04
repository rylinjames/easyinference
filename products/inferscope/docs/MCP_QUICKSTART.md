# InferScope MCP Quickstart

This is the fastest supported path to connect InferScope to Cursor or Claude Desktop.

Use this guide if you want to:

- generate a working MCP config in one command
- connect InferScope over stdio
- confirm the server is visible in your MCP client
- run the first profiling tool call against a live endpoint

## Before You Start

You need:

- a local checkout of `EasyInference/products/inferscope`
- `uv`
- an MCP client such as Cursor or Claude Desktop
- a running inference endpoint with a Prometheus-style `/metrics` surface

The supported source-checkout install path is:

```bash
git clone https://github.com/rylinjames/easyinference.git
cd easyinference/products/inferscope
uv sync --dev --no-editable
```

## 1. Generate The MCP Config

Run:

```bash
uv run --no-editable python -m inferscope.cli connect --project-dir "$(pwd)"
```

This prints JSON like:

```json
{
  "mcpServers": {
    "InferScope": {
      "command": "uv",
      "args": [
        "run",
        "--no-editable",
        "--directory",
        "/absolute/path/to/EasyInference/products/inferscope",
        "inferscope",
        "serve"
      ]
    }
  }
}
```

That is the copy-pasteable stdio config for both Cursor and Claude Desktop.

## 2. Add It To Your MCP Client

Paste the generated JSON into your MCP client config.

For Cursor:

- open Cursor MCP settings
- add the generated `InferScope` server entry
- save and reload MCP servers

For Claude Desktop:

- open the Claude Desktop MCP config file
- add the generated `InferScope` server entry
- restart Claude Desktop

If you prefer to write the config manually, use the generated JSON as the source of truth.

## 3. Verify The Server Command

Before debugging the client, verify the server command itself:

```bash
uv run --no-editable inferscope serve --help
```

Success looks like help output for:

- `serve`
- `--transport`
- `--port`

If that command fails, fix the local install before touching the client config.

## 4. Confirm MCP Handshake

After adding the config, the MCP client should show an `InferScope` server with available tools.

The first tools you should expect to see are:

- `tool_profile_runtime`
- `tool_audit_deployment`
- `tool_check_deployment`
- `tool_resolve_benchmark_plan`
- `tool_run_benchmark`
- `tool_compare_benchmarks`

If the server appears but no tools are listed, reload the client and verify the generated JSON was copied exactly.

## 5. Run The First Tool Call

Use this first call:

- `tool_profile_runtime`

Recommended minimal input:

```json
{
  "endpoint": "http://localhost:8000"
}
```

Better input if you know the deployment details:

```json
{
  "endpoint": "http://localhost:8000",
  "gpu_arch": "sm_100",
  "model_name": "Kimi-K2.5",
  "quantization": "fp8",
  "kv_cache_dtype": "fp8_e4m3"
}
```

Success looks like:

- a summary
- confidence
- runtime metrics
- findings or bottlenecks
- tuning-relevant evidence

## 6. What To Do If It Fails

If the MCP client cannot start the server:

- rerun `uv sync --dev --no-editable`
- rerun `uv run --no-editable inferscope connect --project-dir "$(pwd)"`
- make sure the config uses the absolute `products/inferscope` path

If the server starts but tools do not appear:

- reload the MCP client
- confirm the JSON is valid
- confirm the `args` list still contains `--no-editable`

If `tool_profile_runtime` fails:

- verify the endpoint is reachable
- verify `/metrics` exists
- verify any required auth is available to the local machine running the MCP client

## Next Docs

- [Quickstart](./QUICKSTART.md)
- [Runtime Profiling](./PROFILING.md)
- [Benchmark Plan](./BENCHMARK-PLAN.md)
