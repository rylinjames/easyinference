# Runtime Profiling

InferScope ships a Prometheus-first runtime profiling surface for live inference deployments.

This is currently the strongest product-aligned subsystem in InferScope.

## Public surfaces

- CLI: `inferscope profile-runtime`
- MCP: `tool_profile_runtime`

`inferscope profile` remains a static model-intel helper. `profile-runtime` is the live diagnostic path.

## Product purpose

The profiling path exists to answer operator questions such as:

- is this deployment queue-bound, cache-bound, or memory-bound?
- is KV reuse working well enough to matter?
- is the current topology introducing migrations or handoff pain?
- what tuning change is worth testing next?

It is not meant to be a generic metrics browser.

## Current flow

1. scrape `/metrics`
2. normalize engine-specific metrics into a shared runtime schema
3. classify workload heuristically
4. run deployment checks
5. group findings into bottlenecks
6. preview tuning recommendations
7. optionally enrich runtime identity from `/v1/models`

## Current scope

The product contract is narrowed around:

- `Kimi-K2.5`
- Hopper / Blackwell (`h100`, `h200`, `b200`, `b300`)
- `dynamo` as the production lane
- `vllm` as the comparison lane

For low-cost validation, the same profiling surface is also used against preview smoke endpoints such as the checked-in Modal `a10g` demo, typically with `--metrics-endpoint` and a higher `--scrape-timeout-seconds`.

The profiling system is still broader internally than the public product contract, but the public direction is clear: profiling should support deployment diagnosis for the supported product lane first.

## Report contents

Current runtime profiles include:

- normalized runtime metrics
- health summary
- memory pressure analysis
- cache effectiveness analysis
- heuristic workload classification
- audit findings
- grouped bottlenecks
- optional tuning preview
- optional runtime identity enrichment

The runtime profile uses the same `MetricSnapshot` schema family as probe artifacts so live runtime evidence and saved benchmark evidence can converge over time.

This profiling surface is advisory. It can point to likely next tests or configuration changes, but automated execution belongs outside InferScope.

## CLI examples

```bash
uv run inferscope profile-runtime http://localhost:8000
```

```bash
uv run inferscope profile-runtime http://localhost:8000 \
  --gpu-arch sm_100 \
  --model-name Kimi-K2.5 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3
```

## MCP notes

`tool_profile_runtime` is the MCP-safe wrapper around the same profiling core.

- private IP ranges are blocked by default
- metrics auth is resolved from the MCP payload
- runtime identity enrichment follows the same network policy

## Current limits

v1 is still intentionally Prometheus-first.

It does **not** yet provide the full product-grade evidence model InferScope eventually needs, including:

- phase-aware prefill/decode/handoff timing
- explicit LMCache / offload / transfer metrics everywhere
- persistent runtime profiles by default
- stronger recommendation logic that closes the evidence loop for operators

That missing work belongs under `src/inferscope/profiling/`, `src/inferscope/telemetry/`, and future diagnostics layers — not in a generic dashboard system.
