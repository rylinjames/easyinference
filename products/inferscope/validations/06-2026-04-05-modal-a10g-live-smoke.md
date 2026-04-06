# Modal A10G live smoke validation

Date: **April 5, 2026**

## Scope

This validation records the low-cost hosted smoke path for InferScope and ISB-1 using the checked-in Modal demo endpoint.

Primary endpoint:

- <https://hikaflow--easyinference-demo-serve.modal.run>

## What the run validated

- the Modal demo served an OpenAI-compatible endpoint on `/v1/*`
- `/metrics` was reachable and exposed vLLM Prometheus metrics
- `inferscope benchmark-plan coding-smoke ... --gpu a10g --num-gpus 1` succeeded
- `inferscope profile-runtime ... --scrape-timeout-seconds 90` succeeded against the same endpoint
- `uv run isb1 quick-bench ...` completed for both simple and coding smoke requests after the CLI timeout/warmup hardening

## Run shape

- platform: **Modal**
- physical machine shape: **1x A10G**
- served model: `Qwen/Qwen2.5-7B-Instruct`
- engine: `vllm`
- InferScope workload pack: `coding-smoke`

## Captured outcomes

InferScope:

- smoke plan support: `degraded`
- reason: preview smoke lane on `vllm` + `a10g`, not the production-validated Kimi lane
- runtime profile status: `healthy`
- metrics scrape timeout used: `90s`
- observed metrics scrape time: about `39.8s`

ISB-1 simple quick bench:

- requests: `1`
- duration: `120s`
- completed: `1`
- failed: `0`
- wall time: `9.0s`
- TTFT p50: `6.338s`
- throughput: `7 tok/s`

ISB-1 coding quick bench:

- requests: `1`
- duration: `120s`
- completed: `1`
- failed: `0`
- wall time: `25.5s`
- TTFT p50: `0.992s`
- throughput: `28 tok/s`

## Known limitations

- this was a hosted smoke validation, not a publishable benchmark claim
- serverless cold starts and scheduling variance materially affected latency
- only summary metrics were preserved locally for the Modal run, not a full raw artifact export

## Local record

The machine-readable summary stored alongside this note is:

- `docs/examples/modal-a10g-live-smoke-summary.json`
