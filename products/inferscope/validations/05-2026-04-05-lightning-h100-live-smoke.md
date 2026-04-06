# Lightning H100 live smoke validation

Date: **April 5, 2026**

## Scope

This validation records the last known successful Lightning.ai smoke benchmark for InferScope on a single remote **H100** machine.

Primary experiment URL:

- <https://lightning.ai/romirj/easyinference-evaluation-project/experiments/inferscope-h100-live-smoke>

Related proof run:

- <https://lightning.ai/romirj/easyinference-evaluation-project/experiments/inferscope-ssh-proof-nologcap>

## What the run validated

- Lightning SSH access and remote execution worked
- a remote vLLM endpoint was reachable on `http://localhost:8000`
- `inferscope experiment-run` successfully logged to Lightning Experiments
- benchmark summary metrics and remote artifact paths were emitted

## Run shape

- physical machine shape: **1x H100**
- benchmark flags used for the packaged lane: `--gpu h100 --num-gpus 8`
- served model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- served alias used for the probe path: `Kimi-K2.5`

This was a smoke validation of the tooling path, not a truthful production benchmark for Kimi on 8 GPUs.

## Captured benchmark summary

- total requests: `2`
- succeeded: `2`
- failed: `0`
- concurrency: `4`
- wall time: `1726.57 ms`
- average latency: `863.05 ms`
- p95 latency: `890.15 ms`
- p99 latency: `892.56 ms`
- average TTFT: `53.12 ms`
- p90 TTFT: `77.21 ms`
- p95 TTFT: `80.22 ms`
- p99 TTFT: `82.63 ms`
- prompt tokens: `2443`
- completion tokens: `256`
- total tokens: `2699`
- metrics targets total: `2`
- metrics targets with errors: `2`
- metrics capture complete: `false`

## Remote artifact paths observed during the run

- run directory: `/home/zeus/content/easyinference/products/inferscope/lightning_logs/inferscope-h100-live-smoke/`
- benchmark artifact: `/teamspace/studios/this_studio/.inferscope/benchmarks/kimi-k2-long-context-coding-6dcf5b04e52d-kimi-k2-long-context-coding.json`

## Known limitations

- the benchmark lane required packaging assumptions that did not match the actual hardware shape, so the command used `--num-gpus 8` while the physical box was a single H100
- metrics capture was incomplete because the smoke environment used one local vLLM endpoint rather than the multi-target production metrics layout expected by the probe plan
- the raw remote artifact files were not copied into the local repo at run time

## Local record

The machine-readable summary stored alongside this note is:

- `docs/examples/lightning-h100-live-smoke-summary.json`
- `docs/examples/lightning-h100-live-smoke-runtime-profile.json`
- `docs/examples/lightning-h100-live-smoke-benchmark-plan.json`
- `docs/examples/lightning-h100-live-smoke-benchmark-artifact.json`
- `validations/05-2026-04-05-lightning-h100-live-smoke-console.txt`

## Authenticated export check

On April 5, 2026, the public experiment URL was still reachable and returned HTTP `200`, but it only served the Lightning web app shell at fetch time rather than directly exposing the experiment artifact JSON through a simple unauthenticated request.

An authenticated SDK export was then completed successfully using the local Lightning credentials flow. The exported `benchmark-artifact.json` matches the original artifact-named file byte-for-byte, so only the normalized benchmark artifact file is stored locally.

The exported console log artifact exists in Lightning but is empty in this run.
