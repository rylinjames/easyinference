# Modal Dynamo/Kimi Architecture

This document explains the exact Modal architecture needed to make a
**true InferScope Kimi/Dynamo lane** possible, and what each InferScope term
means in that deployment.

It is intentionally separate from the low-cost preview smoke path in
[demo/modal_vllm.py](/Users/romirjain/Desktop/building%20projects/axion_compute/EasyInference/demo/modal_vllm.py).

## What the terms mean

### Metrics scraping

InferScope reads Prometheus metrics from one or more HTTP endpoints.

In code, that happens in
[telemetry/prometheus.py](/Users/romirjain/Desktop/building%20projects/axion_compute/EasyInference/products/inferscope/src/inferscope/telemetry/prometheus.py).

For a Dynamo deployment, the important metric groups are:

- frontend queue and latency: `dynamo_frontend_*`
- worker request metrics: `dynamo_component_*`
- KV cache metrics: `dynamo_component_kvstats_*`
- routing overhead histograms: `dynamo_router_overhead_*`
- LMCache metrics: `lmcache:*` (LMCache is an upstream project with its own `/metrics` endpoint — Dynamo does NOT re-export them under a `dynamo_lmcache_*` prefix)
- KVBM tiering metrics: `kvbm_*` (on a separate port, default `6880` via `DYN_KVBM_METRICS_PORT`, requires `DYN_KVBM_METRICS=true` at launch)
- NIXL transfer metrics: on a separate port via `NIXL_TELEMETRY_PROMETHEUS_PORT`; schema not yet pinned down in InferScope

There are no server-side `dynamo_slo_*` or `dynamo_grove_*` metrics. "Grove"
is NVIDIA Dynamo's Kubernetes gang-scheduling component (topology-aware pod
placement) and has nothing to do with KV tiering. SLO violation accounting
is a client/harness-side concern and must be computed from TTFT/ITL
histograms rather than read from server counters.

### Plan resolution

Plan resolution is the step that turns:

- a workload pack such as `kimi-k2-long-context-coding`
- an endpoint URL
- a GPU type and GPU count
- an experiment name
- optional artifact inputs

into one concrete `BenchmarkRunPlan`.

That logic lives in
[benchmarks/probe_resolution.py](/Users/romirjain/Desktop/building%20projects/axion_compute/EasyInference/products/inferscope/src/inferscope/benchmarks/probe_resolution.py).

The plan includes:

- the resolved model and engine
- topology mode
- cache strategy
- metrics targets
- support tier
- preflight validation

### Benchmark replay

Benchmark replay is the part that actually sends requests to the model server.

InferScope takes each request from the workload pack, applies the planned
concurrency and warmup settings, sends the requests to the request endpoint,
records latency and token outputs, and captures metrics before and after.

The runtime is implemented in
[benchmarks/runtime.py](/Users/romirjain/Desktop/building%20projects/axion_compute/EasyInference/products/inferscope/src/inferscope/benchmarks/runtime.py).

### Artifact generation

Artifact generation means writing one JSON file that captures the whole run.

The artifact model is
[BenchmarkArtifact in benchmarks/models.py](/Users/romirjain/Desktop/building%20projects/axion_compute/EasyInference/products/inferscope/src/inferscope/benchmarks/models.py).

That file stores:

- the resolved run plan
- provenance for workload and experiment
- lane classification
- metrics snapshots
- per-request results
- summary rollups

## Why the current Modal smoke app is not enough

The existing Modal smoke path is real, but it is a different lane:

- model: `Qwen2.5-7B-Instruct`
- engine: `vllm`
- GPU: `A10G`
- topology: single endpoint
- claim scope: preview or smoke only

The canonical production lane in InferScope is:

- model: `Kimi-K2.5`
- engine: `dynamo`
- GPU family: `h100`, `h200`, `b200`, `b300`
- topology: `single_endpoint` or `prefill_decode_split`
- cache strategy: `lmcache`

That contract is defined in
[production_target.py](/Users/romirjain/Desktop/building%20projects/axion_compute/EasyInference/products/inferscope/src/inferscope/production_target.py).

## Recommended first production Modal shape

Start with the **aggregated** lane, not the split prefill/decode lane.

Why:

- it is already first-class in the shipped experiment specs
- it only needs two metrics targets: `frontend` and `worker`
- it matches the current product contract cleanly
- it avoids multi-node or router orchestration on day one

The matching experiment spec is
[dynamo-aggregated-lmcache-kimi-k2.yaml](/Users/romirjain/Desktop/building%20projects/axion_compute/EasyInference/products/inferscope/src/inferscope/benchmarks/experiment_specs/dynamo-aggregated-lmcache-kimi-k2.yaml).

## Exact aggregated Modal architecture

### Components

1. One persistent Modal Volume with Kimi weights and the artifact manifest.
2. One GPU-backed Modal service running:
   - a local Dynamo frontend process
   - a local aggregated Dynamo worker process
   - LMCache configured for the lane
3. One public ASGI gateway that exposes:
   - `/v1/*`
   - `/frontend/metrics`
   - `/worker/metrics`

The implementation scaffold for that is in
[demo/modal_dynamo_kimi.py](/Users/romirjain/Desktop/building%20projects/axion_compute/EasyInference/demo/modal_dynamo_kimi.py).

### Public URLs

If deployed, the public Modal base URL should provide:

- `https://<workspace>--inferscope-kimi-dynamo-serve.modal.run/v1/*`
- `https://<workspace>--inferscope-kimi-dynamo-serve.modal.run/frontend/metrics`
- `https://<workspace>--inferscope-kimi-dynamo-serve.modal.run/worker/metrics`
- `https://<workspace>--inferscope-kimi-dynamo-serve.modal.run/healthz`

### Local process layout inside the GPU container

The aggregated scaffold assumes these local process surfaces:

- request traffic: `http://127.0.0.1:8001`
- frontend metrics: `http://127.0.0.1:8001/metrics`
- worker metrics: `http://127.0.0.1:9200/metrics`

The Modal gateway proxies public paths to those local URLs.

### Why this counts as a true lane

If the deployment is actually:

- `Kimi-K2.5`
- `dynamo`
- `lmcache`
- 4 supported GPUs
- with distinct frontend and worker metrics

then InferScope can truthfully classify the resulting artifact as production
candidate evidence for the aggregated lane.

## Required launch responsibilities

The scaffold intentionally does not invent Dynamo startup commands.

You must provide:

- `INFERSCOPE_DYNAMO_FRONTEND_CMD`
- `INFERSCOPE_DYNAMO_WORKER_CMD`

## Candidate Dynamo commands

These are the commands I would give the scaffold first.

They are based on the current Dynamo docs shape and are the best available
starting point for an aggregated Kimi lane on one 4-GPU Modal host.

### Frontend

```bash
export INFERSCOPE_DYNAMO_FRONTEND_CMD='python3 -m dynamo.frontend --http-port 8001 --discovery-backend file'
```

Meaning:

- `dynamo.frontend`: the HTTP entrypoint
- `--http-port 8001`: internal request and frontend metrics surface
- `--discovery-backend file`: simple single-host service discovery

### Worker

```bash
export INFERSCOPE_DYNAMO_WORKER_CMD='DYN_SYSTEM_PORT=9200 python3 -m dynamo.vllm --model /model-weights/Kimi-K2-Instruct --tensor-parallel-size 4 --served-model-name Kimi-K2.5 --max-model-len 131072 --gpu-memory-utilization 0.92 --enable-prefix-caching --trust-remote-code --discovery-backend file --connector lmcache'
```

Meaning:

- `DYN_SYSTEM_PORT=9200`: the worker metrics and health surface
- `dynamo.vllm`: a Dynamo-managed worker using the vLLM backend
- `--model /model-weights/Kimi-K2-Instruct`: load Kimi from the Modal Volume
- `--tensor-parallel-size 4`: shard across the 4 GPUs in the Modal container
- `--served-model-name Kimi-K2.5`: make the request model name match the InferScope lane
- `--max-model-len 131072`: 128K context lane
- `--gpu-memory-utilization 0.92`: reserve most HBM for model + KV
- `--enable-prefix-caching`: required for the lane
- `--discovery-backend file`: simple single-host discovery
- `--connector lmcache`: enable the required cache connector

### Important caveat

The worker command is a **best current draft**, not a guaranteed final runtime.

The parts most likely to need adjustment after the first live boot are:

- whether `dynamo.vllm` accepts `--served-model-name` directly
- the exact LMCache connector flag spelling
- any extra frontend or worker registration arguments required by the current Dynamo build

The right way to use these commands is:

1. boot the scaffold
2. inspect `/healthz`, `/frontend/metrics`, and `/worker/metrics`
3. adjust only the launch args that the real runtime rejects

Those commands should:

- load the Kimi weights from `/model-weights/Kimi-K2-Instruct`
- serve `Kimi-K2.5`
- expose the request surface on `127.0.0.1:8001`
- expose frontend metrics on `127.0.0.1:8001/metrics`
- expose worker metrics on `127.0.0.1:9200/metrics`
- preserve `X-Session-ID` stickiness
- enable LMCache for the lane

## Artifact manifest

The scaffold writes:

- `/model-weights/Kimi-K2-Instruct/artifact-manifest.yaml`

That manifest encodes the lane contract:

- model
- engine
- production target name
- tensor parallel size
- topology mode
- cache strategy
- session header name

## InferScope commands against this Modal lane

### Benchmark plan

```bash
uv run inferscope benchmark-plan \
  kimi-k2-long-context-coding \
  https://<workspace>--inferscope-kimi-dynamo-serve.modal.run \
  --gpu h200 \
  --num-gpus 4 \
  --model-artifact-path /model-weights/Kimi-K2-Instruct \
  --artifact-manifest /model-weights/Kimi-K2-Instruct/artifact-manifest.yaml
```

### Runtime profile

For runtime profiling, use the frontend metrics surface first:

```bash
uv run inferscope profile-runtime \
  https://<workspace>--inferscope-kimi-dynamo-serve.modal.run \
  --metrics-endpoint https://<workspace>--inferscope-kimi-dynamo-serve.modal.run/frontend/metrics
```

### Full benchmark replay

```bash
uv run inferscope benchmark \
  kimi-k2-long-context-coding \
  https://<workspace>--inferscope-kimi-dynamo-serve.modal.run \
  --experiment dynamo-aggregated-lmcache-kimi-k2 \
  --gpu h200 \
  --num-gpus 4 \
  --metrics-target frontend=https://<workspace>--inferscope-kimi-dynamo-serve.modal.run/frontend/metrics \
  --metrics-target worker=https://<workspace>--inferscope-kimi-dynamo-serve.modal.run/worker/metrics \
  --model-artifact-path /model-weights/Kimi-K2-Instruct \
  --artifact-manifest /model-weights/Kimi-K2-Instruct/artifact-manifest.yaml \
  --output artifact.json
```

### Production-lane validation

```bash
uv run inferscope validate-production-lane artifact.json
```

## What a split prefill/decode Modal lane would add later

The split production lane is the next step, not the first step.

That lane would require three scrape targets:

- `frontend`
- `prefill`
- `decode`

and it maps to
[dynamo-disagg-lmcache-kimi-k2.yaml](/Users/romirjain/Desktop/building%20projects/axion_compute/EasyInference/products/inferscope/src/inferscope/benchmarks/experiment_specs/dynamo-disagg-lmcache-kimi-k2.yaml).

It is harder because:

- the frontend must act as a KV-aware router
- prefill and decode need separate metrics ports
- LMCache must preserve cross-stage reuse
- the infrastructure needs to behave like a real split-serving system

## Operational boundaries

This Modal scaffold is the right way to test a true Kimi/Dynamo lane on Modal,
but it is not automatically equivalent to a multi-host production cluster.

It proves the right contract only if the runtime really matches the lane.

Do not classify results as production evidence unless all of these are true:

- model is `Kimi-K2.5`
- engine is `dynamo`
- cache strategy is `lmcache`
- GPU shape meets the lane minimum
- metrics capture is complete
- artifact preflight is valid
- `validate-production-lane` returns `valid: true`
