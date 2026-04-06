# InferScope Example Results

These files show what InferScope output looks like before you point it at a real deployment.

All examples in this directory are **reference data**:

- they are synthetic or fixture-backed
- they are intended to show output shape and interpretation
- they are not claims about real production benchmark performance

Checked-in live smoke exports sit beside those reference fixtures, but they stay clearly labeled as smoke validations with documented limitations.

## Files

- [examples/runtime-profile-example.json](examples/runtime-profile-example.json)
  Representative output for `uv run inferscope profile-runtime http://localhost:8000`
- [examples/benchmark-plan-example.json](examples/benchmark-plan-example.json)
  Representative output for `uv run inferscope benchmark-plan ...`
- [examples/benchmark-artifact-baseline.json](examples/benchmark-artifact-baseline.json)
  A reference benchmark artifact for a baseline run
- [examples/benchmark-artifact-candidate.json](examples/benchmark-artifact-candidate.json)
  A reference benchmark artifact for a candidate run
- [examples/benchmark-comparison-example.json](examples/benchmark-comparison-example.json)
  Representative output for `uv run inferscope benchmark-compare before.json after.json`
- [examples/artifact-manifest-example.yaml](examples/artifact-manifest-example.yaml)
  Example JSON/YAML manifest for `--artifact-manifest` preflight validation
- [examples/lightning-h100-live-smoke-summary.json](examples/lightning-h100-live-smoke-summary.json)
  Authenticated export summary from the Lightning H100 smoke validation
- [examples/modal-a10g-live-smoke-summary.json](examples/modal-a10g-live-smoke-summary.json)
  Local summary of the Modal A10G smoke validation
- [examples/kimi-dynamo-production-reference-summary.json](examples/kimi-dynamo-production-reference-summary.json)
  Canonical production-lane reference corpus for the Kimi/Dynamo path

## How To Use These

Use the examples to answer three questions quickly:

1. What shape of JSON does InferScope produce?
2. Which sections matter when profiling or comparing a deployment?
3. What does a successful run look like before I have my own artifacts?

## Live smoke validations

These files are different from the synthetic fixtures above:

- `lightning-h100-live-smoke-*` preserves an authenticated export from the Lightning smoke run
- `modal-a10g-live-smoke-summary.json` records the low-cost Modal smoke path that validated `coding-smoke`, runtime profiling, and ISB-1 quick benches
- `kimi-dynamo-production-reference-summary.json` pins the checked-in baseline/candidate artifact corpus for the production-validated Kimi lane

Use them to answer a narrower question:

1. What did a real smoke validation look like on the currently documented hosted paths?

Do not use them as publishable benchmark claims. Each summary calls out the exact limitations.

## What To Look At First

If you are new to InferScope, start here:

1. `runtime-profile-example.json`
   This is the fastest way to understand the live profiling surface.
2. `benchmark-plan-example.json`
   This shows how InferScope resolves one supported probe lane into a concrete run plan.
3. `benchmark-comparison-example.json`
   This shows the kind of before/after summary InferScope gives once you have two artifacts.
4. `kimi-dynamo-production-reference-summary.json`
   This tells you which checked-in fixtures define the canonical production lane and what acceptance claims they are allowed to make.

## How These Map To Commands

### Runtime profiling

```bash
uv run inferscope profile-runtime http://localhost:8000
```

Relevant example:

- [examples/runtime-profile-example.json](examples/runtime-profile-example.json)

Focus on:

- `summary`
- `confidence`
- `bottlenecks`
- `tuning_preview`
- `cache_effectiveness`

### Probe plan resolution

```bash
uv run inferscope benchmark-plan \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --gpu b200 \
  --num-gpus 8
```

Relevant example:

- [examples/benchmark-plan-example.json](examples/benchmark-plan-example.json)
- [examples/lightning-h100-live-smoke-benchmark-plan.json](examples/lightning-h100-live-smoke-benchmark-plan.json)
- [examples/artifact-manifest-example.yaml](examples/artifact-manifest-example.yaml)

Focus on:

- `run_plan`
- `support`
- `production_target`
- `preflight_validation`

### Probe artifact comparison

```bash
uv run inferscope benchmark-compare before.json after.json
```

Relevant examples:

- [examples/benchmark-artifact-baseline.json](examples/benchmark-artifact-baseline.json)
- [examples/benchmark-artifact-candidate.json](examples/benchmark-artifact-candidate.json)
- [examples/benchmark-comparison-example.json](examples/benchmark-comparison-example.json)
- [examples/lightning-h100-live-smoke-benchmark-artifact.json](examples/lightning-h100-live-smoke-benchmark-artifact.json)

Focus on:

- `summary`
- `compatibility`
- `deltas`
- `ratios`
- `baseline.lane` / `candidate.lane`

The benchmark artifact examples now also carry explicit provenance:

- `provenance.workload`
- `provenance.experiment`
- `provenance.lane`

If you are benchmarking from local model or engine directories, the checked-in
artifact manifest example shows the minimal contract for `--artifact-manifest`:

- `artifact_kind`
- `model`
- `engine`
- `quantization`
- `tensor_parallel_size`
- `gpu_family`

## Notes

- The benchmark artifact examples are valid against the checked-in `BenchmarkArtifact` model.
- The comparison example is generated from those two artifact shapes and is meant to stay close to the real comparison helper.
- The Lightning runtime profile and benchmark artifact exports are also validated in the test suite.
- The Kimi/Dynamo production reference summary defines the canonical checked-in acceptance corpus for the production-validated lane.
- If the product contract changes, update these examples along with the docs and validation notes.
