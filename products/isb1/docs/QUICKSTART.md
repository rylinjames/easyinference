# ISB-1 Quick Start

This guide gets you from install to a first reproducible benchmark run.

## Prerequisites

- Python 3.10+
- NVIDIA GPU(s) supported by the configs you intend to run
- CUDA and driver stack compatible with your vLLM install
- access to the model weights you plan to serve

## Install

```bash
git clone https://github.com/OCWC22/EasyInference.git
cd EasyInference/products/isb1
pip install -e ".[dev,quality]"
```

Confirm the CLI is available:

```bash
isb1 --help
```

## Validate the benchmark configs

```bash
isb1 validate --sweep configs/sweep/core.yaml
isb1 validate --all-yaml --config-root configs
```

This checks the benchmark matrix, config integrity, and rough memory-fit constraints.

## Preview the execution plan

```bash
isb1 plan --config configs/sweep/core.yaml
```

Use this before long runs. It lets you inspect the matrix without launching anything.

## Run a single cell

```bash
isb1 run-cell \
  --gpu h100 \
  --model llama70b \
  --workload chat \
  --mode mode_a \
  --quantization fp8 \
  --output results/
```

What happens during a cell run:

1. configs are loaded and validated
2. a deterministic request pool is materialized from the workload config
3. `trace.jsonl` is written into the run directory
4. the serving stack is launched
5. the internal replay client executes the configured rate sweep
6. manifests, raw results, and lockfiles are written to disk

## Run the core sweep

```bash
isb1 run --config configs/sweep/core.yaml --output results/
```

Add `--dry-run` if you want the orchestrator plan without execution.

## Aggregate results

```bash
isb1 analyze --results-dir results/ --output analysis.json
isb1 claims --results-dir results/
isb1 leaderboard --analysis analysis.json
isb1 report --analysis analysis.json --output report.html
```

## What to inspect after a run

Look at:

- the saved `trace.jsonl`
- raw per-rate result JSON files
- `manifest.json`
- the generated lockfile
- aggregated analysis output

Those files are the minimum reproducibility surface for a benchmark claim or an internal rollout review.

## Where to go next

- [METHODOLOGY.md](METHODOLOGY.md) — benchmark contract and metrics
- [ARCHITECTURE.md](ARCHITECTURE.md) — code structure and lifecycle
- [ECOSYSTEM.md](ECOSYSTEM.md) — relationship to InferenceX and InferScope
- [CONTRIBUTING.md](CONTRIBUTING.md) — submissions and code changes
