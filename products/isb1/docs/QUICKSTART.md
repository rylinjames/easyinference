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
uv sync --dev --extra quality --no-editable
```

Confirm the CLI is available:

```bash
uv run --no-sync isb1 --help
```

If you prefer a checked-in wrapper:

```bash
./scripts/isb1.sh --help
```

## Validate the benchmark configs

```bash
uv run --no-sync isb1 validate --sweep configs/sweep/core.yaml
uv run --no-sync isb1 validate --all-yaml --config-root configs
```

This checks the benchmark matrix, config integrity, and rough memory-fit constraints.

## Preview the execution plan

```bash
uv run --no-sync isb1 plan --config configs/sweep/core.yaml
```

Use this before long runs. It lets you inspect the matrix without launching anything.

## Run a single cell

```bash
uv run --no-sync isb1 run-cell \
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
uv run --no-sync isb1 run --config configs/sweep/core.yaml --output results/
```

Add `--dry-run` if you want the orchestrator plan without execution.

## Aggregate results

```bash
uv run --no-sync isb1 analyze --results-dir results/ --output analysis.json
uv run --no-sync isb1 claims --results-dir results/
uv run --no-sync isb1 leaderboard --analysis analysis.json
uv run --no-sync isb1 report --analysis analysis.json --output report.html
```

## Serverless smoke bench

Use this when you want a cheap validation run against a hosted OpenAI-compatible endpoint such as Modal.

```bash
uv run --no-sync isb1 quick-bench \
  https://<endpoint> \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --requests 1 \
  --duration 120

uv run --no-sync isb1 quick-bench \
  https://<endpoint> \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --workload coding \
  --requests 1 \
  --duration 120
```

Notes:

- `quick-bench` now retries model detection and sends a short warmup request by default
- remote and serverless endpoints get larger timeout defaults automatically
- pass `--no-warmup` or explicit timeout flags only when you have a reason to override the defaults

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
