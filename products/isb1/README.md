# ISB-1: Inference Serving Benchmark Standard 1

**Version 1.0.0** | **License: Apache-2.0**

ISB-1 is the benchmark standard inside EasyInference. It exists to produce reproducible, reviewable measurements of LLM inference serving behavior under realistic workload families.

Within this monorepo:

- **InferenceX** remains the external public reference for market-wide hardware and framework comparisons.
- **ISB-1** is the controlled benchmark standard for methodology, replay, publication, and validation.
- **InferScope** is the operator-facing CLI and MCP product that consumes benchmark ideas and exposes replay workflows.

The local `inferscope-bench/` tree is a donor foundation for workload patterns. It is not a third product and is not the public benchmark surface.

---

## What ISB-1 is for

ISB-1 is designed for teams that need:

- deterministic workload traces
- reproducible benchmark runs with lockfiles and manifests
- neutral workload families that can be reused across publications and internal reviews
- a path from raw replay data to aggregated metrics, claims, and reports

Today the execution harness launches a vLLM server and measures it through an internal OpenAI-compatible replay client. The methodology is broader than that single implementation, but the current product code is explicit about the runtime it validates.

---

## Benchmark families

ISB-1 defines four canonical workload families:

| Family | Purpose | Notes |
| --- | --- | --- |
| `chat` | high-concurrency conversational serving | short-to-medium context, throughput-oriented |
| `agent` | tool-calling and MCP-style workflows | structured tool schemas, growing context |
| `rag` | long-context retrieval serving | prefill-heavy, TTFT-sensitive |
| `coding` | repository-context assistance | long-context coding and prefix reuse |

These families are intentionally stable. More specific scenarios can sit on top of them without fragmenting the standard.

Examples of downstream mapping:

- InferScope `tool-agent` maps into the **agent** family.
- InferScope `coding-long-context` maps into the **coding** family.

---

## Execution model

The current ISB-1 implementation has four phases.

1. **Materialize traces** from workload configs into deterministic JSONL request pools.
2. **Launch the serving stack** for a benchmark cell.
3. **Replay requests** through the internal OpenAI-compatible client while collecting telemetry and engine metrics.
4. **Aggregate and publish** the resulting metrics, manifests, and lockfiles.

Important implementation details:

- traces are generated from `configs/workloads/*.yaml`
- `trace.jsonl` is persisted for each run
- the replay path is internal to ISB-1 and no longer depends on `vllm.benchmarks.benchmark_serving`
- lockfiles record benchmark runner identity and trace hashes for reproduction

---

## Quick start

```bash
git clone <repository-url>
cd EasyInference/products/isb1
pip install -e ".[dev,quality]"

# validate config shape
isb1 validate --sweep configs/sweep/core.yaml

# inspect the planned matrix
isb1 plan --config configs/sweep/core.yaml

# run a single cell
isb1 run-cell \
  --gpu h100 \
  --model llama70b \
  --workload chat \
  --mode mode_a \
  --quantization fp8 \
  --output results/

# aggregate finished runs
isb1 analyze --results-dir results/ --output analysis.json
```

For a fuller walkthrough, see [docs/QUICKSTART.md](docs/QUICKSTART.md).

---

## Repository layout

```text
products/isb1/
├── workloads/      # canonical generators and trace materialization
├── harness/        # replay execution, server lifecycle, telemetry, manifests
├── analysis/       # metric computation and statistics
├── quality/        # quality-side checks
├── configs/        # GPUs, models, workloads, sweep definitions
├── publication/    # report templates
├── scripts/        # setup and utility scripts
└── tests/          # benchmark-local regression tests
```

Key modules:

- `workloads/materialize.py` — generates deterministic request pools from workload configs
- `harness/replay_client.py` — OpenAI-compatible replay runner with TTFT / token timestamp capture
- `harness/runner.py` — single-cell lifecycle, trace persistence, manifest and lockfile production
- `analysis/metrics.py` — raw-result to aggregated-metric computation

---

## Output model

Each benchmark cell produces:

- a persisted request trace
- raw per-rate replay result JSON
- a manifest describing the run
- a lockfile capturing software, hardware, config hashes, benchmark runner, and trace hash
- optional telemetry and engine-metric snapshots

Primary metrics include:

- TTFT, TPOT, ITL, and E2E latency percentiles
- request throughput and generation throughput
- goodput and SLO attainment
- prefix-cache, queue-depth, and KV cache signals where available
- power and efficiency metrics where available

---

## Relationship to InferScope

InferScope is the product that turns these ideas into an operator workflow.

- ISB-1 owns the benchmark standard and canonical workload families.
- InferScope packages practical built-in workloads and experiments for CLI/MCP use.
- InferScope reuses the benchmark concepts for recommendation validation, artifact comparison, and endpoint replay.

If you want benchmark methodology or benchmark publication work, start here.
If you want benchmark replay through an MCP, start in `products/inferscope/`.

---

## Documentation

- [docs/QUICKSTART.md](docs/QUICKSTART.md)
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ECOSYSTEM.md](docs/ECOSYSTEM.md)
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

## Validation

```bash
python -m ruff check .
python -m black --check .
python -m pytest tests/ -v --tb=short
python -m harness.config_validator --all-yaml --config-root configs
python -m harness.config_validator --sweep configs/sweep/core.yaml --config-root configs
```

---

## License

Apache-2.0
