# EasyInference

EasyInference is a two-product monorepo.

- **ISB-1** — the benchmark standard, harness, configs, analysis, and publication layer in `products/isb1/`
- **InferScope** — the operator-facing runtime profiling and narrow probe product in `products/inferscope/`

## Product positioning

EasyInference is designed to be complementary to public frontier benchmarking, not a clone of it.

- **InferenceX** owns the broad public frontier benchmark problem
- **ISB-1** owns reproducible benchmark methodology in this repo
- **InferScope** owns deployment-specific diagnostics, profiling, and probe execution

That separation matters.

This repo should not collapse into one generic benchmark-and-MCP playground.

## What each product is for

### ISB-1

Use `products/isb1/` when you need:

- reproducible benchmark configs and manifests
- benchmark harness execution
- analysis, statistics, and publication support
- broad workload-family handling across chat, coding, agent, and RAG scenarios

### InferScope

Use `products/inferscope/` when you need:

- live runtime profiling of a deployment
- narrow benchmark probe execution against a real endpoint
- artifact comparison before and after a change
- an MCP surface for production-truth diagnostics

InferScope is intentionally narrower than a generic benchmark product.
It is being shaped around KV-cache, offload, and disaggregated-serving analysis.

## Repository layout

```text
EasyInference/
├── products/
│   ├── isb1/         # benchmark standard and harness
│   └── inferscope/   # operator diagnostics, profiling, and probe tooling
├── demo/
├── .github/workflows/
├── docs/
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── VALIDATION.md
└── Makefile
```

## Quick start

### ISB-1

```bash
cd products/isb1
pip install -e ".[dev,quality]"
pytest tests/ -v --tb=short
```

### InferScope

```bash
cd products/inferscope
uv sync --dev
PYTHONPATH=src uv run python -m inferscope.cli profile-runtime http://localhost:8000
PYTHONPATH=src uv run python -m inferscope.cli benchmark-plan kimi-k2-long-context-coding http://localhost:8000 --gpu b200 --num-gpus 8
```

For the full guided onboarding path, use [products/inferscope/docs/QUICKSTART.md](products/inferscope/docs/QUICKSTART.md).

## Monorepo rules

- Two products only: **ISB-1** and **InferScope**
- `inferscope-bench/` is a donor harness, not a product surface
- product docs and CI should preserve the ISB-1 vs InferScope boundary
- benchmark-standard work belongs in ISB-1
- operator-diagnostics work belongs in InferScope

## Documentation

- [Monorepo architecture](ARCHITECTURE.md)
- [InferScope README](products/inferscope/README.md)
- [InferScope architecture](products/inferscope/ARCHITECTURE.md)
- [ISB-1 README](products/isb1/README.md)

## License

This repository contains multiple licenses:

- `products/isb1/` — Apache-2.0
- `products/inferscope/` — MIT
