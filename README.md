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

## Current State vs Long-Term Direction

### What exists today

- **ISB-1** is the benchmark-standard and workload-harness product in this repo.
- **InferScope** is the shipped operator surface for runtime profiling, narrow probe execution, artifact comparison, CLI usage, and MCP usage.
- The current supported human workflows are CLI-first and MCP-second.

### What we are building toward

EasyInference is intended to become a platform, not a terminal-only product.

- **Primary interface:** managed web product
- **Enterprise option:** self-hosted or local deployment of the same platform
- **Secondary interfaces:** CLI and MCP
- **Shared core:** one analysis engine and one API model underneath all interfaces

That means the web-first platform is the long-term product shape, while CLI and MCP are the strongest supported interfaces today.

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
uv sync --dev --no-editable
uv run inferscope profile-runtime http://localhost:8000
uv run inferscope benchmark-plan kimi-k2-long-context-coding http://localhost:8000 --gpu b200 --num-gpus 8
```

For the full guided onboarding path, use [products/inferscope/docs/QUICKSTART.md](products/inferscope/docs/QUICKSTART.md).
For the MCP-specific onboarding path, use [products/inferscope/docs/MCP_QUICKSTART.md](products/inferscope/docs/MCP_QUICKSTART.md).
For reference output examples, use [products/inferscope/docs/EXAMPLE_RESULTS.md](products/inferscope/docs/EXAMPLE_RESULTS.md).

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
