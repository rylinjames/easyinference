# EasyInference Quick Start Index

This monorepo has two products. Choose the right quickstart for your task.

## InferScope (operator CLI + MCP)

Use InferScope if you need to recommend, validate, profile, benchmark, or tune inference serving.

```bash
cd products/inferscope
uv sync --dev
uv run inferscope --help
```

Full guide: [InferScope README](../products/inferscope/README.md)

## ISB-1 (benchmark standard)

Use ISB-1 if you need reproducible benchmark execution, analysis, claims, or publication.

```bash
cd products/isb1
pip install -e ".[dev,quality]"
isb1 --help
```

Full guide: [ISB-1 Quick Start](../products/isb1/docs/QUICKSTART.md)

## Prerequisites

- **Python 3.11+** (InferScope) or **Python 3.10+** (ISB-1)
- **[uv](https://docs.astral.sh/uv/)** — required for InferScope
- **pip** — required for ISB-1
- **NVIDIA GPU + CUDA** or **AMD GPU + ROCm** — only for live benchmark execution and profiling
- See the root [README](../README.md) for full details.
