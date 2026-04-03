# AGENTS.md — Machine-Readable Onboarding for Coding Agents

This file helps AI coding agents (Claude Code, Codex, Copilot Workspace, etc.) navigate the EasyInference monorepo without prior context.

## Repository Identity

- **Name:** EasyInference
- **Remote:** https://github.com/OCWC22/EasyInference.git
- **Structure:** Two-product monorepo with a shared root for docs, CI, and Make targets
- **Primary language:** Python 3.11+
- **Package manager:** `uv` (InferScope), `pip` (ISB-1)

## Products

| Product | Path | Purpose | Package Manager | CLI Entry |
|---------|------|---------|----------------|-----------|
| **InferScope** | `products/inferscope/` | Operator CLI + MCP for inference tuning, profiling, benchmarks | `uv` | `inferscope` |
| **ISB-1** | `products/isb1/` | Reproducible benchmark standard, harness, analysis | `pip` | `isb1` |

**`inferscope-bench/`** is a local donor harness (InferenceX fork). It is NOT a product — do not modify it for feature work. Its ideas are absorbed into `products/inferscope/src/inferscope/benchmarks/`.

## Routing: Where to Make Changes

| Task | Target Directory |
|------|-----------------|
| Recommendation / optimization / engine logic | `products/inferscope/src/inferscope/optimization/`, `engines/` |
| Benchmark workloads, replay, artifacts | `products/inferscope/src/inferscope/benchmarks/` |
| Runtime profiling | `products/inferscope/src/inferscope/profiling/` |
| GPU / model metadata | `products/inferscope/src/inferscope/hardware/`, `models/` |
| CLI commands | `products/inferscope/src/inferscope/cli*.py` |
| MCP tools | `products/inferscope/src/inferscope/server*.py` |
| ISB-1 benchmark harness | `products/isb1/harness/` |
| ISB-1 workload generators | `products/isb1/workloads/` |
| ISB-1 analysis / reporting | `products/isb1/analysis/` |
| Root docs / CI / Makefile | Repository root |

## Dependency Direction (InferScope)

```
hardware ─┐
models ───┤
           ├─→ optimization ─→ engines
           │         │
           │         ▼
           ├─→ telemetry ─→ profiling
           │         │
           │         ▼
           └─→ benchmarks ─→ tools
                     │
                     ▼
              cli*.py / server*.py
```

- `optimization` does NOT depend on `benchmarks`
- `benchmarks` depends on `telemetry` and `optimization`
- CLI and MCP compose everything — they are leaf nodes

## GPU Platform Coverage

| Vendor | Supported GPUs | ISA | Status |
|--------|---------------|-----|--------|
| NVIDIA Hopper | H100, H200, GH200 | sm_90a | Production |
| NVIDIA Blackwell | B200, B300, GB200, GB300 | sm_100, sm_103 | Production |
| AMD CDNA3 | MI300X | gfx942 | Day-one support |
| AMD CDNA4 | MI355X | gfx950 | Day-one support |

NVIDIA Hopper/Blackwell is the primary validation path. AMD is supported for planning, benchmark gating, and support assessment.

## Development Setup

```bash
# InferScope (most common)
cd products/inferscope
uv sync --dev
cp .env.example .env  # optional, for INFERSCOPE_DEBUG etc.

# ISB-1
cd products/isb1
pip install -e ".[dev,quality]"
```

## Validation Commands

### InferScope (run from `products/inferscope/`)

```bash
uv run ruff check src/ tests/          # lint
uv run ruff format --check src/ tests/  # format check
uv run mypy src/inferscope/             # type check
uv run pytest tests/ -v --tb=short      # unit tests
uv run bandit -r src/inferscope/ -c pyproject.toml -ll  # security
```

### ISB-1 (run from `products/isb1/`)

```bash
python -m ruff check .
python -m black --check .
pytest tests/ -v --tb=short
```

### From the monorepo root

```bash
make all-checks  # runs both products
```

## Test Conventions (InferScope)

- Tests live in `products/inferscope/tests/`
- `conftest.py` adds `src/` to `sys.path` — no editable install required
- Use `pytest.mark.asyncio` for async tests
- Use `httpx.MockTransport` for mocking HTTP endpoints (not `unittest.mock.patch`)
- Use `pytest.mark.integration` for cross-module tests
- Use `pytest.mark.live_engine` for tests requiring a running inference endpoint
- Never import from `inferscope-bench/` — it is not on the Python path
- Test files follow `test_<module>_<area>.py` naming

## Commit and PR Conventions

- Branch from `main`
- Keep scope tight — one concern per PR
- Update docs when CLI or MCP behavior changes
- Run the relevant product validation before opening a PR
- Do not commit `.env` files, `inferscope-bench.tar.gz`, or the `inferscope-bench/` donor tree

## Files to Never Modify for Feature Work

- `inferscope-bench/` — donor harness, read-only reference
- `inferscope-bench.tar.gz` — archive of the donor
- `.env` files — local configuration, gitignored
- `uv.lock` — managed by `uv`, do not hand-edit

## Key Contracts

- `WorkloadPack` — the standard benchmark workload container
- `BenchmarkArtifact` — the standard benchmark output container
- `ServingProfile` → `ConfigCompiler` → `EngineConfig` — the recommendation pipeline
- `MetricSnapshot` — shared telemetry schema used by both profiling and benchmarks
- Packaged workloads: `src/inferscope/benchmarks/workloads/*.yaml`
- Packaged experiments: `src/inferscope/benchmarks/experiment_specs/*.yaml`
