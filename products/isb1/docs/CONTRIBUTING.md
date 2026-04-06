# Contributing to ISB-1

ISB-1 is the benchmark standard inside EasyInference. This guide covers both benchmark code changes and Mode C operator submissions.

## Scope

Use `products/isb1/` when your change affects:

- canonical benchmark workload families
- trace materialization or replay methodology
- manifests, lockfiles, metrics, or analysis
- benchmark publication templates or claim evaluation

If your goal is operator-facing CLI or MCP replay, use `products/inferscope/` instead.

## Mode C submissions

Mode C exists for operator- or vendor-submitted serving configurations evaluated under the same benchmark methodology.

A submission should include:

- operator metadata
- the targeted GPU / model / workload cells
- the engine arguments or launch settings required for reproduction
- any notes needed to understand the optimization strategy

Submission rules:

1. the configuration must be reproducible
2. the configuration must not change benchmark workload semantics
3. the configuration must not bypass warmup, replay, or instrumentation
4. the configuration must pass `isb1 validate`

## Canonical workload taxonomy

ISB-1 keeps a stable family taxonomy:

- `chat`
- `agent`
- `rag`
- `coding`

More specific scenario names should map back to one of these families instead of creating a parallel benchmark taxonomy.

Examples:

- MCP / tool-calling scenarios belong to `agent`
- long-context repository review belongs to `coding`

## Development setup

```bash
git clone <repository-url>
cd EasyInference/products/isb1
uv sync --dev --extra quality --no-editable
```

## Required local checks

```bash
uv run --no-sync ruff check .
uv run --no-sync black --check .
uv run --no-sync python -m pytest tests/ -v --tb=short
uv run --no-sync isb1 validate --all-yaml --config-root configs
uv run --no-sync isb1 validate --sweep configs/sweep/core.yaml --config-root configs
```

## Pull request expectations

1. keep changes scoped
2. update docs when public benchmark behavior changes
3. include regression tests for replay, trace materialization, or analysis changes
4. call out compatibility or methodology impact in the PR description

## Reporting issues

When reporting a benchmark issue, include:

- the workload, model, and GPU involved
- the config files or sweep used
- the manifest and lockfile if available
- the raw replay output or a reduced reproducer
