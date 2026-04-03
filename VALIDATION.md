# EasyInference Validation

This repository validates two separate products.

## ISB-1 benchmark

Run benchmark-local checks from the monorepo root or inside `products/isb1/`.

From the root:

```bash
make validate
make isb1-lint
make isb1-format-check
make test
```

Directly in the product:

```bash
cd products/isb1
pip install -e ".[dev]"
python -m ruff check .
python -m black --check .
pytest tests/ -v --tb=short
```

## InferScope

```bash
make inferscope-lint
make inferscope-typecheck
make inferscope-security
make inferscope-package-smoke
make inferscope-test
```

Or directly in the product:

```bash
cd products/inferscope
uv sync --dev
uv run ruff check src/
uv run ruff format --check src/
uv run mypy src/inferscope/
uv run bandit -r src/inferscope/ -c pyproject.toml -ll
```

## CI workflows

As of **March 25, 2026**, the repository-level CI entrypoints are:
- `.github/workflows/isb1-ci.yml`
- `.github/workflows/inferscope-ci.yml`

## When to run both

Run both product validations when you change:
- root documentation
- root workflows
- root Make targets
- shared repository structure under `products/`
