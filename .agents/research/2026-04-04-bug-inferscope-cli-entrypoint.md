# Bug Report: InferScope CLI Entrypoint From Source Checkout

**Date:** 2026-04-04
**Symptom:** `uv run inferscope --help` failed from `products/inferscope/` with `ModuleNotFoundError: No module named 'inferscope'`

## Root Cause

- The source-checkout install path was relying on editable-install machinery in the virtualenv.
- On this macOS environment, the generated editable `.pth` file in `site-packages` was marked `hidden`, so Python startup skipped it.
- That prevented the source package from being added to `sys.path`, which broke the generated `inferscope` console entrypoint.

## Evidence

- Failing command before fix:
  - `uv run inferscope --help`
- Python startup trace showed:
  - `Skipping hidden .pth file: '.../_inferscope.pth'`
- File flags confirmed:
  - `.venv/lib/python3.12/site-packages/_inferscope.pth` had the macOS `hidden` flag

## Pattern Comparison

- Direct execution with an explicit source path worked:
  - `PYTHONPATH=src uv run python -m inferscope.cli --help`
- A non-editable install populated a real `site-packages/inferscope/` package and restored the console entrypoint path.

## Fix

1. Added a dedicated launcher package in `src/inferscope_launcher/`.
2. Switched the project script target in `pyproject.toml` to `inferscope_launcher:main`.
3. Added regression coverage for the generated console entrypoint in `tests/test_cli_entrypoint.py`.
4. Updated the docs to use the supported source-checkout install path:
   - `uv sync --dev --no-editable`
   - `uv run inferscope ...`

## Verification

- `uv sync --dev --no-editable`
- `.venv/bin/inferscope --help`
- `uv run inferscope --help`
- `uv run pytest tests/test_cli_entrypoint.py tests/test_cli_profiling.py tests/test_cli_benchmarks.py -v --tb=short`

## Remaining Caveat

- The default editable source-checkout path is still sensitive to the hidden `.pth` behavior in this environment.
- The documented supported path is now `uv sync --dev --no-editable` until that upstream/tooling behavior is addressed more cleanly.
