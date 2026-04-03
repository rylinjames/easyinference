# Validation Reports

InferScope validation reports are versioned snapshots.

Within the EasyInference monorepo, this product lives at `products/inferscope/`.

## Naming convention

```text
NN-YYYY-MM-DD-name.md
```

## Latest reports

- [01-2026-03-23-ai-first-validation.md](validations/01-2026-03-23-ai-first-validation.md)
- [02-2026-03-23-benchmark-and-stress-test-plan.md](validations/02-2026-03-23-benchmark-and-stress-test-plan.md)
- [03-2026-03-25-runtime-profiling-v1.md](validations/03-2026-03-25-runtime-profiling-v1.md)
- [04-2026-03-25-hopper-blackwell-hardening.md](validations/04-2026-03-25-hopper-blackwell-hardening.md)

## Current validation contract

As of **March 28, 2026**, InferScope should be considered valid when the checks below pass.

### Automated checks (copy-paste runnable)

Run these from `products/inferscope/`:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/inferscope/
uv run pytest tests/ -v --tb=short
uv run bandit -r src/inferscope/ -c pyproject.toml -ll
```

Or from the monorepo root:

```bash
make inferscope-lint inferscope-typecheck inferscope-security inferscope-test inferscope-package-smoke
```

The `inferscope-package-smoke` target builds a wheel, installs it in an isolated venv, and verifies that packaged benchmark workloads and experiment specs are loadable.

### Coverage expectations (reviewer checklist)

The test suite should cover the following areas. If your change touches any of these, verify the relevant tests still pass or add new ones:

- [ ] Runtime profiling: telemetry capture, profiling orchestration, CLI registration, MCP-safe wrapper behavior
- [ ] Production target contract: `production_target.py` remains the single scope authority
- [ ] Supported InferScope lane: `Kimi-K2.5` on `h100`, `h200`, `b200`, `b300`
- [ ] Benchmark probe resolution: shared `probe_resolution.py` logic for CLI + MCP
- [ ] Benchmark support: GPU/model/topology gating for the narrowed probe lane
- [ ] Benchmark runtime: TTFT / TPOT / ITL / throughput / session-failure semantics
- [ ] Procedural benchmark expansion: packaged built-ins only, with CLI-only `context_file`
- [ ] MCP narrowing: no benchmark matrix / strategy / stack-plan tools remain in the public server surface

## Validation report cross-references

| Validation Report | Related Changelog Entries |
|-------------------|--------------------------|
| [01 — AI-first validation](validations/01-2026-03-23-ai-first-validation.md) | Initial release `[0.1.0]` |
| [02 — Benchmark stress test plan](validations/02-2026-03-23-benchmark-and-stress-test-plan.md) | Initial release `[0.1.0]` |
| [03 — Runtime profiling v1](validations/03-2026-03-25-runtime-profiling-v1.md) | `[Unreleased]` runtime profiling, narrowed probe surface |
| [04 — Hopper/Blackwell hardening](validations/04-2026-03-25-hopper-blackwell-hardening.md) | `[Unreleased]` platform policy, scope hardening, compiler regression |

## Notes

- Treat files in `validations/` as point-in-time reports, not permanent guarantees.
- If docs drift from code, trust the current packaged CLI and MCP surfaces.
- Procedural benchmark generation is limited to selected packaged built-ins, not arbitrary YAML file paths.
- Runtime profiling is Prometheus-first in v1 and does not persist profiles to disk by default.
- InferScope's public benchmark and MCP contract is now intentionally narrower than the broader planning code still present in the repo.
- If docs drift from code, trust `production_target.py`, the retained CLI benchmark commands, and the retained MCP benchmark tools.
