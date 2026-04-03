# Runtime Profiling v1 Validation

Date: **March 25, 2026**

## Scope

This validation covers the first production runtime profiling surface for InferScope:

- shared telemetry capture models reused by benchmarks and profiling
- shared profiling core under `src/inferscope/profiling/`
- additive CLI command: `profile-runtime`
- additive MCP helper/tool path for runtime profiling
- refactoring of diagnose/audit/live-tuner onto the shared runtime analysis path

## Automated validation

Validated in `products/inferscope/`:

```bash
uv run ruff check src tests
uv run pytest tests -q
```

Observed result:

- `ruff check` ✅
- `pytest` ✅ (`21 passed`)

## Coverage included

- telemetry snapshot capture compatibility
- expected-engine mismatch handling
- runtime profiling report structure
- bottleneck grouping against the shared bottleneck enum
- degraded identity-enrichment handling
- legacy tool response-shape compatibility
- CLI registration for `profile-runtime`
- MCP-safe runtime profiling helper behavior

## Manual validation checklist

- [ ] local vLLM endpoint profile via CLI
- [ ] MCP private-endpoint rejection
- [ ] auth-protected metrics endpoint
- [ ] `/metrics` input still resolves adapter base URL correctly
- [ ] adapter `/v1/models` failure degrades without breaking the report
- [ ] benchmark artifact telemetry remains readable and unchanged in shape

## Notes

- Runtime profiling is Prometheus-first in v1.
- Runtime profiles are not persisted to disk by default.
- Direct `nsys` / `rocprofv3` execution remains future work behind the profiling boundary.
