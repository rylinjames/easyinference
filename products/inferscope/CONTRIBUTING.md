# Contributing to InferScope

InferScope is the operator-facing CLI and MCP product in EasyInference.

Use this product when your change affects:

- recommendation and validation flows
- operator diagnostics and audits
- packaged benchmark workloads or experiment specs
- benchmark replay through the CLI or MCP server
- benchmark artifact handling or comparison

## Development setup

```bash
git clone https://github.com/OCWC22/EasyInference.git
cd EasyInference/products/inferscope
uv sync --dev
cp .env.example .env  # optional
```

## Required local checks

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/inferscope/
uv run pytest tests/ -v --tb=short
```

If your change affects package security posture, also run:

```bash
uv run bandit -r src/inferscope/ -c pyproject.toml -ll
```

## Benchmark contribution rules

Packaged benchmark assets belong only under:

- `src/inferscope/benchmarks/workloads/`
- `src/inferscope/benchmarks/experiment_specs/`

Do not introduce repo-root mirrors of packaged built-ins.

If you modify the benchmark bridge, verify at minimum:

```bash
uv run inferscope benchmark-plan \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --gpu b200 \
  --num-gpus 8 \
  --synthetic-requests 2 || true

uv run inferscope benchmark \
  kimi-k2-long-context-coding \
  http://localhost:8000 \
  --experiment dynamo-aggregated-lmcache-kimi-k2 \
  --gpu b200 \
  --num-gpus 8 \
  --synthetic-requests 2 || true
```

The benchmark surface is intentionally narrow. Do not add matrix, strategy, or stack-plan abstractions back into the public CLI or MCP contract without revisiting the product boundary.

## Test conventions

Tests live in `tests/` and run with `uv run pytest tests/ -v --tb=short`.

**Key patterns:**

- `conftest.py` adds `src/` to `sys.path` so no editable install is required.
- Use `pytest.mark.asyncio` for all async test functions.
- Mock HTTP endpoints with `httpx.MockTransport` + `httpx.AsyncClient`, not `unittest.mock.patch`. Example from `test_benchmark_runtime.py`:

```python
def handler(request: httpx.Request) -> httpx.Response:
    return _sse_response({"choices": [{"delta": {"content": "ok"}}]})

client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
```

- Use `pytest.mark.integration` for broader cross-module tests.
- Use `pytest.mark.live_engine` for tests that require a running inference endpoint.
- Never import from `inferscope-bench/` — it is a donor harness and not on the Python path.
- Test file naming: `test_<module>_<area>.py` (e.g., `test_benchmark_support.py`).
- For benchmark tests, use `load_workload()` / `load_experiment()` to load packaged assets.
- Use `build_run_plan()` to construct run plans for replay tests.
- SSE streaming responses can be mocked with a `DelayedSSE(httpx.AsyncByteStream)` helper — see `test_benchmark_runtime.py`.

**Fixture and marker summary:**

| Marker | When to use |
|--------|------------|
| `@pytest.mark.asyncio` | Any `async def test_*` function |
| `@pytest.mark.integration` | Tests spanning multiple subsystems |
| `@pytest.mark.live_engine` | Tests requiring a real running serving endpoint |

## Pull request expectations

1. keep changes scoped
2. update docs when CLI or MCP behavior changes
3. add tests when benchmark plan resolution, replay contracts, or artifact structure changes
4. call out rollout implications in the PR description
5. run `uv run pytest tests/ -v --tb=short` and `uv run ruff check src/ tests/` before pushing
