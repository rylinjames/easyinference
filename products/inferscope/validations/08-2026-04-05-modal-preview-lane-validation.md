# Modal Preview-Lane Validation

Date: 2026-04-05

## Goal

Validate the new `validate-production-lane` command against a live Modal smoke
artifact, and confirm that the command rejects preview-smoke runs even when the
underlying replay succeeds.

## Live endpoint

- Modal app: `easyinference-demo`
- URL: `https://hikaflow--easyinference-demo-serve.modal.run`
- GPU: `A10G`
- Served model alias: `Qwen2.5-7B-Instruct`
- Backing weights: `Qwen/Qwen2.5-7B-Instruct`

## Repo change required for the live run

The original Modal demo exposed `Qwen/Qwen2.5-7B-Instruct` as the served model
name, while the packaged `coding-smoke` workload targets `Qwen2.5-7B-Instruct`.
That mismatch caused zero successful replay requests. The demo was updated to
serve the short alias that InferScope and ISB-1 expect.

Files changed for that fix:

- `demo/modal_vllm.py`
- `demo/run_low_cost_smoke.sh`

## Commands used

Deploy:

```bash
cd EasyInference
/tmp/modal-setup-venv/bin/python -m modal deploy demo/modal_vllm.py
```

Warm:

```bash
python3 - <<'PY'
import urllib.request
with urllib.request.urlopen("https://hikaflow--easyinference-demo-serve.modal.run/v1/models", timeout=180) as r:
    print(r.status)
    print(r.read().decode("utf-8")[:500])
PY
```

Benchmark:

```bash
cd EasyInference/products/inferscope
PYTHONPATH=src uv run python -m inferscope.cli benchmark \
  coding-smoke \
  https://hikaflow--easyinference-demo-serve.modal.run \
  --gpu a10g \
  --num-gpus 1 \
  --metrics-endpoint https://hikaflow--easyinference-demo-serve.modal.run \
  --output docs/examples/modal-a10g-production-lane-check-artifact.json
```

Validate:

```bash
cd EasyInference/products/inferscope
PYTHONPATH=src uv run python -m inferscope.cli validate-production-lane \
  docs/examples/modal-a10g-production-lane-check-artifact.json
```

## Live benchmark result

- Artifact: `docs/examples/modal-a10g-production-lane-check-artifact.json`
- Requests: `2/2` succeeded
- Wall time: `7937.9 ms`
- p95 latency: `6480.4 ms`
- p95 TTFT: `984.5 ms`
- Output throughput: `23.94 tok/s`
- Metrics capture: complete

## Production-lane validation result

The new validator correctly rejected the artifact for production-lane claims.

Reasons:

1. Lane class was `preview_smoke`, not `production_validated`.
2. Claim scope was `toolchain_validation_only`, not `production_comparable`.
3. The artifact did not target `dynamo_long_context_coding`.

Important nuance:

- `production_readiness.ready` was still `true`.
- `preflight_validation.valid` was still `true`.

This is the intended behavior: a smoke run can be healthy and reproducible while
still being ineligible for production-lane claims.
