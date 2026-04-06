"""Deploy vLLM on Modal for EasyInference low-cost smoke validation.

Usage:
    modal deploy demo/modal_vllm.py

Cost controls (enforced — not optional):
    timeout=1800        hard 30-min container kill, no overnight billing
    scaledown_window=60 scales to 0 within 60s of no traffic

Endpoint URL:
    https://<workspace>--easyinference-demo-serve.modal.run

What this deploy guarantees:
    /v1/models         OpenAI-compatible model discovery
    /v1/chat/completions
    /metrics           Prometheus metrics emitted by the embedded vLLM server

Served model:
    Qwen2.5-7B-Instruct (backed by Qwen/Qwen2.5-7B-Instruct) on 1x A10G

Warm the endpoint once:
    curl -sS https://<workspace>--easyinference-demo-serve.modal.run/v1/models

Run a quick smoke test:
    cd products/isb1
    uv sync --dev --no-editable
    uv run --no-sync isb1 quick-bench \
      https://<url> \
      --model-id Qwen2.5-7B-Instruct \
      --workload coding \
      --requests 1 \
      --duration 120

InferScope smoke lane:
    cd products/inferscope
    uv sync --dev --no-editable
    uv run inferscope benchmark-plan coding-smoke https://<url> --gpu a10g --num-gpus 1
    uv run inferscope profile-runtime https://<url> --metrics-endpoint https://<url> --scrape-timeout-seconds 90
"""

import modal

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SERVED_MODEL_NAME = "Qwen2.5-7B-Instruct"

app = modal.App("easyinference-demo")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.8.0")
)


@app.function(
    image=vllm_image,
    gpu="A10G",
    # Hard 30-min cap — smoke test only, never leave running
    timeout=1800,
    # Scale to 0 within 60s of last request
    scaledown_window=60,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    import subprocess

    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--served-model-name", SERVED_MODEL_NAME,
        "--port", "8000",
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
        "--max-model-len", "4096",
        "--dtype", "auto",
    ])
