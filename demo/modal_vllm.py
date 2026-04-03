"""Deploy vLLM on Modal for EasyInference smoke testing (Qwen2.5-7B on A10G).

Usage:
    modal deploy demo/modal_vllm.py

Cost controls (enforced — not optional):
    timeout=1800        hard 30-min container kill, no overnight billing
    scaledown_window=60 scales to 0 within 60s of no traffic

Endpoint URL:
    https://<workspace>--easyinference-demo-serve.modal.run

Run a quick smoke test:
    cd products/isb1
    uv run isb1 quick-bench https://<url>/v1 --workload simple --requests 10
"""

import modal

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

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
        "--port", "8000",
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
        "--max-model-len", "4096",
        "--dtype", "auto",
    ])
