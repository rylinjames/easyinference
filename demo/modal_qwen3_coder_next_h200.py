"""Deploy Qwen3-Coder-Next on Modal H200 SXM with vLLM for KV cache benchmarking.

Qwen3-Coder-Next (80B-A3B hybrid MoE):
  - 80B total, 3B active per token, 512 experts (10+1 active)
  - Hybrid attention: 12 Gated Attention layers + 36 Gated DeltaNet layers
  - KV cache: 24 KB/token (BF16, 12 attention layers only)
  - DeltaNet state: ~18 MB/sequence (fixed, independent of seq length)
  - FP8 KV NOT supported (vLLM #26646) — BF16 KV required
  - TP=1 fits on single H200 at FP8 (~80 GB weights, 48 GB KV headroom)

Cost controls (enforced):
    serve.timeout=1800        hard 30-min container kill
    serve.scaledown_window=60 scales to 0 within 60s of no traffic

Workflow:
    # Deploy + download weights:
    modal deploy demo/modal_qwen3_coder_next_h200.py
    modal run demo/modal_qwen3_coder_next_h200.py

    # Check download progress:
    modal volume ls qwen3-coder-next-weights /Qwen3-Coder-Next | wc -l

    # Benchmark with InferScope:
    cd products/inferscope
    uv run inferscope  # MCP server — connect from Cursor/Claude

    # Or quick smoke test:
    cd products/isb1
    uv run isb1 quick-bench https://ocwc22--qwen3-coder-next-h200-serve.modal.run/v1 \\
        --workload coding --requests 10

Endpoint: https://ocwc22--qwen3-coder-next-h200-serve.modal.run
"""

import modal

MODEL_ID = "Qwen/Qwen3-Coder-Next"
MODEL_ID_FP8 = "Qwen/Qwen3-Coder-Next-FP8"
APP_NAME = "qwen3-coder-next-h200"
_WEIGHT_DIR = "/model-weights/Qwen3-Coder-Next-FP8"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .pip_install(
        "vllm>=0.10.0",
        "transformers>=4.51.0",
        "huggingface-hub[hf_transfer]>=0.20.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "CUDA_HOME": "/usr/local/cuda",
    })
)

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name("qwen3-coder-next-weights", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=None,
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/model-weights": model_volume},
)
def download_weights() -> None:
    """Pull Qwen3-Coder-Next-FP8 weights to Modal Volume. Idempotent."""
    import os
    from huggingface_hub import snapshot_download

    dest = _WEIGHT_DIR
    if os.path.exists(dest):
        safetensors = [f for f in os.listdir(dest) if f.endswith(".safetensors")]
        print(f"Found {len(safetensors)} safetensors shards at {dest}.")
        if len(safetensors) >= 5:
            print("Weights look complete — nothing to do.")
            return
        print("Partial download — resuming.")

    print(f"Downloading {MODEL_ID_FP8} → {dest} ...")
    snapshot_download(
        repo_id=MODEL_ID_FP8,
        local_dir=dest,
        ignore_patterns=["*.pt", "*.bin", "original/**"],
    )
    model_volume.commit()
    safetensors = [f for f in os.listdir(dest) if f.endswith(".safetensors")]
    print(f"Done — {len(safetensors)} shards committed to volume.")


@app.local_entrypoint()
def main() -> None:
    """Spawn weight download (CPU-only) and exit immediately."""
    print("Spawning weight download on Modal (CPU-only, no GPU billing)...")
    download_weights.spawn()
    print("Done. Download running independently — close your terminal freely.")
    print(f"Track: modal volume ls qwen3-coder-next-weights /Qwen3-Coder-Next-FP8 | wc -l")
    print(f"Endpoint: https://ocwc22--{APP_NAME}-serve.modal.run")


@app.function(
    image=vllm_image,
    gpu="H200",                 # Single H200 — 80B FP8 fits with 48 GB KV headroom
    timeout=1800,               # Hard 30-min kill
    scaledown_window=60,        # Scale to 0 within 60s
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/model-weights": model_volume},
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    import subprocess

    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", _WEIGHT_DIR,
        "--download-dir", "/model-weights",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "auto",
        # BF16 KV required — FP8 KV not supported for Qwen3-Next hybrid attention
        "--kv-cache-dtype", "auto",
        "--max-model-len", "131072",
        "--gpu-memory-utilization", "0.92",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-seqs", "64",
        "--served-model-name", "Qwen3-Coder-Next",
        "--trust-remote-code",
    ])
