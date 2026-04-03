"""Deploy Kimi-K2.5 on Modal H200 SXM (Hopper) with vLLM + prefix caching.

NOTE: H200 required (141 GiB HBM3e). Kimi K2.5 is 671B total MoE params; at FP8
that's ~84 GiB/GPU with TP=8 — exceeds H100 80GiB. H200 has 60+ GiB headroom for KV.

Cost controls (enforced — not optional):
    serve.timeout=1800        hard 30-min container kill
    serve.scaledown_window=60 scales to 0 within 60s of no traffic
    download_weights.gpu=None CPU-only download — zero billing during weight pull

Workflow (two commands, no CLI connection required after):
    # Step 1 — deploy + kick off weight download (exits in seconds):
    modal deploy demo/modal_kimi_k2_hopper.py
    modal run demo/modal_kimi_k2_hopper.py

    # Step 2 — once volume shows 61/61 shards, trigger the benchmark:
    cd products/isb1
    uv run isb1 quick-bench https://ocwc22--kimi-k2-hopper-serve.modal.run \\
        --workload swebench --requests 20 --context-bucket 32k --sessions 4

Check download progress:
    modal volume ls kimi-k2-weights /Kimi-K2-Instruct | grep safetensors | wc -l

Volume: kimi-k2-weights (persists across deploys — weights downloaded once)
Endpoint: https://ocwc22--kimi-k2-hopper-serve.modal.run
"""

import modal

MODEL_ID = "moonshotai/Kimi-K2-Instruct"
APP_NAME = "kimi-k2-hopper"
_WEIGHT_DIR = "/model-weights/Kimi-K2-Instruct"
_SHARD_COUNT = 61  # Kimi-K2.5 safetensors shards

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.8.5",              # pinned — 0.18.0 has trust_remote_code regression
        "transformers==4.51.1",     # pinned — minimum vLLM 0.8.5 requires; later 4.51.x/4.52+ breaks DeepseekVLV2Config dataclass ordering
        "huggingface-hub[hf_transfer]>=0.20.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
)

app = modal.App(APP_NAME)

# Persistent — weights survive app stop/restart, no re-download
model_volume = modal.Volume.from_name("kimi-k2-weights", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=None,               # CPU only — zero H100 billing during weight pull
    timeout=21600,          # 6 hrs — 420GB at authenticated speeds takes ~1hr, keep headroom
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/model-weights": model_volume},
)
def download_weights() -> None:
    """Pull Kimi-K2.5 weights to Modal Volume. Idempotent — safe to re-run."""
    import os
    from huggingface_hub import snapshot_download

    dest = _WEIGHT_DIR
    if os.path.exists(dest):
        shard_count = len([f for f in os.listdir(dest) if f.endswith(".safetensors")])
        print(f"Found {shard_count}/{_SHARD_COUNT} safetensors shards at {dest}.")
        if shard_count >= _SHARD_COUNT:
            print("Weights complete — nothing to do.")
            return
        print("Partial download — resuming.")

    print(f"Downloading {MODEL_ID} → {dest} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=dest,
        ignore_patterns=["*.pt", "*.bin"],  # safetensors only
    )
    model_volume.commit()
    shard_count = len([f for f in os.listdir(dest) if f.endswith(".safetensors")])
    print(f"Done — {shard_count}/{_SHARD_COUNT} shards committed to volume.")


@app.local_entrypoint()
def main() -> None:
    """Spawn weight download (CPU-only) and exit immediately.

    The download runs independently on Modal — no CLI connection required.
    Run this once after `modal deploy`. Re-running is safe (idempotent).
    """
    print("Spawning weight download on Modal (CPU-only, no H100 billing)...")
    download_weights.spawn()
    print("Done. Download running independently — close your terminal freely.")
    print("Track progress: modal volume ls kimi-k2-weights /Kimi-K2-Instruct | grep safetensors | wc -l")
    print(f"Serve endpoint (idle until triggered): https://ocwc22--kimi-k2-hopper-serve.modal.run")


@app.function(
    image=vllm_image,
    gpu="H200:8",           # H200 required — 671B FP8 model needs ~84 GiB/GPU, H100 80GiB OOMs
    # COST CONTROLS — do not remove
    timeout=1800,           # hard 30-min kill — no accidental overnight billing
    scaledown_window=60,    # scale to 0 within 60s of last request
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/model-weights": model_volume},
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    import subprocess

    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", _WEIGHT_DIR,
        "--download-dir", "/model-weights",
        "--port", "8000",
        "--tensor-parallel-size", "8",
        "--dtype", "auto",               # native fp8 from checkpoint
        "--kv-cache-dtype", "fp8",
        "--max-model-len", "131072",     # 128K context — H200 has headroom
        "--gpu-memory-utilization", "0.92",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-seqs", "64",
        "--served-model-name", "Kimi-K2.5",
        "--trust-remote-code",
    ])
