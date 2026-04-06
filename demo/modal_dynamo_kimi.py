"""Modal skeleton for a true InferScope Kimi/Dynamo production lane.

This is an architecture scaffold, not a turnkey one-click deploy.

What it gives you:
    - a persistent Volume for Kimi weights
    - a checked-in artifact manifest written into that Volume
    - one public ASGI service that proxies:
        /v1/*              -> local Dynamo frontend
        /frontend/metrics  -> local frontend Prometheus port
        /worker/metrics    -> local worker Prometheus port
    - a deployment shape that matches the InferScope aggregated production lane

What you still need to provide:
    - the actual Dynamo frontend launch command
    - the actual Dynamo worker launch command
    - any Dynamo/LMCache runtime images, wheels, or binaries you depend on

Required environment variables at deploy/run time:
    INFERSCOPE_DYNAMO_FRONTEND_CMD
    INFERSCOPE_DYNAMO_WORKER_CMD

Optional environment variables:
    INFERSCOPE_DYNAMO_FRONTEND_URL       default: http://127.0.0.1:8001
    INFERSCOPE_DYNAMO_FRONTEND_METRICS   default: <frontend-url>/metrics
    INFERSCOPE_DYNAMO_WORKER_METRICS     default: http://127.0.0.1:9200/metrics
    INFERSCOPE_DYNAMO_HEALTHCHECK_URL    default: <frontend-url>/metrics

Suggested operator flow:
    modal deploy demo/modal_dynamo_kimi.py
    modal run demo/modal_dynamo_kimi.py

    # After the stack is live:
    cd products/inferscope
    uv sync --dev --no-editable
    uv run inferscope benchmark \
      kimi-k2-long-context-coding \
      https://<workspace>--inferscope-kimi-dynamo-serve.modal.run \
      --experiment dynamo-aggregated-lmcache-kimi-k2 \
      --gpu h200 \
      --num-gpus 4 \
      --metrics-target frontend=https://<workspace>--inferscope-kimi-dynamo-serve.modal.run/frontend/metrics \
      --metrics-target worker=https://<workspace>--inferscope-kimi-dynamo-serve.modal.run/worker/metrics \
      --model-artifact-path /model-weights/Kimi-K2-Instruct \
      --artifact-manifest /model-weights/Kimi-K2-Instruct/artifact-manifest.yaml
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from pathlib import Path

import modal

MODEL_ID = "moonshotai/Kimi-K2-Instruct"
SERVED_MODEL_NAME = "Kimi-K2.5"
APP_NAME = "inferscope-kimi-dynamo"
WEIGHT_DIR = "/model-weights/Kimi-K2-Instruct"
MANIFEST_PATH = f"{WEIGHT_DIR}/artifact-manifest.yaml"
SHARD_COUNT = 61

DEFAULT_FRONTEND_URL = "http://127.0.0.1:8001"
DEFAULT_WORKER_METRICS_URL = "http://127.0.0.1:9200/metrics"

dynamo_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .pip_install(
        "fastapi>=0.115.0",
        "httpx>=0.28.0",
        "huggingface-hub[hf_transfer]>=0.20.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("kimi-k2-weights", create_if_missing=True)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required for demo/modal_dynamo_kimi.py. "
            "Set it to the exact shell command that starts the local Dynamo process."
        )
    return value


def _start_process(command: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


async def _wait_for_http(url: str, timeout_seconds: float) -> None:
    import httpx

    deadline = asyncio.get_event_loop().time() + timeout_seconds
    last_error = "endpoint did not become ready"
    async with httpx.AsyncClient(timeout=5.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                response = await client.get(url)
                if response.status_code < 500:
                    return
                last_error = f"{url} returned {response.status_code}"
            except Exception as exc:  # pragma: no cover - startup integration path
                last_error = str(exc)
            await asyncio.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def _render_manifest() -> str:
    return """schema_version: "1"
model: Kimi-K2.5
engine: dynamo
lane_class: production_validated
claim_scope: production_comparable
production_target_name: dynamo_long_context_coding
artifact_type: model_weights
artifact_path: /model-weights/Kimi-K2-Instruct
served_model_name: Kimi-K2.5
tensor_parallel_size: 4
topology_mode: single_endpoint
cache_strategy: lmcache
cache_connector: LMCacheConnectorV1
session_header_name: X-Session-ID
target_gpu_family: hopper
notes:
  - Modal aggregated production-lane scaffold.
  - Public request path is proxied at /v1/*.
  - Public metrics paths are /frontend/metrics and /worker/metrics.
"""


@app.function(
    image=dynamo_image,
    gpu=None,
    timeout=21600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/model-weights": model_volume},
)
def download_weights() -> None:
    """Pull Kimi weights into a persistent Modal Volume."""
    from huggingface_hub import snapshot_download

    dest = Path(WEIGHT_DIR)
    if dest.exists():
        shard_count = len([file for file in dest.iterdir() if file.suffix == ".safetensors"])
        print(f"Found {shard_count}/{SHARD_COUNT} safetensors shards at {dest}.")
        if shard_count >= SHARD_COUNT:
            print("Weights complete; nothing to do.")
            return
        print("Partial download detected; resuming.")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(dest),
        ignore_patterns=["*.pt", "*.bin"],
    )
    model_volume.commit()
    shard_count = len([file for file in dest.iterdir() if file.suffix == ".safetensors"])
    print(f"Done; {shard_count}/{SHARD_COUNT} shards committed.")


@app.function(
    image=dynamo_image,
    gpu=None,
    timeout=300,
    volumes={"/model-weights": model_volume},
)
def write_artifact_manifest() -> None:
    """Write the InferScope artifact manifest next to the Kimi weights."""
    path = Path(MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_manifest())
    model_volume.commit()
    print(f"Wrote {path}.")


@app.local_entrypoint()
def main() -> None:
    """Spawn weight download + manifest creation and exit immediately."""
    print("Spawning Kimi weight download on Modal (CPU-only)...")
    download_weights.spawn()
    print("Writing artifact manifest into the Modal Volume...")
    write_artifact_manifest.remote()
    print("Done. The production-lane scaffold is now ready for the Dynamo launch commands.")
    print(
        "Next: deploy with INFERSCOPE_DYNAMO_FRONTEND_CMD and "
        "INFERSCOPE_DYNAMO_WORKER_CMD set, then point InferScope at "
        f"https://<workspace>--{APP_NAME}-serve.modal.run"
    )


@app.function(
    image=dynamo_image,
    gpu="H200:4",
    timeout=1800,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/model-weights": model_volume},
)
@modal.asgi_app()
def serve():
    """Serve the aggregated Dynamo lane behind a single public Modal endpoint."""
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    import httpx

    frontend_url = os.environ.get("INFERSCOPE_DYNAMO_FRONTEND_URL", DEFAULT_FRONTEND_URL).rstrip("/")
    frontend_metrics_url = os.environ.get(
        "INFERSCOPE_DYNAMO_FRONTEND_METRICS",
        f"{frontend_url}/metrics",
    )
    worker_metrics_url = os.environ.get(
        "INFERSCOPE_DYNAMO_WORKER_METRICS",
        DEFAULT_WORKER_METRICS_URL,
    )
    healthcheck_url = os.environ.get(
        "INFERSCOPE_DYNAMO_HEALTHCHECK_URL",
        frontend_metrics_url,
    )

    api = FastAPI(title="InferScope Modal Dynamo/Kimi Aggregated Skeleton")

    def _forward_headers(headers: httpx.Headers) -> dict[str, str]:
        blocked = {"content-encoding", "transfer-encoding", "connection", "host"}
        return {key: value for key, value in headers.items() if key.lower() not in blocked}

    async def _proxy(url: str, request: Request) -> Response:
        content = await request.body()
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length"}
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream = await client.request(
                request.method,
                url,
                params=request.query_params,
                headers=headers,
                content=content,
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_forward_headers(upstream.headers),
            media_type=upstream.headers.get("content-type"),
        )

    @api.on_event("startup")
    async def startup() -> None:  # pragma: no cover - exercised in live deploys
        Path(WEIGHT_DIR).mkdir(parents=True, exist_ok=True)
        api.state.frontend_process = _start_process(_required_env("INFERSCOPE_DYNAMO_FRONTEND_CMD"))
        api.state.worker_process = _start_process(_required_env("INFERSCOPE_DYNAMO_WORKER_CMD"))
        await _wait_for_http(healthcheck_url, timeout_seconds=900.0)
        await _wait_for_http(frontend_metrics_url, timeout_seconds=120.0)
        await _wait_for_http(worker_metrics_url, timeout_seconds=120.0)

    @api.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - exercised in live deploys
        for attr in ("frontend_process", "worker_process"):
            process = getattr(api.state, attr, None)
            if process is None or process.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    @api.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "served_model_name": SERVED_MODEL_NAME,
            "frontend_url": frontend_url,
            "frontend_metrics_url": frontend_metrics_url,
            "worker_metrics_url": worker_metrics_url,
            "artifact_manifest": MANIFEST_PATH,
        }

    @api.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy_api(path: str, request: Request) -> Response:
        return await _proxy(f"{frontend_url}/v1/{path}", request)

    @api.get("/frontend/metrics")
    async def frontend_metrics(request: Request) -> Response:
        return await _proxy(frontend_metrics_url, request)

    @api.get("/worker/metrics")
    async def worker_metrics(request: Request) -> Response:
        return await _proxy(worker_metrics_url, request)

    @api.get("/")
    async def root() -> dict[str, str]:
        return {
            "message": "InferScope Modal Dynamo/Kimi aggregated scaffold",
            "request_surface": "/v1/*",
            "frontend_metrics": "/frontend/metrics",
            "worker_metrics": "/worker/metrics",
        }

    @api.exception_handler(RuntimeError)
    async def runtime_error_handler(_: Request, exc: RuntimeError) -> Response:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return api
