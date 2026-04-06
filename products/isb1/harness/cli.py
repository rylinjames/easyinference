"""ISB-1 Benchmark Harness CLI — unified command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import click

from harness.paths import (
    default_config_root,
    default_results_root,
    resolve_existing_path,
    resolve_path,
)
from harness.replay_client import _normalize_base_url

logger = logging.getLogger(__name__)

_SERVERLESS_HOST_SUFFIXES = (
    ".modal.run",
    ".modal.com",
    ".lightning.ai",
)
_LOCAL_ENDPOINT_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SERVERLESS_MIN_REQUEST_TIMEOUT_SECONDS = 180
_REMOTE_MIN_REQUEST_TIMEOUT_SECONDS = 120
_SERVERLESS_WARMUP_TIMEOUT_SECONDS = 180
_MODEL_DETECT_TIMEOUT_SECONDS = 30
_MODEL_DETECT_RETRIES = 2


def _resolve_existing_click_path(
    _ctx: click.Context, _param: click.Parameter, value: Path | None
) -> Path | None:
    if value is None:
        return None
    try:
        return resolve_existing_path(value)
    except FileNotFoundError as exc:
        raise click.BadParameter(str(exc)) from exc


def _resolve_click_path(
    _ctx: click.Context, _param: click.Parameter, value: Path | None
) -> Path | None:
    if value is None:
        return None
    return resolve_path(value)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _endpoint_hostname(endpoint: str) -> str:
    parsed = urlparse(_normalize_base_url(endpoint))
    return (parsed.hostname or "").lower()


def _is_local_endpoint(endpoint: str) -> bool:
    host = _endpoint_hostname(endpoint)
    return host in _LOCAL_ENDPOINT_HOSTS or host.endswith(".local")


def _is_serverless_endpoint(endpoint: str) -> bool:
    host = _endpoint_hostname(endpoint)
    return any(host.endswith(suffix) for suffix in _SERVERLESS_HOST_SUFFIXES)


def _default_request_timeout_seconds(endpoint: str, workload_type: str, duration: int) -> int:
    timeout = min(max(duration * 2, 60), 300)
    if workload_type in {"coding", "agent", "swebench", "coderforge"}:
        timeout = max(timeout, 120)
    if _is_serverless_endpoint(endpoint):
        timeout = max(timeout, _SERVERLESS_MIN_REQUEST_TIMEOUT_SECONDS)
    elif not _is_local_endpoint(endpoint):
        timeout = max(timeout, _REMOTE_MIN_REQUEST_TIMEOUT_SECONDS)
    return timeout


def _default_total_timeout_seconds(
    request_timeout_seconds: int,
    duration: int,
    num_requests: int,
) -> int:
    burst_budget = request_timeout_seconds * min(max(num_requests, 1), 4)
    return max(duration * 3, burst_budget + 60)


def _detect_model_id(
    endpoint: str,
    *,
    auth_headers: dict[str, str] | None,
    timeout_seconds: int,
    retries: int,
) -> tuple[str | None, str | None]:
    import time

    import requests as req

    detect_headers: dict[str, str] = {}
    if auth_headers:
        detect_headers.update(auth_headers)

    url = f"{_normalize_base_url(endpoint)}/v1/models"
    attempts = max(1, retries + 1)
    last_error: str | None = None

    for attempt in range(1, attempts + 1):
        try:
            resp = req.get(url, headers=detect_headers, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data:
                model_id = data[0].get("id")
                if isinstance(model_id, str) and model_id.strip():
                    return model_id, None
            last_error = "no model entries returned"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 5))

    return None, last_error


def _warm_endpoint(
    endpoint: str,
    *,
    model_id: str,
    auth_headers: dict[str, str] | None,
    timeout_seconds: int,
) -> tuple[bool, float | None, str | None]:
    import time

    import requests as req

    headers = {"Content-Type": "application/json"}
    if auth_headers:
        headers.update(auth_headers)

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Reply with OK only."},
        ],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
    }
    url = f"{_normalize_base_url(endpoint)}/v1/chat/completions"
    started = time.perf_counter()

    try:
        resp = req.post(url, json=payload, headers=headers, timeout=timeout_seconds)
        resp.raise_for_status()
        elapsed = time.perf_counter() - started
        return True, elapsed, None
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        return False, elapsed, f"{type(exc).__name__}: {exc}"


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.version_option(version="1.0.0", prog_name="isb1")
def main(verbose: bool) -> None:
    """ISB-1: Inference Serving Benchmark Standard 1."""
    _setup_logging(verbose)


# ── validate ─────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--sweep",
    "sweep_path",
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to sweep config YAML.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
@click.option("--all-yaml", is_flag=True, help="Parse-check every YAML under config root.")
def validate(sweep_path: str | None, config_root: str, all_yaml: bool) -> None:
    """Validate ISB-1 configuration files."""
    from harness.config_validator import ConfigValidator

    validator = ConfigValidator(config_root)

    if all_yaml:
        click.echo("Checking all YAML files...")
        res = validator.validate_all_yamls()
        click.echo(res.summary())

    if sweep_path:
        click.echo(f"Validating sweep: {sweep_path}")
        res = validator.validate_sweep(sweep_path)
        click.echo(res.summary())
        if not res.ok:
            raise SystemExit(1)
        click.echo("Sweep validation passed.")

    if not sweep_path and not all_yaml:
        click.echo("No action specified. Use --sweep or --all-yaml.")
        raise SystemExit(1)


# ── plan ─────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--config",
    "--sweep",
    "config_path",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to sweep config YAML.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
def plan(config_path: str, config_root: str) -> None:
    """Print the sweep matrix without executing."""
    from harness.sweep import SweepOrchestrator

    orchestrator = SweepOrchestrator(
        sweep_path=config_path,
        config_root=config_root,
        dry_run=True,
    )
    click.echo(orchestrator.plan())


# ── run ──────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--config",
    "--sweep",
    "config_path",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to sweep config YAML.",
)
@click.option(
    "--output",
    "output_dir",
    default=default_results_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_click_path,
    help="Output directory. Defaults to the product-local results/ tree.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
@click.option("--dry-run", is_flag=True, help="Print plan without executing.")
@click.option("--resume", "do_resume", is_flag=True, help="Resume a previous sweep.")
def run(
    config_path: str,
    output_dir: str,
    config_root: str,
    dry_run: bool,
    do_resume: bool,
) -> None:
    """Execute an ISB-1 benchmark sweep."""
    from harness.sweep import SweepOrchestrator

    orchestrator = SweepOrchestrator(
        sweep_path=config_path,
        output_dir=output_dir,
        config_root=config_root,
        dry_run=dry_run,
    )

    if dry_run:
        click.echo(orchestrator.plan())
        return

    if do_resume:
        summary = orchestrator.resume()
    else:
        summary = orchestrator.execute()

    click.echo("\nSweep complete:")
    click.echo(f"  Total cells:  {summary.total_cells}")
    click.echo(f"  Completed:    {summary.completed}")
    click.echo(f"  Failed:       {summary.failed}")
    click.echo(f"  Skipped:      {summary.skipped}")

    if summary.failed > 0:
        raise SystemExit(1)


# ── run-cell ─────────────────────────────────────────────────────────────


@main.command("run-cell")
@click.option("--gpu", required=True, help="GPU short name (e.g. h100).")
@click.option("--model", required=True, help="Model short name (e.g. llama70b).")
@click.option("--workload", required=True, help="Workload name (e.g. chat).")
@click.option("--mode", required=True, help="Mode name (e.g. mode_a).")
@click.option("--quantization", default="fp8", help="Quantization format.")
@click.option("--trial", default=1, type=int, help="Trial number.")
@click.option("--gpu-count", default=None, type=int, help="Number of GPUs (auto-detected if omitted).")
@click.option(
    "--output",
    "output_dir",
    default=default_results_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_click_path,
    help="Output directory. Defaults to the product-local results/ tree.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
@click.option(
    "--endpoint",
    default=None,
    help="External endpoint URL (e.g. https://my-modal-app.modal.run). "
    "Skips local vLLM server launch and GPU telemetry.",
)
@click.option(
    "--model-id",
    default=None,
    help="HuggingFace model ID served by the endpoint (e.g. Qwen/Qwen2.5-7B-Instruct). "
    "Required when --endpoint is used with a model not in ISB-1 configs.",
)
def run_cell(
    gpu: str,
    model: str,
    workload: str,
    mode: str,
    quantization: str,
    trial: int,
    gpu_count: int | None,
    output_dir: str,
    config_root: str,
    endpoint: str | None,
    model_id: str | None,
) -> None:
    """Execute a single benchmark cell."""
    from harness.config_validator import ConfigValidator
    from harness.runner import BenchmarkRunner, CellConfig

    validator = ConfigValidator(config_root)

    # Resolve model info
    model_cfg: dict = {}
    model_hf_id = model_id or ""
    try:
        model_cfg = validator.load_model(model)
        if not model_hf_id:
            model_hf_id = model_cfg.get("hf_model_id", "")
    except FileNotFoundError:
        if not endpoint:
            click.echo(f"ERROR: Model config for '{model}' not found.", err=True)
            raise SystemExit(1)
        # External endpoint mode — config is optional if --model-id is provided
        if not model_hf_id:
            click.echo(
                f"WARNING: No config for '{model}' and no --model-id provided. "
                "Attempting to auto-detect from endpoint...",
                err=True,
            )
            try:
                import httpx
                resp = httpx.get(f"{_normalize_base_url(endpoint)}/v1/models", timeout=30)
                data = resp.json().get("data", [])
                if data:
                    model_hf_id = data[0].get("id", model)
                    click.echo(f"  Detected model: {model_hf_id}", err=True)
                else:
                    model_hf_id = model
            except Exception:
                model_hf_id = model

    # Auto-detect gpu_count if not provided
    if gpu_count is None:
        if model_cfg:
            quant_key = "fp8" if quantization.startswith("fp8") else quantization
            min_gpus = model_cfg.get("min_gpus", {})
            quant_map = min_gpus.get(quant_key, min_gpus.get("bf16", {}))
            gpu_count = quant_map.get(gpu, 1)
        else:
            gpu_count = 1

    # Resolve topology
    if model_cfg:
        rec = model_cfg.get("recommended_topology", {})
        quant_key = "fp8" if quantization.startswith("fp8") else quantization
        topology = rec.get(quant_key, rec.get("bf16", {})).get(gpu, f"tp{gpu_count}")
    else:
        topology = f"tp{gpu_count}"

    # Resolve workload
    try:
        wl_cfg = validator.load_workload(workload)
    except FileNotFoundError:
        click.echo(f"ERROR: Workload config for '{workload}' not found.", err=True)
        raise SystemExit(1)

    rate_sweep = wl_cfg.get("arrival", {}).get("rate_sweep", [1.0])
    num_prompts = int(wl_cfg.get("trace", {}).get("num_requests", 1000))
    arrival_cfg = wl_cfg.get("arrival", {})
    arrival_model = str(arrival_cfg.get("model", "poisson"))
    arrival_shape = float(arrival_cfg["shape"]) if "shape" in arrival_cfg else None
    goodput_slo = wl_cfg.get("slo") or None

    cell = CellConfig(
        gpu=gpu,
        gpu_count=gpu_count,
        model=model,
        model_hf_id=model_hf_id,
        workload=workload,
        mode=mode,
        quantization=quantization,
        topology=topology,
        trial_number=trial,
        num_prompts=num_prompts,
        rate_sweep=rate_sweep,
        seed=42 + trial,
        arrival_model=arrival_model,
        arrival_shape=arrival_shape,
        goodput_slo=goodput_slo,
        output_dir=output_dir,
        config_root=config_root,
        external_endpoint=endpoint,
    )

    runner = BenchmarkRunner(cell)
    result = runner.run()

    click.echo(f"\nRun complete: {result.run_id}")
    click.echo(f"  Status: {result.status}")
    if result.error_message:
        click.echo(f"  Error: {result.error_message}")
    click.echo(f"  Manifest: {result.manifest_path}")

    if result.status == "failed":
        raise SystemExit(1)


# ── quick-bench ──────────────────────────────────────────────────────────


@main.command("quick-bench")
@click.argument("endpoint")
@click.option("--requests", "num_requests", default=20, type=int, help="Number of requests to send.")
@click.option("--duration", default=30, type=int, help="Measurement duration in seconds.")
@click.option("--rate", default=4.0, type=float, help="Request rate (req/s).")
@click.option("--model-id", default=None, help="Model ID served by the endpoint. Auto-detected if omitted.")
@click.option(
    "--request-timeout-seconds",
    default=None,
    type=int,
    help="Per-request timeout. Defaults scale up automatically for remote and serverless endpoints.",
)
@click.option(
    "--total-timeout-seconds",
    default=None,
    type=int,
    help="Overall benchmark timeout. Defaults scale up automatically from request timeout and request count.",
)
@click.option(
    "--detect-timeout-seconds",
    default=_MODEL_DETECT_TIMEOUT_SECONDS,
    show_default=True,
    type=int,
    help="Timeout for model auto-detection via /v1/models.",
)
@click.option(
    "--model-detect-retries",
    default=_MODEL_DETECT_RETRIES,
    show_default=True,
    type=int,
    help="Additional retries for model auto-detection on cold-start-heavy endpoints.",
)
@click.option(
    "--warmup/--no-warmup",
    default=True,
    help="Issue one short request before the measured run to warm serverless or cold endpoints.",
)
@click.option(
    "--warmup-timeout-seconds",
    default=_SERVERLESS_WARMUP_TIMEOUT_SECONDS,
    show_default=True,
    type=int,
    help="Timeout for the pre-benchmark warmup request.",
)
@click.option(
    "--workload",
    "workload_type",
    default="simple",
    type=click.Choice(["simple", "chat", "coding", "agent", "swebench", "coderforge"]),
    help="Workload type: simple (default), chat (ShareGPT), coding (synthetic repo), "
    "agent (tool calling), swebench (real GitHub issues), coderforge (real agent trajectories).",
)
@click.option("--api-key", default=None, help="Bearer token for authenticated endpoints (Modal, Fireworks, etc.).")
@click.option(
    "--context-bucket",
    default="32k",
    type=click.Choice(["8k", "16k", "32k", "64k", "128k"]),
    help="Context length bucket for swebench workload (default: 32k).",
)
@click.option("--sessions", default=4, type=int, help="Max concurrent sessions (for multi-turn workloads).")
def quick_bench(
    endpoint: str,
    num_requests: int,
    duration: int,
    rate: float,
    model_id: str | None,
    request_timeout_seconds: int | None,
    total_timeout_seconds: int | None,
    detect_timeout_seconds: int,
    model_detect_retries: int,
    warmup: bool,
    warmup_timeout_seconds: int,
    workload_type: str,
    api_key: str | None,
    context_bucket: str,
    sessions: int,
) -> None:
    """Fast smoke test against a live endpoint. Not publishable, but good for comparing configs.

    \b
    Examples:
      isb1 quick-bench https://my-endpoint.modal.run
      isb1 quick-bench https://api.openai.com/v1 --workload swebench --api-key $KEY --context-bucket 64k
      isb1 quick-bench https://my-endpoint.modal.run --workload coderforge --api-key $TOKEN
      isb1 quick-bench https://my-endpoint.modal.run --workload coding --sessions 8
    """
    import asyncio
    import time

    from analysis.metrics import _compute_itl_gaps, _compute_tpot, _safe_percentile
    from harness.replay_client import run_rate
    from workloads.base import Request, _new_request_id

    # Build auth headers
    auth_headers: dict[str, str] | None = None
    if api_key:
        auth_headers = {"Authorization": f"Bearer {api_key}"}

    # Auto-detect model
    if not model_id:
        model_id, detect_error = _detect_model_id(
            endpoint,
            auth_headers=auth_headers,
            timeout_seconds=detect_timeout_seconds,
            retries=model_detect_retries,
        )
        if model_id:
            click.echo(f"Detected model: {model_id}")
        else:
            model_id = "unknown"
            message = "Could not auto-detect model. Use --model-id to specify."
            if detect_error:
                message += f" Last error: {detect_error}"
            click.echo(message)

    if warmup and model_id != "unknown":
        click.echo("Warming endpoint before the measured run...")
        warm_ok, warm_elapsed, warm_error = _warm_endpoint(
            endpoint,
            model_id=model_id,
            auth_headers=auth_headers,
            timeout_seconds=warmup_timeout_seconds,
        )
        if warm_ok:
            click.echo(f"Warmup completed in {warm_elapsed:.2f}s")
        else:
            click.echo(
                "Warmup did not complete cleanly; continuing anyway. "
                f"Observed after {warm_elapsed:.2f}s: {warm_error}"
            )

    if request_timeout_seconds is None:
        request_timeout_seconds = _default_request_timeout_seconds(endpoint, workload_type, duration)
    if total_timeout_seconds is None:
        total_timeout_seconds = _default_total_timeout_seconds(
            request_timeout_seconds,
            duration,
            num_requests,
        )

    # Generate request pool based on workload type
    slo = {"ttft_p95_ms": 2000, "tpot_p95_ms": 100}

    if workload_type == "swebench":
        from workloads.swebench import SWEBenchCodingGenerator

        click.echo(f"Generating SWE-bench coding workload (context: {context_bucket})...")
        gen = SWEBenchCodingGenerator(seed=42, context_bucket=context_bucket, max_sessions=sessions)
        requests_pool = gen.generate(num_requests)
        slo = {"ttft_p95_ms": 3000 if context_bucket in ("8k", "16k", "32k") else 6000, "tpot_p95_ms": 60}

    elif workload_type == "coderforge":
        from workloads.coderforge import CoderForgeAgentGenerator

        click.echo("Generating CoderForge agent workload...")
        gen = CoderForgeAgentGenerator(seed=42, max_sessions=sessions)
        requests_pool = gen.generate(num_requests)
        slo = {"ttft_p95_ms": 1500, "tpot_p95_ms": 80}

    elif workload_type == "coding":
        from workloads.coding import CodingTraceGenerator

        click.echo("Generating synthetic coding workload...")
        gen = CodingTraceGenerator(seed=42)
        requests_pool = gen.generate(num_requests)
        slo = {"ttft_p95_ms": 3000, "tpot_p95_ms": 60}

    elif workload_type == "chat":
        from workloads.chat import ChatTraceGenerator

        click.echo("Generating chat workload (ShareGPT)...")
        gen = ChatTraceGenerator(seed=42)
        requests_pool = gen.generate(num_requests)
        slo = {"ttft_p95_ms": 2000, "tpot_p95_ms": 100}

    elif workload_type == "agent":
        from workloads.agent import AgentTraceGenerator

        click.echo("Generating agent workload (tool calling)...")
        gen = AgentTraceGenerator(seed=42)
        requests_pool = gen.generate(num_requests)
        slo = {"ttft_p95_ms": 1500, "tpot_p95_ms": 80}

    else:
        # Simple (original behavior)
        requests_pool = []
        for i in range(num_requests):
            requests_pool.append(
                Request(
                    request_id=_new_request_id(),
                    messages=[
                        {"role": "user", "content": f"Explain concept {i} in 2-3 sentences."},
                    ],
                    expected_output_tokens=128,
                    metadata={"workload": "quick_bench"},
                )
            )

    click.echo(f"Running quick bench: {num_requests} requests at {rate} req/s for {duration}s")
    click.echo(f"Endpoint: {endpoint}")
    click.echo(f"Workload: {workload_type}")
    if auth_headers:
        click.echo("Auth: Bearer token provided")
    if _is_serverless_endpoint(endpoint):
        click.echo("Profile: serverless endpoint heuristics enabled")
    elif not _is_local_endpoint(endpoint):
        click.echo("Profile: remote endpoint heuristics enabled")
    click.echo(f"Request timeout: {request_timeout_seconds}s")
    click.echo(f"Total timeout:   {total_timeout_seconds}s")
    click.echo()

    start = time.time()
    result = asyncio.run(
        run_rate(
            base_url=endpoint.rstrip("/"),
            model=model_id,
            request_pool=requests_pool,
            request_count=num_requests,
            request_rate=rate,
            arrival_model="poisson",
            arrival_shape=None,
            seed=42,
            concurrency=sessions,
            request_timeout_seconds=request_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            goodput_slo=slo,
            extra_headers=auth_headers,
        )
    )
    elapsed = time.time() - start

    # Compute quick metrics
    ok = [r for r in result.per_request if not r.error]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    tpots = [
        _compute_tpot(r.e2e_latency, r.ttft, r.output_tokens)
        for r in ok
        if r.ttft is not None
    ]
    tpots = [t for t in tpots if t is not None]
    itl_gaps = []
    for r in ok:
        itl_gaps.extend(_compute_itl_gaps(r.token_timestamps))

    click.echo("━" * 60)
    click.echo(f"  Workload:   {workload_type}")
    click.echo(f"  Completed:  {result.completed}/{len(result.per_request)} ({result.error_rate:.0%} errors)")
    click.echo(f"  Duration:   {elapsed:.1f}s")
    click.echo()
    click.echo(f"  TTFT p50:   {_safe_percentile(ttfts, 50):.3f}s")
    click.echo(f"  TTFT p95:   {_safe_percentile(ttfts, 95):.3f}s")
    click.echo(f"  TPOT p50:   {_safe_percentile(tpots, 50) * 1000:.1f}ms")
    click.echo(f"  TPOT p95:   {_safe_percentile(tpots, 95) * 1000:.1f}ms")
    click.echo(f"  ITL  p50:   {_safe_percentile(itl_gaps, 50) * 1000:.1f}ms")
    click.echo(f"  ITL  p95:   {_safe_percentile(itl_gaps, 95) * 1000:.1f}ms")
    click.echo(f"  Throughput: {result.output_throughput:.0f} tok/s")
    click.echo(f"  Goodput:    {result.goodput:.1f} req/s ({result.slo_attainment:.0%} SLO)")

    # Try to scrape KV cache metrics from /metrics endpoint
    _scrape_kv_metrics(endpoint, auth_headers)

    click.echo("━" * 60)


def _scrape_kv_metrics(endpoint: str, auth_headers: dict[str, str] | None) -> None:
    """Attempt to scrape KV cache metrics from the endpoint's /metrics path."""
    import re

    try:
        import requests as req

        headers = {}
        if auth_headers:
            headers.update(auth_headers)

        # Try both /metrics and the base endpoint /metrics
        base = endpoint.rstrip("/")
        metrics_urls = [f"{base}/metrics"]
        # If endpoint ends with /v1, also try without /v1
        if base.endswith("/v1"):
            metrics_urls.append(f"{base[:-3]}/metrics")

        text = ""
        for url in metrics_urls:
            try:
                resp = req.get(url, headers=headers, timeout=10)
                if resp.ok and "# HELP" in resp.text:
                    text = resp.text
                    break
            except Exception:
                continue

        if not text:
            return

        # Detect engine
        engine = "unknown"
        if "vllm:" in text:
            engine = "vllm"
        elif "dynamo_" in text:
            engine = "dynamo"
        elif "sglang:" in text:
            engine = "sglang"

        # Extract KV cache metrics
        kv_usage = _extract_gauge(text, [
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
            "dynamo_component_kvstats_gpu_cache_usage_percent",
        ])
        prefix_hit = _extract_gauge(text, [
            "vllm:gpu_prefix_cache_hit_rate",
            "dynamo_component_kvstats_gpu_prefix_cache_hit_rate",
        ])
        # Try counter-based prefix hit rate for vLLM v0.18+
        if prefix_hit is None:
            hits = _extract_gauge(text, ["vllm:prefix_cache_hits_total"])
            queries = _extract_gauge(text, ["vllm:prefix_cache_queries_total"])
            if hits is not None and queries is not None and queries > 0:
                prefix_hit = hits / queries

        if kv_usage is not None or prefix_hit is not None:
            click.echo()
            click.echo("  KV Cache:")
            if kv_usage is not None:
                click.echo(f"    Utilization:      {kv_usage:.0%}")
            if prefix_hit is not None:
                click.echo(f"    Prefix Hit Rate:  {prefix_hit:.0%}")
            click.echo(f"    Engine:           {engine}")

    except Exception:
        pass  # Metrics endpoint is optional


def _extract_gauge(text: str, metric_names: list[str]) -> float | None:
    """Extract the first matching gauge value from Prometheus text."""
    import re

    for name in metric_names:
        # Match: metric_name{labels} value  OR  metric_name value
        pattern = re.escape(name) + r"(?:\{[^}]*\})?\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


# ── analyze ──────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--results-dir",
    "--input",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Directory containing benchmark results.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Output file for analysis results (JSON).",
)
def analyze(results_dir: str, output_path: str | None) -> None:
    """Analyze benchmark results and compute metrics."""
    from analysis.metrics import MetricComputer

    results_path = Path(results_dir)

    all_metrics: list[dict] = []
    for manifest_path in sorted(results_path.rglob("manifest.json")):
        run_dir = manifest_path.parent
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest_data.get("status") != "completed":
            continue

        # Load benchmark results
        bench_dir = run_dir / "benchmark"
        if not bench_dir.exists():
            continue

        json_files = sorted(bench_dir.glob("*.json"))
        if not json_files:
            continue

        raw = json.loads(json_files[-1].read_text(encoding="utf-8"))
        per_request = raw.get("per_request", [])

        # Load engine metrics
        em_path = run_dir / "engine_metrics.jsonl"
        engine_data: list[dict] = []
        if em_path.exists():
            for line in em_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    engine_data.append(json.loads(line))

        # Load telemetry
        telem_data: list[dict] = []
        telem_path = run_dir / "telemetry.csv"
        if telem_path.exists():
            import csv

            with open(telem_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        telem_data.append(
                            {
                                "power_watts": float(row.get("power_draw_watts", 0) or 0),
                                "timestamp": float(row.get("timestamp", 0) or 0),
                            }
                        )
                    except (ValueError, TypeError):
                        pass

        computer = MetricComputer(
            gpu_name=manifest_data.get("gpu", ""),
            gpu_count=manifest_data.get("gpu_count", 1),
        )
        metrics = computer.compute(per_request, engine_data, telem_data)
        entry = {**manifest_data, **metrics.to_dict()}
        all_metrics.append(entry)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8"
        )
        click.echo(f"Analysis written to {output_path} ({len(all_metrics)} cells)")
    else:
        click.echo(json.dumps(all_metrics, indent=2))


# ── claims ───────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--results-dir",
    "--input",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Directory containing benchmark results.",
)
def claims(results_dir: str) -> None:
    """Extract publishable performance claims from results."""
    results_path = Path(results_dir)
    manifests: list[dict] = []

    for mp in sorted(results_path.rglob("manifest.json")):
        data = json.loads(mp.read_text(encoding="utf-8"))
        if data.get("status") == "completed":
            manifests.append(data)

    if not manifests:
        click.echo("No completed runs found.")
        return

    click.echo(f"Found {len(manifests)} completed runs.")
    click.echo("\nPublishable claims require:")
    click.echo("  - Minimum 3 trials per configuration")
    click.echo("  - CV < 10% across trials")
    click.echo("  - Stable warmup achieved")
    click.echo("\nRun 'isb1 analyze' first to compute full metrics.")


# ── leaderboard ──────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--analysis",
    "--input",
    "analysis_path",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to analysis JSON output.",
)
@click.option("--sort-by", default="generation_throughput", help="Metric to sort by.")
@click.option("--top", default=20, type=int, help="Number of entries to show.")
def leaderboard(analysis_path: str, sort_by: str, top: int) -> None:
    """Display a leaderboard from analysis results."""
    data = json.loads(Path(analysis_path).read_text(encoding="utf-8"))

    if not data:
        click.echo("No data to display.")
        return

    # Sort by the specified metric (descending)
    try:
        sorted_data = sorted(data, key=lambda x: x.get(sort_by, 0), reverse=True)
    except TypeError:
        click.echo(f"Cannot sort by '{sort_by}'.")
        return

    header = (
        f"{'Rank':<5} {'GPU':<8} {'Model':<15} {'Workload':<10} "
        f"{'Quant':<6} {sort_by:<25} {'Status':<10}"
    )
    click.echo(header)
    click.echo("-" * len(header))

    for i, entry in enumerate(sorted_data[:top], 1):
        val = entry.get(sort_by, "N/A")
        if isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)
        click.echo(
            f"{i:<5} {entry.get('gpu', ''):<8} {entry.get('model', ''):<15} "
            f"{entry.get('workload', ''):<10} {entry.get('quantization', ''):<6} "
            f"{val_str:<25} {entry.get('status', ''):<10}"
        )


# ── report ───────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--analysis",
    "--input",
    "analysis_path",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to analysis JSON output.",
)
@click.option(
    "--output",
    "output_path",
    default="report.html",
    type=click.Path(),
    help="Output HTML report path.",
)
@click.option(
    "--template",
    default=None,
    type=click.Path(path_type=Path),
    callback=_resolve_click_path,
    help="Jinja2 template file.",
)
def report(analysis_path: str, output_path: str, template: str | None) -> None:
    """Generate an HTML report from analysis results."""
    import jinja2

    data = json.loads(Path(analysis_path).read_text(encoding="utf-8"))

    if template:
        with open(template, "r", encoding="utf-8") as fh:
            tmpl = jinja2.Template(fh.read())
    else:
        tmpl = jinja2.Template(_DEFAULT_REPORT_TEMPLATE)

    completed_results = [r for r in data if r.get("status") == "completed"]
    throughputs = [r.get("generation_throughput", 0) for r in completed_results]
    goodputs = [r.get("goodput", 0) for r in completed_results]
    economic_data = [r for r in completed_results if r.get("cost_per_million_tokens", 0) > 0]

    html = tmpl.render(
        title="ISB-1 Benchmark Report",
        results=data,
        total_cells=len(data),
        completed=len(completed_results),
        failed=sum(1 for r in data if r.get("status") == "failed"),
        best_throughput=max(throughputs) if throughputs else 0,
        best_goodput=max(goodputs) if goodputs else 0,
        max_throughput=max(throughputs) if throughputs else 1,
        max_goodput=max(goodputs) if goodputs else 1,
        economic_data=economic_data,
    )

    Path(output_path).write_text(html, encoding="utf-8")
    click.echo(f"Report written to {output_path}")


_DEFAULT_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<title>{{ title }}</title>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 2em; background: #0d1117; color: #c9d1d9; }
  h1 { color: #58a6ff; margin-bottom: 0.5em; font-size: 1.8em; }
  h2 { color: #8b949e; margin: 1.5em 0 0.5em; font-size: 1.3em; }
  .summary { display: flex; gap: 1.5em; margin: 1em 0 2em; flex-wrap: wrap; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1em 1.5em; min-width: 140px; }
  .card .label { font-size: 0.8em; color: #8b949e; }
  .card .value { font-size: 1.6em; font-weight: 600; color: #58a6ff; }
  .card.green .value { color: #3fb950; }
  .card.yellow .value { color: #d29922; }
  .card.red .value { color: #f85149; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; background: #161b22; border-radius: 8px; overflow: hidden; }
  th { background: #21262d; color: #8b949e; padding: 10px 12px; text-align: left; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; }
  td { padding: 10px 12px; border-top: 1px solid #21262d; font-size: 0.9em; }
  tr:hover td { background: #1c2128; }
  .good { color: #3fb950; }
  .warn { color: #d29922; }
  .bad { color: #f85149; }
  .bar { display: inline-block; height: 16px; border-radius: 3px; min-width: 2px; }
  .bar-throughput { background: #58a6ff; }
  .bar-goodput { background: #3fb950; }
  .meta { color: #484f58; font-size: 0.8em; margin-top: 2em; }
  a { color: #58a6ff; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p style="color: #8b949e;">Generated by ISB-1 (Inference Serving Benchmark Standard 1)</p>

<div class="summary">
  <div class="card"><div class="label">Total Cells</div><div class="value">{{ total_cells }}</div></div>
  <div class="card green"><div class="label">Completed</div><div class="value">{{ completed }}</div></div>
  {% if failed > 0 %}<div class="card red"><div class="label">Failed</div><div class="value">{{ failed }}</div></div>{% endif %}
  {% if best_throughput %}<div class="card"><div class="label">Peak Throughput</div><div class="value">{{ "%.0f"|format(best_throughput) }} tok/s</div></div>{% endif %}
  {% if best_goodput %}<div class="card green"><div class="label">Peak Goodput</div><div class="value">{{ "%.1f"|format(best_goodput) }} req/s</div></div>{% endif %}
</div>

<h2>Results</h2>
<table>
<tr>
  <th>GPU</th><th>Model</th><th>Workload</th><th>Mode</th><th>Quant</th>
  <th>Throughput (tok/s)</th><th>TTFT p95</th><th>TPOT p95</th><th>ITL p95</th>
  <th>Goodput</th><th>SLO %</th><th>Errors</th>
</tr>
{% for r in results %}
<tr>
  <td>{{ r.gpu|upper }}</td>
  <td>{{ r.model }}</td>
  <td>{{ r.workload }}</td>
  <td>{{ r.mode }}</td>
  <td>{{ r.quantization }}</td>
  <td>
    <span class="bar bar-throughput" style="width: {{ [r.generation_throughput|default(0) / (max_throughput or 1) * 100, 100]|min }}px"></span>
    {{ "%.0f"|format(r.generation_throughput|default(0)) }}
  </td>
  <td class="{{ 'good' if r.ttft_p95|default(0) < 1 else 'warn' if r.ttft_p95|default(0) < 5 else 'bad' }}">
    {{ "%.2f"|format(r.ttft_p95|default(0)) }}s
  </td>
  <td class="{{ 'good' if r.tpot_p95|default(0) < 0.05 else 'warn' if r.tpot_p95|default(0) < 0.1 else 'bad' }}">
    {{ "%.1f"|format(r.tpot_p95|default(0) * 1000) }}ms
  </td>
  <td>{{ "%.1f"|format(r.itl_p95|default(0) * 1000) }}ms</td>
  <td>
    <span class="bar bar-goodput" style="width: {{ [r.goodput|default(0) / (max_goodput or 1) * 100, 100]|min }}px"></span>
    {{ "%.1f"|format(r.goodput|default(0)) }}
  </td>
  <td class="{{ 'good' if r.slo_attainment|default(0) > 0.9 else 'warn' if r.slo_attainment|default(0) > 0.7 else 'bad' }}">
    {{ "%.0f"|format(r.slo_attainment|default(0) * 100) }}%
  </td>
  <td class="{{ 'good' if r.error_rate|default(0) < 0.01 else 'warn' if r.error_rate|default(0) < 0.05 else 'bad' }}">
    {{ "%.1f"|format(r.error_rate|default(0) * 100) }}%
  </td>
</tr>
{% endfor %}
</table>

{% if economic_data %}
<h2>Economics</h2>
<table>
<tr><th>GPU</th><th>Model</th><th>Quant</th><th>tok/s</th><th>$/M tokens</th><th>tok/watt</th></tr>
{% for r in economic_data %}
<tr>
  <td>{{ r.gpu|upper }}</td>
  <td>{{ r.model }}</td>
  <td>{{ r.quantization }}</td>
  <td>{{ "%.0f"|format(r.generation_throughput|default(0)) }}</td>
  <td>{{ "%.2f"|format(r.cost_per_million_tokens|default(0)) }}</td>
  <td>{{ "%.2f"|format(r.tokens_per_watt|default(0)) }}</td>
</tr>
{% endfor %}
</table>
{% endif %}

<p class="meta">
  ISB-1 Benchmark Report | {{ total_cells }} cells |
  <a href="https://github.com/OCWC22/EasyInference">github.com/OCWC22/EasyInference</a>
</p>
</body>
</html>
"""


# ── import-results ───────────────────────────────────────────────────────


@main.command("import-results")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--format", "fmt", default="auto", help="Format: auto, vllm_json, genai_perf_csv, jsonl")
@click.option("--output", "output_path", default=None, type=click.Path(), help="Output ISB-1 JSON file.")
@click.option("--gpu", default="unknown", help="GPU used (for metadata).")
@click.option("--model", default="unknown", help="Model name (for metadata).")
@click.option("--workload", default="chat", help="Workload type.")
@click.option("--quantization", default="unknown", help="Quantization method.")
def import_results(
    input_path: str,
    fmt: str,
    output_path: str | None,
    gpu: str,
    model: str,
    workload: str,
    quantization: str,
) -> None:
    """Import external benchmark results (vLLM, GenAI-Perf, JSONL) into ISB-1 format.

    Lets you use ISB-1's analysis, comparison, and leaderboard tools
    on results from any benchmark tool.

    Examples:
      isb1 import-results vllm_output.json --gpu h100 --model llama70b
      isb1 import-results genai_perf.csv --format genai_perf_csv --gpu a100
    """
    from analysis.importers import detect_format, import_genai_perf_csv, import_jsonl, import_vllm_benchmark
    from analysis.metrics import MetricComputer

    if fmt == "auto":
        fmt = detect_format(input_path)
        click.echo(f"Detected format: {fmt}")

    if fmt == "vllm_json":
        per_request = import_vllm_benchmark(input_path)
    elif fmt == "genai_perf_csv":
        per_request = import_genai_perf_csv(input_path)
    elif fmt == "jsonl":
        per_request = import_jsonl(input_path)
    else:
        click.echo(f"ERROR: Unknown format '{fmt}'", err=True)
        raise SystemExit(1)

    click.echo(f"Imported {len(per_request)} requests")

    computer = MetricComputer(gpu_name=gpu)
    metrics = computer.compute(per_request)

    result = {
        "gpu": gpu,
        "model": model,
        "workload": workload,
        "quantization": quantization,
        "mode": "imported",
        "status": "completed",
        "source": str(input_path),
        "source_format": fmt,
        **metrics.to_dict(),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps([result], indent=2) + "\n", encoding="utf-8"
        )
        click.echo(f"Written to {output_path}")
    else:
        click.echo()
        click.echo(f"  TTFT p95:    {metrics.ttft_p95:.3f}s")
        click.echo(f"  TPOT p95:    {metrics.tpot_p95 * 1000:.1f}ms")
        click.echo(f"  Throughput:  {metrics.generation_throughput:.0f} tok/s")
        click.echo(f"  Goodput:     {metrics.goodput:.1f} req/s")
        click.echo(f"  SLO:         {metrics.slo_attainment:.0%}")
        click.echo(f"  Error rate:  {metrics.error_rate:.1%}")
        click.echo()
        click.echo("Use --output results.json to save, then 'isb1 leaderboard --input results.json' to rank.")


# ── quality ──────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--results-dir",
    "--input",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Directory containing benchmark results.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
@click.option(
    "--quality-config",
    default=None,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to quality config YAML.",
)
def quality(results_dir: str, config_root: str, quality_config: str | None) -> None:
    """Run quality checks on benchmark outputs.

    Validates that inference outputs match reference quality expectations
    (ROUGE scores, accuracy thresholds, etc.).
    """
    results_path = Path(results_dir)

    # Find all completed runs
    manifests: list[dict] = []
    for mp in sorted(results_path.rglob("manifest.json")):
        data = json.loads(mp.read_text(encoding="utf-8"))
        if data.get("status") == "completed":
            manifests.append(data)

    if not manifests:
        click.echo("No completed runs found for quality checking.")
        return

    click.echo(f"Found {len(manifests)} completed runs for quality validation.")

    # Load quality configs
    quality_dir = Path(config_root) / "quality"
    if quality_dir.exists():
        import yaml

        quality_checks: list[dict] = []
        for qf in sorted(quality_dir.glob("*.yaml")):
            with open(qf, "r", encoding="utf-8") as fh:
                quality_checks.append(yaml.safe_load(fh) or {})
        click.echo(f"Loaded {len(quality_checks)} quality check definitions.")
    else:
        click.echo("No quality config directory found.")
        return

    click.echo("\nQuality validation requires model outputs to be captured.")
    click.echo("Run benchmarks with output capture enabled, then re-run this command.")


if __name__ == "__main__":
    main()
