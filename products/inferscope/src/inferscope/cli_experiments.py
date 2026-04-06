"""Lightning Experiments orchestration for InferScope CLI runs."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from inferscope._cli_helpers import parse_json_option as _parse_json_option
from inferscope.benchmarks import build_default_artifact_path, run_openai_replay
from inferscope.benchmarks.probe_resolution import resolve_probe_plan
from inferscope.endpoint_auth import parse_header_values
from inferscope.production_target import build_benchmark_readiness_summary
from inferscope.tools.profiling import profile_runtime


def _default_experiment_name() -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return f"inferscope-{timestamp}"


def _ensure_litlogger() -> Any:
    try:
        return importlib.import_module("litlogger")
    except ImportError as exc:  # pragma: no cover - exercised through CLI test
        raise typer.BadParameter(
            "litlogger is not installed. Install it with `uv pip install --python .venv/bin/python litlogger`."
        ) from exc


def _json_dump_path(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _log_scalar(experiment: Any, name: str, value: Any, *, step: int = 0) -> None:
    if isinstance(value, bool):
        experiment[name].append(int(value), step=step)
        return
    if isinstance(value, (int, float)) and math.isfinite(value):
        experiment[name].append(value, step=step)


def _log_benchmark_summary(experiment: Any, benchmark: Any) -> None:
    summary = benchmark.summary
    _log_scalar(experiment, "benchmark_total_requests", summary.total_requests)
    _log_scalar(experiment, "benchmark_succeeded", summary.succeeded)
    _log_scalar(experiment, "benchmark_failed", summary.failed)
    _log_scalar(experiment, "benchmark_concurrency", summary.concurrency)
    _log_scalar(experiment, "benchmark_wall_time_ms", summary.wall_time_ms)
    _log_scalar(experiment, "benchmark_latency_avg_ms", summary.latency_avg_ms)
    _log_scalar(experiment, "benchmark_latency_p50_ms", summary.latency_p50_ms)
    _log_scalar(experiment, "benchmark_latency_p95_ms", summary.latency_p95_ms)
    _log_scalar(experiment, "benchmark_latency_p99_ms", summary.latency_p99_ms)
    _log_scalar(experiment, "benchmark_ttft_avg_ms", summary.ttft_avg_ms)
    _log_scalar(experiment, "benchmark_ttft_p90_ms", summary.ttft_p90_ms)
    _log_scalar(experiment, "benchmark_ttft_p95_ms", summary.ttft_p95_ms)
    _log_scalar(experiment, "benchmark_ttft_p99_ms", summary.ttft_p99_ms)
    _log_scalar(experiment, "benchmark_prompt_tokens", summary.prompt_tokens)
    _log_scalar(experiment, "benchmark_completion_tokens", summary.completion_tokens)
    _log_scalar(experiment, "benchmark_total_tokens", summary.total_tokens)
    _log_scalar(experiment, "benchmark_metrics_targets_total", summary.metrics_targets_total)
    _log_scalar(experiment, "benchmark_metrics_targets_with_errors", summary.metrics_targets_with_errors)
    _log_scalar(experiment, "benchmark_metrics_capture_complete", summary.metrics_capture_complete)


def register_experiment_commands(
    app: typer.Typer,
    *,
    print_result: Callable[[dict[str, Any]], None],
    resolve_metrics_auth: Callable[..., Any],
) -> None:
    """Register the Lightning Experiments wrapper command."""

    @app.command(name="experiment-run")
    def experiment_run_cmd(
        endpoint: Annotated[str, typer.Argument(help="Inference endpoint URL (e.g., http://localhost:8000)")],
        teamspace: Annotated[str, typer.Option(help="Lightning teamspace to log into")] = "",
        name: Annotated[str, typer.Option(help="Lightning experiment name")] = "",
        workload: Annotated[
            str,
            typer.Option(help="Supported workload file path or built-in workload name"),
        ] = "kimi-k2-long-context-coding",
        benchmark_experiment: Annotated[
            str,
            typer.Option("--benchmark-experiment", help="Optional supported probe experiment name"),
        ] = "",
        root_dir: Annotated[Path, typer.Option(help="Local directory for experiment JSON outputs")] = Path(
            "lightning_logs"
        ),
        benchmark: Annotated[
            bool,
            typer.Option("--benchmark/--no-benchmark", help="Run the benchmark after profiling and plan resolution"),
        ] = False,
        model_artifact_path: Annotated[
            str,
            typer.Option(help="Optional local model or engine directory to validate before plan resolution"),
        ] = "",
        artifact_manifest: Annotated[
            str,
            typer.Option(help="Optional JSON/YAML artifact manifest for stricter model/engine compatibility checks"),
        ] = "",
        save_terminal_logs: Annotated[
            bool,
            typer.Option("--save-terminal-logs/--no-save-terminal-logs", help="Upload captured terminal logs"),
        ] = False,
        output: Annotated[
            Path | None,
            typer.Option(help="Where to write the benchmark artifact JSON when --benchmark is enabled"),
        ] = None,
        gpu: Annotated[str, typer.Option(help="Concrete GPU SKU for support validation")] = "",
        num_gpus: Annotated[int | None, typer.Option(help="Concrete GPU count for support validation", min=1)] = None,
        gpu_arch: Annotated[str, typer.Option(help="GPU arch (sm_90a, sm_100, etc.)")] = "",
        gpu_name: Annotated[str, typer.Option(help="GPU SKU or deployment label")] = "",
        model_name: Annotated[str, typer.Option(help="Model name for context")] = "",
        model_type: Annotated[str, typer.Option(help="Model type: dense or moe")] = "",
        attention_type: Annotated[str, typer.Option(help="Attention: GQA, MLA, MHA")] = "",
        experts_total: Annotated[int, typer.Option(help="Total experts for MoE models", min=0)] = 0,
        tp: Annotated[int, typer.Option(help="Tensor parallelism degree", min=1)] = 1,
        ep: Annotated[int, typer.Option(help="Expert parallelism degree", min=0)] = 0,
        quantization: Annotated[str, typer.Option(help="Current quantization (fp8, bf16, etc.)")] = "",
        kv_cache_dtype: Annotated[str, typer.Option(help="KV cache dtype (fp8_e4m3, auto)")] = "",
        gpu_memory_utilization: Annotated[
            float,
            typer.Option(help="Configured GPU memory utilization if known", min=0.0, max=1.0),
        ] = 0.0,
        block_size: Annotated[int, typer.Option(help="Attention block size if known", min=0)] = 0,
        has_rdma: Annotated[bool, typer.Option(help="RDMA available between inference nodes")] = False,
        split_prefill_decode: Annotated[bool, typer.Option(help="Deployment uses split prefill/decode")] = False,
        current_scheduler: Annotated[
            str,
            typer.Option(help="Optional scheduler JSON object, e.g. '{\"batched_token_budget\":8192}'"),
        ] = "",
        current_cache: Annotated[
            str,
            typer.Option(help="Optional cache JSON object, e.g. '{\"gpu_memory_utilization\":0.92}'"),
        ] = "",
        provider: Annotated[str, typer.Option(help="Managed provider preset (fireworks, baseten, huggingface)")] = "",
        metrics_api_key: Annotated[
            str,
            typer.Option(help="API key for scraping authenticated metrics endpoints"),
        ] = "",
        metrics_auth_scheme: Annotated[
            str,
            typer.Option(help="Metrics auth scheme: bearer, api-key, x-api-key, raw"),
        ] = "",
        metrics_auth_header_name: Annotated[str, typer.Option(help="Override metrics auth header name")] = "",
        metrics_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional metrics headers as Header=Value. Repeat for multiple headers."),
        ] = None,
        concurrency: Annotated[int | None, typer.Option(help="Override concurrency", min=1)] = None,
        metrics_endpoint: Annotated[str | None, typer.Option(help="Optional default Prometheus base URL")] = None,
        metrics_target: Annotated[
            list[str] | None,
            typer.Option(
                help="Additional metrics targets as name=url. Example: --metrics-target router=http://host:9000"
            ),
        ] = None,
        request_rate: Annotated[float | None, typer.Option(help="Request rate for scheduled replay", min=0.0)] = None,
        arrival_model: Annotated[str, typer.Option(help="Arrival model: immediate, poisson, gamma")] = "immediate",
        arrival_shape: Annotated[float | None, typer.Option(help="Gamma arrival shape", min=0.0001)] = None,
        warmup_requests: Annotated[int, typer.Option(help="Warmup requests before measurement", min=0)] = 0,
        goodput_slo: Annotated[str, typer.Option(help="Optional JSON object for goodput thresholds")] = "",
        strict_support: Annotated[
            bool,
            typer.Option(
                "--strict-support/--no-strict-support",
                help="Fail plan resolution and benchmark execution on unsupported combinations",
            ),
        ] = True,
        synthetic_requests: Annotated[
            int | None,
            typer.Option(help="Procedurally expand the workload to this many requests", min=1),
        ] = None,
        synthetic_input_tokens: Annotated[
            int | None,
            typer.Option(help="Approximate input token target for procedural workloads", min=64),
        ] = None,
        synthetic_output_tokens: Annotated[
            int | None,
            typer.Option(help="Approximate output token target for procedural workloads", min=32),
        ] = None,
        synthetic_seed: Annotated[int, typer.Option(help="Seed for procedural workload expansion", min=0)] = 42,
        context_file: Annotated[
            str,
            typer.Option(help="Optional repo/context file used for procedural workload expansion"),
        ] = "",
        api_key: Annotated[
            str,
            typer.Option(envvar="OPENAI_API_KEY", help="API key or token for the request endpoint"),
        ] = "",
        auth_scheme: Annotated[str, typer.Option(help="Request auth scheme: bearer, api-key, x-api-key, raw")] = "",
        auth_header_name: Annotated[str, typer.Option(help="Override request auth header name")] = "",
        request_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional request headers as Header=Value. Repeat for multiple headers."),
        ] = None,
        metrics_provider: Annotated[
            str,
            typer.Option(
                help="Managed provider preset for the metrics endpoint if different from the request endpoint"
            ),
        ] = "",
        capture_metrics: Annotated[
            bool,
            typer.Option(help="Capture Prometheus snapshots before and after the run"),
        ] = True,
    ) -> None:
        """Run InferScope and log profiling plus probe artifacts to Lightning Experiments."""
        litlogger = _ensure_litlogger()
        experiment_name = name or _default_experiment_name()
        run_dir = root_dir / experiment_name
        experiment = litlogger.init(
            name=experiment_name,
            root_dir=str(root_dir),
            teamspace=teamspace or None,
            metadata={
                key: value
                for key, value in {
                    "endpoint": endpoint,
                    "workload": workload,
                    "benchmark_experiment": benchmark_experiment,
                    "gpu": gpu,
                    "num_gpus": str(num_gpus) if num_gpus is not None else "",
                    "model_name": model_name,
                    "quantization": quantization,
                    "model_artifact_path": model_artifact_path,
                    "artifact_manifest": artifact_manifest,
                }.items()
                if value
            },
            save_logs=save_terminal_logs,
            print_url=False,
            verbose=False,
        )

        experiment_url = getattr(experiment, "url", None)
        profile_path: Path | None = None
        plan_path: Path | None = None
        benchmark_path: Path | None = None

        try:
            profile_started = time.perf_counter()
            profile_result = asyncio.run(
                profile_runtime(
                    endpoint,
                    gpu_arch=gpu_arch,
                    gpu_name=gpu_name,
                    model_name=model_name,
                    model_type=model_type,
                    attention_type=attention_type,
                    experts_total=experts_total,
                    tp=tp,
                    ep=ep,
                    quantization=quantization,
                    kv_cache_dtype=kv_cache_dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    block_size=block_size,
                    has_rdma=has_rdma,
                    split_prefill_decode=split_prefill_decode,
                    current_scheduler=_parse_json_option(current_scheduler, option_name="current scheduler"),
                    current_cache=_parse_json_option(current_cache, option_name="current cache"),
                    allow_private=True,
                    metrics_auth=resolve_metrics_auth(
                        provider=provider,
                        metrics_api_key=metrics_api_key,
                        metrics_auth_scheme=metrics_auth_scheme,
                        metrics_auth_header_name=metrics_auth_header_name,
                        metrics_header=metrics_header,
                    ),
                    include_identity=True,
                    include_tuning_preview=True,
                    include_raw_metrics=False,
                    include_samples=False,
                )
            )
            _log_scalar(experiment, "profile_runtime_seconds", time.perf_counter() - profile_started)
            _log_scalar(experiment, "profile_confidence", profile_result.get("confidence"))
            profile_path = _json_dump_path(run_dir / "profile-runtime.json", profile_result)
            experiment.log_file(str(profile_path))

            plan_started = time.perf_counter()
            resolved = resolve_probe_plan(
                workload,
                endpoint,
                experiment=benchmark_experiment,
                gpu=gpu,
                num_gpus=num_gpus,
                concurrency=concurrency,
                metrics_endpoint=metrics_endpoint,
                metrics_target=metrics_target,
                request_rate=request_rate,
                arrival_model=arrival_model,
                arrival_shape=arrival_shape,
                warmup_requests=warmup_requests,
                goodput_slo=_parse_json_option(goodput_slo, option_name="goodput_slo"),
                strict_support=strict_support,
                synthetic_requests=synthetic_requests,
                synthetic_input_tokens=synthetic_input_tokens,
                synthetic_output_tokens=synthetic_output_tokens,
                synthetic_seed=synthetic_seed,
                context_file=context_file,
                model_artifact_path=model_artifact_path,
                artifact_manifest=artifact_manifest,
                allow_context_file=True,
            )
            _log_scalar(experiment, "benchmark_plan_seconds", time.perf_counter() - plan_started)
            preflight_validation = (
                resolved.run_plan.preflight_validation.model_dump(mode="json")
                if resolved.run_plan.preflight_validation is not None
                else None
            )
            plan_payload = {
                "run_plan": resolved.run_plan.model_dump(mode="json"),
                "support": resolved.support.model_dump(mode="json"),
                "preflight_validation": preflight_validation,
            }
            plan_path = _json_dump_path(run_dir / "benchmark-plan.json", plan_payload)
            experiment.log_file(str(plan_path))
            if artifact_manifest:
                manifest_path = Path(artifact_manifest).expanduser()
                if manifest_path.exists() and manifest_path.is_file():
                    experiment.log_file(str(manifest_path))

            result: dict[str, Any] = {
                "summary": f"Logged InferScope run to Lightning experiment {experiment_name}",
                "experiment_url": experiment_url,
                "lightning_run_dir": str(run_dir),
                "profile_result_path": str(profile_path),
                "benchmark_plan_path": str(plan_path),
                "run_plan": plan_payload["run_plan"],
                "support": plan_payload["support"],
                "preflight_validation": preflight_validation,
            }

            if benchmark:
                try:
                    metrics_headers = parse_header_values(metrics_header, option_name="metrics header")
                    request_headers = parse_header_values(request_header, option_name="request header")
                except ValueError as exc:
                    raise typer.BadParameter(str(exc)) from exc

                benchmark_started = time.perf_counter()
                artifact = asyncio.run(
                    run_openai_replay(
                        resolved.workload_pack,
                        endpoint,
                        metrics_endpoint=metrics_endpoint,
                        run_plan=resolved.run_plan,
                        workload_ref=resolved.workload_reference,
                        api_key=(api_key or None),
                        provider=provider,
                        metrics_provider=metrics_provider,
                        auth_scheme=auth_scheme,
                        auth_header_name=auth_header_name,
                        extra_headers=request_headers,
                        metrics_api_key=(metrics_api_key or None),
                        metrics_auth_scheme=metrics_auth_scheme,
                        metrics_auth_header_name=metrics_auth_header_name,
                        metrics_headers=metrics_headers,
                        capture_metrics=capture_metrics,
                        allow_private=True,
                    )
                )
                _log_scalar(experiment, "benchmark_seconds", time.perf_counter() - benchmark_started)
                _log_benchmark_summary(experiment, artifact)
                readiness = build_benchmark_readiness_summary(artifact)
                _log_scalar(experiment, "benchmark_ready", readiness["ready"])
                _log_scalar(experiment, "benchmark_readiness_issue_count", len(readiness["issues"]))
                benchmark_path = artifact.save_json(output or build_default_artifact_path(artifact))
                experiment.log_file(str(benchmark_path))
                benchmark_json_path = _json_dump_path(run_dir / "benchmark-artifact.json", artifact.model_dump(mode="json"))
                experiment.log_file(str(benchmark_json_path))
                result["benchmark_artifact_path"] = str(benchmark_path)
                result["benchmark_summary"] = artifact.summary.model_dump(mode="json")
                result["production_readiness"] = readiness
                if not readiness["ready"]:
                    result["summary"] = (
                        f"Logged InferScope run to Lightning experiment {experiment_name} "
                        "with benchmark observability/reliability issues"
                    )

            experiment.finalize("success")
            print_result(result)
        except Exception as exc:  # noqa: BLE001
            error_path = _json_dump_path(
                run_dir / "error.json",
                {
                    "error": str(exc),
                    "experiment_url": experiment_url,
                    "profile_result_path": str(profile_path) if profile_path else None,
                    "benchmark_plan_path": str(plan_path) if plan_path else None,
                    "benchmark_artifact_path": str(benchmark_path) if benchmark_path else None,
                },
            )
            experiment.log_file(str(error_path))
            experiment.finalize("failed")
            print_result(
                {
                    "summary": f"Lightning experiment failed: {exc}",
                    "experiment_url": experiment_url,
                    "error_path": str(error_path),
                    "error": str(exc),
                }
            )
            raise typer.Exit(code=1) from exc
