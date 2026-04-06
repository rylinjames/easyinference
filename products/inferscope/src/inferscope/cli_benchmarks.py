"""Narrowed benchmark command registration for the InferScope CLI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from inferscope.benchmarks import (
    build_default_artifact_path,
    compare_benchmark_artifacts,
    load_benchmark_artifact,
    run_openai_replay,
)
from inferscope.benchmarks.probe_resolution import resolve_probe_plan
from inferscope.endpoint_auth import parse_header_values
from inferscope.production_target import (
    build_benchmark_readiness_summary,
    build_lane_summary,
    build_production_contract,
    validate_production_lane_artifact,
)


def _parse_json_option(raw: str, *, option_name: str) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{option_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise typer.BadParameter(f"{option_name} must be a JSON object")
    return {str(key): val for key, val in value.items()}


def register_benchmark_commands(
    app: typer.Typer,
    *,
    print_result: Callable[[dict[str, Any]], None],
) -> None:
    """Register narrowed benchmark probe commands on the main CLI app."""

    @app.command(name="benchmark-plan")
    def benchmark_plan_cmd(
        workload: Annotated[str, typer.Argument(help="Supported workload file path or built-in workload name")],
        endpoint: Annotated[str, typer.Argument(help="OpenAI-compatible request endpoint base URL")],
        experiment: Annotated[str, typer.Option(help="Optional supported probe experiment name")] = "",
        gpu: Annotated[str, typer.Option(help="Concrete GPU SKU for support validation")] = "",
        num_gpus: Annotated[int | None, typer.Option(help="Concrete GPU count for support validation", min=1)] = None,
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
                help="Fail plan resolution on unsupported GPU/probe combinations",
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
        model_artifact_path: Annotated[
            str,
            typer.Option(help="Optional local model artifact or engine directory to validate before launch"),
        ] = "",
        artifact_manifest: Annotated[
            str,
            typer.Option(help="Optional JSON/YAML artifact manifest for strict compatibility checks"),
        ] = "",
    ):
        """Resolve the supported InferScope probe into a concrete benchmark run plan."""
        try:
            resolved = resolve_probe_plan(
                workload,
                endpoint,
                experiment=experiment,
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
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc

        print_result(
            {
                "summary": f"Resolved probe plan for {resolved.workload_reference}",
                "lane": build_lane_summary(
                    model_name=resolved.run_plan.model,
                    workload_pack=resolved.workload_pack.name,
                ),
                "run_plan": resolved.run_plan.model_dump(mode="json"),
                "support": resolved.support.model_dump(mode="json"),
                "preflight_validation": (
                    resolved.run_plan.preflight_validation.model_dump(mode="json")
                    if resolved.run_plan.preflight_validation is not None
                    else None
                ),
                "production_target": build_production_contract(),
            }
        )

    @app.command(name="benchmark")
    def benchmark_cmd(
        workload: Annotated[str, typer.Argument(help="Supported workload file path or built-in workload name")],
        endpoint: Annotated[str, typer.Argument(help="OpenAI-compatible request endpoint base URL")],
        experiment: Annotated[str, typer.Option(help="Optional supported probe experiment name")] = "",
        gpu: Annotated[str, typer.Option(help="Concrete GPU SKU for support validation")] = "",
        num_gpus: Annotated[int | None, typer.Option(help="Concrete GPU count for support validation", min=1)] = None,
        output: Annotated[Path | None, typer.Option(help="Where to write the benchmark artifact JSON")] = None,
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
                help="Fail benchmark execution on unsupported GPU/probe combinations",
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
        model_artifact_path: Annotated[
            str,
            typer.Option(help="Optional local model artifact or engine directory to validate before launch"),
        ] = "",
        artifact_manifest: Annotated[
            str,
            typer.Option(help="Optional JSON/YAML artifact manifest for strict compatibility checks"),
        ] = "",
        provider: Annotated[
            str,
            typer.Option(help="Managed provider preset for auth defaults (fireworks, baseten, huggingface)"),
        ] = "",
        metrics_provider: Annotated[
            str,
            typer.Option(
                help="Managed provider preset for the metrics endpoint if different from the request endpoint"
            ),
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
        metrics_api_key: Annotated[
            str,
            typer.Option(envvar="INFERSCOPE_METRICS_API_KEY", help="API key for authenticated metrics endpoints"),
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
        capture_metrics: Annotated[
            bool,
            typer.Option(help="Capture Prometheus snapshots before and after the run"),
        ] = True,
    ):
        """Run the supported InferScope probe and save an artifact."""
        try:
            resolved = resolve_probe_plan(
                workload,
                endpoint,
                experiment=experiment,
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
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc

        try:
            metrics_headers = parse_header_values(metrics_header, option_name="metrics header")
            request_headers = parse_header_values(request_header, option_name="request header")
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

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
        artifact_path = output or build_default_artifact_path(artifact)
        saved_path = artifact.save_json(artifact_path)
        print_result(
            {
                "summary": (
                    (
                        f"Probe completed: {artifact.summary.succeeded}/"
                        f"{artifact.summary.total_requests} requests succeeded "
                        f"| p95 latency={artifact.summary.latency_p95_ms:.1f} ms"
                    )
                    if artifact.summary.latency_p95_ms is not None
                    else (
                        f"Probe completed: {artifact.summary.succeeded}/"
                        f"{artifact.summary.total_requests} requests succeeded"
                    )
                ),
                "artifact_path": str(saved_path),
                "lane": build_lane_summary(
                    model_name=resolved.run_plan.model,
                    workload_pack=resolved.workload_pack.name,
                ),
                "run_plan": resolved.run_plan.model_dump(mode="json"),
                "support": resolved.support.model_dump(mode="json"),
                "preflight_validation": (
                    resolved.run_plan.preflight_validation.model_dump(mode="json")
                    if resolved.run_plan.preflight_validation is not None
                    else None
                ),
                "observed_runtime": (artifact.run_plan or {}).get("observed_runtime", {}),
                "production_readiness": build_benchmark_readiness_summary(artifact),
                "production_target": build_production_contract(),
                "benchmark": artifact.model_dump(mode="json"),
            }
        )

    @app.command(name="benchmark-compare")
    def benchmark_compare_cmd(
        baseline: Annotated[Path, typer.Argument(help="Baseline benchmark artifact JSON path")],
        candidate: Annotated[Path, typer.Argument(help="Candidate benchmark artifact JSON path")],
    ):
        """Compare two benchmark artifacts."""
        try:
            baseline_artifact = load_benchmark_artifact(baseline)
            candidate_artifact = load_benchmark_artifact(candidate)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc
        print_result(compare_benchmark_artifacts(baseline_artifact, candidate_artifact))

    @app.command(name="validate-production-lane")
    def validate_production_lane_cmd(
        candidate: Annotated[Path, typer.Argument(help="Candidate benchmark artifact JSON path")],
        baseline: Annotated[
            Path | None,
            typer.Option(help="Optional baseline benchmark artifact JSON path for comparison checks"),
        ] = None,
    ):
        """Validate that an artifact belongs to the canonical production lane."""
        try:
            candidate_artifact = load_benchmark_artifact(candidate)
            baseline_artifact = load_benchmark_artifact(baseline) if baseline is not None else None
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc
        print_result(
            validate_production_lane_artifact(
                candidate_artifact,
                baseline=baseline_artifact,
            )
        )
