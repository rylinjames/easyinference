"""Narrowed benchmark MCP tool registration for the InferScope server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastmcp import FastMCP

from inferscope.benchmarks import (
    build_default_artifact_path,
    compare_benchmark_artifacts,
    load_benchmark_artifact,
    run_openai_replay,
)
from inferscope.benchmarks.probe_resolution import (
    BenchmarkSupportError,
    ProbeResolutionError,
    ProductionTargetValidationError,
    resolve_probe_plan,
)
from inferscope.config import settings
from inferscope.endpoint_auth import resolve_auth_payload
from inferscope.production_target import (
    build_benchmark_readiness_summary,
    build_lane_summary,
    build_production_contract,
    supported_configuration_hint,
    validate_production_lane_artifact,
)


def _resolve_artifact_path_for_mcp(path_or_name: str) -> Path:
    """Resolve an artifact path under the benchmark directory only."""
    artifact_root = settings.benchmark_dir.resolve()
    candidate = Path(path_or_name)
    if not candidate.is_absolute():
        candidate = artifact_root / candidate
    resolved = candidate.resolve()
    if artifact_root not in resolved.parents and resolved != artifact_root:
        raise ValueError(f"Artifact path must stay under {artifact_root}")
    return resolved


def _production_error(errors: list[str]) -> dict[str, Any]:
    return {
        "error": "; ".join(errors),
        "summary": "❌ Unsupported production target",
        "production_target": build_production_contract(),
        "next_steps": [
            "Switch to a production-validated Kimi lane or a benchmark-supported public-model lane "
            "on H100, H200, B200, or B300.",
            supported_configuration_hint(),
        ],
        "confidence": 1.0,
        "evidence": "production_target_validation",
    }


def _support_error(message: str, support: Any | None = None) -> dict[str, Any]:
    return {
        "error": message,
        "support": support.model_dump(mode="json") if support is not None else None,
        "summary": "❌ Unsupported benchmark configuration",
        "production_target": build_production_contract(),
        "next_steps": [
            "Adjust the requested GPU, engine, topology, or experiment to match one of the "
            "shipped production or benchmark contracts.",
            supported_configuration_hint(),
        ],
        "confidence": 1.0,
        "evidence": "benchmark_support_assessment",
    }


def _resolution_error(message: str) -> dict[str, Any]:
    return {
        "error": message,
        "summary": "❌ Failed to resolve benchmark plan",
        "production_target": build_production_contract(),
        "next_steps": [
            "Verify the workload, experiment, and metrics target inputs first.",
            supported_configuration_hint(),
        ],
        "confidence": 1.0,
        "evidence": "benchmark_plan_resolution",
    }


def _resolve_benchmark_plan(
    workload: str,
    endpoint: str,
    *,
    experiment: str = "",
    gpu: str = "",
    num_gpus: int = 0,
    metrics_endpoint: str = "",
    concurrency: int = 0,
    metrics_target_overrides: dict[str, str] | None = None,
    request_rate: float = 0.0,
    arrival_model: str = "immediate",
    arrival_shape: float = 0.0,
    warmup_requests: int = 0,
    goodput_slo: dict[str, Any] | None = None,
    strict_support: bool = True,
    synthetic_requests: int = 0,
    synthetic_input_tokens: int = 0,
    synthetic_output_tokens: int = 0,
    synthetic_seed: int = 42,
    context_file: str = "",
    model_artifact_path: str = "",
    artifact_manifest: str = "",
):
    try:
        resolved = resolve_probe_plan(
            workload,
            endpoint,
            experiment=experiment,
            gpu=gpu,
            num_gpus=(num_gpus or None),
            metrics_endpoint=(metrics_endpoint or None),
            metrics_target_overrides=metrics_target_overrides,
            concurrency=(concurrency or None),
            request_rate=(request_rate or None),
            arrival_model=arrival_model,
            arrival_shape=(arrival_shape or None),
            warmup_requests=warmup_requests,
            goodput_slo=goodput_slo,
            strict_support=strict_support,
            synthetic_requests=(synthetic_requests or None),
            synthetic_input_tokens=(synthetic_input_tokens or None),
            synthetic_output_tokens=(synthetic_output_tokens or None),
            synthetic_seed=synthetic_seed,
            context_file=context_file,
            model_artifact_path=model_artifact_path,
            artifact_manifest=artifact_manifest,
            allow_context_file=False,
        )
    except ProductionTargetValidationError as exc:
        return _production_error(exc.errors), None, None, None, None
    except BenchmarkSupportError as exc:
        return _support_error(str(exc), exc.support), None, None, None, None
    except ProbeResolutionError as exc:
        return _resolution_error(str(exc)), None, None, None, None
    except Exception as exc:  # noqa: BLE001
        return _resolution_error(str(exc)), None, None, None, None

    return None, resolved.workload_reference, resolved.workload_pack, resolved.run_plan, resolved.support


def register_benchmark_tools(mcp: FastMCP) -> None:
    """Register narrowed benchmark MCP tools."""

    @mcp.tool()
    async def tool_get_production_contract() -> dict[str, Any]:
        """Return the supported InferScope production contract for MCP clients."""
        contract = build_production_contract()
        return {
            "summary": (
                "InferScope MCP exposes three surfaces: the production-validated Kimi lane, "
                "benchmark-supported public-model comparison lanes, and a preview-only low-cost A10G smoke path."
            ),
            "production_target": contract,
            "confidence": 1.0,
            "evidence": "production_target_contract",
        }

    @mcp.tool()
    async def tool_resolve_benchmark_plan(
        workload: str,
        endpoint: str,
        experiment: str = "",
        gpu: str = "",
        num_gpus: int = 0,
        metrics_endpoint: str = "",
        concurrency: int = 0,
        metrics_target_overrides: dict[str, str] | None = None,
        request_rate: float = 0.0,
        arrival_model: str = "immediate",
        arrival_shape: float = 0.0,
        warmup_requests: int = 0,
        goodput_slo: dict[str, Any] | None = None,
        strict_support: bool = True,
        synthetic_requests: int = 0,
        synthetic_input_tokens: int = 0,
        synthetic_output_tokens: int = 0,
        synthetic_seed: int = 42,
        context_file: str = "",
        model_artifact_path: str = "",
        artifact_manifest: str = "",
    ) -> dict[str, Any]:
        """Resolve the supported InferScope probe into a concrete run plan."""
        error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
            workload,
            endpoint,
            experiment=experiment,
            gpu=gpu,
            num_gpus=num_gpus,
            metrics_endpoint=metrics_endpoint,
            concurrency=concurrency,
            metrics_target_overrides=metrics_target_overrides,
            request_rate=request_rate,
            arrival_model=arrival_model,
            arrival_shape=arrival_shape,
            warmup_requests=warmup_requests,
            goodput_slo=goodput_slo,
            strict_support=strict_support,
            synthetic_requests=synthetic_requests,
            synthetic_input_tokens=synthetic_input_tokens,
            synthetic_output_tokens=synthetic_output_tokens,
            synthetic_seed=synthetic_seed,
            context_file=context_file,
            model_artifact_path=model_artifact_path,
            artifact_manifest=artifact_manifest,
        )
        if error is not None:
            return cast(dict[str, Any], error)
        return {
            "summary": f"Resolved probe plan for {workload_reference}",
            "lane": build_lane_summary(
                model_name=run_plan.model,
                workload_pack=workload_pack.name,
            ),
            "run_plan": cast(dict[str, Any], run_plan.model_dump(mode="json")),
            "support": cast(dict[str, Any], support.model_dump(mode="json")) if support is not None else None,
            "preflight_validation": (
                cast(dict[str, Any], run_plan.preflight_validation.model_dump(mode="json"))
                if run_plan.preflight_validation is not None
                else None
            ),
            "production_target": build_production_contract(),
            "confidence": 0.95,
            "evidence": "benchmark_plan_resolution",
        }

    @mcp.tool()
    async def tool_run_benchmark(
        workload: str,
        endpoint: str,
        experiment: str = "",
        gpu: str = "",
        num_gpus: int = 0,
        metrics_endpoint: str = "",
        concurrency: int = 0,
        capture_metrics: bool = True,
        save_artifact: bool = True,
        metrics_target_overrides: dict[str, str] | None = None,
        request_rate: float = 0.0,
        arrival_model: str = "immediate",
        arrival_shape: float = 0.0,
        warmup_requests: int = 0,
        goodput_slo: dict[str, Any] | None = None,
        strict_support: bool = True,
        synthetic_requests: int = 0,
        synthetic_input_tokens: int = 0,
        synthetic_output_tokens: int = 0,
        synthetic_seed: int = 42,
        context_file: str = "",
        model_artifact_path: str = "",
        artifact_manifest: str = "",
        provider: str = "",
        metrics_provider: str = "",
        request_auth: dict | None = None,
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Run the supported InferScope probe against an OpenAI-compatible endpoint."""
        error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
            workload,
            endpoint,
            experiment=experiment,
            gpu=gpu,
            num_gpus=num_gpus,
            metrics_endpoint=metrics_endpoint,
            concurrency=concurrency,
            metrics_target_overrides=metrics_target_overrides,
            request_rate=request_rate,
            arrival_model=arrival_model,
            arrival_shape=arrival_shape,
            warmup_requests=warmup_requests,
            goodput_slo=goodput_slo,
            strict_support=strict_support,
            synthetic_requests=synthetic_requests,
            synthetic_input_tokens=synthetic_input_tokens,
            synthetic_output_tokens=synthetic_output_tokens,
            synthetic_seed=synthetic_seed,
            context_file=context_file,
            model_artifact_path=model_artifact_path,
            artifact_manifest=artifact_manifest,
        )
        if error is not None:
            return cast(dict[str, Any], error)

        try:
            request_auth_config = resolve_auth_payload(request_auth, provider=provider)
            metrics_auth_config = resolve_auth_payload(
                metrics_auth,
                provider=metrics_provider or provider,
            )
            artifact = await run_openai_replay(
                workload_pack,
                endpoint,
                metrics_endpoint=(metrics_endpoint or None),
                run_plan=run_plan,
                workload_ref=workload_reference,
                provider=provider,
                metrics_provider=metrics_provider,
                api_key=(request_auth_config.api_key or None) if request_auth_config else None,
                auth_scheme=request_auth_config.auth_scheme if request_auth_config else "",
                auth_header_name=request_auth_config.auth_header_name if request_auth_config else "",
                extra_headers=request_auth_config.headers if request_auth_config else None,
                metrics_api_key=(metrics_auth_config.api_key or None) if metrics_auth_config else None,
                metrics_auth_scheme=metrics_auth_config.auth_scheme if metrics_auth_config else "",
                metrics_auth_header_name=(metrics_auth_config.auth_header_name if metrics_auth_config else ""),
                metrics_headers=metrics_auth_config.headers if metrics_auth_config else None,
                capture_metrics=capture_metrics,
                allow_private=False,
            )
            artifact_path = ""
            if save_artifact:
                artifact_path = str(artifact.save_json(build_default_artifact_path(artifact)))
        except Exception as exc:  # noqa: BLE001
            return {
                "error": str(exc),
                "summary": "❌ Benchmark run failed",
                "production_target": build_production_contract(),
                "confidence": 1.0,
                "evidence": "live_benchmark_replay",
            }
        return {
            "summary": (
                f"Probe completed: {artifact.summary.succeeded}/{artifact.summary.total_requests} requests succeeded"
            ),
            "artifact_path": artifact_path,
            "benchmark_id": artifact.benchmark_id,
            "lane": build_lane_summary(
                model_name=run_plan.model,
                workload_pack=workload_pack.name,
            ),
            "run_plan": cast(dict[str, Any], run_plan.model_dump(mode="json")),
            "support": cast(dict[str, Any], support.model_dump(mode="json")) if support is not None else None,
            "preflight_validation": (
                cast(dict[str, Any], run_plan.preflight_validation.model_dump(mode="json"))
                if run_plan.preflight_validation is not None
                else None
            ),
            "observed_runtime": (
                cast(dict[str, Any], artifact.run_plan.get("observed_runtime", {})) if artifact.run_plan else {}
            ),
            "benchmark_summary": cast(dict[str, Any], artifact.summary.model_dump(mode="json")),
            "production_readiness": build_benchmark_readiness_summary(artifact),
            "production_target": build_production_contract(),
            "confidence": 0.85,
            "evidence": "live_benchmark_replay",
        }

    @mcp.tool()
    async def tool_compare_benchmarks(baseline_artifact: str, candidate_artifact: str) -> dict[str, Any]:
        """Compare two saved probe artifacts and report latency/TTFT deltas."""
        baseline = load_benchmark_artifact(_resolve_artifact_path_for_mcp(baseline_artifact))
        candidate = load_benchmark_artifact(_resolve_artifact_path_for_mcp(candidate_artifact))
        comparison = compare_benchmark_artifacts(baseline, candidate)
        comparison["confidence"] = 0.9
        comparison["evidence"] = "benchmark_artifact_comparison"
        return comparison

    @mcp.tool()
    async def tool_get_benchmark_artifact(artifact_name: str) -> dict[str, Any]:
        """Read a saved probe artifact by filename from the benchmark directory."""
        artifact = load_benchmark_artifact(_resolve_artifact_path_for_mcp(artifact_name))
        return {
            "summary": f"Loaded benchmark artifact {artifact.default_filename}",
            "artifact": cast(dict[str, Any], artifact.model_dump(mode="json")),
            "observed_runtime": (
                cast(dict[str, Any], artifact.run_plan.get("observed_runtime", {})) if artifact.run_plan else {}
            ),
            "production_readiness": build_benchmark_readiness_summary(artifact),
            "production_target": build_production_contract(),
            "confidence": 1.0,
            "evidence": "saved_benchmark_artifact",
        }

    @mcp.tool()
    async def tool_validate_production_lane(
        candidate_artifact: str,
        baseline_artifact: str = "",
    ) -> dict[str, Any]:
        """Validate whether saved artifacts belong to the canonical production lane."""
        candidate = load_benchmark_artifact(_resolve_artifact_path_for_mcp(candidate_artifact))
        baseline = (
            load_benchmark_artifact(_resolve_artifact_path_for_mcp(baseline_artifact))
            if baseline_artifact
            else None
        )
        payload = validate_production_lane_artifact(candidate, baseline=baseline)
        payload["confidence"] = 1.0
        payload["evidence"] = "production_lane_validation"
        return payload
