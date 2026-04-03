"""Compatibility wrapper for the packaged benchmark runtime."""

from __future__ import annotations

from pathlib import Path

import httpx

from inferscope.benchmarks.experiments import BenchmarkRunPlan, build_run_plan
from inferscope.benchmarks.models import BenchmarkArtifact, WorkloadPack
from inferscope.benchmarks.runtime import run_benchmark_runtime
from inferscope.config import settings
from inferscope.endpoint_auth import resolve_auth_config, same_origin


def build_default_artifact_path(artifact: BenchmarkArtifact) -> Path:
    """Build a default artifact path under the benchmark cache directory."""
    root = settings.benchmark_dir
    root.mkdir(parents=True, exist_ok=True)
    return root / artifact.default_filename


async def run_openai_replay(
    workload: WorkloadPack,
    endpoint: str,
    *,
    model: str | None = None,
    metrics_endpoint: str | None = None,
    run_plan: BenchmarkRunPlan | None = None,
    workload_ref: str = "",
    api_key: str | None = None,
    provider: str = "",
    metrics_provider: str = "",
    auth_scheme: str = "",
    auth_header_name: str = "",
    metrics_api_key: str | None = None,
    metrics_auth_scheme: str = "",
    metrics_auth_header_name: str = "",
    metrics_headers: dict[str, str] | None = None,
    concurrency: int | None = None,
    allow_private: bool = True,
    capture_metrics: bool = True,
    extra_headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> BenchmarkArtifact:
    """Replay a workload pack against an OpenAI-compatible endpoint."""
    resolved_plan = (
        run_plan.model_copy(deep=True)
        if run_plan is not None
        else build_run_plan(
            workload,
            endpoint,
            workload_ref=workload_ref or workload.name,
            model=model,
            concurrency=concurrency,
            metrics_endpoint=metrics_endpoint,
        )
    )
    if run_plan is not None:
        resolved_plan.request_endpoint = endpoint

    request_auth = resolve_auth_config(
        api_key,
        provider=provider,
        auth_scheme=auth_scheme,
        auth_header_name=auth_header_name,
        default_scheme="bearer",
    )
    reuse_request_auth_for_metrics = (
        metrics_api_key is None and not metrics_headers and same_origin(endpoint, metrics_endpoint or endpoint)
    )
    metrics_auth = resolve_auth_config(
        metrics_api_key if metrics_api_key is not None else (api_key if reuse_request_auth_for_metrics else None),
        provider=metrics_provider or provider,
        auth_scheme=metrics_auth_scheme,
        auth_header_name=metrics_auth_header_name,
        headers=metrics_headers,
        default_scheme="bearer",
    )
    execution_result = await run_benchmark_runtime(
        workload,
        endpoint,
        run_plan=resolved_plan,
        request_auth=request_auth,
        metrics_auth=metrics_auth,
        allow_private=allow_private,
        capture_metrics=capture_metrics,
        extra_headers=extra_headers,
        client=client,
    )
    return execution_result.artifact
