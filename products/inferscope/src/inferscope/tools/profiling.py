"""Operator-facing runtime profiling tool wrappers."""

from __future__ import annotations

from typing import Any

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.production_target import supported_configuration_hint
from inferscope.profiling.models import RuntimeContextHints
from inferscope.profiling.runtime import build_runtime_profile


def _profile_error_next_steps(error: str) -> list[str]:
    lowered = error.lower()
    if "metrics" in lowered or "scrape" in lowered:
        return [
            "Confirm the endpoint exposes a Prometheus-style /metrics surface and that it is "
            "reachable from this machine.",
            "If the request API and metrics surface live at different URLs, rerun with --metrics-endpoint.",
            "If the platform cold-starts containers, rerun with a larger "
            "--scrape-timeout-seconds value after warming the endpoint.",
            "If metrics are authenticated, pass the correct metrics auth settings or headers.",
            supported_configuration_hint(),
        ]
    if "auth" in lowered or "401" in lowered or "403" in lowered:
        return [
            "Check the request or metrics auth configuration and retry with the correct header scheme.",
            supported_configuration_hint(),
        ]
    return [
        "Confirm the endpoint is reachable and returns metrics before retrying.",
        supported_configuration_hint(),
    ]


def _profile_error(endpoint: str, error: str) -> dict[str, Any]:
    return {
        "error": error,
        "endpoint": endpoint,
        "summary": f"❌ Runtime profiling failed: {error}",
        "next_steps": _profile_error_next_steps(error),
        "confidence": 0.0,
        "evidence": "scrape_failure",
    }


async def profile_runtime(
    endpoint: str,
    *,
    metrics_endpoint: str | None = None,
    engine: str = "",
    gpu_arch: str = "",
    gpu_name: str = "",
    model_name: str = "",
    model_type: str = "",
    attention_type: str = "",
    experts_total: int = 0,
    tp: int = 1,
    ep: int = 0,
    quantization: str = "",
    kv_cache_dtype: str = "",
    gpu_memory_utilization: float = 0.0,
    block_size: int = 0,
    has_rdma: bool = False,
    split_prefill_decode: bool = False,
    env_vars: dict[str, str] | None = None,
    prefix_caching: bool = True,
    current_scheduler: dict[str, Any] | None = None,
    current_cache: dict[str, Any] | None = None,
    allow_private: bool = True,
    metrics_auth: EndpointAuthConfig | None = None,
    include_identity: bool = True,
    include_tuning_preview: bool = True,
    include_raw_metrics: bool = False,
    include_samples: bool = False,
    scrape_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Build the unified runtime profile payload."""
    hints = RuntimeContextHints(
        engine=engine,
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
        env_vars=env_vars or {},
        prefix_caching=prefix_caching,
    )
    try:
        report = await build_runtime_profile(
            endpoint,
            metrics_endpoint=metrics_endpoint,
            context_hints=hints,
            current_scheduler=current_scheduler,
            current_cache=current_cache,
            allow_private=allow_private,
            metrics_auth=metrics_auth,
            include_identity=include_identity,
            include_tuning_preview=include_tuning_preview,
            include_raw_metrics=include_raw_metrics,
            include_samples=include_samples,
            scrape_timeout_seconds=scrape_timeout_seconds,
        )
    except RuntimeError as exc:
        return _profile_error(endpoint, str(exc))
    return report.model_dump(mode="json", exclude_none=True)
