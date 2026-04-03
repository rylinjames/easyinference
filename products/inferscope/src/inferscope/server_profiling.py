"""Runtime profiling MCP tool registration for the InferScope server."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from inferscope.endpoint_auth import resolve_auth_payload
from inferscope.tools.audit import audit_deployment
from inferscope.tools.diagnose import (
    check_deployment,
    check_memory_pressure,
    get_cache_effectiveness,
)
from inferscope.tools.live_tuner import auto_tune_deployment
from inferscope.tools.profiling import profile_runtime


async def _profile_runtime_for_mcp(
    endpoint: str,
    *,
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
    current_scheduler: dict[str, Any] | None = None,
    current_cache: dict[str, Any] | None = None,
    include_identity: bool = True,
    include_tuning_preview: bool = True,
    include_raw_metrics: bool = False,
    include_samples: bool = False,
    provider: str = "",
    metrics_auth: dict | None = None,
) -> dict[str, Any]:
    """MCP-safe runtime profiling helper."""
    return await profile_runtime(
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
        current_scheduler=current_scheduler,
        current_cache=current_cache,
        allow_private=False,
        metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
        include_identity=include_identity,
        include_tuning_preview=include_tuning_preview,
        include_raw_metrics=include_raw_metrics,
        include_samples=include_samples,
    )


def register_profiling_tools(mcp: FastMCP) -> None:
    """Register runtime analysis MCP tools."""

    @mcp.tool()
    async def tool_profile_runtime(
        endpoint: str,
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
        current_scheduler: dict[str, Any] | None = None,
        current_cache: dict[str, Any] | None = None,
        include_identity: bool = True,
        include_tuning_preview: bool = True,
        include_raw_metrics: bool = False,
        include_samples: bool = False,
        provider: str = "",
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Build a unified live runtime profile for a running inference endpoint."""
        return await _profile_runtime_for_mcp(
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
            current_scheduler=current_scheduler,
            current_cache=current_cache,
            include_identity=include_identity,
            include_tuning_preview=include_tuning_preview,
            include_raw_metrics=include_raw_metrics,
            include_samples=include_samples,
            provider=provider,
            metrics_auth=metrics_auth,
        )

    @mcp.tool()
    async def tool_audit_deployment(
        endpoint: str,
        gpu_arch: str = "",
        model_name: str = "",
        model_type: str = "",
        attention_type: str = "",
        experts_total: int = 0,
        tp: int = 1,
        quantization: str = "",
        kv_cache_dtype: str = "",
        provider: str = "",
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Run all audit checks against a live vLLM/SGLang/ATOM endpoint."""
        return await audit_deployment(
            endpoint,
            gpu_arch=gpu_arch,
            model_name=model_name,
            model_type=model_type,
            attention_type=attention_type,
            experts_total=experts_total,
            tp=tp,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            allow_private=False,
            metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
        )

    @mcp.tool()
    async def tool_check_deployment(
        endpoint: str,
        provider: str = "",
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Scrape a live endpoint and return a health snapshot."""
        return await check_deployment(
            endpoint,
            allow_private=False,
            metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
        )

    @mcp.tool()
    async def tool_check_memory_pressure(
        endpoint: str,
        provider: str = "",
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Analyze KV cache utilization and preemption rates from live metrics."""
        return await check_memory_pressure(
            endpoint,
            allow_private=False,
            metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
        )

    @mcp.tool()
    async def tool_get_cache_effectiveness(
        endpoint: str,
        provider: str = "",
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Measure prefix cache hit rate and cache-aware routing effectiveness."""
        return await get_cache_effectiveness(
            endpoint,
            allow_private=False,
            metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
        )

    @mcp.tool()
    async def tool_auto_tune_deployment(
        endpoint: str,
        current_engine: str = "",
        current_workload: str = "",
        current_scheduler: dict | None = None,
        current_cache: dict | None = None,
        provider: str = "",
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Analyze a live endpoint and recommend config adjustments."""
        return await auto_tune_deployment(
            endpoint,
            current_engine=current_engine,
            current_workload=current_workload,
            current_scheduler=current_scheduler,
            current_cache=current_cache,
            allow_private=False,
            metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
        )
