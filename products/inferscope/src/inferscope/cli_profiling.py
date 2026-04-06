"""Runtime profiling command registration for the InferScope CLI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Annotated, Any

import typer

from inferscope.tools.audit import audit_deployment
from inferscope.tools.diagnose import (
    check_deployment,
    check_memory_pressure,
    get_cache_effectiveness,
)
from inferscope.tools.profiling import profile_runtime


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


def register_profiling_commands(
    app: typer.Typer,
    *,
    print_result: Callable[[dict[str, Any]], None],
    resolve_metrics_auth: Callable[..., Any],
) -> None:
    """Register runtime profiling and live diagnostics commands."""

    @app.command(name="profile-runtime")
    def profile_runtime_cmd(
        endpoint: Annotated[str, typer.Argument(help="Inference endpoint URL (e.g., http://localhost:8000)")],
        metrics_endpoint: Annotated[
            str,
            typer.Option(help="Optional Prometheus base URL if metrics live at a different endpoint"),
        ] = "",
        gpu_arch: Annotated[str, typer.Option(help="GPU arch (sm_90a, gfx950, etc.)")] = "",
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
        include_identity: Annotated[bool, typer.Option(help="Enrich profile with runtime identity/config")] = True,
        include_tuning_preview: Annotated[
            bool,
            typer.Option(help="Include tuning preview derived from audit findings"),
        ] = True,
        include_raw_metrics: Annotated[bool, typer.Option(help="Include raw Prometheus metric values")] = False,
        include_samples: Annotated[
            bool,
            typer.Option(help="Include persisted metric samples in the profile snapshot"),
        ] = False,
        scrape_timeout_seconds: Annotated[
            float,
            typer.Option(help="HTTP timeout for metrics scraping, useful for cold-start-heavy platforms", min=1.0),
        ] = 30.0,
        provider: Annotated[str, typer.Option(help="Managed provider preset (fireworks, baseten, huggingface)")] = "",
        metrics_api_key: Annotated[
            str,
            typer.Option(help="API key for scraping authenticated metrics endpoints"),
        ] = "",
        metrics_auth_scheme: Annotated[
            str,
            typer.Option(help="Metrics auth scheme: bearer, api-key, x-api-key, raw"),
        ] = "",
        metrics_auth_header_name: Annotated[
            str,
            typer.Option(help="Override metrics auth header name"),
        ] = "",
        metrics_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional metrics headers as Header=Value. Repeat for multiple headers."),
        ] = None,
    ) -> None:
        """Build a unified live runtime profile for an inference endpoint."""
        result = asyncio.run(
            profile_runtime(
                endpoint,
                metrics_endpoint=(metrics_endpoint or None),
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
                include_identity=include_identity,
                include_tuning_preview=include_tuning_preview,
                include_raw_metrics=include_raw_metrics,
                include_samples=include_samples,
                scrape_timeout_seconds=scrape_timeout_seconds,
            )
        )
        print_result(result)

    @app.command(name="audit")
    def audit_cmd(
        endpoint: Annotated[str, typer.Argument(help="Inference endpoint URL (e.g., http://localhost:8000)")],
        metrics_endpoint: Annotated[
            str,
            typer.Option(help="Optional Prometheus base URL if metrics live at a different endpoint"),
        ] = "",
        gpu_arch: Annotated[str, typer.Option(help="GPU arch (sm_90a, gfx950, etc.) for richer checks")] = "",
        model_name: Annotated[str, typer.Option(help="Model name for context")] = "",
        model_type: Annotated[str, typer.Option(help="Model type: dense or moe")] = "",
        attention_type: Annotated[str, typer.Option(help="Attention: GQA, MLA, MHA")] = "",
        tp: Annotated[int, typer.Option(help="Tensor parallelism degree")] = 1,
        quantization: Annotated[str, typer.Option(help="Current quantization (fp8, bf16, etc.)")] = "",
        kv_cache_dtype: Annotated[str, typer.Option(help="KV cache dtype (fp8_e4m3, auto)")] = "",
        provider: Annotated[str, typer.Option(help="Managed provider preset (fireworks, baseten, huggingface)")] = "",
        metrics_api_key: Annotated[
            str,
            typer.Option(help="API key for scraping authenticated metrics endpoints"),
        ] = "",
        metrics_auth_scheme: Annotated[
            str,
            typer.Option(help="Metrics auth scheme: bearer, api-key, x-api-key, raw"),
        ] = "",
        metrics_auth_header_name: Annotated[
            str,
            typer.Option(help="Override metrics auth header name"),
        ] = "",
        metrics_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional metrics headers as Header=Value. Repeat for multiple headers."),
        ] = None,
        scrape_timeout_seconds: Annotated[
            float,
            typer.Option(help="HTTP timeout for metrics scraping, useful for cold-start-heavy platforms", min=1.0),
        ] = 30.0,
    ) -> None:
        """Run all audit checks against a live endpoint."""
        result = asyncio.run(
            audit_deployment(
                endpoint,
                metrics_endpoint=(metrics_endpoint or None),
                gpu_arch=gpu_arch,
                model_name=model_name,
                model_type=model_type,
                attention_type=attention_type,
                tp=tp,
                quantization=quantization,
                kv_cache_dtype=kv_cache_dtype,
                allow_private=True,
                metrics_auth=resolve_metrics_auth(
                    provider=provider,
                    metrics_api_key=metrics_api_key,
                    metrics_auth_scheme=metrics_auth_scheme,
                    metrics_auth_header_name=metrics_auth_header_name,
                    metrics_header=metrics_header,
                ),
                scrape_timeout_seconds=scrape_timeout_seconds,
            )
        )
        print_result(result)

    @app.command(name="check")
    def check_cmd(
        endpoint: Annotated[str, typer.Argument(help="Inference endpoint URL (e.g., http://localhost:8000)")],
        metrics_endpoint: Annotated[
            str,
            typer.Option(help="Optional Prometheus base URL if metrics live at a different endpoint"),
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
        metrics_auth_header_name: Annotated[
            str,
            typer.Option(help="Override metrics auth header name"),
        ] = "",
        metrics_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional metrics headers as Header=Value. Repeat for multiple headers."),
        ] = None,
        scrape_timeout_seconds: Annotated[
            float,
            typer.Option(help="HTTP timeout for metrics scraping, useful for cold-start-heavy platforms", min=1.0),
        ] = 30.0,
    ) -> None:
        """Scrape a live endpoint and show health snapshot."""
        result = asyncio.run(
            check_deployment(
                endpoint,
                metrics_endpoint=(metrics_endpoint or None),
                metrics_auth=resolve_metrics_auth(
                    provider=provider,
                    metrics_api_key=metrics_api_key,
                    metrics_auth_scheme=metrics_auth_scheme,
                    metrics_auth_header_name=metrics_auth_header_name,
                    metrics_header=metrics_header,
                ),
                scrape_timeout_seconds=scrape_timeout_seconds,
            )
        )
        print_result(result)

    @app.command(name="memory")
    def memory_cmd(
        endpoint: Annotated[str, typer.Argument(help="Inference endpoint URL")],
        metrics_endpoint: Annotated[
            str,
            typer.Option(help="Optional Prometheus base URL if metrics live at a different endpoint"),
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
        metrics_auth_header_name: Annotated[
            str,
            typer.Option(help="Override metrics auth header name"),
        ] = "",
        metrics_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional metrics headers as Header=Value. Repeat for multiple headers."),
        ] = None,
        scrape_timeout_seconds: Annotated[
            float,
            typer.Option(help="HTTP timeout for metrics scraping, useful for cold-start-heavy platforms", min=1.0),
        ] = 30.0,
    ) -> None:
        """Check KV cache memory pressure on a live endpoint."""
        result = asyncio.run(
            check_memory_pressure(
                endpoint,
                metrics_endpoint=(metrics_endpoint or None),
                metrics_auth=resolve_metrics_auth(
                    provider=provider,
                    metrics_api_key=metrics_api_key,
                    metrics_auth_scheme=metrics_auth_scheme,
                    metrics_auth_header_name=metrics_auth_header_name,
                    metrics_header=metrics_header,
                ),
                scrape_timeout_seconds=scrape_timeout_seconds,
            )
        )
        print_result(result)

    @app.command(name="cache")
    def cache_cmd(
        endpoint: Annotated[str, typer.Argument(help="Inference endpoint URL")],
        metrics_endpoint: Annotated[
            str,
            typer.Option(help="Optional Prometheus base URL if metrics live at a different endpoint"),
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
        metrics_auth_header_name: Annotated[
            str,
            typer.Option(help="Override metrics auth header name"),
        ] = "",
        metrics_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional metrics headers as Header=Value. Repeat for multiple headers."),
        ] = None,
        scrape_timeout_seconds: Annotated[
            float,
            typer.Option(help="HTTP timeout for metrics scraping, useful for cold-start-heavy platforms", min=1.0),
        ] = 30.0,
    ) -> None:
        """Measure prefix cache hit rate on a live endpoint."""
        result = asyncio.run(
            get_cache_effectiveness(
                endpoint,
                metrics_endpoint=(metrics_endpoint or None),
                metrics_auth=resolve_metrics_auth(
                    provider=provider,
                    metrics_api_key=metrics_api_key,
                    metrics_auth_scheme=metrics_auth_scheme,
                    metrics_auth_header_name=metrics_auth_header_name,
                    metrics_header=metrics_header,
                ),
                scrape_timeout_seconds=scrape_timeout_seconds,
            )
        )
        print_result(result)
