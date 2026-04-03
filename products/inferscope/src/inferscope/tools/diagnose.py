"""Live deployment diagnostics built on the shared runtime analysis core."""

from __future__ import annotations

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.profiling.runtime import analyze_runtime, build_health_summary


async def check_deployment(
    endpoint: str,
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
) -> dict:
    """Scrape a live inference endpoint and return normalized health snapshot."""
    bundle = await analyze_runtime(
        endpoint,
        allow_private=allow_private,
        metrics_auth=metrics_auth,
        include_workload=False,
        include_identity=False,
        include_findings=False,
        include_tuning_preview=False,
    )

    if bundle.snapshot.error:
        return {
            "error": bundle.snapshot.error,
            "endpoint": endpoint,
            "summary": f"❌ Cannot reach {endpoint}/metrics — {bundle.snapshot.error}",
            "confidence": 0.0,
            "evidence": "scrape_failure",
        }

    return {
        "metrics": bundle.normalized.to_dict(),
        "health": bundle.health,
        "summary": build_health_summary(bundle.normalized, bundle.health),
        "confidence": 0.9 if bundle.normalized.engine != "unknown" else 0.5,
        "evidence": "live_prometheus_scrape",
    }


async def check_memory_pressure(
    endpoint: str,
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
) -> dict:
    """Analyze KV cache utilization and preemption rates from live metrics."""
    bundle = await analyze_runtime(
        endpoint,
        allow_private=allow_private,
        metrics_auth=metrics_auth,
        include_workload=False,
        include_identity=False,
        include_findings=False,
        include_tuning_preview=False,
    )

    if bundle.snapshot.error:
        return {"error": bundle.snapshot.error, "confidence": 0.0}

    memory_pressure = dict(bundle.memory_pressure)
    findings = list(memory_pressure.pop("findings", []))
    summary = str(memory_pressure.pop("summary", "Memory pressure analysis complete"))
    return {
        "memory_pressure": memory_pressure,
        "findings": findings,
        "engine": bundle.normalized.engine,
        "summary": summary,
        "confidence": 0.85,
        "evidence": "live_kv_cache_metrics",
    }


async def get_cache_effectiveness(
    endpoint: str,
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
) -> dict:
    """Measure prefix cache hit rate and cache-aware routing effectiveness."""
    bundle = await analyze_runtime(
        endpoint,
        allow_private=allow_private,
        metrics_auth=metrics_auth,
        include_workload=False,
        include_identity=False,
        include_findings=False,
        include_tuning_preview=False,
    )

    if bundle.snapshot.error:
        return {"error": bundle.snapshot.error, "confidence": 0.0}

    cache_effectiveness = dict(bundle.cache_effectiveness)
    recommendations = list(cache_effectiveness.pop("recommendations", []))
    summary = str(cache_effectiveness.pop("summary", "Cache analysis complete"))
    return {
        "cache": cache_effectiveness,
        "recommendations": recommendations,
        "engine": bundle.normalized.engine,
        "summary": summary,
        "confidence": 0.8,
        "evidence": "live_cache_metrics",
    }
