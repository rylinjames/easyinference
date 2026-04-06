"""Live deployment audit built on the shared runtime analysis core."""

from __future__ import annotations

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.logging import get_logger, sanitize_log_text
from inferscope.profiling.models import RuntimeContextHints
from inferscope.profiling.runtime import analyze_runtime, build_audit_payload

log = get_logger(component="audit")


async def audit_deployment(
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
    allow_private: bool = True,
    metrics_endpoint: str | None = None,
    metrics_auth: EndpointAuthConfig | None = None,
    scrape_timeout_seconds: float = 30.0,
) -> dict:
    """Run all audit checks against a live inference endpoint."""
    audit_log = log.bind(
        endpoint=endpoint,
        gpu_arch=gpu_arch,
        gpu_name=gpu_name,
        model_name=model_name,
        allow_private=allow_private,
    )
    audit_log.info(
        "audit_started",
        split_prefill_decode=split_prefill_decode,
        has_rdma=has_rdma,
        tp=tp,
        ep=ep,
    )

    bundle = await analyze_runtime(
        endpoint,
        metrics_endpoint=metrics_endpoint,
        context_hints=RuntimeContextHints(
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
        ),
        allow_private=allow_private,
        metrics_auth=metrics_auth,
        include_workload=True,
        include_identity=False,
        include_findings=True,
        include_tuning_preview=False,
        scrape_timeout_seconds=scrape_timeout_seconds,
    )

    if bundle.snapshot.error:
        audit_log.error(
            "audit_failed",
            error_type="scrape_error",
            error_summary=sanitize_log_text(bundle.snapshot.error),
        )
        return {
            "error": bundle.snapshot.error,
            "endpoint": endpoint,
            "summary": f"❌ Audit failed: {bundle.snapshot.error}",
            "confidence": 0.0,
            "evidence": "scrape_failure",
        }

    workload = bundle.workload.to_dict() if bundle.workload is not None else {}
    audit = build_audit_payload(bundle.findings)

    audit_log.info(
        "audit_completed",
        engine=bundle.normalized.engine,
        total_findings=audit["total"],
        critical=audit["critical"],
        warnings=audit["warnings"],
        info=audit["info"],
        workload_mode=workload.get("mode", "unknown"),
        workload_confidence=workload.get("confidence", 0.0),
    )

    return {
        "audit": audit,
        "workload": workload,
        "metrics": bundle.normalized.to_dict(),
        "engine": bundle.normalized.engine,
        "endpoint": endpoint,
        "summary": (
            f"{bundle.normalized.engine.upper()} audit: {audit['total']} finding(s) "
            f"({audit['critical']} critical, {audit['warnings']} warnings, {audit['info']} info) | "
            f"Workload: {workload.get('mode', 'unknown')} ({float(workload.get('confidence', 0.0)):.0%} confidence)"
        ),
        "confidence": 0.85 if gpu_arch else 0.65,
        "evidence": "live_audit_checks",
    }
