"""Live auto-tuner built on the shared runtime analysis core."""

from __future__ import annotations

from typing import Any

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.profiling.models import RuntimeContextHints
from inferscope.profiling.runtime import analyze_runtime


async def auto_tune_deployment(
    endpoint: str,
    current_engine: str = "",
    current_workload: str = "",
    current_scheduler: dict[str, Any] | None = None,
    current_cache: dict[str, Any] | None = None,
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
) -> dict[str, Any]:
    """Analyze a live endpoint and recommend config adjustments."""
    del current_workload  # reserved for future workload override support

    bundle = await analyze_runtime(
        endpoint,
        context_hints=RuntimeContextHints(engine=current_engine),
        current_scheduler=current_scheduler,
        current_cache=current_cache,
        allow_private=allow_private,
        metrics_auth=metrics_auth,
        include_workload=True,
        include_identity=False,
        include_findings=True,
        include_tuning_preview=True,
    )

    if bundle.snapshot.error:
        return {
            "error": bundle.snapshot.error,
            "endpoint": endpoint,
            "summary": f"Cannot reach {endpoint}/metrics — {bundle.snapshot.error}",
            "confidence": 0.0,
        }

    preview = bundle.tuning_preview
    adjustments = list(preview.adjustments) if preview is not None else []

    reasoning: list[str] = []
    if not bundle.findings:
        reasoning.append("No failure modes detected — current config appears healthy.")
    for finding in bundle.findings:
        reasoning.append(f"Detected {finding.check_id} ({finding.severity}): {finding.title}")
    for adjustment in adjustments:
        reasoning.append(
            f"Adjustment: {adjustment.parameter} {adjustment.current_value} → {adjustment.recommended_value} "
            f"(trigger: {adjustment.trigger}, confidence: {adjustment.confidence:.0%})"
        )
    if preview is not None and preview.error:
        reasoning.append(f"Tuning preview degraded: {preview.error}")

    return {
        "detections": [finding.to_dict() for finding in bundle.findings],
        "adjustments": [adjustment.to_dict() for adjustment in adjustments],
        "updated_scheduler": (preview.updated_scheduler if preview is not None else (current_scheduler or {})),
        "updated_cache": (preview.updated_cache if preview is not None else (current_cache or {})),
        "reasoning": reasoning,
        "metrics_snapshot": bundle.normalized.to_dict(),
        "summary": (
            f"{len(bundle.findings)} issue(s) detected, {len(adjustments)} adjustment(s) recommended"
            if bundle.findings
            else "No issues detected — deployment appears healthy"
        ),
        "confidence": 0.85 if adjustments else 0.9,
        "evidence": "live_metrics_analysis",
    }
