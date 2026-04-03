"""Shared tuning preview logic for live analysis surfaces."""

from __future__ import annotations

from typing import Any

from inferscope.optimization.checks import AuditFinding
from inferscope.profiling.models import TuningAdjustment, TuningPreview
from inferscope.telemetry.normalizer import NormalizedMetrics

DEFAULT_SCHEDULER_PREVIEW_CONFIG: dict[str, Any] = {
    "batched_token_budget": 8192,
    "max_num_seqs": 256,
    "decode_priority": 0.5,
    "chunked_prefill": True,
    "prefill_decode_isolation": "colocated",
    "prefill_lane_budget": 0,
    "decode_lane_budget": 0,
    "max_prefill_chunk_ratio": 0.5,
}

DEFAULT_CACHE_PREVIEW_CONFIG: dict[str, Any] = {
    "gpu_memory_utilization": 0.92,
    "offload_policy": "cold_only",
    "kv_compaction_trigger": 0.4,
    "fragmentation_check": False,
    "pcie_utilization_cap": 0.7,
}


def derive_adjustments(
    findings: list[AuditFinding],
    metrics: NormalizedMetrics,
    current_scheduler: dict[str, Any],
    current_cache: dict[str, Any],
) -> list[TuningAdjustment]:
    """Map detected failure modes to specific config adjustments."""
    adjustments: list[TuningAdjustment] = []
    seen_params: set[str] = set()

    for finding in findings:
        new_adjustments = _adjustments_for_finding(finding, metrics, current_scheduler, current_cache)
        for adjustment in new_adjustments:
            if adjustment.parameter not in seen_params:
                adjustments.append(adjustment)
                seen_params.add(adjustment.parameter)

    return adjustments


def _adjustments_for_finding(
    finding: AuditFinding,
    metrics: NormalizedMetrics,
    scheduler: dict[str, Any],
    cache: dict[str, Any],
) -> list[TuningAdjustment]:
    """Generate adjustments for a specific finding."""
    del metrics  # preserved for future tuning heuristics

    adjustments: list[TuningAdjustment] = []
    check_id = finding.check_id

    if check_id == "DECODE_STARVATION":
        current_priority = scheduler.get("decode_priority", 0.5)
        adjustments.append(
            TuningAdjustment(
                parameter="scheduler.decode_priority",
                current_value=current_priority,
                recommended_value=min(0.9, current_priority + 0.2),
                reason="Decode tokens starved by prefill batches — increase decode scheduling priority",
                confidence=0.8,
                trigger=check_id,
            )
        )
        current_ratio = scheduler.get("max_prefill_chunk_ratio", 0.5)
        adjustments.append(
            TuningAdjustment(
                parameter="scheduler.max_prefill_chunk_ratio",
                current_value=current_ratio,
                recommended_value=max(0.2, current_ratio - 0.15),
                reason="Limit prefill batch fraction to free scheduler budget for decode",
                confidence=0.75,
                trigger=check_id,
            )
        )

    elif check_id == "PREFILL_STARVATION":
        current_priority = scheduler.get("decode_priority", 0.5)
        adjustments.append(
            TuningAdjustment(
                parameter="scheduler.decode_priority",
                current_value=current_priority,
                recommended_value=max(0.2, current_priority - 0.2),
                reason="Prefill queued behind heavy decode — reduce decode priority to admit new requests",
                confidence=0.8,
                trigger=check_id,
            )
        )
        current_budget = scheduler.get("prefill_lane_budget", 0)
        target_budget = scheduler.get("batched_token_budget", 8192)
        adjustments.append(
            TuningAdjustment(
                parameter="scheduler.prefill_lane_budget",
                current_value=current_budget,
                recommended_value=target_budget // 2,
                reason="Reserve dedicated prefill budget to prevent decode from starving new requests",
                confidence=0.7,
                trigger=check_id,
            )
        )

    elif check_id == "PCIE_OFFLOAD_THRASH":
        adjustments.append(
            TuningAdjustment(
                parameter="cache.offload_policy",
                current_value=cache.get("offload_policy", "cold_only"),
                recommended_value="disabled",
                reason="PCIe transfer during active decode causes latency explosion — disable offloading",
                confidence=0.85,
                trigger=check_id,
            )
        )

    elif check_id == "GPU_UNDERUTILIZATION":
        current_seqs = scheduler.get("max_num_seqs", 256)
        adjustments.append(
            TuningAdjustment(
                parameter="scheduler.max_num_seqs",
                current_value=current_seqs,
                recommended_value=min(512, current_seqs * 2),
                reason="GPU has KV headroom but scheduler is limiting admissions",
                confidence=0.8,
                trigger=check_id,
            )
        )
        current_budget = scheduler.get("batched_token_budget", 8192)
        adjustments.append(
            TuningAdjustment(
                parameter="scheduler.batched_token_budget",
                current_value=current_budget,
                recommended_value=min(32768, current_budget * 2),
                reason="Raise batch budget to admit more tokens per step while GPU has capacity",
                confidence=0.75,
                trigger=check_id,
            )
        )

    elif check_id == "OOM_DESPITE_FREE":
        adjustments.append(
            TuningAdjustment(
                parameter="cache.fragmentation_check",
                current_value=cache.get("fragmentation_check", False),
                recommended_value=True,
                reason="Preemptions despite free KV — enable fragmentation monitoring",
                confidence=0.85,
                trigger=check_id,
            )
        )
        adjustments.append(
            TuningAdjustment(
                parameter="cache.kv_compaction_trigger",
                current_value=cache.get("kv_compaction_trigger", 0.4),
                recommended_value=0.3,
                reason="Lower compaction trigger to reclaim fragmented blocks earlier",
                confidence=0.75,
                trigger=check_id,
            )
        )

    elif check_id == "KV_FRAGMENTATION_HIGH":
        adjustments.append(
            TuningAdjustment(
                parameter="cache.kv_compaction_trigger",
                current_value=cache.get("kv_compaction_trigger", 0.4),
                recommended_value=0.5,
                reason="High KV usage with low active sequences — raise compaction trigger",
                confidence=0.7,
                trigger=check_id,
            )
        )

    elif check_id == "HIGH_TTFT":
        current_chunked = scheduler.get("chunked_prefill", True)
        if current_chunked:
            adjustments.append(
                TuningAdjustment(
                    parameter="scheduler.chunked_prefill",
                    current_value=True,
                    recommended_value=False,
                    reason="High TTFT may be caused by chunked prefill splitting — try contiguous prefill",
                    confidence=0.65,
                    trigger=check_id,
                )
            )

    elif check_id == "KV_CACHE_CRITICAL":
        current_util = cache.get("gpu_memory_utilization", 0.92)
        adjustments.append(
            TuningAdjustment(
                parameter="cache.gpu_memory_utilization",
                current_value=current_util,
                recommended_value=max(0.85, current_util - 0.03),
                reason="KV cache saturated — lower memory utilization to create headroom",
                confidence=0.8,
                trigger=check_id,
            )
        )

    return adjustments


def apply_adjustments(
    adjustments: list[TuningAdjustment],
    scheduler: dict[str, Any],
    cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply adjustments to copies of scheduler/cache dicts (for preview)."""
    updated_scheduler = dict(scheduler)
    updated_cache = dict(cache)

    for adjustment in adjustments:
        prefix, key = adjustment.parameter.split(".", 1)
        if prefix == "scheduler":
            updated_scheduler[key] = adjustment.recommended_value
        elif prefix == "cache":
            updated_cache[key] = adjustment.recommended_value

    return updated_scheduler, updated_cache


def build_tuning_preview(
    findings: list[AuditFinding],
    metrics: NormalizedMetrics,
    current_scheduler: dict[str, Any] | None = None,
    current_cache: dict[str, Any] | None = None,
) -> TuningPreview:
    """Build a structured tuning preview from findings."""
    scheduler = dict(DEFAULT_SCHEDULER_PREVIEW_CONFIG)
    if current_scheduler:
        scheduler.update(current_scheduler)

    cache = dict(DEFAULT_CACHE_PREVIEW_CONFIG)
    if current_cache:
        cache.update(current_cache)

    try:
        adjustments = derive_adjustments(findings, metrics, scheduler, cache)
        updated_scheduler, updated_cache = apply_adjustments(adjustments, scheduler, cache)
    except Exception as exc:  # noqa: BLE001
        return TuningPreview(
            adjustments=[],
            updated_scheduler=scheduler,
            updated_cache=cache,
            summary="Failed to derive tuning preview",
            error=str(exc),
        )

    summary = f"{len(adjustments)} config adjustment(s) suggested" if adjustments else "No config adjustments suggested"
    return TuningPreview(
        adjustments=adjustments,
        updated_scheduler=updated_scheduler,
        updated_cache=updated_cache,
        summary=summary,
    )
