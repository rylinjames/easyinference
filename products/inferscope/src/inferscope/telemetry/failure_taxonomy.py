"""Failure-mode classification helpers for normalized runtime telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

from inferscope.telemetry.normalizer import NormalizedMetrics


class FailureMode(StrEnum):
    """Canonical failure modes for InferScope runtime diagnosis."""

    PREFILL_STARVATION = "prefill_starvation"
    DECODE_QUEUE_BACKUP = "decode_queue_backup"
    KV_TRANSFER_TIMEOUT = "kv_transfer_timeout"
    NIXL_FAILURE = "nixl_failure"
    WORKER_CRASH = "worker_crash"
    ROUTER_OVERLOAD = "router_overload"
    LMCACHE_MISS_STORM = "lmcache_miss_storm"


@dataclass(slots=True)
class ClassifiedFailure:
    """A structured failure-mode classification derived from runtime metrics."""

    mode: FailureMode
    severity: str
    confidence: float
    description: str
    evidence: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _append_if(condition: bool, findings: list[ClassifiedFailure], finding: ClassifiedFailure) -> None:
    if condition:
        findings.append(finding)


def classify_failure_modes(metrics: NormalizedMetrics) -> list[ClassifiedFailure]:
    """Classify likely failure modes from normalized telemetry."""

    findings: list[ClassifiedFailure] = []

    _append_if(
        metrics.ttft_avg_s is not None and metrics.itl_avg_s is not None and metrics.ttft_avg_s > 5.0 and metrics.itl_avg_s < 0.03,
        findings,
        ClassifiedFailure(
            mode=FailureMode.PREFILL_STARVATION,
            severity="warning",
            confidence=0.82,
            description="High TTFT with low ITL indicates decode-heavy batches blocking new prefills.",
            evidence=[
                f"ttft_avg_s={metrics.ttft_avg_s}",
                f"itl_avg_s={metrics.itl_avg_s}",
            ],
        ),
    )
    _append_if(
        metrics.requests_waiting > 10 and metrics.queue_time_avg_s is not None and metrics.queue_time_avg_s > 1.0,
        findings,
        ClassifiedFailure(
            mode=FailureMode.DECODE_QUEUE_BACKUP,
            severity="critical",
            confidence=0.8,
            description="Queue depth and queue time indicate decode-side backlog or scheduler overload.",
            evidence=[
                f"requests_waiting={metrics.requests_waiting}",
                f"queue_time_avg_s={metrics.queue_time_avg_s}",
            ],
        ),
    )
    _append_if(
        metrics.nixl_transfer_latency_s is not None and metrics.nixl_transfer_latency_s > 0.5,
        findings,
        ClassifiedFailure(
            mode=FailureMode.KV_TRANSFER_TIMEOUT,
            severity="warning",
            confidence=0.74,
            description="NIXL/KV transfer latency is high enough to dominate decode-side work.",
            evidence=[f"nixl_transfer_latency_s={metrics.nixl_transfer_latency_s}"],
        ),
    )
    _append_if(
        metrics.nixl_transfer_failures > 0,
        findings,
        ClassifiedFailure(
            mode=FailureMode.NIXL_FAILURE,
            severity="critical",
            confidence=0.93,
            description="NIXL transfer failures were observed during runtime.",
            evidence=[f"nixl_transfer_failures={metrics.nixl_transfer_failures}"],
        ),
    )
    _append_if(
        bool(metrics.scrape_error) or metrics.disconnected_clients > 0,
        findings,
        ClassifiedFailure(
            mode=FailureMode.WORKER_CRASH,
            severity="critical",
            confidence=0.7,
            description="Scrape failures or disconnected clients suggest worker instability.",
            evidence=[
                f"scrape_error={metrics.scrape_error}",
                f"disconnected_clients={metrics.disconnected_clients}",
            ],
        ),
    )
    _append_if(
        metrics.request_migrations_total > 0 or metrics.requests_waiting > 25,
        findings,
        ClassifiedFailure(
            mode=FailureMode.ROUTER_OVERLOAD,
            severity="warning",
            confidence=0.69,
            description="Request migrations or sustained queue depth suggest router overload.",
            evidence=[
                f"request_migrations_total={metrics.request_migrations_total}",
                f"requests_waiting={metrics.requests_waiting}",
            ],
        ),
    )
    _append_if(
        metrics.lmcache_hit_rate > 0 and metrics.lmcache_hit_rate < 0.2,
        findings,
        ClassifiedFailure(
            mode=FailureMode.LMCACHE_MISS_STORM,
            severity="warning",
            confidence=0.72,
            description="LMCache hit rate is low enough that remote cache fetches are likely not paying off.",
            evidence=[f"lmcache_hit_rate={metrics.lmcache_hit_rate}"],
        ),
    )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda item: (severity_order.get(item.severity, 3), -item.confidence))
    return findings


def dominant_failure_mode(metrics: NormalizedMetrics) -> ClassifiedFailure | None:
    """Return the highest-severity classified failure mode, if any."""

    findings = classify_failure_modes(metrics)
    return findings[0] if findings else None
