"""KV cache pressure ramp — degradation profiling at increasing load levels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PressurePoint:
    """Metrics at a single pressure level."""
    pressure_pct: float
    concurrency: int
    ttft_p50_ms: float | None = None
    ttft_p99_ms: float | None = None
    tpot_p50_ms: float | None = None
    tpot_p99_ms: float | None = None
    error_rate: float = 0.0
    preemption_count: float = 0.0
    kv_cache_usage: float = 0.0
    queue_depth_avg: float = 0.0
    throughput_tps: float = 0.0
    confidence_kind: str = "direct"


@dataclass
class KVPressureProfileResult:
    """Result of a pressure ramp across multiple load levels."""
    model_name: str
    isl: int
    base_capacity: int
    pressure_points: list[PressurePoint] = field(default_factory=list)
    cliff_pressure_pct: float | None = None
    cliff_type: str = ""  # "preemption" | "latency" | "error" | "queue"
    support_tier: str = ""
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "isl": self.isl,
            "base_capacity": self.base_capacity,
            "pressure_points": [
                {
                    "pressure_pct": p.pressure_pct,
                    "concurrency": p.concurrency,
                    "ttft_p50_ms": p.ttft_p50_ms,
                    "ttft_p99_ms": p.ttft_p99_ms,
                    "error_rate": round(p.error_rate, 4),
                    "preemption_count": p.preemption_count,
                    "kv_cache_usage": round(p.kv_cache_usage, 3),
                    "queue_depth_avg": round(p.queue_depth_avg, 1),
                    "throughput_tps": round(p.throughput_tps, 1),
                    "confidence_kind": p.confidence_kind,
                }
                for p in self.pressure_points
            ],
            "cliff_pressure_pct": self.cliff_pressure_pct,
            "cliff_type": self.cliff_type,
            "support_tier": self.support_tier,
            "warnings": self.warnings,
            "summary": self.summary,
        }
