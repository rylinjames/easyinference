"""KV cache behavior analysis — cold start, prefix reuse, session growth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColdStartResult:
    """Cold start warmup measurement."""
    warmup_requests: int
    ttft_first_ms: float | None = None
    ttft_steady_ms: float | None = None
    warmup_speedup: float | None = None
    confidence_kind: str = "direct"


@dataclass
class PrefixReusePoint:
    """TTFT at a specific prefix sharing ratio."""
    prefix_ratio: float
    ttft_avg_ms: float | None = None
    ttft_p99_ms: float | None = None
    speedup_vs_zero: float | None = None
    prefix_cache_hit_rate: float | None = None
    confidence_kind: str = "direct"


@dataclass
class SessionGrowthPoint:
    """TTFT at a specific conversation turn."""
    turn: int
    accumulated_context_tokens: int
    ttft_ms: float | None = None
    confidence_kind: str = "direct"


@dataclass
class KVCacheBehaviorResult:
    """Combined cache behavior analysis."""
    model_name: str
    cold_start: ColdStartResult | None = None
    prefix_reuse: list[PrefixReusePoint] = field(default_factory=list)
    session_growth: list[SessionGrowthPoint] = field(default_factory=list)
    session_ttft_growth_ms_per_turn: float | None = None
    support_tier: str = ""
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "cold_start": {
                "warmup_requests": self.cold_start.warmup_requests,
                "ttft_first_ms": self.cold_start.ttft_first_ms,
                "ttft_steady_ms": self.cold_start.ttft_steady_ms,
                "warmup_speedup": self.cold_start.warmup_speedup,
                "confidence_kind": self.cold_start.confidence_kind,
            } if self.cold_start else None,
            "prefix_reuse": [
                {
                    "prefix_ratio": p.prefix_ratio,
                    "ttft_avg_ms": p.ttft_avg_ms,
                    "speedup_vs_zero": p.speedup_vs_zero,
                    "prefix_cache_hit_rate": p.prefix_cache_hit_rate,
                    "confidence_kind": p.confidence_kind,
                }
                for p in self.prefix_reuse
            ],
            "session_growth": [
                {
                    "turn": s.turn,
                    "accumulated_context_tokens": s.accumulated_context_tokens,
                    "ttft_ms": s.ttft_ms,
                    "confidence_kind": s.confidence_kind,
                }
                for s in self.session_growth
            ],
            "session_ttft_growth_ms_per_turn": self.session_ttft_growth_ms_per_turn,
            "support_tier": self.support_tier,
            "warnings": self.warnings,
            "summary": self.summary,
        }
