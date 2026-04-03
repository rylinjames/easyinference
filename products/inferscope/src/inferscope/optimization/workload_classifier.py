"""Workload classifier — detects coding/chat/agent from live metrics.

Uses request patterns, prefix reuse, session length, and queue behavior
to classify the dominant workload mode without user declaration.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferscope.optimization.serving_profile import WorkloadMode
from inferscope.telemetry.normalizer import NormalizedMetrics


@dataclass
class WorkloadClassification:
    """Result of workload classification from live metrics."""

    mode: WorkloadMode
    confidence: float  # 0-1, computed from signal strength
    signals: dict[str, str]  # signal_name → observation

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "confidence": round(self.confidence, 2),
            "signals": self.signals,
        }


def classify_workload(metrics: NormalizedMetrics) -> WorkloadClassification:
    """Classify the dominant workload from live normalized metrics.

    Signals used:
    - High prefix cache hits + moderate queue → coding
    - High concurrency + low queue time + short requests → chat
    - Session stickiness + tool-call patterns + long sessions → agent
    """
    signals: dict[str, str] = {}
    scores = {"coding": 0.0, "chat": 0.0, "agent": 0.0}

    # Signal 1: Prefix cache hit rate
    if metrics.prefix_cache_hit_rate > 0.7:
        scores["coding"] += 2.0
        signals["prefix_reuse"] = f"High ({metrics.prefix_cache_hit_rate:.0%}) — coding/agent pattern"
    elif metrics.prefix_cache_hit_rate > 0.3:
        scores["coding"] += 1.0
        scores["agent"] += 0.5
        signals["prefix_reuse"] = f"Moderate ({metrics.prefix_cache_hit_rate:.0%})"
    else:
        scores["chat"] += 1.0
        signals["prefix_reuse"] = f"Low ({metrics.prefix_cache_hit_rate:.0%}) — likely unique prompts"

    # Signal 2: Queue depth vs running — concurrency pattern
    if metrics.requests_running > 0:
        queue_ratio = metrics.requests_waiting / max(metrics.requests_running, 1)
        if queue_ratio > 2.0:
            scores["chat"] += 1.5  # high concurrency pressure = chat
            signals["concurrency"] = f"High queue pressure (ratio {queue_ratio:.1f}) — chat pattern"
        elif queue_ratio < 0.5 and metrics.requests_running < 20:
            scores["coding"] += 1.0  # low concurrency = interactive coding
            signals["concurrency"] = f"Low concurrency ({metrics.requests_running:.0f} running) — coding pattern"
        else:
            scores["agent"] += 0.5
            signals["concurrency"] = (
                f"Moderate ({metrics.requests_running:.0f} running, {metrics.requests_waiting:.0f} waiting)"
            )

    # Signal 3: TTFT (time to first token)
    if metrics.ttft_avg_s is not None:
        if metrics.ttft_avg_s < 0.3:
            scores["coding"] += 1.0  # fast TTFT = latency-sensitive coding
            signals["ttft"] = f"Fast ({metrics.ttft_avg_s * 1000:.0f}ms) — latency-sensitive"
        elif metrics.ttft_avg_s > 3.0:
            scores["agent"] += 1.0  # slow TTFT = long-context agent
            signals["ttft"] = f"Slow ({metrics.ttft_avg_s * 1000:.0f}ms) — long-context/agent"
        else:
            scores["chat"] += 0.5
            signals["ttft"] = f"Moderate ({metrics.ttft_avg_s * 1000:.0f}ms)"

    # Signal 4: KV cache usage pattern
    if metrics.kv_cache_usage > 0.8:
        scores["agent"] += 1.5  # high KV = long sessions
        signals["kv_pressure"] = f"High ({metrics.kv_cache_usage:.0%}) — long-lived sessions"
    elif metrics.kv_cache_usage < 0.4:
        scores["chat"] += 1.0  # low KV = short requests
        signals["kv_pressure"] = f"Low ({metrics.kv_cache_usage:.0%}) — short requests"
    else:
        signals["kv_pressure"] = f"Moderate ({metrics.kv_cache_usage:.0%})"

    # Signal 5: Generation throughput (SGLang-specific)
    if metrics.gen_throughput_tps > 0 and metrics.gen_throughput_tps > 5000:
        scores["chat"] += 1.0  # high throughput = batch chat
        signals["throughput"] = f"High ({metrics.gen_throughput_tps:.0f} tok/s) — batch processing"

    # Pick winner
    winner = max(scores, key=scores.get)  # type: ignore[arg-type]
    total = sum(scores.values())
    confidence = scores[winner] / total if total > 0 else 0.33

    return WorkloadClassification(
        mode=WorkloadMode(winner),
        confidence=min(confidence, 0.95),  # cap — never claim certainty from heuristics
        signals=signals,
    )
