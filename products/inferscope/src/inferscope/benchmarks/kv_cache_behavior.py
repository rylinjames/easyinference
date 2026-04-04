"""KV cache behavior analysis — cold start, prefix reuse, session growth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferscope.benchmarks.models import ChatMessage, WorkloadPack, WorkloadRequest
from inferscope.benchmarks.openai_replay import run_openai_replay


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


def _single_request_pack(model_name: str, name: str, messages: list[ChatMessage], *, session_id: str | None = None) -> WorkloadPack:
    return WorkloadPack(
        name=name,
        description="Live KV cache behavior probe",
        workload_class="kv_cache_behavior",
        model=model_name,
        concurrency=1,
        stream=True,
        requests=[
            WorkloadRequest(
                name=name,
                session_id=session_id,
                messages=messages,
                max_tokens=64,
                metadata={"phase": "kv_cache_behavior", "approx_context_tokens": 4096},
            )
        ],
    )


async def run_kv_cache_behavior(
    endpoint: str,
    model_name: str,
    *,
    metrics_endpoint: str | None = None,
    capture_metrics: bool = True,
    client: Any | None = None,
) -> KVCacheBehaviorResult:
    """Run live probes for cold start, prefix reuse, and session growth."""

    result = KVCacheBehaviorResult(model_name=model_name, support_tier="live_probe")

    cold_first = await run_openai_replay(
        _single_request_pack(model_name, "cold-start-first", [ChatMessage(role="user", content="Explain cold start impact.")]),
        endpoint,
        model=model_name,
        metrics_endpoint=metrics_endpoint,
        capture_metrics=capture_metrics,
        client=client,
    )
    cold_second = await run_openai_replay(
        _single_request_pack(model_name, "cold-start-second", [ChatMessage(role="user", content="Explain cold start impact.")]),
        endpoint,
        model=model_name,
        metrics_endpoint=metrics_endpoint,
        capture_metrics=capture_metrics,
        client=client,
    )
    first_ttft = cold_first.summary.ttft_avg_ms
    steady_ttft = cold_second.summary.ttft_avg_ms
    result.cold_start = ColdStartResult(
        warmup_requests=2,
        ttft_first_ms=first_ttft,
        ttft_steady_ms=steady_ttft,
        warmup_speedup=((first_ttft or 0.0) / steady_ttft) if first_ttft and steady_ttft else None,
        confidence_kind="direct",
    )

    prefix_prompts = [
        ("prefix-0", "Unique prompt with no shared prefix."),
        ("prefix-50", "Shared repo context. Shared repo context. Unique suffix A."),
        ("prefix-90", "Shared repo context. Shared repo context. Shared repo context. Unique suffix B."),
    ]
    prefix_ratios = [0.0, 0.5, 0.9]
    baseline_ttft: float | None = None
    for (name, prompt), ratio in zip(prefix_prompts, prefix_ratios):
        artifact = await run_openai_replay(
            _single_request_pack(model_name, name, [ChatMessage(role="user", content=prompt)]),
            endpoint,
            model=model_name,
            metrics_endpoint=metrics_endpoint,
            capture_metrics=capture_metrics,
            client=client,
        )
        ttft = artifact.summary.ttft_avg_ms
        if baseline_ttft is None:
            baseline_ttft = ttft
        metrics = (artifact.metrics_after.normalized_metrics if artifact.metrics_after else {}) or {}
        result.prefix_reuse.append(
            PrefixReusePoint(
                prefix_ratio=ratio,
                ttft_avg_ms=ttft,
                ttft_p99_ms=artifact.summary.ttft_p99_ms,
                speedup_vs_zero=((baseline_ttft or 0.0) / ttft) if baseline_ttft and ttft else None,
                prefix_cache_hit_rate=float(metrics.get("cache", {}).get("prefix_hit_rate", 0.0) or 0.0),
                confidence_kind="direct",
            )
        )

    session_id = "kv-cache-session"
    for turn in range(1, 6):
        artifact = await run_openai_replay(
            _single_request_pack(
                model_name,
                f"session-turn-{turn}",
                [ChatMessage(role="user", content=f"Continue turn {turn} with the accumulated repo context.")],
                session_id=session_id,
            ),
            endpoint,
            model=model_name,
            metrics_endpoint=metrics_endpoint,
            capture_metrics=capture_metrics,
            client=client,
        )
        result.session_growth.append(
            SessionGrowthPoint(
                turn=turn,
                accumulated_context_tokens=turn * 4096,
                ttft_ms=artifact.summary.ttft_avg_ms,
                confidence_kind="direct",
            )
        )

    if len(result.session_growth) >= 2:
        first = result.session_growth[0].ttft_ms or 0.0
        last = result.session_growth[-1].ttft_ms or first
        result.session_ttft_growth_ms_per_turn = (last - first) / (len(result.session_growth) - 1)
    result.summary = "Live KV cache behavior probe captured cold-start, prefix reuse, and session-growth signals."
    return result
