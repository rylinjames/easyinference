"""ClaimEvaluator — evaluates ISB-1 benchmark claims against measured data.

The ISB-1 standard defines 13 reproducible claims about inference serving
performance.  The evaluator checks each claim against benchmark results
and returns a verdict: SUPPORTED, REFUTED, or INCONCLUSIVE.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ClaimVerdict(str, Enum):
    """Possible outcomes for a claim evaluation."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ClaimResult:
    """Result of evaluating a single claim."""

    claim_id: str
    claim_text: str
    verdict: ClaimVerdict
    evidence: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


# ---------------------------------------------------------------------------
# The 13 ISB-1 claims
# ---------------------------------------------------------------------------

_CLAIMS: dict[str, dict[str, Any]] = {
    "C1": {
        "text": "TTFT P99 is under the specified SLO threshold.",
        "metric": "ttft_p99",
        "direction": "less",
    },
    "C2": {
        "text": "TPOT P99 is under the specified SLO threshold.",
        "metric": "tpot_p99",
        "direction": "less",
    },
    "C3": {
        "text": "Goodput meets or exceeds the target requests per second.",
        "metric": "goodput",
        "direction": "greater",
    },
    "C4": {
        "text": "SLO attainment exceeds 95%.",
        "metric": "slo_attainment",
        "direction": "greater",
        "default_threshold": 0.95,
    },
    "C5": {
        "text": "Error rate is below 1%.",
        "metric": "error_rate",
        "direction": "less",
        "default_threshold": 0.01,
    },
    "C6": {
        "text": "Generation throughput meets the target tokens per second.",
        "metric": "generation_throughput",
        "direction": "greater",
    },
    "C7": {
        "text": "Request throughput meets the target requests per second.",
        "metric": "request_throughput",
        "direction": "greater",
    },
    "C8": {
        "text": "Prefix cache hit rate exceeds the target.",
        "metric": "prefix_cache_hit_rate",
        "direction": "greater",
    },
    "C9": {
        "text": "KV cache utilization stays below saturation threshold.",
        "metric": "kv_cache_utilization_p95",
        "direction": "less",
        "default_threshold": 0.95,
    },
    "C10": {
        "text": "Preemptions per minute are below the acceptable limit.",
        "metric": "preemptions_per_minute",
        "direction": "less",
        "default_threshold": 5.0,
    },
    "C11": {
        "text": "Power efficiency (watts per token) is within budget.",
        "metric": "watts_per_token",
        "direction": "less",
    },
    "C12": {
        "text": "ITL P99 is under the specified threshold.",
        "metric": "itl_p99",
        "direction": "less",
    },
    "C13": {
        "text": "E2E latency P99 is under the specified threshold.",
        "metric": "e2e_p99",
        "direction": "less",
    },
}


class ClaimEvaluator:
    """Evaluate benchmark results against ISB-1 claims.

    Parameters
    ----------
    thresholds : dict, optional
        Per-claim thresholds keyed by claim ID (e.g. ``{"C1": 2.0}``).
        Falls back to ``default_threshold`` in the claim definition if
        not provided.
    min_requests : int
        Minimum number of successful requests required to evaluate a
        claim (default 30).  Below this, the verdict is INCONCLUSIVE.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        min_requests: int = 30,
    ) -> None:
        self.thresholds = thresholds or {}
        self.min_requests = min_requests

    @staticmethod
    def all_claim_ids() -> List[str]:
        """Return all defined claim IDs."""
        return list(_CLAIMS.keys())

    def evaluate(
        self,
        claim_id: str,
        metrics: Dict[str, Any],
    ) -> ClaimResult:
        """Evaluate a single claim against measured metrics.

        Parameters
        ----------
        claim_id : str
            One of C1-C13.
        metrics : dict
            Benchmark metrics (e.g. from ``CellMetrics.to_dict()``).

        Returns
        -------
        ClaimResult
        """
        if claim_id not in _CLAIMS:
            raise ValueError(f"Unknown claim: {claim_id}")

        claim = _CLAIMS[claim_id]
        metric_key = claim["metric"]
        claim_text = claim["text"]

        # Check for sufficient data
        total = metrics.get("successful_requests", metrics.get("total_requests", 0))
        if total < self.min_requests:
            return ClaimResult(
                claim_id=claim_id,
                claim_text=claim_text,
                verdict=ClaimVerdict.INCONCLUSIVE,
                evidence=f"Insufficient data: {total} requests (need {self.min_requests}).",
            )

        # Get metric value
        metric_value = metrics.get(metric_key)
        if metric_value is None:
            return ClaimResult(
                claim_id=claim_id,
                claim_text=claim_text,
                verdict=ClaimVerdict.INCONCLUSIVE,
                evidence=f"Metric '{metric_key}' not found in results.",
            )

        # Get threshold
        threshold = self.thresholds.get(claim_id, claim.get("default_threshold"))
        if threshold is None:
            return ClaimResult(
                claim_id=claim_id,
                claim_text=claim_text,
                verdict=ClaimVerdict.INCONCLUSIVE,
                evidence=f"No threshold defined for claim {claim_id}.",
            )

        # Evaluate direction
        direction = claim["direction"]
        if direction == "less":
            passed = metric_value < threshold
        else:  # "greater"
            passed = metric_value >= threshold

        verdict = ClaimVerdict.SUPPORTED if passed else ClaimVerdict.REFUTED
        evidence = (
            f"{metric_key}={metric_value:.4f} {'<' if direction == 'less' else '>='} "
            f"{threshold:.4f} -> {verdict.value}"
        )

        return ClaimResult(
            claim_id=claim_id,
            claim_text=claim_text,
            verdict=verdict,
            evidence=evidence,
            metric_value=float(metric_value),
            threshold=float(threshold),
        )

    def evaluate_all(
        self,
        metrics: Dict[str, Any],
    ) -> List[ClaimResult]:
        """Evaluate all 13 claims against the provided metrics."""
        return [self.evaluate(cid, metrics) for cid in _CLAIMS]
