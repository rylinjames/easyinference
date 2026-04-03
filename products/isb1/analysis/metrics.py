"""CellMetrics dataclass and MetricComputer for ISB-1 benchmark analysis.

CRITICAL invariant: TTFT is EXCLUDED from TPOT and ITL calculations.
  TPOT = (e2e_latency - ttft) / (output_tokens - 1)
  ITL  = per-token inter-token gaps starting from token index 2
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class CellMetrics:
    """All metrics for a single benchmark cell (gpu x model x workload x mode x quant)."""

    # ── latency: Time To First Token (seconds) ──────────────────────────
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0

    # ── latency: Time Per Output Token (seconds) ────────────────────────
    tpot_p50: float = 0.0
    tpot_p95: float = 0.0
    tpot_p99: float = 0.0

    # ── latency: Inter-Token Latency (seconds) ──────────────────────────
    itl_p50: float = 0.0
    itl_p95: float = 0.0
    itl_p99: float = 0.0

    # ── latency: End-to-End (seconds) ───────────────────────────────────
    e2e_p50: float = 0.0
    e2e_p95: float = 0.0
    e2e_p99: float = 0.0

    # ── throughput ──────────────────────────────────────────────────────
    generation_throughput: float = 0.0  # tokens/s across all concurrent requests
    request_throughput: float = 0.0     # requests/s

    # ── goodput & SLO ───────────────────────────────────────────────────
    goodput: float = 0.0               # good requests / wall-clock second
    slo_attainment: float = 0.0        # fraction of requests meeting SLOs

    # ── caching ─────────────────────────────────────────────────────────
    prefix_cache_hit_rate: float = 0.0

    # ── KV cache utilisation ────────────────────────────────────────────
    kv_cache_utilization_p50: float = 0.0
    kv_cache_utilization_p95: float = 0.0

    # ── scheduling ──────────────────────────────────────────────────────
    preemptions_per_minute: float = 0.0
    queue_depth_p50: float = 0.0
    queue_depth_p95: float = 0.0

    # ── errors ──────────────────────────────────────────────────────────
    error_rate: float = 0.0

    # ── GPU utilization ─────────────────────────────────────────────────
    gpu_utilization_mean: float = 0.0   # mean SM utilization 0.0–1.0 across GPUs
    gpu_utilization_p5: float = 0.0    # p5 (idle floor, useful for disagg)

    # ── power ───────────────────────────────────────────────────────────
    avg_power_watts: float = 0.0
    watts_per_token: float = 0.0

    # ── economics ─────────────────────────────────────────────────────
    tokens_per_dollar_hour: float = 0.0   # gen_throughput * 3600 / gpu_hourly_cost
    cost_per_million_tokens: float = 0.0  # (gpu_hourly_cost / 3600) / gen_throughput * 1e6
    tokens_per_watt: float = 0.0          # gen_throughput / avg_power_watts

    # ── goodput economics (derived from inferencebreakpoints/14-inference-economics) ──
    goodput_per_dollar_hour: float = 0.0  # goodput * 3600 / gpu_hourly_cost
    cost_per_million_good_tokens: float = 0.0  # cost accounting for waste
    waste_ratio: float = 0.0              # 1 - (goodput / throughput)

    # ── request counts ──────────────────────────────────────────────────
    total_requests: int = 0
    successful_requests: int = 0

    # ── measurement ─────────────────────────────────────────────────────
    measurement_duration_seconds: float = 0.0

    # ── serialisation helpers ───────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    def to_csv_row(self) -> str:
        """Return a single CSV row (with header available via csv_header())."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([getattr(self, f.name) for f in fields(self)])
        return buf.getvalue().rstrip("\r\n")

    @classmethod
    def csv_header(cls) -> str:
        """Return the CSV header row matching to_csv_row() column order."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([f.name for f in fields(cls)])
        return buf.getvalue().rstrip("\r\n")

    @classmethod
    def from_dict(cls, data: dict) -> "CellMetrics":
        """Construct from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ── Helpers ──────────────────────────────────────────────────────────────


def _safe_percentile(values: Sequence[float], q: float) -> float:
    """Compute percentile, returning 0.0 for empty sequences."""
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, q))


def _compute_tpot(e2e: float, ttft: float, output_tokens: int) -> Optional[float]:
    """TPOT = (e2e - ttft) / (output_tokens - 1).  Returns None if < 2 tokens."""
    if output_tokens < 2:
        return None
    decode_time = e2e - ttft
    if decode_time < 0:
        return None
    return decode_time / (output_tokens - 1)


def _compute_itl_gaps(token_timestamps: List[float]) -> List[float]:
    """Compute inter-chunk latency gaps from SSE token timestamps.

    vLLM and other engines batch multiple tokens into single SSE chunks,
    producing sub-millisecond gaps between tokens in the same chunk.
    This function filters out intra-chunk gaps (< 1ms) to measure the
    actual inter-chunk delivery latency that users experience.

    token_timestamps[0] is the first token (TTFT boundary).
    Gaps are computed from index 1 onward, excluding the TTFT boundary.
    """
    if len(token_timestamps) < 2:
        return []
    # Compute all gaps from index 1 onward (skip TTFT gap at index 0->1)
    raw_gaps = [
        token_timestamps[i] - token_timestamps[i - 1]
        for i in range(2, len(token_timestamps))
    ]
    # Filter out intra-chunk gaps (< 1ms) — these are multiple tokens
    # batched into a single SSE event, not real decode latency
    return [g for g in raw_gaps if g >= 0.001]


# ── GPU cost table (per-GPU on-demand $/hr from public cloud pricing) ────

_GPU_HOURLY_COST: Dict[str, float] = {
    # NVIDIA Hopper
    "h100": 3.50,
    "h100_sxm": 3.50,
    "h100_pcie": 2.50,
    "h100_nvl": 3.50,
    "h200": 5.50,
    "h200_sxm": 5.50,
    "h800": 3.00,
    "h20": 1.50,
    "gh200": 6.00,
    # NVIDIA Blackwell
    "b100": 5.00,
    "b200": 6.50,
    "b300": 9.00,
    "gb200": 8.00,
    "gb300": 10.00,
    # NVIDIA Ampere
    "a100": 2.00,
    "a100_80gb": 2.00,
    "a100_40gb": 1.50,
    "a10g": 0.75,
    # AMD
    "mi300x": 3.00,
    "mi325x": 4.00,
    "mi350x": 5.50,
    "mi355x": 5.50,
}


def get_gpu_hourly_cost(gpu_name: str) -> float:
    """Look up on-demand hourly cost for a GPU. Returns 0.0 if unknown."""
    key = gpu_name.lower().replace("-", "_").replace(" ", "_")
    if key in _GPU_HOURLY_COST:
        return _GPU_HOURLY_COST[key]
    # Fuzzy match: prefer longest matching key to avoid h20-in-h200 collisions
    best_match = ""
    best_cost = 0.0
    for k, v in _GPU_HOURLY_COST.items():
        if k in key and len(k) > len(best_match):
            best_match = k
            best_cost = v
    return best_cost


# ── Main computer ────────────────────────────────────────────────────────


class MetricComputer:
    """Compute CellMetrics from raw benchmark data.

    Parameters
    ----------
    ttft_slo : float
        TTFT SLO threshold in seconds (default 2.0s).
    tpot_slo : float
        TPOT SLO threshold in seconds (default 0.1s).
    gpu_hourly_cost : float
        Per-GPU on-demand hourly cost in USD.  If 0.0, economic metrics
        are skipped.  When *gpu_name* is provided and cost is 0.0, the
        cost is looked up from the built-in table.
    gpu_name : str
        Optional GPU name for automatic cost lookup.
    gpu_count : int
        Number of GPUs used (cost is multiplied by this).
    """

    def __init__(
        self,
        ttft_slo: float = 2.0,
        tpot_slo: float = 0.1,
        gpu_hourly_cost: float = 0.0,
        gpu_name: str = "",
        gpu_count: int = 1,
    ) -> None:
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        if gpu_hourly_cost > 0:
            self.total_hourly_cost = gpu_hourly_cost * gpu_count
        elif gpu_name:
            self.total_hourly_cost = get_gpu_hourly_cost(gpu_name) * gpu_count
        else:
            self.total_hourly_cost = 0.0

    def compute(
        self,
        latency_data: List[Dict[str, Any]],
        engine_metrics: Optional[List[Dict[str, Any]]] = None,
        gpu_telemetry: Optional[List[Dict[str, Any]]] = None,
    ) -> CellMetrics:
        """Compute all metrics from raw measurement data.

        Parameters
        ----------
        latency_data : list[dict]
            Per-request records.  Expected keys:
            - ttft: float (seconds)
            - e2e_latency: float (seconds)
            - output_tokens: int
            - token_timestamps: list[float] (optional, absolute times per token)
            - error: bool (optional, default False)
            - input_tokens: int (optional)
            - timestamp: float (request start epoch, optional)

        engine_metrics : list[dict], optional
            Time-series snapshots from the inference engine.  Expected keys:
            - kv_cache_utilization: float (0-1)
            - prefix_cache_hit_rate: float (0-1)
            - preemptions: int (cumulative or per-snapshot)
            - queue_depth: int
            - timestamp: float

        gpu_telemetry : list[dict], optional
            Time-series GPU readings.  Expected keys:
            - power_watts: float
            - timestamp: float
        """
        engine_metrics = engine_metrics or []
        gpu_telemetry = gpu_telemetry or []

        # ── separate successful / failed requests ────────────────────────
        successful = [r for r in latency_data if not r.get("error", False)]
        total = len(latency_data)
        n_success = len(successful)
        n_errors = total - n_success

        # ── measurement duration ─────────────────────────────────────────
        wall_clock = self._wall_clock(latency_data)

        # ── TTFT ─────────────────────────────────────────────────────────
        ttfts = [r["ttft"] for r in successful if "ttft" in r]

        # ── TPOT (TTFT excluded) ─────────────────────────────────────────
        tpots: List[float] = []
        for r in successful:
            t = _compute_tpot(
                r.get("e2e_latency", 0.0),
                r.get("ttft", 0.0),
                r.get("output_tokens", 0),
            )
            if t is not None:
                tpots.append(t)

        # ── ITL (TTFT excluded) ──────────────────────────────────────────
        all_itl_gaps: List[float] = []
        for r in successful:
            ts = r.get("token_timestamps", [])
            all_itl_gaps.extend(_compute_itl_gaps(ts))

        # ── E2E latency ──────────────────────────────────────────────────
        e2es = [r["e2e_latency"] for r in successful if "e2e_latency" in r]

        # ── throughput ───────────────────────────────────────────────────
        total_output_tokens = sum(r.get("output_tokens", 0) for r in successful)
        gen_throughput = total_output_tokens / wall_clock if wall_clock > 0 else 0.0
        req_throughput = n_success / wall_clock if wall_clock > 0 else 0.0

        # ── goodput (both TTFT and TPOT SLOs must be met) ────────────────
        good_count = 0
        for r in successful:
            ttft_threshold = r.get("ttft_slo_seconds", self.ttft_slo)
            tpot_threshold = r.get("tpot_slo_seconds", self.tpot_slo)
            ttft_val = r.get("ttft")
            ttft_ok = ttft_val is not None and ttft_val <= ttft_threshold
            tpot_val = _compute_tpot(
                r.get("e2e_latency", 0.0),
                r.get("ttft", 0.0),
                r.get("output_tokens", 0),
            )
            tpot_ok = tpot_val is not None and tpot_val <= tpot_threshold
            if ttft_ok and tpot_ok:
                good_count += 1
        goodput = good_count / wall_clock if wall_clock > 0 else 0.0
        slo_attainment = good_count / n_success if n_success > 0 else 0.0

        # ── engine metrics ───────────────────────────────────────────────
        kv_utils = [e["kv_cache_utilization"] for e in engine_metrics if "kv_cache_utilization" in e]
        cache_hits = [e["prefix_cache_hit_rate"] for e in engine_metrics if "prefix_cache_hit_rate" in e]
        preemptions = [e.get("preemptions", 0) for e in engine_metrics]
        queue_depths = [e.get("queue_depth", 0) for e in engine_metrics]

        prefix_hit = float(np.mean(cache_hits)) if cache_hits else 0.0

        total_preemptions = max(preemptions) - min(preemptions) if len(preemptions) >= 2 else 0
        preempt_per_min = (total_preemptions / (wall_clock / 60.0)) if wall_clock > 0 else 0.0

        # ── GPU telemetry ────────────────────────────────────────────────
        powers = [g["power_watts"] for g in gpu_telemetry if "power_watts" in g]
        avg_power = float(np.mean(powers)) if powers else 0.0
        w_per_tok = avg_power / gen_throughput if gen_throughput > 0 else 0.0

        # ── economics ─────────────────────────────────────────────────
        tok_per_watt = gen_throughput / avg_power if avg_power > 0 else 0.0
        if self.total_hourly_cost > 0 and gen_throughput > 0:
            tok_per_dollar_hr = gen_throughput * 3600.0 / self.total_hourly_cost
            cost_per_m_tok = (self.total_hourly_cost / 3600.0) / gen_throughput * 1e6
        else:
            tok_per_dollar_hr = 0.0
            cost_per_m_tok = 0.0

        # ── goodput economics (grounded in inferencebreakpoints/14-inference-economics) ──
        # Waste = compute spent on preempted, SLO-violating, or failed requests
        waste_ratio = 1.0 - (goodput / req_throughput) if req_throughput > 0 else 0.0
        waste_ratio = max(0.0, min(waste_ratio, 1.0))
        if self.total_hourly_cost > 0 and goodput > 0:
            goodput_per_dollar_hr = goodput * 3600.0 / self.total_hourly_cost
            cost_per_m_good_tok = (self.total_hourly_cost / 3600.0) / goodput * 1e6
        else:
            goodput_per_dollar_hr = 0.0
            cost_per_m_good_tok = 0.0

        return CellMetrics(
            ttft_p50=_safe_percentile(ttfts, 50),
            ttft_p95=_safe_percentile(ttfts, 95),
            ttft_p99=_safe_percentile(ttfts, 99),
            tpot_p50=_safe_percentile(tpots, 50),
            tpot_p95=_safe_percentile(tpots, 95),
            tpot_p99=_safe_percentile(tpots, 99),
            itl_p50=_safe_percentile(all_itl_gaps, 50),
            itl_p95=_safe_percentile(all_itl_gaps, 95),
            itl_p99=_safe_percentile(all_itl_gaps, 99),
            e2e_p50=_safe_percentile(e2es, 50),
            e2e_p95=_safe_percentile(e2es, 95),
            e2e_p99=_safe_percentile(e2es, 99),
            generation_throughput=gen_throughput,
            request_throughput=req_throughput,
            goodput=goodput,
            slo_attainment=slo_attainment,
            prefix_cache_hit_rate=prefix_hit,
            kv_cache_utilization_p50=_safe_percentile(kv_utils, 50),
            kv_cache_utilization_p95=_safe_percentile(kv_utils, 95),
            preemptions_per_minute=preempt_per_min,
            queue_depth_p50=_safe_percentile(queue_depths, 50),
            queue_depth_p95=_safe_percentile(queue_depths, 95),
            error_rate=n_errors / total if total > 0 else 0.0,
            avg_power_watts=avg_power,
            watts_per_token=w_per_tok,
            tokens_per_dollar_hour=tok_per_dollar_hr,
            cost_per_million_tokens=cost_per_m_tok,
            tokens_per_watt=tok_per_watt,
            goodput_per_dollar_hour=goodput_per_dollar_hr,
            cost_per_million_good_tokens=cost_per_m_good_tok,
            waste_ratio=waste_ratio,
            total_requests=total,
            successful_requests=n_success,
            measurement_duration_seconds=wall_clock,
        )

    # ── internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _wall_clock(latency_data: List[Dict[str, Any]]) -> float:
        """Estimate measurement wall-clock duration from request data.

        Prefers explicit 'timestamp' + 'e2e_latency' fields.
        Falls back to sum of e2e latencies as rough approximation.
        """
        if not latency_data:
            return 0.0

        # Try timestamp-based calculation
        starts = [r["timestamp"] for r in latency_data if "timestamp" in r]
        ends = [
            r["timestamp"] + r.get("e2e_latency", 0.0)
            for r in latency_data
            if "timestamp" in r and "e2e_latency" in r
        ]
        if starts and ends:
            return max(ends) - min(starts)

        # Fallback: use explicit duration if present on any record
        durations = [r.get("measurement_duration_seconds", 0.0) for r in latency_data]
        if any(d > 0 for d in durations):
            return max(durations)

        # Last resort: sum of e2e (sequential assumption)
        return sum(r.get("e2e_latency", 0.0) for r in latency_data)
