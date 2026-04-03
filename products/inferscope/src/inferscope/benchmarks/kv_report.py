"""KV cache benchmark report generation — markdown/JSON from phase results."""

from __future__ import annotations

from typing import Any

from inferscope.benchmarks.kv_capacity_probe import KVCapacityProbeResult
from inferscope.benchmarks.kv_pressure_ramp import KVPressureProfileResult
from inferscope.benchmarks.kv_cache_behavior import KVCacheBehaviorResult
from inferscope.benchmarks.kv_disagg_transfer import KVDisaggTransferResult


def generate_kv_report(
    *,
    capacity: KVCapacityProbeResult | None = None,
    pressure: KVPressureProfileResult | None = None,
    cache_behavior: KVCacheBehaviorResult | None = None,
    disagg_transfer: KVDisaggTransferResult | None = None,
) -> dict[str, Any]:
    """Generate a combined KV benchmark report from phase results."""
    report: dict[str, Any] = {
        "phases_completed": [],
        "recommendations": [],
        "warnings": [],
    }

    if capacity:
        report["phases_completed"].append("kv_capacity_probe")
        report["capacity"] = capacity.to_dict()
        report["warnings"].extend(capacity.warnings)

        # Generate capacity recommendations
        if capacity.capacity_curve:
            worst = min(capacity.capacity_curve, key=lambda p: p.max_concurrent)
            if worst.max_concurrent < 5:
                report["recommendations"].append(
                    f"At {worst.isl // 1024}K context, only {worst.max_concurrent} concurrent sessions fit. "
                    "Consider enabling KV cache offloading to CPU DRAM via LMCache."
                )

    if pressure:
        report["phases_completed"].append("kv_pressure_profile")
        report["pressure"] = pressure.to_dict()
        report["warnings"].extend(pressure.warnings)

        if pressure.cliff_pressure_pct:
            report["recommendations"].append(
                f"Performance cliff at {pressure.cliff_pressure_pct:.0f}% capacity "
                f"(type: {pressure.cliff_type}). Set admission control at "
                f"{pressure.cliff_pressure_pct * 0.8:.0f}% to maintain SLO."
            )

    if cache_behavior:
        report["phases_completed"].append("kv_cache_behavior")
        report["cache_behavior"] = cache_behavior.to_dict()
        report["warnings"].extend(cache_behavior.warnings)

        if cache_behavior.session_ttft_growth_ms_per_turn:
            growth = cache_behavior.session_ttft_growth_ms_per_turn
            report["recommendations"].append(
                f"TTFT grows {growth:.0f}ms per conversation turn. "
                f"At 500ms SLO, sessions are sustainable for ~{int(500 / growth)} turns."
            )

    if disagg_transfer:
        report["phases_completed"].append("kv_disagg_transfer")
        report["disagg_transfer"] = disagg_transfer.to_dict()
        report["warnings"].extend(disagg_transfer.warnings)

        if disagg_transfer.avg_decode_idle_fraction and disagg_transfer.avg_decode_idle_fraction > 0.3:
            report["recommendations"].append(
                f"Decode workers idle {disagg_transfer.avg_decode_idle_fraction:.0%} of the time "
                "waiting for KV transfer. Consider reducing prefill batch size or "
                "increasing prefill worker count."
            )

    report["summary"] = (
        f"{len(report['phases_completed'])} phases completed. "
        f"{len(report['recommendations'])} recommendations. "
        f"{len(report['warnings'])} warnings."
    )

    return report
