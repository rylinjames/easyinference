"""ComparisonGenerator -- compares benchmark results across modes or configurations.

Produces per-metric deltas, statistical significance, and a structured report
for Mode A vs Mode B (or any two configuration sets).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from analysis.statistical import (
    PairedTTestResult,
    coefficient_of_variation,
    paired_ttest,
)

# Metrics where lower is better (latency, error rate, power, preemptions)
_LOWER_IS_BETTER = frozenset({
    "ttft_p50", "ttft_p95", "ttft_p99",
    "tpot_p50", "tpot_p95", "tpot_p99",
    "itl_p50", "itl_p95", "itl_p99",
    "e2e_p50", "e2e_p95", "e2e_p99",
    "error_rate",
    "preemptions_per_minute",
    "kv_cache_utilization_p95",
    "queue_depth_p95",
    "avg_power_watts",
    "watts_per_token",
})

# Metrics where higher is better (throughput, goodput, cache hits)
_HIGHER_IS_BETTER = frozenset({
    "generation_throughput",
    "request_throughput",
    "goodput",
    "slo_attainment",
    "prefix_cache_hit_rate",
})

# Core metrics included in summary reports by default
_SUMMARY_METRICS = (
    "generation_throughput",
    "request_throughput",
    "goodput",
    "slo_attainment",
    "ttft_p95",
    "tpot_p95",
    "itl_p95",
    "e2e_p95",
    "error_rate",
    "prefix_cache_hit_rate",
)


@dataclass
class MetricDelta:
    """Comparison result for a single metric."""

    metric: str
    baseline_value: float
    candidate_value: float
    delta: float
    delta_pct: float
    direction: str  # "improved", "regressed", "unchanged"
    significant: bool  # paired t-test p < 0.05 (requires multi-trial)
    t_test: PairedTTestResult | None = None
    baseline_cv: float = 0.0
    candidate_cv: float = 0.0


@dataclass
class ComparisonReport:
    """Full comparison between two configurations."""

    baseline_label: str
    candidate_label: str
    cell_key: str
    metric_deltas: list[MetricDelta] = field(default_factory=list)
    improvements: int = 0
    regressions: int = 0
    unchanged: int = 0
    publishable: bool = False
    publish_blockers: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [
            f"Comparison: {self.baseline_label} vs {self.candidate_label}",
            f"Cell: {self.cell_key}",
            f"Results: {self.improvements} improved, "
            f"{self.regressions} regressed, {self.unchanged} unchanged",
        ]
        if not self.publishable:
            parts.append(f"Publish blockers: {', '.join(self.publish_blockers)}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "cell_key": self.cell_key,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "unchanged": self.unchanged,
            "publishable": self.publishable,
            "publish_blockers": self.publish_blockers,
            "metric_deltas": [],
        }
        for md in self.metric_deltas:
            entry: dict[str, Any] = {
                "metric": md.metric,
                "baseline_value": md.baseline_value,
                "candidate_value": md.candidate_value,
                "delta": md.delta,
                "delta_pct": md.delta_pct,
                "direction": md.direction,
                "significant": md.significant,
                "baseline_cv": md.baseline_cv,
                "candidate_cv": md.candidate_cv,
            }
            if md.t_test is not None:
                entry["t_test"] = {
                    "t_statistic": md.t_test.t_statistic,
                    "p_value": md.t_test.p_value,
                    "ci_95_lower": md.t_test.ci_95_lower,
                    "ci_95_upper": md.t_test.ci_95_upper,
                }
            d["metric_deltas"].append(entry)
        return d


def _classify_direction(metric: str, delta: float) -> str:
    """Classify a delta as improved, regressed, or unchanged."""
    if abs(delta) < 1e-9:
        return "unchanged"
    if metric in _LOWER_IS_BETTER:
        return "improved" if delta < 0 else "regressed"
    if metric in _HIGHER_IS_BETTER:
        return "improved" if delta > 0 else "regressed"
    # Unknown metric -- assume higher is better
    return "improved" if delta > 0 else "regressed"


def _cell_key_from_entry(entry: dict[str, Any]) -> str:
    return (
        f"{entry.get('gpu', '')}_{entry.get('model', '')}_"
        f"{entry.get('workload', '')}_{entry.get('quantization', '')}"
    )


class ComparisonGenerator:
    """Compares benchmark results between two modes or configurations.

    Typical usage::

        gen = ComparisonGenerator(analysis_data)
        report = gen.compare(mode_a="mode_a", mode_b="mode_b")
        print(report.summary)
    """

    def __init__(self, analysis_data: list[dict[str, Any]]) -> None:
        self._data = analysis_data

    @classmethod
    def from_json_file(cls, path: str | Path) -> ComparisonGenerator:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
        return cls(data)

    def compare(
        self,
        *,
        mode_a: str = "mode_a",
        mode_b: str = "mode_b",
        metrics: Sequence[str] = _SUMMARY_METRICS,
        min_trials_for_significance: int = 3,
    ) -> list[ComparisonReport]:
        """Compare all cells that have both *mode_a* and *mode_b* results.

        Returns one ``ComparisonReport`` per (gpu, model, workload, quant) cell.
        """
        # Group by (gpu, model, workload, quant) -> mode -> list of trial entries
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for entry in self._data:
            cell = _cell_key_from_entry(entry)
            mode = entry.get("mode", "")
            grouped.setdefault(cell, {}).setdefault(mode, []).append(entry)

        reports: list[ComparisonReport] = []
        for cell_key, modes in grouped.items():
            if mode_a not in modes or mode_b not in modes:
                continue

            a_trials = modes[mode_a]
            b_trials = modes[mode_b]

            report = self._compare_cell(
                cell_key=cell_key,
                baseline_label=mode_a,
                candidate_label=mode_b,
                baseline_trials=a_trials,
                candidate_trials=b_trials,
                metrics=metrics,
                min_trials=min_trials_for_significance,
            )
            reports.append(report)

        return reports

    def compare_two_sets(
        self,
        baseline: list[dict[str, Any]],
        candidate: list[dict[str, Any]],
        *,
        baseline_label: str = "baseline",
        candidate_label: str = "candidate",
        metrics: Sequence[str] = _SUMMARY_METRICS,
    ) -> ComparisonReport:
        """Compare two explicit sets of trial results for the same cell."""
        cell_key = _cell_key_from_entry(baseline[0]) if baseline else "unknown"
        return self._compare_cell(
            cell_key=cell_key,
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            baseline_trials=baseline,
            candidate_trials=candidate,
            metrics=metrics,
            min_trials=2,
        )

    def _compare_cell(
        self,
        *,
        cell_key: str,
        baseline_label: str,
        candidate_label: str,
        baseline_trials: list[dict[str, Any]],
        candidate_trials: list[dict[str, Any]],
        metrics: Sequence[str],
        min_trials: int,
    ) -> ComparisonReport:
        report = ComparisonReport(
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            cell_key=cell_key,
        )

        # Check publishability
        n_a = len(baseline_trials)
        n_b = len(candidate_trials)
        if n_a < 3:
            report.publish_blockers.append(f"{baseline_label} has {n_a} trials (need 3+)")
        if n_b < 3:
            report.publish_blockers.append(f"{candidate_label} has {n_b} trials (need 3+)")

        for metric in metrics:
            a_values = [
                float(t.get(metric, 0))
                for t in baseline_trials
                if _is_numeric(t.get(metric))
            ]
            b_values = [
                float(t.get(metric, 0))
                for t in candidate_trials
                if _is_numeric(t.get(metric))
            ]

            if not a_values or not b_values:
                continue

            a_mean = sum(a_values) / len(a_values)
            b_mean = sum(b_values) / len(b_values)
            delta = b_mean - a_mean
            delta_pct = (delta / a_mean * 100.0) if a_mean != 0 else 0.0

            a_cv = coefficient_of_variation(a_values) if len(a_values) >= 2 else 0.0
            b_cv = coefficient_of_variation(b_values) if len(b_values) >= 2 else 0.0

            # Statistical significance via paired t-test (only if enough trials)
            t_test_result: PairedTTestResult | None = None
            significant = False
            if len(a_values) >= min_trials and len(b_values) >= min_trials:
                n_pairs = min(len(a_values), len(b_values))
                try:
                    t_test_result = paired_ttest(a_values[:n_pairs], b_values[:n_pairs])
                    significant = t_test_result.significant
                except ValueError:
                    pass

            direction = _classify_direction(metric, delta)

            # Check CV for publishability
            if a_cv > 0.10:
                blocker = f"{baseline_label} {metric} CV={a_cv:.1%} (need <10%)"
                if blocker not in report.publish_blockers:
                    report.publish_blockers.append(blocker)
            if b_cv > 0.10:
                blocker = f"{candidate_label} {metric} CV={b_cv:.1%} (need <10%)"
                if blocker not in report.publish_blockers:
                    report.publish_blockers.append(blocker)

            md = MetricDelta(
                metric=metric,
                baseline_value=a_mean,
                candidate_value=b_mean,
                delta=delta,
                delta_pct=delta_pct,
                direction=direction,
                significant=significant,
                t_test=t_test_result,
                baseline_cv=a_cv,
                candidate_cv=b_cv,
            )
            report.metric_deltas.append(md)

            if direction == "improved":
                report.improvements += 1
            elif direction == "regressed":
                report.regressions += 1
            else:
                report.unchanged += 1

        report.publishable = len(report.publish_blockers) == 0
        return report

    def write_report_json(
        self,
        path: str | Path,
        reports: list[ComparisonReport],
    ) -> Path:
        """Write comparison reports to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in reports]
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path


def _is_numeric(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return False
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False
