"""LeaderboardGenerator -- ranks benchmark configurations and produces leaderboard artifacts.

Consumes the JSON output of ``isb1 analyze`` and produces:
- Ranked list of configurations by any metric
- CSV suitable for the leaderboard heatmap plot
- Mode A vs Mode B delta analysis when both modes are present
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class LeaderboardEntry:
    """A single row in the leaderboard."""

    rank: int
    gpu: str
    model: str
    workload: str
    mode: str
    quantization: str
    metric_name: str
    metric_value: float
    status: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class ModeComparison:
    """Delta between Mode A and Mode B for one (gpu, model, workload, quant) cell."""

    gpu: str
    model: str
    workload: str
    quantization: str
    mode_a_value: float
    mode_b_value: float
    delta: float
    delta_pct: float
    metric_name: str


class LeaderboardGenerator:
    """Generates ranked leaderboards and mode comparisons from analysis results."""

    def __init__(self, analysis_data: list[dict[str, Any]]) -> None:
        self._data = analysis_data

    @classmethod
    def from_json_file(cls, path: str | Path) -> LeaderboardGenerator:
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
        return cls(data)

    def rank(
        self,
        sort_by: str = "generation_throughput",
        *,
        top: int = 0,
        ascending: bool = False,
        filters: dict[str, str] | None = None,
    ) -> list[LeaderboardEntry]:
        """Return entries ranked by *sort_by*.

        Parameters
        ----------
        sort_by:
            Name of a metric field (e.g. ``generation_throughput``, ``goodput``,
            ``ttft_p95``).
        top:
            If > 0, return only the top *top* entries.
        ascending:
            If True, rank from lowest to highest (useful for latency metrics).
        filters:
            Optional key-value pairs to filter entries before ranking.
            Keys are field names (``gpu``, ``model``, ``workload``, ``mode``,
            ``quantization``); values are case-insensitive substrings.
        """
        entries = self._data
        if filters:
            entries = [
                e
                for e in entries
                if all(
                    v.lower() in str(e.get(k, "")).lower()
                    for k, v in filters.items()
                )
            ]

        def _sort_key(entry: dict[str, Any]) -> float:
            val = entry.get(sort_by, 0)
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        ranked = sorted(entries, key=_sort_key, reverse=not ascending)
        if top > 0:
            ranked = ranked[:top]

        return [
            LeaderboardEntry(
                rank=i,
                gpu=e.get("gpu", ""),
                model=e.get("model", ""),
                workload=e.get("workload", ""),
                mode=e.get("mode", ""),
                quantization=e.get("quantization", ""),
                metric_name=sort_by,
                metric_value=_sort_key(e),
                status=e.get("status", ""),
                raw=e,
            )
            for i, e in enumerate(ranked, 1)
        ]

    def compare_modes(
        self,
        metric: str = "goodput",
        *,
        mode_a: str = "mode_a",
        mode_b: str = "mode_b",
    ) -> list[ModeComparison]:
        """Compute deltas between two modes for each (gpu, model, workload, quant) cell.

        Returns a list sorted by ``delta_pct`` descending (biggest improvements first).
        """
        by_cell: dict[tuple[str, str, str, str], dict[str, float]] = {}
        for entry in self._data:
            key = (
                entry.get("gpu", ""),
                entry.get("model", ""),
                entry.get("workload", ""),
                entry.get("quantization", ""),
            )
            mode = entry.get("mode", "")
            try:
                val = float(entry.get(metric, 0))
            except (TypeError, ValueError):
                continue
            by_cell.setdefault(key, {})[mode] = val

        results: list[ModeComparison] = []
        for (gpu, model, workload, quant), modes in by_cell.items():
            if mode_a not in modes or mode_b not in modes:
                continue
            a_val = modes[mode_a]
            b_val = modes[mode_b]
            delta = b_val - a_val
            delta_pct = (delta / a_val * 100.0) if a_val != 0 else 0.0
            results.append(
                ModeComparison(
                    gpu=gpu,
                    model=model,
                    workload=workload,
                    quantization=quant,
                    mode_a_value=a_val,
                    mode_b_value=b_val,
                    delta=delta,
                    delta_pct=delta_pct,
                    metric_name=metric,
                )
            )

        results.sort(key=lambda c: c.delta_pct, reverse=True)
        return results

    def write_leaderboard_csv(
        self,
        path: str | Path,
        sort_by: str = "generation_throughput",
        *,
        top: int = 0,
        ascending: bool = False,
    ) -> Path:
        """Write a ranked leaderboard CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = self.rank(sort_by, top=top, ascending=ascending)
        if not entries:
            return path

        metric_fields = [
            "ttft_p95", "tpot_p95", "itl_p95", "e2e_p95",
            "generation_throughput", "request_throughput",
            "goodput", "slo_attainment", "prefix_cache_hit_rate",
            "error_rate",
        ]
        fieldnames = ["rank", "gpu", "model", "workload", "mode", "quantization"] + metric_fields

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                row: dict[str, Any] = {
                    "rank": e.rank,
                    "gpu": e.gpu,
                    "model": e.model,
                    "workload": e.workload,
                    "mode": e.mode,
                    "quantization": e.quantization,
                }
                for mf in metric_fields:
                    val = e.raw.get(mf, 0)
                    row[mf] = f"{val:.4f}" if isinstance(val, float) else val
                writer.writerow(row)

        return path

    def write_comparison_csv(
        self,
        path: str | Path,
        metrics: Sequence[str] = ("goodput", "generation_throughput", "ttft_p95", "tpot_p95"),
        *,
        mode_a: str = "mode_a",
        mode_b: str = "mode_b",
    ) -> Path:
        """Write a Mode A vs Mode B comparison CSV for the heatmap plot.

        Produces one row per (gpu, model, workload, quant) cell with
        ``{metric}_delta_pct`` columns.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        comparisons_by_metric: dict[str, list[ModeComparison]] = {}
        for metric in metrics:
            comparisons_by_metric[metric] = self.compare_modes(
                metric, mode_a=mode_a, mode_b=mode_b,
            )

        # Build a unified cell index
        cells: dict[tuple[str, str, str, str], dict[str, float]] = {}
        for metric, comps in comparisons_by_metric.items():
            for c in comps:
                key = (c.gpu, c.model, c.workload, c.quantization)
                cells.setdefault(key, {})[f"{metric}_delta_pct"] = c.delta_pct

        delta_cols = [f"{m}_delta_pct" for m in metrics]
        fieldnames = ["gpu", "model", "workload", "quantization"] + delta_cols

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for (gpu, model, workload, quant), deltas in cells.items():
                row: dict[str, Any] = {
                    "gpu": gpu,
                    "model": model,
                    "workload": workload,
                    "quantization": quant,
                }
                for col in delta_cols:
                    val = deltas.get(col, 0.0)
                    row[col] = f"{val:.1f}"
                writer.writerow(row)

        return path
