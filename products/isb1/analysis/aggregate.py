"""ResultAggregator -- reads raw JSONL trial data and produces per-cell summary CSVs.

Usage as a module::

    python -m analysis.aggregate --input results/raw --output results/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from analysis.metrics import CellMetrics, MetricComputer
from analysis.statistical import bootstrap_ci, coefficient_of_variation


# ---------------------------------------------------------------------------
# Cell key
# ---------------------------------------------------------------------------

CellKey = Tuple[str, str, str, str, str]  # (gpu, model, workload, mode, quant)


def _cell_key(record: Dict[str, Any]) -> CellKey:
    """Extract the 5-tuple cell key from a raw record."""
    return (
        record.get("gpu", "unknown"),
        record.get("model", "unknown"),
        record.get("workload", "unknown"),
        record.get("mode", "unknown"),
        record.get("quant", "none"),
    )


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: {path}:{line_no}: {exc}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class ResultAggregator:
    """Aggregate raw JSONL benchmark results into per-cell summaries.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing ``*.jsonl`` files produced by the benchmark harness.
    output_dir : str or Path
        Directory where summary CSVs will be written.
    ttft_slo : float
        TTFT SLO threshold in seconds (forwarded to MetricComputer).
    tpot_slo : float
        TPOT SLO threshold in seconds (forwarded to MetricComputer).
    """

    def __init__(
        self,
        input_dir: str | Path = "results/raw",
        output_dir: str | Path = "results/summary",
        ttft_slo: float = 2.0,
        tpot_slo: float = 0.1,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.computer = MetricComputer(ttft_slo=ttft_slo, tpot_slo=tpot_slo)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[CellKey, Dict[str, Any]]:
        """Execute the full aggregation pipeline.

        Returns
        -------
        dict
            Mapping from cell key to a dict containing:
            - ``"trials"``: list of ``CellMetrics`` per trial
            - ``"mean"``: dict of mean metric values across trials
            - ``"ci"``: dict mapping metric name -> (lower, upper)
            - ``"cv"``: dict mapping metric name -> coefficient of variation
        """
        raw_records = self._load_all()
        grouped = self._group_by_cell(raw_records)
        summaries = self._compute_summaries(grouped)
        self._write_csvs(summaries)
        return summaries

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_all(self) -> List[Dict[str, Any]]:
        """Load all JSONL files from the input directory."""
        all_records: List[Dict[str, Any]] = []
        if not self.input_dir.exists():
            print(f"WARNING: input directory {self.input_dir} does not exist", file=sys.stderr)
            return all_records
        for jsonl_path in sorted(self.input_dir.glob("*.jsonl")):
            all_records.extend(_read_jsonl(jsonl_path))
        print(f"Loaded {len(all_records)} records from {self.input_dir}")
        return all_records

    @staticmethod
    def _group_by_cell(
        records: List[Dict[str, Any]],
    ) -> Dict[CellKey, Dict[int, List[Dict[str, Any]]]]:
        """Group records by cell key and trial number.

        Returns mapping: cell_key -> {trial_id -> [records]}.
        """
        grouped: Dict[CellKey, Dict[int, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in records:
            key = _cell_key(rec)
            trial = rec.get("trial", 0)
            grouped[key][trial].append(rec)
        return grouped

    def _compute_summaries(
        self,
        grouped: Dict[CellKey, Dict[int, List[Dict[str, Any]]]],
    ) -> Dict[CellKey, Dict[str, Any]]:
        """Compute per-cell, per-trial metrics and across-trial statistics."""
        summaries: Dict[CellKey, Dict[str, Any]] = {}

        for cell_key, trials_dict in grouped.items():
            trial_metrics: List[CellMetrics] = []
            for _trial_id, records in sorted(trials_dict.items()):
                # Split record streams by type
                latency_data = [r for r in records if r.get("record_type", "latency") == "latency"]
                engine_data = [r for r in records if r.get("record_type") == "engine"]
                gpu_data = [r for r in records if r.get("record_type") == "gpu"]

                # If no explicit record_type, treat all as latency data
                if not latency_data and not engine_data and not gpu_data:
                    latency_data = records

                metrics = self.computer.compute(
                    latency_data=latency_data,
                    engine_metrics=engine_data if engine_data else None,
                    gpu_telemetry=gpu_data if gpu_data else None,
                )
                trial_metrics.append(metrics)

            # Across-trial statistics
            metric_names = [f.name for f in fields(CellMetrics)]
            means: Dict[str, float] = {}
            cis: Dict[str, Tuple[float, float]] = {}
            cvs: Dict[str, float] = {}

            for name in metric_names:
                values = [float(getattr(m, name)) for m in trial_metrics]
                means[name] = float(np.mean(values)) if values else 0.0
                if len(values) >= 2:
                    ci_result = bootstrap_ci(values, n_bootstrap=2000, rng_seed=42)
                    cis[name] = (ci_result.lower, ci_result.upper)
                    cvs[name] = coefficient_of_variation(values)
                else:
                    cis[name] = (means[name], means[name])
                    cvs[name] = 0.0

            summaries[cell_key] = {
                "trials": trial_metrics,
                "mean": means,
                "ci": cis,
                "cv": cvs,
            }

        return summaries

    def _write_csvs(self, summaries: Dict[CellKey, Dict[str, Any]]) -> None:
        """Write per-cell summary CSVs to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Per-cell detail files ---
        for cell_key, summary in summaries.items():
            gpu, model, workload, mode, quant = cell_key
            safe_name = f"{gpu}_{model}_{workload}_{mode}_{quant}".replace("/", "_")
            cell_path = self.output_dir / f"{safe_name}.csv"

            with open(cell_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["trial"] + CellMetrics.csv_header().split(","))
                for i, m in enumerate(summary["trials"]):
                    writer.writerow([i] + m.to_csv_row().split(","))

        # --- Combined summary file ---
        summary_path = self.output_dir / "summary.csv"
        metric_names = [f.name for f in fields(CellMetrics)]

        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            header = ["gpu", "model", "workload", "mode", "quant", "n_trials"]
            for name in metric_names:
                header.extend([f"{name}_mean", f"{name}_ci_lower", f"{name}_ci_upper", f"{name}_cv"])
            writer.writerow(header)

            for cell_key, summary in sorted(summaries.items()):
                gpu, model, workload, mode, quant = cell_key
                n_trials = len(summary["trials"])
                row: List[Any] = [gpu, model, workload, mode, quant, n_trials]
                for name in metric_names:
                    row.append(f"{summary['mean'][name]:.6f}")
                    ci_lo, ci_hi = summary["ci"][name]
                    row.append(f"{ci_lo:.6f}")
                    row.append(f"{ci_hi:.6f}")
                    row.append(f"{summary['cv'][name]:.6f}")
                writer.writerow(row)

        print(f"Wrote {len(summaries)} cell summaries to {self.output_dir}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate raw ISB-1 benchmark JSONL results into summary CSVs."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/raw",
        help="Directory containing raw *.jsonl files (default: results/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/summary",
        help="Output directory for summary CSVs (default: results/summary)",
    )
    parser.add_argument("--ttft-slo", type=float, default=2.0, help="TTFT SLO in seconds")
    parser.add_argument("--tpot-slo", type=float, default=0.1, help="TPOT SLO in seconds")
    args = parser.parse_args()

    aggregator = ResultAggregator(
        input_dir=args.input,
        output_dir=args.output,
        ttft_slo=args.ttft_slo,
        tpot_slo=args.tpot_slo,
    )
    aggregator.run()


if __name__ == "__main__":
    main()
