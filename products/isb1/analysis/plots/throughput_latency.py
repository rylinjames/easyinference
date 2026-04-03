"""Pareto frontier curves: throughput (x) vs latency percentiles (y) for each mode."""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt


def plot_throughput_latency(
    aggregated_dir: Path,
    output_dir: Path,
    metric: str = "tpot_p95_ms",
) -> Path:
    """Generate throughput vs latency Pareto frontier plot.

    Args:
        aggregated_dir: Directory containing per-cell summary JSON files.
        output_dir: Directory to save the plot.
        metric: Latency metric for y-axis (default: tpot_p95_ms).

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = list(aggregated_dir.glob("*.json"))

    # Group by (gpu, model, workload) and mode
    groups: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for summary_path in summaries:
        with open(summary_path) as f:
            data = json.load(f)
        key = (data.get("gpu", ""), data.get("model", ""), data.get("workload", ""))
        mode = data.get("mode", "unknown")
        groups.setdefault(key, {}).setdefault(mode, []).append(data)

    figures = []
    for (gpu, model, workload), mode_data in groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        mode_colors = {"mode_a": "#1f77b4", "mode_b": "#ff7f0e", "mode_c": "#2ca02c"}

        for mode, cells in sorted(mode_data.items()):
            throughputs = []
            latencies = []
            for cell in cells:
                tp = cell.get("generation_throughput_toks", 0)
                lat = cell.get(metric, 0)
                if tp > 0 and lat > 0:
                    throughputs.append(tp)
                    latencies.append(lat)

            if not throughputs:
                continue

            # Sort by throughput for Pareto frontier
            sorted_pairs = sorted(zip(throughputs, latencies))
            tp_sorted = [p[0] for p in sorted_pairs]
            lat_sorted = [p[1] for p in sorted_pairs]

            # Compute Pareto frontier
            pareto_tp, pareto_lat = _pareto_frontier(tp_sorted, lat_sorted)

            color = mode_colors.get(mode, "#7f7f7f")
            label = mode.replace("_", " ").title()
            ax.scatter(throughputs, latencies, color=color, alpha=0.5, s=30)
            ax.plot(pareto_tp, pareto_lat, color=color, linewidth=2, label=f"{label} (Pareto)")

        ax.set_xlabel("Generation Throughput (tokens/s)", fontsize=12)
        ax.set_ylabel(f"{metric} (ms)", fontsize=12)
        ax.set_title(f"Throughput vs Latency — {gpu.upper()} / {model} / {workload}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig_path = output_dir / f"throughput_latency_{gpu}_{model}_{workload}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig_path)

    return output_dir


def _pareto_frontier(
    x: list[float], y: list[float], maximize_x: bool = True, minimize_y: bool = True
) -> tuple[list[float], list[float]]:
    """Compute Pareto frontier from (x, y) pairs."""
    sorted_pairs = sorted(zip(x, y), key=lambda p: (-p[0] if maximize_x else p[0]))
    pareto_x, pareto_y = [], []
    best_y = float("inf") if minimize_y else float("-inf")

    for px, py in sorted_pairs:
        if (minimize_y and py <= best_y) or (not minimize_y and py >= best_y):
            pareto_x.append(px)
            pareto_y.append(py)
            best_y = py

    return pareto_x, pareto_y


@click.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path())
@click.option("--metric", default="tpot_p95_ms")
def main(input_dir: str, output_dir: str, metric: str) -> None:
    """Generate throughput vs latency Pareto frontier plots."""
    plot_throughput_latency(Path(input_dir), Path(output_dir), metric)
    click.echo(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
