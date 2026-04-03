"""Rate sweep plots: request rate (x) vs metrics (y) showing saturation behavior."""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt


def plot_concurrency_sweep(
    aggregated_dir: Path,
    output_dir: Path,
) -> Path:
    """Generate rate sweep plots showing saturation behavior.

    For each (gpu, model, workload), plot request rate vs key metrics
    with separate lines for each mode.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = list(aggregated_dir.glob("*.json"))

    # Group by (gpu, model, workload, mode) -> list of rate-point results
    groups: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for summary_path in summaries:
        with open(summary_path) as f:
            data = json.load(f)
        key = (data.get("gpu", ""), data.get("model", ""), data.get("workload", ""))
        mode = data.get("mode", "unknown")
        groups.setdefault(key, {}).setdefault(mode, []).append(data)

    metrics_to_plot = [
        ("generation_throughput_toks", "Throughput (tok/s)"),
        ("ttft_p95_ms", "TTFT P95 (ms)"),
        ("tpot_p95_ms", "TPOT P95 (ms)"),
        ("goodput_reqs", "Goodput (req/s)"),
    ]
    mode_colors = {"mode_a": "#1f77b4", "mode_b": "#ff7f0e", "mode_c": "#2ca02c"}

    for (gpu, model, workload), mode_data in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Rate Sweep — {gpu.upper()} / {model} / {workload}",
            fontsize=14,
            fontweight="bold",
        )

        for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
            ax = axes[idx // 2][idx % 2]

            for mode, cells in sorted(mode_data.items()):
                rates = []
                values = []
                for cell in cells:
                    rate = cell.get("request_rate", 0)
                    val = cell.get(metric_key, 0)
                    if rate > 0:
                        rates.append(rate)
                        values.append(val)

                if not rates:
                    continue

                sorted_pairs = sorted(zip(rates, values))
                r = [p[0] for p in sorted_pairs]
                v = [p[1] for p in sorted_pairs]

                color = mode_colors.get(mode, "#7f7f7f")
                label = mode.replace("_", " ").title()
                ax.plot(r, v, "o-", color=color, label=label, markersize=4, linewidth=1.5)

            ax.set_xlabel("Request Rate (req/s)", fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_xscale("log", base=2)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = output_dir / f"rate_sweep_{gpu}_{model}_{workload}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return output_dir


@click.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path())
def main(input_dir: str, output_dir: str) -> None:
    """Generate rate sweep plots."""
    plot_concurrency_sweep(Path(input_dir), Path(output_dir))
    click.echo(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
