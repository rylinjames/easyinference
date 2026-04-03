"""Quality metrics vs quantization level plots."""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt


def plot_quality_degradation(
    quality_results_dir: Path,
    output_dir: Path,
) -> Path:
    """Generate quality degradation plots across quantization levels.

    Shows how quality metrics (ROUGE-L, HumanEval pass@1, MMLU-Pro accuracy)
    degrade as quantization becomes more aggressive.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result_files = list(quality_results_dir.glob("*.json"))

    if not result_files:
        return output_dir

    # Group by model
    model_data: dict[str, dict[str, dict]] = {}
    for result_path in result_files:
        with open(result_path) as f:
            data = json.load(f)
        model = data.get("model", "unknown")
        quant = data.get("quantization", "unknown")
        model_data.setdefault(model, {})[quant] = data

    quant_order = ["bf16", "fp8", "nvfp4", "int4"]
    quant_labels = {"bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4", "int4": "INT4"}
    metrics_to_plot = [
        ("rouge_l_fmeasure", "ROUGE-L F1"),
        ("humaneval_pass_at_1", "HumanEval pass@1"),
        ("mmlu_pro_accuracy", "MMLU-Pro Accuracy"),
    ]

    for model, quant_results in model_data.items():
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        fig.suptitle(f"Quality vs Quantization — {model}", fontsize=14, fontweight="bold")

        for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
            quants = []
            values = []
            for q in quant_order:
                if q in quant_results and metric_key in quant_results[q]:
                    quants.append(quant_labels.get(q, q))
                    values.append(quant_results[q][metric_key])

            if not quants:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(metric_label)
                continue

            colors = []
            bf16_val = quant_results.get("bf16", {}).get(metric_key)
            for v in values:
                if bf16_val is not None and bf16_val > 0:
                    delta = (v - bf16_val) / bf16_val
                    if delta >= -0.01:
                        colors.append("#2ca02c")  # green
                    elif delta >= -0.02:
                        colors.append("#ff7f0e")  # yellow/orange
                    else:
                        colors.append("#d62728")  # red
                else:
                    colors.append("#1f77b4")

            bars = ax.bar(quants, values, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_title(metric_label, fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig.tight_layout()
        fig_path = output_dir / f"quality_degradation_{model}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return output_dir


@click.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path())
def main(input_dir: str, output_dir: str) -> None:
    """Generate quality degradation plots."""
    plot_quality_degradation(Path(input_dir), Path(output_dir))
    click.echo(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
