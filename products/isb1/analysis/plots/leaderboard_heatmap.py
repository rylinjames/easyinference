"""Visual heatmap of the ISB-1 leaderboard."""

from __future__ import annotations

import csv
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_leaderboard_heatmap(
    leaderboard_csv: Path,
    output_dir: Path,
    metric: str = "goodput_delta_pct",
) -> Path:
    """Generate a heatmap visualization of the ISB-1 leaderboard.

    Rows: GPU × Model combinations
    Columns: Workloads
    Cell values: Mode B vs Mode A delta percentage for the specified metric
    Color: Green (improvement) → Red (regression)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read leaderboard CSV
    rows = []
    with open(leaderboard_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return output_dir

    # Build matrix
    row_labels = []
    col_labels = sorted(set(r.get("workload", "") for r in rows))
    row_label_set = set()

    for r in rows:
        label = f"{r.get('gpu', '').upper()} / {r.get('model', '')}"
        if label not in row_label_set:
            row_labels.append(label)
            row_label_set.add(label)

    matrix = np.full((len(row_labels), len(col_labels)), np.nan)
    annotations = [[" " for _ in col_labels] for _ in row_labels]

    for r in rows:
        label = f"{r.get('gpu', '').upper()} / {r.get('model', '')}"
        ri = row_labels.index(label)
        ci = col_labels.index(r.get("workload", ""))
        val = r.get(metric, "")
        try:
            matrix[ri][ci] = float(val)
            annotations[ri][ci] = f"{float(val):+.1f}%"
        except (ValueError, TypeError):
            annotations[ri][ci] = "N/A"

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 2.5), max(6, len(row_labels) * 0.6)))

    if HAS_SEABORN:
        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=np.array(annotations),
            fmt="",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": f"{metric} (%)"},
            vmin=-30,
            vmax=30,
        )
    else:
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-30, vmax=30)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, label=f"{metric} (%)")

    ax.set_title(
        "ISB-1 Leaderboard — Mode B vs Mode A Improvement",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Workload", fontsize=12)
    ax.set_ylabel("GPU / Model", fontsize=12)

    fig.tight_layout()
    fig_path = output_dir / "leaderboard_heatmap.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fig_path


@click.command()
@click.option("--input", "input_csv", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path())
@click.option("--metric", default="goodput_delta_pct")
def main(input_csv: str, output_dir: str, metric: str) -> None:
    """Generate leaderboard heatmap."""
    plot_leaderboard_heatmap(Path(input_csv), Path(output_dir), metric)
    click.echo(f"Heatmap saved to {output_dir}")


if __name__ == "__main__":
    main()
