"""GPU utilization, temperature, and power draw time series plots."""

from __future__ import annotations

import csv
from pathlib import Path

import click
import matplotlib.pyplot as plt


def plot_gpu_telemetry(
    telemetry_csv: Path,
    output_dir: Path,
    run_id: str = "",
) -> Path:
    """Generate GPU telemetry time series plots from CSV data.

    Args:
        telemetry_csv: Path to GPU telemetry CSV file.
        output_dir: Directory to save the plot.
        run_id: Run ID for the plot title.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV telemetry data
    timestamps = []
    gpu_util = []
    mem_util = []
    power_draw = []
    temperature = []
    gpu_clock = []

    with open(telemetry_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row.get("timestamp", 0))
            timestamps.append(ts)
            gpu_util.append(float(row.get("gpu_utilization_pct", 0)))
            mem_util.append(float(row.get("memory_utilization_pct", 0)))
            power_draw.append(float(row.get("power_draw_watts", 0)))
            temperature.append(float(row.get("temperature_gpu_celsius", 0)))
            gpu_clock.append(float(row.get("gpu_clock_mhz", 0)))

    if not timestamps:
        return output_dir

    # Normalize timestamps to start at 0
    t0 = timestamps[0]
    time_s = [t - t0 for t in timestamps]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"GPU Telemetry — {run_id}" if run_id else "GPU Telemetry", fontsize=14)

    # GPU Utilization
    axes[0].plot(time_s, gpu_util, color="#1f77b4", linewidth=0.8)
    axes[0].fill_between(time_s, gpu_util, alpha=0.2, color="#1f77b4")
    axes[0].set_ylabel("GPU Utilization (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)

    # Memory Utilization
    axes[1].plot(time_s, mem_util, color="#ff7f0e", linewidth=0.8)
    axes[1].fill_between(time_s, mem_util, alpha=0.2, color="#ff7f0e")
    axes[1].set_ylabel("Memory Utilization (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    # Power Draw
    axes[2].plot(time_s, power_draw, color="#2ca02c", linewidth=0.8)
    axes[2].fill_between(time_s, power_draw, alpha=0.2, color="#2ca02c")
    axes[2].set_ylabel("Power Draw (W)")
    axes[2].grid(True, alpha=0.3)

    # Temperature
    axes[3].plot(time_s, temperature, color="#d62728", linewidth=0.8)
    axes[3].fill_between(time_s, temperature, alpha=0.2, color="#d62728")
    axes[3].set_ylabel("Temperature (°C)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    suffix = f"_{run_id}" if run_id else ""
    fig_path = output_dir / f"gpu_telemetry{suffix}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fig_path


@click.command()
@click.option("--input", "input_csv", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path())
@click.option("--run-id", default="")
def main(input_csv: str, output_dir: str, run_id: str) -> None:
    """Generate GPU telemetry time series plots."""
    plot_gpu_telemetry(Path(input_csv), Path(output_dir), run_id)
    click.echo(f"Plot saved to {output_dir}")


if __name__ == "__main__":
    main()
