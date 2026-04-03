"""Roofline model analysis for compute/memory bound classification."""

from __future__ import annotations

from dataclasses import dataclass

from inferscope.hardware.gpu_profiles import GPUProfile


@dataclass
class RooflineResult:
    """Result of a roofline analysis."""

    operational_intensity: float  # FLOPs per byte
    ridge_point: float  # OI where compute meets memory
    is_compute_bound: bool
    is_memory_bound: bool
    achievable_tflops: float
    peak_tflops: float
    memory_bandwidth_tb_s: float
    bottleneck: str  # "compute" | "memory"
    utilization_pct: float


def analyze_roofline(
    gpu: GPUProfile,
    flops_per_token: float,
    bytes_per_token: float,
    precision: str = "fp16",
) -> RooflineResult:
    """Analyze whether a workload is compute or memory bound on a GPU.

    Args:
        gpu: GPU hardware profile
        flops_per_token: FLOPs required per output token
        bytes_per_token: Bytes of memory accessed per output token
        precision: Compute precision (determines peak TFLOPS)
    """
    # Get peak TFLOPS for this precision
    tflops_key = f"{precision}_tc" if f"{precision}_tc" in gpu.peak_tflops else "fp16_tc"
    peak_tflops = gpu.peak_tflops.get(tflops_key, gpu.peak_tflops.get("fp16_tc", 100))

    # Ridge point: where compute = memory
    # ridge = peak_tflops / bandwidth_tb_s (in FLOP/byte)
    ridge = peak_tflops / gpu.memory_bandwidth_tb_s if gpu.memory_bandwidth_tb_s > 0 else 0

    # Operational intensity
    oi = flops_per_token / bytes_per_token if bytes_per_token > 0 else float("inf")

    is_compute = oi >= ridge
    is_memory = not is_compute

    # Achievable performance
    achievable = oi * gpu.memory_bandwidth_tb_s if is_memory else peak_tflops

    utilization = (achievable / peak_tflops * 100) if peak_tflops > 0 else 0

    return RooflineResult(
        operational_intensity=oi,
        ridge_point=ridge,
        is_compute_bound=is_compute,
        is_memory_bound=is_memory,
        achievable_tflops=achievable,
        peak_tflops=peak_tflops,
        memory_bandwidth_tb_s=gpu.memory_bandwidth_tb_s,
        bottleneck="compute" if is_compute else "memory",
        utilization_pct=min(100, utilization),
    )
