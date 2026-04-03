"""Hardware intelligence tools — GPU specs, comparison, etc."""

from __future__ import annotations

from inferscope.hardware.gpu_profiles import GPUProfile, get_gpu_profile, list_gpus


def get_gpu_specs(gpu: str) -> dict:
    """Return complete ISA-level specs for a GPU.

    Includes tensor core instructions, memory hierarchy, cache sizes,
    FP8/FP4 support, interconnect bandwidth, and inference-specific notes.
    """
    profile = get_gpu_profile(gpu)
    if profile is None:
        available = list_gpus()
        return {
            "error": f"Unknown GPU: '{gpu}'",
            "available_gpus": available,
            "summary": f"GPU '{gpu}' not found. Available: {', '.join(available)}",
            "confidence": 0.0,
            "evidence": "lookup_failure",
        }

    return {
        "gpu": profile.to_dict(),
        "summary": _gpu_summary(profile),
        "confidence": 1.0,
        "evidence": "hardware_knowledge_base",
    }


def compare_gpus(gpu_a: str, gpu_b: str, workload: str = "inference") -> dict:
    """Side-by-side GPU comparison with inference-relevant metrics."""
    pa = get_gpu_profile(gpu_a)
    pb = get_gpu_profile(gpu_b)

    if pa is None or pb is None:
        missing = []
        if pa is None:
            missing.append(gpu_a)
        if pb is None:
            missing.append(gpu_b)
        return {
            "error": f"Unknown GPU(s): {missing}",
            "available_gpus": list_gpus(),
            "confidence": 0.0,
        }

    comparison = {
        "gpu_a": pa.to_dict(),
        "gpu_b": pb.to_dict(),
        "comparison": {
            "memory_gb": {
                "a": pa.memory_gb,
                "b": pb.memory_gb,
                "ratio": pb.memory_gb / pa.memory_gb if pa.memory_gb else 0,
            },
            "bandwidth_tb_s": {
                "a": pa.memory_bandwidth_tb_s,
                "b": pb.memory_bandwidth_tb_s,
                "ratio": pb.memory_bandwidth_tb_s / pa.memory_bandwidth_tb_s if pa.memory_bandwidth_tb_s else 0,
            },
            "fp8_tflops": {
                "a": pa.peak_tflops.get("fp8_tc", 0),
                "b": pb.peak_tflops.get("fp8_tc", 0),
            },
            "fp4_support": {"a": pa.fp4_support, "b": pb.fp4_support},
            "fp8_format": {"a": pa.fp8_format, "b": pb.fp8_format},
            "interconnect_bandwidth_gb_s": {
                "a": pa.nvlink_bandwidth_gb_s or pa.if_bandwidth_gb_s,
                "b": pb.nvlink_bandwidth_gb_s or pb.if_bandwidth_gb_s,
            },
        },
        "summary": _comparison_summary(pa, pb),
        "confidence": 1.0,
        "evidence": "hardware_knowledge_base",
    }
    return comparison


def _gpu_summary(p: GPUProfile) -> str:
    """Generate a human-readable summary of a GPU."""
    parts = [f"{p.name} ({p.architecture}, {p.compute_capability})"]
    parts.append(f"{p.memory_gb} GB {p.memory_type} @ {p.memory_bandwidth_tb_s} TB/s")
    if p.fp8_support:
        parts.append(f"FP8 ({p.fp8_format}): {p.peak_tflops.get('fp8_tc', 0):.0f} TFLOPS")
    if p.fp4_support:
        parts.append(
            f"FP4 ({p.fp4_format}): {p.peak_tflops.get('fp4_tc', p.peak_tflops.get('mxfp4_tc', 0)):.0f} TFLOPS"
        )
    if p.nvlink_bandwidth_gb_s:
        parts.append(f"NVLink {p.nvlink_version}: {p.nvlink_bandwidth_gb_s} GB/s")
    elif p.if_bandwidth_gb_s:
        parts.append(f"Infinity Fabric {p.infinity_fabric_version}: {p.if_bandwidth_gb_s} GB/s")
    return " | ".join(parts)


def _comparison_summary(a: GPUProfile, b: GPUProfile) -> str:
    """Generate a comparison summary."""
    parts = [f"{a.name} vs {b.name}:"]

    mem_ratio = b.memory_gb / a.memory_gb if a.memory_gb else 0
    if mem_ratio > 1:
        parts.append(f"{b.name} has {mem_ratio:.1f}x more memory")
    elif mem_ratio < 1:
        parts.append(f"{a.name} has {1 / mem_ratio:.1f}x more memory")

    bw_ratio = b.memory_bandwidth_tb_s / a.memory_bandwidth_tb_s if a.memory_bandwidth_tb_s else 0
    if bw_ratio > 1.1:
        parts.append(f"{b.name} has {bw_ratio:.1f}x more bandwidth")
    elif bw_ratio < 0.9:
        parts.append(f"{a.name} has {1 / bw_ratio:.1f}x more bandwidth")

    if a.fp8_support != b.fp8_support:
        has = a.name if a.fp8_support else b.name
        lacks = b.name if a.fp8_support else a.name
        parts.append(f"{has} has FP8 support, {lacks} does not")

    if a.fp8_format and b.fp8_format and a.fp8_format != b.fp8_format:
        parts.append(f"CAUTION: Different FP8 formats ({a.name}: {a.fp8_format}, {b.name}: {b.fp8_format})")

    return " ".join(parts)
