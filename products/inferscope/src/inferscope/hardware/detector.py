"""GPU auto-detection from telemetry endpoints."""

from __future__ import annotations

from inferscope.hardware.gpu_profiles import GPUProfile, get_gpu_profile


def detect_gpu_from_name(name: str) -> GPUProfile | None:
    """Best-effort GPU detection from a name string (e.g., from nvidia-smi output)."""
    name_lower = name.lower()

    # Common patterns
    mappings = [
        ("a100-sxm", "a100_sxm_80gb"),
        ("a100-pcie", "a100_pcie_80gb"),
        ("a100 80gb", "a100_sxm_80gb"),
        ("a100 40gb", "a100_40gb"),
        ("a100", "a100"),
        ("a10g", "a10g"),
        ("h100 sxm", "h100_sxm"),
        ("h100 nvl", "h100_nvl"),
        ("h100 pcie", "h100_pcie"),
        ("h100", "h100"),
        ("h200 nvl", "h200_nvl"),
        ("h200", "h200"),
        ("gh200", "gh200"),
        ("grace hopper", "gh200"),
        ("b200", "b200"),
        ("b300", "b300"),
        ("gb200", "gb200"),
        ("grace blackwell", "gb200"),
        ("gb300", "gb300"),
        ("mi300x", "mi300x"),
        ("mi325x", "mi325x"),
        ("mi355x", "mi355x"),
    ]

    for pattern, key in mappings:
        if pattern in name_lower:
            return get_gpu_profile(key)

    return None
