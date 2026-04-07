"""GPU auto-detection from telemetry endpoints."""

from __future__ import annotations

from inferscope.hardware.gpu_profiles import GPUProfile, get_gpu_profile


def _normalize_for_match(s: str) -> str:
    """Normalize separators so a single pattern matches both `H100-SXM5` and
    `H100 SXM5`. Replaces hyphens and underscores with spaces, lowercases.

    Closes the snapshot v1.0.0 P1 bug `gpu_detector_correctness` sub-bug 2
    (the original implementation only `.lower()`'d the input but had mixed
    hyphen and space patterns, so half the conventions silently fell
    through to the bare-arch fallback).
    """
    return s.lower().replace("-", " ").replace("_", " ")


def detect_gpu_from_name(name: str) -> GPUProfile | None:
    """Best-effort GPU detection from a name string (e.g., from nvidia-smi output).

    Closes the snapshot v1.0.0 P1 bug `gpu_detector_correctness`. Three
    sub-bugs: missing B100/H800/H20/MI350X mappings, separator-convention
    mismatch (sub-bug 2), and ordering for `grace blackwell` vs `gb300`
    (sub-bug 3).
    """
    name_norm = _normalize_for_match(name)

    # Patterns are normalized to all-space convention to match _normalize_for_match.
    #
    # Order matters: more-specific patterns come BEFORE more-generic ones.
    # Critically, the Grace+Hopper/Blackwell SUPERCHIP patterns (`gh200`,
    # `gb200`, `gb300`) must be checked BEFORE the bare GPU patterns
    # (`h200`, `b200`, `b300`) because the bare names are substrings of
    # the superchip names. Sub-bug 3 in the original code: a `nvidia gh200`
    # input fell into the bare `h200` branch because `h200` was listed first.
    mappings = [
        # Grace+Blackwell / Grace+Hopper superchips — checked FIRST
        # because their names contain the bare GPU names as substrings.
        # Within this group, more-specific names come first
        # (`grace blackwell ultra` before `grace blackwell`,
        # `gb300` before `grace blackwell`, etc.).
        ("grace blackwell ultra", "gb300"),
        ("gb300", "gb300"),
        ("gb200", "gb200"),
        ("grace blackwell", "gb200"),
        ("gh200", "gh200"),
        ("grace hopper", "gh200"),
        # NVIDIA Ampere
        ("a100 sxm", "a100_sxm_80gb"),
        ("a100 pcie", "a100_pcie_80gb"),
        ("a100 80gb", "a100_sxm_80gb"),
        ("a100 40gb", "a100_40gb"),
        ("a100", "a100"),
        ("a10g", "a10g"),
        # NVIDIA Hopper
        ("h100 sxm", "h100_sxm"),
        ("h100 nvl", "h100_nvl"),
        ("h100 pcie", "h100_pcie"),
        ("h100", "h100"),
        ("h200 nvl", "h200_nvl"),
        ("h200", "h200"),
        ("h800", "h800"),  # sub-bug 1: previously missing
        ("h20", "h20"),    # sub-bug 1: previously missing
        # NVIDIA Blackwell
        ("b200", "b200"),
        ("b300", "b300"),
        ("b100", "b100"),  # sub-bug 1: previously missing
        # AMD CDNA
        ("mi300x", "mi300x"),
        ("mi325x", "mi325x"),
        ("mi350x", "mi350x"),  # sub-bug 1: previously missing
        ("mi355x", "mi355x"),
    ]

    for pattern, key in mappings:
        if pattern in name_norm:
            return get_gpu_profile(key)

    return None
