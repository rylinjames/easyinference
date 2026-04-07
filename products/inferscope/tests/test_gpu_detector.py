"""Regression coverage for `hardware/detector.py::detect_gpu_from_name`.

Closes the snapshot v1.0.0 P1 bug `gpu_detector_correctness.md`, which
documented 3 sub-bugs:

- Sub-bug 1: missing B100/H800/H20/MI350X mappings (the registry has
  these but the detector did not).
- Sub-bug 2: separator-convention mismatch (the patterns mixed hyphens
  and spaces, but the input was only `.lower()`'d, not normalized).
- Sub-bug 3: ordering bug — `grace blackwell` was checked before
  `gb300`, so a `Grace Blackwell GB300` device was misclassified as GB200.
"""

from __future__ import annotations

import pytest

from inferscope.hardware.detector import detect_gpu_from_name


# ----------------------------------------------------------------------------
# Sub-bug 1 — missing SKUs
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_sku",
    [
        ("NVIDIA H800 80GB HBM3", "H800"),
        ("NVIDIA H20 96GB HBM3", "H20"),
        ("NVIDIA B100", "B100"),
        ("AMD Instinct MI350X", "MI350X"),
    ],
)
def test_detector_finds_previously_missing_skus(name: str, expected_sku: str) -> None:
    """The detector previously had no mappings for B100, H800, H20, MI350X
    even though all 4 exist in `GPU_REGISTRY`. Now they should match."""
    profile = detect_gpu_from_name(name)
    assert profile is not None, f"Detector did not match {name!r}"
    assert profile.name == expected_sku


# ----------------------------------------------------------------------------
# Sub-bug 2 — separator convention
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_sku_substring",
    [
        # Hyphen convention
        ("NVIDIA A100-SXM4-80GB", "A100"),
        ("NVIDIA H100-SXM5", "H100"),
        ("NVIDIA H100-PCIe", "H100"),
        # Space convention
        ("NVIDIA A100 SXM 80GB", "A100"),
        ("NVIDIA H100 SXM5", "H100"),
        ("NVIDIA H100 PCIe", "H100"),
        # Underscore convention (less common but legal)
        ("NVIDIA H100_SXM5", "H100"),
    ],
)
def test_detector_normalizes_separators(name: str, expected_sku_substring: str) -> None:
    """The detector must normalize hyphens, underscores, and spaces to a
    single convention before matching, so a single pattern catches all
    three input forms."""
    profile = detect_gpu_from_name(name)
    assert profile is not None, f"Detector did not match {name!r}"
    assert expected_sku_substring in profile.name


def test_detector_classifies_a100_sxm_with_hyphen_correctly() -> None:
    """The dash form should match the SXM-specific pattern, not the bare A100."""
    profile = detect_gpu_from_name("NVIDIA A100-SXM4-80GB")
    assert profile is not None
    assert "SXM" in profile.name


def test_detector_classifies_a100_sxm_with_space_correctly() -> None:
    """Same scenario in space form — must also match the SXM-specific pattern."""
    profile = detect_gpu_from_name("NVIDIA A100 SXM4 80GB")
    assert profile is not None
    assert "SXM" in profile.name


# ----------------------------------------------------------------------------
# Sub-bug 3 — `gb300` must be checked before `grace blackwell`
# ----------------------------------------------------------------------------


def test_detector_distinguishes_gb300_from_gb200() -> None:
    """The headline scenario: a device string mentioning both
    'Grace Blackwell' and 'GB300' must classify as GB300, not GB200.

    GB200 vs GB300 differ in HBM (288 vs 192 GB), TDP (1400W vs 1000W),
    sm_103 vs sm_100, and accelerated_softmax — misclassifying changes
    the recommendation envelope significantly."""
    profile = detect_gpu_from_name("NVIDIA Grace Blackwell GB300 Superchip")
    assert profile is not None
    assert profile.name == "GB300", f"Got {profile.name} — should be GB300"


def test_detector_grace_blackwell_alone_falls_back_to_gb200() -> None:
    """A device string with only `Grace Blackwell` (no GB300/GB200 SKU)
    should fall back to GB200 (the generic Grace Blackwell)."""
    profile = detect_gpu_from_name("NVIDIA Grace Blackwell Superchip")
    assert profile is not None
    assert profile.name == "GB200"


def test_detector_grace_blackwell_ultra_classifies_as_gb300() -> None:
    """`Grace Blackwell Ultra` is the marketing name for GB300."""
    profile = detect_gpu_from_name("NVIDIA Grace Blackwell Ultra Superchip")
    assert profile is not None
    assert profile.name == "GB300"


# ----------------------------------------------------------------------------
# Regressions — existing patterns still work
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_sku_substring",
    [
        ("NVIDIA H100", "H100"),
        ("NVIDIA H100 NVL", "H100"),  # NVL variant
        ("NVIDIA H200", "H200"),
        ("NVIDIA B200", "B200"),
        ("NVIDIA B300", "B300"),
        ("NVIDIA GH200", "GH200"),
        ("NVIDIA GB200", "GB200"),
        ("NVIDIA A10G", "A10G"),
        ("AMD MI300X", "MI300X"),
        ("AMD MI325X", "MI325X"),
        ("AMD MI355X", "MI355X"),
    ],
)
def test_detector_existing_patterns_still_work(name: str, expected_sku_substring: str) -> None:
    profile = detect_gpu_from_name(name)
    assert profile is not None, f"Detector did not match {name!r}"
    assert expected_sku_substring in profile.name


def test_detector_returns_none_for_unknown_gpu() -> None:
    """Unrecognized strings should return None, not raise."""
    assert detect_gpu_from_name("Definitely Not A GPU") is None
    assert detect_gpu_from_name("") is None
