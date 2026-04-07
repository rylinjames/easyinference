"""KV cache math regression coverage.

Closes Tier 1 Item 2 from `improvements/easyinference/bugs/tests_zero_coverage_pile.md`
(`tools/kv_cache.py` previously had zero direct test coverage). Provides
regression coverage for two paired snapshot v1.0.0 P0 bugs:

- `bugs/hybrid_attention_kv_undercount.md` — `kv_cache_bytes_per_token`
  had no hybrid branch, so `Qwen3-Coder-Next` was overstated by 4x.
- `bugs/recompute_vs_transfer_kv_size_hardcoded.md` — `_estimate_recompute_vs_transfer`
  hardcoded `kv_bytes_per_token = 512`, ~250x off for Kimi-K2.5.
"""

from __future__ import annotations

import pytest

from inferscope.models.registry import get_model_variant
from inferscope.tools.kv_cache import _estimate_recompute_vs_transfer


# ----------------------------------------------------------------------------
# hybrid_attention_kv_undercount: Qwen3-Coder-Next must use 12 KV layers
# ----------------------------------------------------------------------------


def test_qwen3_coder_next_hybrid_uses_kv_layers_not_total_layers() -> None:
    """`Qwen3-Coder-Next` has `attention_type='hybrid'` plus `kv_layers: 12,
    deltanet_layers: 36, layers: 48`. The whole-model per-token KV must use
    12 layers, not 48.

    Per-layer (BF16, kv_heads=2, head_dim=256): 2 * 2 * 256 * 2 = 2048 bytes/layer
    Whole-model with 12 KV layers: 2048 * 12 = 24576 bytes/token (~24 KB)
    Whole-model with 48 layers (the BUG): 2048 * 48 = 98304 bytes/token (~96 KB)
    """
    variant = get_model_variant("Qwen3-Coder-Next")
    assert variant is not None
    assert variant.attention_type == "hybrid"
    assert variant.serving.get("kv_layers") == 12
    assert variant.layers == 48

    per_layer_bf16 = variant.kv_cache_bytes_per_token("bf16")
    assert per_layer_bf16 == 2 * 2 * 256 * 2  # = 2048

    total_bf16 = variant.kv_cache_bytes_per_token_total("bf16")
    assert total_bf16 == per_layer_bf16 * 12, (
        f"Hybrid total must use kv_layers=12, got {total_bf16} (per-layer={per_layer_bf16})"
    )

    # Defense-in-depth: ensure we are NOT returning the buggy 4x overstatement
    bug_value = per_layer_bf16 * variant.layers
    assert total_bf16 != bug_value, (
        f"Total {total_bf16} matches the buggy per-layer*all-layers result"
    )
    assert bug_value == 4 * total_bf16, "sanity: buggy answer is 4x the correct one"


def test_dense_gqa_total_uses_all_layers() -> None:
    """For dense GQA models like Kimi-K2.5, the total must equal per-layer × layers."""
    variant = get_model_variant("Kimi-K2.5")
    assert variant is not None
    assert variant.attention_type == "GQA"

    per_layer_fp8 = variant.kv_cache_bytes_per_token("fp8")
    total_fp8 = variant.kv_cache_bytes_per_token_total("fp8")

    assert total_fp8 == per_layer_fp8 * variant.layers


def test_mla_model_total_uses_all_layers_with_latent_dim() -> None:
    """For MLA models (DeepSeek-V3 family), the total uses the latent dim per layer
    times all layers. The MLA branch in kv_cache_bytes_per_token already returns
    the compressed per-layer figure; the total just multiplies by layers."""
    # Find any MLA model in the registry; use it for this round-trip check
    candidates = ["DeepSeek-V3.2", "DeepSeek-R1"]
    variant = None
    for name in candidates:
        variant = get_model_variant(name)
        if variant is not None:
            break
    if variant is None:
        pytest.skip("No MLA model in the registry to test")

    if variant.attention_type != "MLA":
        pytest.skip(f"{variant.name} is not MLA in current source")

    per_layer = variant.kv_cache_bytes_per_token("fp8")
    total = variant.kv_cache_bytes_per_token_total("fp8")
    assert total == per_layer * variant.layers


def test_kv_cache_bytes_per_token_total_falls_back_when_kv_layers_missing() -> None:
    """If a hybrid model is missing the `kv_layers` serving hint, the total
    must fall back to the legacy per-layer × all-layers product (preserves
    the historical answer for any pre-existing hybrid model that doesn't
    have the hint)."""
    from inferscope.models.registry import ModelVariant
    from inferscope.optimization.serving_profile import ModelClass

    fake = ModelVariant(
        name="fake-hybrid",
        family="test",
        model_class=ModelClass.QWEN35_HYBRID,
        params_total_b=10,
        params_active_b=10,
        model_type="dense",
        context_length=4096,
        attention_type="hybrid",
        kv_heads=4,
        head_dim=128,
        layers=20,
        # No serving["kv_layers"] hint
    )
    per_layer = fake.kv_cache_bytes_per_token("fp16")
    total = fake.kv_cache_bytes_per_token_total("fp16")
    assert total == per_layer * 20, "fallback must be per_layer * layers"


# ----------------------------------------------------------------------------
# recompute_vs_transfer_kv_size_hardcoded: real model lookup, not 512
# ----------------------------------------------------------------------------


def test_recompute_vs_transfer_uses_real_model_kv_size_for_kimi() -> None:
    """For Kimi-K2.5 (61 GQA layers, kv_heads=8, head_dim=128), the per-token
    whole-model KV at FP8 is 2 * 8 * 128 * 1 * 61 = 124,928 bytes (~125 KB).
    The legacy hardcoded 512 was off by ~244x. The fix routes through the
    model variant via the new `kv_cache_bytes_per_token_total` helper."""
    from inferscope.hardware.gpu_profiles import get_gpu_profile

    variant = get_model_variant("Kimi-K2.5")
    gpu_profile = get_gpu_profile("b200")
    assert variant is not None
    assert gpu_profile is not None

    expected_kv_bytes = variant.kv_cache_bytes_per_token_total("fp8")
    assert expected_kv_bytes == 2 * 8 * 128 * 1 * 61  # = 124928

    result = _estimate_recompute_vs_transfer(
        avg_prompt_tokens=8192,
        gpu_profile=gpu_profile,
        has_fast_transport=True,
        variant=variant,
    )

    # The crossover should be much smaller than the buggy 4069 because the
    # real KV size is ~244x larger. The estimated_transfer_ms should be
    # ~244x larger too (more bytes to transfer).
    legacy_512_transfer_ms = (8192 * 512) / (25.0 * 1e6)
    fixed_transfer_ms = (8192 * expected_kv_bytes) / (25.0 * 1e6)
    assert fixed_transfer_ms > 100 * legacy_512_transfer_ms, (
        "fixed transfer cost should be >>100x the buggy estimate"
    )
    assert result["estimated_transfer_ms"] >= round(fixed_transfer_ms, 1) - 1.0


def test_recompute_vs_transfer_falls_back_when_no_variant_provided() -> None:
    """Backward compatibility: if no model variant is in scope (e.g. unit tests
    that don't pass `variant=`), the function falls back to the legacy 512-byte
    heuristic. This preserves the API for callers that don't have a model."""
    from inferscope.hardware.gpu_profiles import get_gpu_profile

    gpu_profile = get_gpu_profile("b200")
    assert gpu_profile is not None

    result = _estimate_recompute_vs_transfer(
        avg_prompt_tokens=8192,
        gpu_profile=gpu_profile,
        has_fast_transport=True,
        # variant=None (the default)
    )
    assert "crossover_tokens" in result
    # With the 512-byte fallback the transfer cost is small
    legacy_512_transfer_ms = (8192 * 512) / (25.0 * 1e6)
    assert result["estimated_transfer_ms"] == round(legacy_512_transfer_ms, 1)
