"""Regression tests for `models/registry.py` serving-dict drift bugs.

Closes the snapshot v1.0.0 P1 bug `model_registry_serving_drift`:
- Sub-bug A: `Qwen3-Coder-480B-A35B-Instruct.serving["hf_id_fp8"]` previously
  pointed at `Qwen/Qwen3.5-397B-A17B-FP8` (wrong family, wrong size).
- Sub-bug B: `_minimum_tp_for_gpu` ignored `tp_bf16_*` keys, so models that
  declared `kv_cache_quantizable=False` (e.g. Qwen3-Coder-Next) silently
  fell back to FP8 TP hints that would OOM at long context.
- Sub-bug C: `vocab_size` was set on only 4 of 19 models — readers could
  not distinguish "unknown" from "actually zero".
"""

from __future__ import annotations

from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.models.registry import _MODELS, get_model_variant
from inferscope.production_target import _minimum_tp_for_gpu


# --- Sub-bug A ---------------------------------------------------------------


def test_qwen3_coder_480b_does_not_reference_wrong_fp8_model():
    model = get_model_variant("Qwen3-Coder-480B-A35B-Instruct")
    assert model is not None
    hf_id_fp8 = model.serving.get("hf_id_fp8")
    # Either unset (no first-party FP8 build) or pointing at a Qwen3-Coder
    # model — never the unrelated Qwen3.5-397B-A17B copy-paste.
    assert hf_id_fp8 is None or "Qwen3-Coder" in hf_id_fp8
    assert hf_id_fp8 != "Qwen/Qwen3.5-397B-A17B-FP8"


def test_qwen3_coder_480b_hf_id_still_correct():
    model = get_model_variant("Qwen3-Coder-480B-A35B-Instruct")
    assert model is not None
    assert model.serving.get("hf_id") == "Qwen/Qwen3-Coder-480B-A35B-Instruct"


# --- Sub-bug B ---------------------------------------------------------------


def test_qwen3_coder_next_prefers_bf16_tp_hint_on_h200():
    """Qwen3-Coder-Next declares kv_cache_quantizable=False — recommender
    must return TP=2 (the bf16 hint) instead of TP=1 (the fp8 hint)."""
    model = get_model_variant("Qwen3-Coder-Next")
    h200 = get_gpu_profile("h200")
    assert model is not None and h200 is not None
    assert _minimum_tp_for_gpu(model, h200) == 2


def test_qwen3_coder_next_prefers_bf16_tp_hint_on_h100():
    model = get_model_variant("Qwen3-Coder-Next")
    h100 = get_gpu_profile("h100")
    assert model is not None and h100 is not None
    assert _minimum_tp_for_gpu(model, h100) == 4


def test_kimi_k25_still_uses_fp8_tp_hint_on_h200():
    """Regression guard — Kimi-K2.5 has no bf16 marker, must keep using
    its tp_fp8_h200=4 hint after the bf16 priority change."""
    model = get_model_variant("Kimi-K2.5")
    h200 = get_gpu_profile("h200")
    assert model is not None and h200 is not None
    assert _minimum_tp_for_gpu(model, h200) == 4


def test_kimi_k25_still_uses_fp8_tp_hint_on_h100():
    model = get_model_variant("Kimi-K2.5")
    h100 = get_gpu_profile("h100")
    assert model is not None and h100 is not None
    assert _minimum_tp_for_gpu(model, h100) == 8


def test_minimum_tp_no_gpu_branch_honors_bf16_for_qwen3_coder_next():
    model = get_model_variant("Qwen3-Coder-Next")
    assert model is not None
    # No-GPU summary path should also walk bf16 keys first.
    assert _minimum_tp_for_gpu(model, None) == 2


# --- Sub-bug C ---------------------------------------------------------------


def test_all_registered_models_have_nonzero_vocab_size():
    missing = [m.name for m in _MODELS.values() if m.vocab_size <= 0]
    assert missing == [], f"Models missing vocab_size: {missing}"
