"""Model profile registry — maps model names to serving profiles.

InferScope tunes per MODEL CLASS with family-specific overrides, not per model name.
5 classes: Dense-GQA, Qwen3.5-Hybrid, Frontier-MLA-MoE, Compact-Agentic-MoE, Classical-MoE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferscope.optimization.serving_profile import ModelClass


@dataclass
class ModelVariant:
    """A specific model variant with serving-relevant specs."""

    name: str
    family: str
    model_class: ModelClass
    params_total_b: float
    params_active_b: float  # Same as total for dense models
    model_type: str  # dense | moe
    context_length: int
    attention_type: str  # GQA | MLA | MHA | hybrid
    kv_heads: int = 0
    head_dim: int = 128
    layers: int = 0
    experts_total: int = 0
    experts_active: int = 0
    vocab_size: int = 0
    mtp_speculative: bool = False

    # Serving recommendations
    serving: dict[str, Any] = field(default_factory=dict)

    # Memory estimates (approximate bytes per parameter)
    weight_bytes_fp16: float = 0.0  # Computed from params

    def __post_init__(self) -> None:
        if self.weight_bytes_fp16 == 0.0:
            self.weight_bytes_fp16 = self.params_total_b * 2e9  # 2 bytes per param FP16

    def weight_gb(self, precision: str = "fp16") -> float:
        """Estimated weight memory in GB."""
        multiplier = {
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "int8": 1.0,
            "fp4": 0.5,
            "nvfp4": 0.5,
            "int4": 0.5,
            "mxfp4": 0.5,
            "awq": 0.5,
            "gptq": 0.5,
        }.get(precision, 2.0)
        return self.params_total_b * multiplier

    def kv_cache_bytes_per_token(self, precision: str = "fp16") -> float:
        """KV cache bytes per token per layer."""
        dtype_bytes = {"fp16": 2.0, "bf16": 2.0, "fp8_e4m3": 1.0, "fp8": 1.0, "auto": 2.0}
        bpt = dtype_bytes.get(precision, 2.0)

        if self.attention_type == "MLA":
            # MLA compresses KV ~32x — use latent_dim instead of full dim
            latent_dim_raw = self.serving.get("mla_latent_dim", 512)
            latent_dim = int(latent_dim_raw) if isinstance(latent_dim_raw, int | float) else 512
            return 2 * latent_dim * bpt  # K + V latent vectors
        else:
            # Standard GQA: 2 * kv_heads * head_dim * bytes
            return 2 * self.kv_heads * self.head_dim * bpt

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "model_class": self.model_class.value,
            "params_total_b": self.params_total_b,
            "params_active_b": self.params_active_b,
            "type": self.model_type,
            "context_length": self.context_length,
            "attention_type": self.attention_type,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "layers": self.layers,
            "experts_total": self.experts_total,
            "experts_active": self.experts_active,
            "mtp_speculative": self.mtp_speculative,
            "weight_gb_fp16": round(self.weight_gb("fp16"), 1),
            "weight_gb_fp8": round(self.weight_gb("fp8"), 1),
            "serving": self.serving,
        }


# =============================================================================
# Model Profiles
# =============================================================================

_MODELS: dict[str, ModelVariant] = {}


def _compact_model_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _register(model: ModelVariant) -> ModelVariant:
    _MODELS[model.name.lower()] = model
    return model


# --- Qwen 3.5 family (Hybrid Gated DeltaNet + GQA) ---

_register(
    ModelVariant(
        name="Qwen3.5-32B",
        family="Qwen 3.5",
        model_class=ModelClass.QWEN35_HYBRID,
        params_total_b=32,
        params_active_b=32,
        model_type="dense",
        context_length=131072,
        attention_type="hybrid",
        kv_heads=8,
        head_dim=128,
        layers=64,
        serving={"vllm_flags": "--trust-remote-code", "tp_fp8": 1, "tp_bf16": 2},
    )
)

_register(
    ModelVariant(
        name="Qwen3.5-72B",
        family="Qwen 3.5",
        model_class=ModelClass.QWEN35_HYBRID,
        params_total_b=72,
        params_active_b=72,
        model_type="dense",
        context_length=131072,
        attention_type="hybrid",
        kv_heads=8,
        head_dim=128,
        layers=80,
        serving={"vllm_flags": "--trust-remote-code", "tp_fp8": 2, "tp_bf16": 4},
    )
)

_register(
    ModelVariant(
        name="Qwen3.5-397B-A17B",
        family="Qwen 3.5",
        model_class=ModelClass.QWEN35_HYBRID,
        params_total_b=397,
        params_active_b=17,
        model_type="moe",
        context_length=262144,
        attention_type="hybrid",
        kv_heads=2,
        head_dim=256,
        layers=60,
        experts_total=512,
        experts_active=10,
        mtp_speculative=True,
        serving={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "vllm_flags": "--trust-remote-code",
            "sglang_speculative": "--speculative-algo NEXTN",
            "tp_fp8": 2,
            "tp_bf16": 8,
        },
    )
)

# --- DeepSeek V3 / R1 (Frontier MLA MoE) ---

_register(
    ModelVariant(
        name="DeepSeek-V3",
        family="DeepSeek V3/R1",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        params_total_b=671,
        params_active_b=37,
        model_type="moe",
        context_length=131072,
        attention_type="MLA",
        kv_heads=128,
        head_dim=128,
        layers=61,
        experts_total=256,
        experts_active=8,
        serving={
            "mla_latent_dim": 512,
            "compression_ratio": 32,
            "vllm_flags": "--trust-remote-code --block-size 1",
            "tp_fp8_h200": 4,
            "tp_fp8_h100": 8,
            "tp_bf16": 16,
            "ep_recommended": True,
            "nvidia_fp4": "deepseek-ai/DeepSeek-V3-0324-FP4 -tp 4 --enable-expert-parallel",
        },
    )
)

_register(
    ModelVariant(
        name="DeepSeek-R1",
        family="DeepSeek V3/R1",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        params_total_b=671,
        params_active_b=37,
        model_type="moe",
        context_length=131072,
        attention_type="MLA",
        kv_heads=128,
        head_dim=128,
        layers=61,
        experts_total=256,
        experts_active=8,
        serving={
            "mla_latent_dim": 512,
            "compression_ratio": 32,
            "vllm_flags": "--trust-remote-code --block-size 1",
            "additional_flags": "--enable-reasoning --reasoning-parser deepseek_r1",
            "tp_fp8_h200": 4,
            "tp_fp8_h100": 8,
            "tp_bf16": 16,
            "ep_recommended": True,
        },
    )
)

# --- DeepSeek distills (Dense GQA — NOT MLA) ---

_register(
    ModelVariant(
        name="DeepSeek-R1-Distill-Qwen-7B",
        family="DeepSeek Distills",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=7,
        params_active_b=7,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=4,
        head_dim=128,
        layers=28,
        serving={"tp_fp16": 1, "note": "Standard Qwen architecture — no special flags needed"},
    )
)

_register(
    ModelVariant(
        name="DeepSeek-R1-Distill-Llama-70B",
        family="DeepSeek Distills",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=70,
        params_active_b=70,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=80,
        serving={"tp_fp8_h200": 1, "tp_fp8_h100": 2, "tp_fp16_mi300x": 1},
    )
)

# --- Kimi K2.5 (GQA MoE) ---

_register(
    ModelVariant(
        name="Kimi-K2.5",
        family="Kimi K2/K2.5",
        model_class=ModelClass.CLASSICAL_MOE,
        params_total_b=400,
        params_active_b=50,
        model_type="moe",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=61,
        experts_total=128,
        experts_active=8,
        vocab_size=160000,
        serving={
            "target_profile": "dynamo_long_context_coding",
            "dynamo_backend": "vllm",
            "tp_fp8_h100": 8,
            "tp_fp8_h200": 4,
            "tp_fp8_b200": 4,
            "tp_fp8_b300": 2,
            "tp_fp4_b200": 2,
            "tp_fp4_b300": 1,
            "recommended_topology": {
                "fp8": {"h100": "tp8", "h200": "tp4", "b200": "tp4", "b300": "tp2"},
                "fp4": {"b200": "tp2", "b300": "tp1"},
            },
            "dynamo_notes": [
                "Long-context coding lane expects sticky session routing and LMCache namespace isolation.",
                "Disaggregated plans should use shared LMCache with explicit prefill/decode observability targets.",
            ],
            "vllm_flags": "--trust-remote-code --enforce-eager --tool-call-parser kimi_k2 --reasoning-parser kimi_k2",
            "nvidia_nvfp4": "moonshotai/Kimi-K2.5-NVFP4 -tp 4",
            "eagle3_speculative": '{"model": "lightseekorg/kimi-k2.5-eagle3", "method": "eagle3"}',
            "decode_context_parallel": "--decode-context-parallel-size 8",
        },
    )
)

# --- Qwen3 Coder family (Coding-focused MoE) ---

_register(
    ModelVariant(
        name="Qwen3-Coder-480B-A35B-Instruct",
        family="Qwen3 Coder",
        model_class=ModelClass.CLASSICAL_MOE,
        params_total_b=480,
        params_active_b=35,
        model_type="moe",
        context_length=262144,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=62,
        experts_total=160,
        experts_active=8,
        vocab_size=151936,
        serving={
            "support_tier": "benchmark_supported",
            "kv_estimation_mode": "exact",
            "hf_id": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "hf_id_fp8": "Qwen/Qwen3.5-397B-A17B-FP8",
            "tp_fp8_h100": 8,
            "tp_fp8_h200": 8,
            "tp_fp8_b200": 4,
            "tp_fp8_b300": 4,
            "recommended_topology": {
                "fp8": {"h100": "tp8", "h200": "tp8", "b200": "tp4", "b300": "tp4"},
            },
            "vllm_flags": "--trust-remote-code",
            "dynamo_backend": "vllm",
        },
    )
)

_register(
    ModelVariant(
        name="Qwen3-Coder-30B-A3B-Instruct",
        family="Qwen3 Coder",
        model_class=ModelClass.CLASSICAL_MOE,
        params_total_b=30,
        params_active_b=3,
        model_type="moe",
        context_length=262144,
        attention_type="GQA",
        kv_heads=4,
        head_dim=128,
        layers=48,
        experts_total=128,
        experts_active=8,
        vocab_size=151936,
        serving={
            "support_tier": "benchmark_supported",
            "kv_estimation_mode": "exact",
            "hf_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "tp_fp8_h100": 1,
            "tp_fp8_h200": 1,
            "tp_fp8_b200": 1,
            "tp_fp8_b300": 1,
            "recommended_topology": {
                "fp8": {"h100": "tp1", "h200": "tp1", "b200": "tp1", "b300": "tp1"},
            },
            "vllm_flags": "--trust-remote-code",
            "dynamo_backend": "vllm",
        },
    )
)

_register(
    ModelVariant(
        name="Qwen3-Coder-Next",
        family="Qwen3 Coder",
        model_class=ModelClass.QWEN35_HYBRID,
        params_total_b=80,
        params_active_b=3,
        model_type="moe",
        context_length=262144,
        attention_type="hybrid",
        kv_heads=2,
        head_dim=256,
        layers=48,
        experts_total=512,
        experts_active=11,
        vocab_size=151936,
        serving={
            "support_tier": "benchmark_supported",
            "kv_estimation_mode": "hybrid_exact",
            "hf_id": "Qwen/Qwen3-Coder-Next",
            "hf_id_fp8": "Qwen/Qwen3-Coder-Next-FP8",
            "tp_fp8_h200": 1,
            "tp_fp8_h100": 2,
            "tp_bf16_h200": 2,
            "tp_bf16_h100": 4,
            "full_attention_interval": 4,
            "kv_layers": 12,
            "deltanet_layers": 36,
            # 36 layers × 16 QK heads × 128 key_dim × 128 value_dim × 2 bytes = 18,874,368
            "deltanet_state_bytes_per_seq_bf16": 18874368,
            "kv_cache_quantizable": False,
            "recommended_kv_dtype": "bf16",
            "recommended_topology": {
                "fp8": {"h200": "tp1", "h100": "tp2"},
                "bf16": {"h200": "tp2", "h100": "tp4"},
            },
            "vllm_flags": "--trust-remote-code",
            "dynamo_backend": "vllm",
            "warnings": [
                "Hybrid attention: 12/48 layers use standard KV cache, 36 use Gated DeltaNet (fixed state).",
                "FP8 KV cache not yet supported for this architecture — use BF16 KV.",
            ],
        },
    )
)

# --- GLM family ---

_register(
    ModelVariant(
        name="GLM-5",
        family="GLM",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=70,
        params_active_b=70,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        serving={
            "available": False,
            "min_gpus": {
                "bf16": {"h100": 2, "h200": 1, "b200": 1, "b300": 1},
                "fp8": {"h100": 1, "h200": 1, "b200": 1, "b300": 1},
            },
            "recommended_topology": {
                "fp8": {"h100": "tp1", "h200": "tp1", "b200": "tp1", "b300": "tp1"},
                "bf16": {"h100": "tp2", "h200": "tp1", "b200": "tp1", "b300": "tp1"},
            },
            "vllm_flags": "--trust-remote-code",
        },
    )
)

_register(
    ModelVariant(
        name="GLM-4.7",
        family="GLM",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        params_total_b=355,
        params_active_b=50,
        model_type="moe",
        context_length=1048576,
        attention_type="GQA",
        kv_heads=8,
        layers=60,
        experts_total=160,
        experts_active=8,
        serving={
            "target_profile": "dynamo_long_context_coding",
            "dynamo_backend": "vllm",
            "tp_fp8_h100": 8,
            "tp_fp8_h200": 4,
            "tp_fp8_b200": 4,
            "tp_fp8_b300": 2,
            "recommended_topology": {
                "fp8": {"h100": "tp8", "h200": "tp4", "b200": "tp4", "b300": "tp2"},
            },
            "dynamo_notes": [
                "GLM-4.7 targets long-context coding with LMCache rather than generic chat.",
                "Benchmark disaggregation should preserve session routing for prefix reuse.",
            ],
            "vllm_flags": "--trust-remote-code",
            "mtp_speculative": "--speculative-config.method mtp --speculative-config.num_speculative_tokens 1",
            "mtp_acceptance_rate": ">90%",
        },
    )
)

# --- MiniMax (Compact Agentic MoE) ---

_register(
    ModelVariant(
        name="MiniMax-M2.5",
        family="MiniMax",
        model_class=ModelClass.COMPACT_AGENTIC_MOE,
        params_total_b=230,
        params_active_b=10,
        model_type="moe",
        context_length=1048576,
        attention_type="GQA",
        kv_heads=8,  # Estimated — official architecture docs under-specified
        serving={
            "vllm_flags": (
                "--trust-remote-code --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think"
            ),
            "tp_warning": "Pure TP=8 NOT supported — max pure TP is 4, use EP for 8-GPU",
            "tp_with_ep": "-tp 4 --enable-expert-parallel",
        },
    )
)

# --- Mixtral (Classical MoE) ---

_register(
    ModelVariant(
        name="Mixtral-8x7B",
        family="Mixtral",
        model_class=ModelClass.CLASSICAL_MOE,
        params_total_b=46.7,
        params_active_b=13,
        model_type="moe",
        context_length=32768,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=32,
        experts_total=8,
        experts_active=2,
        serving={"tp_fp8_h100": 1, "tp_fp16_a100": 2},
    )
)

_register(
    ModelVariant(
        name="Mixtral-8x22B",
        family="Mixtral",
        model_class=ModelClass.CLASSICAL_MOE,
        params_total_b=141,
        params_active_b=39,
        model_type="moe",
        context_length=65536,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=56,
        experts_total=8,
        experts_active=2,
        serving={"tp_fp8": 2, "tp_fp16": 4},
    )
)

# --- Llama 3 family (Dense GQA) ---

_register(
    ModelVariant(
        name="Llama-3-8B",
        family="Llama 3",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=8,
        params_active_b=8,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=32,
        serving={"tp_fp16": 1, "tp_fp8": 1},
    )
)

_register(
    ModelVariant(
        name="Llama-3-70B",
        family="Llama 3",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=70,
        params_active_b=70,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=80,
        serving={"tp_fp8_h200": 1, "tp_fp8_h100": 2, "tp_fp16": 4},
    )
)


# =============================================================================
# Lookup functions
# =============================================================================


def get_model_variant(name: str) -> ModelVariant | None:
    """Look up a model by name (case-insensitive, flexible matching)."""
    key = name.lower().strip()
    if not key:
        return None

    normalized = key.replace("/", "-").replace("_", "-")
    compact = _compact_model_key(normalized)

    # Direct match
    if key in _MODELS:
        return _MODELS[key]
    if normalized in _MODELS:
        return _MODELS[normalized]

    # Compact match: ignore punctuation like -, _, ., /
    if compact:
        for model_key, model in _MODELS.items():
            if compact == _compact_model_key(model_key):
                return model

    # Fuzzy match: try removing vendor prefixes, hyphens, etc.
    for model_key, model in _MODELS.items():
        model_compact = _compact_model_key(model_key)
        if normalized in model_key or model_key in normalized:
            return model
        # Check against the full HuggingFace-style name
        if normalized.endswith(model_key):
            return model
        if compact and (compact in model_compact or model_compact in compact):
            return model

    return None


def list_models() -> list[str]:
    """List all known model names."""
    return sorted(set(m.name for m in _MODELS.values()))


def get_models_by_class(model_class: ModelClass) -> list[ModelVariant]:
    """Get all models in a given class."""
    return [m for m in _MODELS.values() if m.model_class == model_class]
