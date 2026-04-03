#!/usr/bin/env python3
"""Generate Mode B (InferScope-optimized) configs from Mode A defaults.

For each Mode A config, runs the InferScope recommendation DAG and writes
a corresponding Mode B YAML with the optimized parameters.

Usage:
    python scripts/generate_mode_b_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Add InferScope to path
_INFERSCOPE_SRC = Path(__file__).resolve().parents[2] / "inferscope" / "src"
sys.path.insert(0, str(_INFERSCOPE_SRC))

from inferscope.hardware.gpu_profiles import get_gpu_profile  # noqa: E402
from inferscope.models.registry import get_model_variant  # noqa: E402
from inferscope.optimization.recommender import recommend  # noqa: E402
from inferscope.optimization.serving_profile import ObjectiveSpec, WorkloadMode  # noqa: E402

_MODE_A_DIR = Path(__file__).resolve().parents[1] / "configs" / "modes" / "mode_a"
_MODE_B_DIR = Path(__file__).resolve().parents[1] / "configs" / "modes" / "mode_b"

# Map ISB-1 model short names to InferScope model registry names
_MODEL_MAP = {
    "dsr1": "DeepSeek-R1",
    "llama70b": "Llama-3-70B",
    "qwen235b": "Qwen3.5-397B-A17B",
}

# Map ISB-1 GPU short names to InferScope GPU registry names
_GPU_MAP = {
    "h100": "h100",
    "h200": "h200",
    "b200": "b200",
}


def _infer_workload(model_short: str) -> WorkloadMode:
    """Pick a representative workload for the recommendation."""
    return WorkloadMode.CHAT


def generate_mode_b(mode_a_path: Path) -> dict | None:
    """Generate a Mode B config from a Mode A config file."""
    mode_a = yaml.safe_load(mode_a_path.read_text(encoding="utf-8"))

    gpu_short = mode_a.get("gpu", "")
    model_short = mode_a.get("model", "")
    quant = mode_a.get("quantization", "fp8")

    gpu_name = _GPU_MAP.get(gpu_short, gpu_short)
    model_name = _MODEL_MAP.get(model_short, model_short)

    gpu = get_gpu_profile(gpu_name)
    model = get_model_variant(model_name)

    if gpu is None:
        print(f"  SKIP: GPU '{gpu_name}' not in InferScope registry")
        return None
    if model is None:
        print(f"  SKIP: Model '{model_name}' not in InferScope registry")
        return None

    # Determine GPU count from Mode A TP (or default to model needs)
    tp = mode_a.get("tensor_parallel_size", 1)
    num_gpus = max(tp, 1)

    workload = _infer_workload(model_short)

    try:
        profile, engine_config, memory_plan = recommend(
            model=model,
            gpu=gpu,
            num_gpus=num_gpus,
            workload=workload,
            engine="auto",
            objective=ObjectiveSpec(),
        )
    except Exception as exc:
        print(f"  ERROR: recommendation failed: {exc}")
        return None

    # Build Mode B config
    mode_b = {
        "mode": "mode_b",
        "description": (
            f"Mode B (InferScope-optimized) — {mode_a.get('model_name', model_short)} "
            f"on {mode_a.get('gpu_name', gpu_short)} "
            f"(baseline {quant} -> optimized {profile.precision.weights})"
        ),
        "baseline_quantization": quant,
        "gpu": gpu_short,
        "gpu_name": mode_a.get("gpu_name", gpu_short),
        "model": model_short,
        "model_name": mode_a.get("model_name", model_short),
        "hf_model_id": mode_a.get("hf_model_id", ""),
        "quantization": profile.precision.weights,
        "tensor_parallel_size": profile.topology.tp,
        "gpu_memory_utilization": profile.cache.gpu_memory_utilization,
        "max_num_seqs": profile.scheduler.max_num_seqs,
        "max_num_batched_tokens": profile.scheduler.batched_token_budget,
        "enable_prefix_caching": profile.cache.prefix_cache,
        "enable_chunked_prefill": profile.scheduler.chunked_prefill,
        "kv_cache_dtype": profile.precision.kv_cache,
        "dtype": "auto",
        "engine": profile.engine.value,
        "optimized_by": "inferscope",
        "reasoning_summary": profile.reasoning_trace[:5] if profile.reasoning_trace else [],
    }

    # Add speculation if enabled
    if profile.speculation and profile.speculation.mode != "off":
        mode_b["speculative_model"] = profile.speculation.method
        mode_b["num_speculative_tokens"] = profile.speculation.num_speculative_tokens

    # Add data parallelism if > 1
    if profile.topology.dp > 1:
        mode_b["data_parallel_size"] = profile.topology.dp
    if profile.topology.ep > 1:
        mode_b["expert_parallel_size"] = profile.topology.ep

    return mode_b


def main() -> None:
    _MODE_B_DIR.mkdir(parents=True, exist_ok=True)

    mode_a_files = sorted(_MODE_A_DIR.glob("mode_a__*.yaml"))
    if not mode_a_files:
        print("No Mode A configs found.")
        return

    generated = 0
    for mode_a_path in mode_a_files:
        name = mode_a_path.stem.replace("mode_a__", "mode_b__")
        print(f"Processing {mode_a_path.name}...")

        mode_b = generate_mode_b(mode_a_path)
        if mode_b is None:
            continue

        out_path = _MODE_B_DIR / f"{name}.yaml"
        out_path.write_text(
            yaml.dump(mode_b, default_flow_style=False, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"  -> {out_path.name}")
        generated += 1

    print(f"\nGenerated {generated} Mode B configs in {_MODE_B_DIR}")


if __name__ == "__main__":
    main()
