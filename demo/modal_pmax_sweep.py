"""Validate adaptive P_max scheduling on Modal with real veRL-style rollouts.

Runs the same model with fixed P_max vs adaptive P_max, compares:
- GPU utilization (tokens generated / tokens budgeted)
- Throughput (tok/s)
- Truncation rate

Usage:
    modal run demo/modal_pmax_sweep.py

Cost: ~$1-2 (5-10 min on A100)
"""

import json
import time

import modal

app = modal.App("inferscope-pmax-sweep")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

PROMPTS_EASY = [
    "What is 17 × 23?",
    "What is the capital of France?",
    "Name three primary colors.",
    "What is 2 + 2?",
    "Is water wet?",
]

PROMPTS_HARD = [
    "Write a complete implementation of a red-black tree in Python with insert, delete, and search operations. Include all rotation cases and color fixes.",
    "Explain the complete mathematical derivation of backpropagation through a 3-layer neural network with ReLU activations, including all partial derivatives.",
    "Design a distributed consensus protocol that handles network partitions, leader election, and log replication. Provide pseudocode for all message handlers.",
    "Write a comprehensive comparison of 8 different garbage collection algorithms, including their time complexity, space overhead, and real-world use cases in production systems.",
    "Implement a complete HTTP/2 parser in Python that handles HPACK header compression, stream multiplexing, flow control, and priority handling.",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.8.0", "torch")
)


def _run_batch(llm, prompts, pmax_per_prompt, system_prompt="You are a helpful assistant."):
    """Run a batch with per-prompt P_max budgets, return utilization stats."""
    from vllm import SamplingParams

    results = []
    for i, (prompt, pmax) in enumerate(zip(prompts, pmax_per_prompt)):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        params = SamplingParams(temperature=0.0, max_tokens=pmax)
        outputs = llm.generate([formatted], params)
        comp = outputs[0].outputs[0]
        tokens_used = len(comp.token_ids)
        truncated = comp.finish_reason == "length"

        results.append({
            "prompt_index": i,
            "pmax": pmax,
            "tokens_used": tokens_used,
            "truncated": truncated,
            "utilization": tokens_used / pmax if pmax > 0 else 0,
        })

    return results


@app.function(image=image, gpu="A100-40GB", timeout=1200, scaledown_window=60)
def run_pmax_comparison():
    """Compare fixed vs adaptive P_max on mixed easy/hard batch."""
    from vllm import LLM

    prompts = PROMPTS_EASY + PROMPTS_HARD
    batch_size = len(prompts)

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        max_model_len=4096,
    )

    # --- Fixed P_max = 2048 for all ---
    print("=== Fixed P_max = 2048 ===")
    fixed_pmax = [2048] * batch_size
    start = time.perf_counter()
    fixed_results = _run_batch(llm, prompts, fixed_pmax)
    fixed_time = time.perf_counter() - start

    fixed_tokens = sum(r["tokens_used"] for r in fixed_results)
    fixed_budget = sum(r["pmax"] for r in fixed_results)
    fixed_truncated = sum(1 for r in fixed_results if r["truncated"])

    print(f"  Tokens: {fixed_tokens}/{fixed_budget} ({fixed_tokens/fixed_budget*100:.1f}% util)")
    print(f"  Truncated: {fixed_truncated}/{batch_size}")
    print(f"  Time: {fixed_time:.2f}s")

    # --- Adaptive: easy=256, hard=2048 ---
    print("\n=== Adaptive P_max (easy=256, hard=2048) ===")
    adaptive_pmax = [256] * len(PROMPTS_EASY) + [2048] * len(PROMPTS_HARD)
    start = time.perf_counter()
    adaptive_results = _run_batch(llm, prompts, adaptive_pmax)
    adaptive_time = time.perf_counter() - start

    adaptive_tokens = sum(r["tokens_used"] for r in adaptive_results)
    adaptive_budget = sum(r["pmax"] for r in adaptive_results)
    adaptive_truncated = sum(1 for r in adaptive_results if r["truncated"])

    print(f"  Tokens: {adaptive_tokens}/{adaptive_budget} ({adaptive_tokens/adaptive_budget*100:.1f}% util)")
    print(f"  Truncated: {adaptive_truncated}/{batch_size}")
    print(f"  Time: {adaptive_time:.2f}s")

    budget_savings = (1 - adaptive_budget / fixed_budget) * 100
    throughput_fixed = fixed_tokens / fixed_time if fixed_time > 0 else 0
    throughput_adaptive = adaptive_tokens / adaptive_time if adaptive_time > 0 else 0

    return {
        "fixed": {
            "results": fixed_results,
            "total_tokens": fixed_tokens,
            "total_budget": fixed_budget,
            "utilization": fixed_tokens / fixed_budget,
            "truncated": fixed_truncated,
            "wall_time_s": round(fixed_time, 2),
            "throughput_tok_s": round(throughput_fixed, 1),
        },
        "adaptive": {
            "results": adaptive_results,
            "total_tokens": adaptive_tokens,
            "total_budget": adaptive_budget,
            "utilization": adaptive_tokens / adaptive_budget,
            "truncated": adaptive_truncated,
            "wall_time_s": round(adaptive_time, 2),
            "throughput_tok_s": round(throughput_adaptive, 1),
        },
        "comparison": {
            "budget_savings_pct": round(budget_savings, 1),
            "throughput_ratio": round(throughput_adaptive / throughput_fixed, 2) if throughput_fixed > 0 else 0,
            "fixed_util_pct": round(fixed_tokens / fixed_budget * 100, 1),
            "adaptive_util_pct": round(adaptive_tokens / adaptive_budget * 100, 1),
        },
    }


@app.local_entrypoint()
def main():
    print("Running P_max sweep on Modal A100...")
    results = run_pmax_comparison.remote()

    path = "/tmp/pmax_sweep_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    c = results["comparison"]
    print(f"\n=== Comparison ===")
    print(f"Budget savings: {c['budget_savings_pct']}%")
    print(f"Fixed utilization: {c['fixed_util_pct']}%")
    print(f"Adaptive utilization: {c['adaptive_util_pct']}%")
    print(f"Throughput ratio: {c['throughput_ratio']}x")
    print(f"\nResults saved: {path}")
