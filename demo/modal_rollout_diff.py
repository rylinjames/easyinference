"""Generate training vs serving rollout logs on Modal for rollout-diff validation.

Captures per-token log-probs from both a simulated training rollout (using vLLM
with different precision/settings) and a serving replay, then downloads both JSONL
files for local `inferscope rollout-diff` analysis.

Usage:
    # Generate both logs on A100 (BF16 training vs FP16 serving to force divergence)
    modal run demo/modal_rollout_diff.py

    # Then locally:
    inferscope rollout-diff /tmp/rollout_training.jsonl /tmp/rollout_serving.jsonl

Cost: ~$1-3 (5-15 min on A100-40GB)
"""

import json
import os
import time

import modal

app = modal.App("inferscope-rollout-diff")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

PROMPTS = [
    "Write a Python function to compute the nth Fibonacci number recursively with memoization.",
    "Explain the difference between TCP and UDP in exactly three sentences.",
    "What is the time complexity of merge sort and why?",
    "Write a bash one-liner that finds all .py files modified in the last 24 hours.",
    "Explain gradient descent to a 10-year-old.",
    "What happens during a TLS handshake? Be concise.",
    "Write a SQL query to find the second highest salary in an employees table.",
    "What is the CAP theorem? Give a real-world example for each tradeoff.",
    "Implement binary search in Python. Include edge cases.",
    "Explain why floating point arithmetic is not associative. Give a concrete example.",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8.0",
        "torch",
    )
)


def _collect_logprobs(llm, prompts, sampling_params, engine_name):
    """Run prompts through vLLM and collect per-token log-probs as JSONL entries."""
    from vllm import SamplingParams

    outputs = llm.generate(prompts, sampling_params)
    entries = []

    for i, output in enumerate(outputs):
        completion = output.outputs[0]
        tokens = []

        if completion.logprobs:
            for pos, logprob_dict in enumerate(completion.logprobs):
                for token_id, logprob_obj in logprob_dict.items():
                    tokens.append({
                        "position": pos,
                        "token_id": token_id,
                        "token_text": logprob_obj.decoded_token,
                        "log_prob": logprob_obj.logprob,
                    })
                    break

        entries.append({
            "request_id": f"req_{i:04d}",
            "prompt": prompts[i],
            "completion": completion.text,
            "model": MODEL_ID,
            "engine": engine_name,
            "tokens": tokens,
            "metadata": {"index": i},
        })

    return entries


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1200,
    scaledown_window=60,
)
def generate_rollout_logs():
    """Generate training and serving rollout logs with different configs to produce divergence."""
    from vllm import LLM, SamplingParams

    results = {}

    # --- Training rollout: BF16, no prefix caching, temperature=0 ---
    print("=== Loading training config (BF16, no prefix cache) ===")
    training_llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        max_model_len=2048,
        enable_prefix_caching=False,
    )
    training_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        logprobs=1,
    )
    training_entries = _collect_logprobs(training_llm, PROMPTS, training_params, "verl_rollout")

    del training_llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    # --- Serving replay: FP16 (half), prefix caching on ---
    print("=== Loading serving config (FP16, prefix cache) ===")
    serving_llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        gpu_memory_utilization=0.45,
        max_model_len=2048,
        enable_prefix_caching=True,
    )
    serving_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        logprobs=1,
    )
    serving_entries = _collect_logprobs(serving_llm, PROMPTS, serving_params, "vllm_serving")

    results["training"] = training_entries
    results["serving"] = serving_entries

    # Print summary
    print(f"\n=== Generated {len(training_entries)} training + {len(serving_entries)} serving entries ===")

    return results


@app.local_entrypoint()
def main():
    """Run on Modal, download logs locally."""
    import tempfile

    print("Generating rollout logs on Modal A100...")
    results = generate_rollout_logs.remote()

    training_path = "/tmp/rollout_training.jsonl"
    serving_path = "/tmp/rollout_serving.jsonl"

    with open(training_path, "w") as f:
        for entry in results["training"]:
            f.write(json.dumps(entry) + "\n")

    with open(serving_path, "w") as f:
        for entry in results["serving"]:
            f.write(json.dumps(entry) + "\n")

    print(f"\nTraining log: {training_path} ({len(results['training'])} entries)")
    print(f"Serving log:  {serving_path} ({len(results['serving'])} entries)")
    print(f"\nRun diff:")
    print(f"  inferscope rollout-diff {training_path} {serving_path}")
