"""Benchmark FP16 vs FP8 KV cache quantization on Modal.

Runs the same model+workload with both KV cache dtypes, captures ISB-1-style
metrics, and outputs two benchmark artifact JSONs for comparison.

Targets wedge: TurboQuant KV Cache Quantization (vLLM #38171, 113 upvotes).

Usage:
    # Run both configs on H100
    modal run demo/modal_kv_quant_bench.py

    # Compare artifacts locally
    inferscope benchmark-compare /tmp/kv_bench_fp16.json /tmp/kv_bench_fp8.json

Cost: ~$7-14 (2-4 hrs H100)
"""

import json
import time
from dataclasses import dataclass

import modal

app = modal.App("inferscope-kv-quant")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Long-context prompts to stress KV cache
PROMPTS = [
    "Summarize the history of computing from 1950 to 2025 in detail, covering major breakthroughs in hardware, software, networking, and AI. Be comprehensive." * 3,
    "Write a detailed technical guide on implementing a distributed key-value store from scratch, covering consistency models, replication, partitioning, and failure handling." * 3,
    "Explain the complete lifecycle of an HTTP request from the browser to a load-balanced backend service and back, including DNS, TLS, TCP, routing, and caching." * 3,
    "Compare and contrast 10 different sorting algorithms, providing pseudocode, time complexity analysis, space complexity, stability properties, and real-world use cases for each." * 2,
    "Write a comprehensive guide to GPU programming with CUDA, covering memory hierarchy, kernel launches, warp scheduling, shared memory, and optimization techniques." * 3,
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.8.0", "torch")
)


@dataclass
class BenchResult:
    kv_dtype: str
    model: str
    total_prompts: int
    total_prompt_tokens: int
    total_completion_tokens: int
    wall_time_s: float
    ttft_ms_avg: float
    ttft_ms_p95: float
    tpot_ms_avg: float
    throughput_tok_s: float
    gpu_memory_peak_mb: float
    gpu_memory_allocated_mb: float


def _run_benchmark(model_id: str, kv_dtype: str, prompts: list[str]) -> dict:
    """Run inference with specified KV cache dtype and collect metrics."""
    import torch
    from vllm import LLM, SamplingParams

    torch.cuda.reset_peak_memory_stats()

    llm = LLM(
        model=model_id,
        dtype="auto",
        kv_cache_dtype=kv_dtype,
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        enable_prefix_caching=True,
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    # Warmup
    print(f"  Warmup ({kv_dtype})...")
    _ = llm.generate(["Hello, world!"], SamplingParams(max_tokens=16, temperature=0.0))

    # Benchmark
    print(f"  Running {len(prompts)} prompts ({kv_dtype})...")
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall_time = time.perf_counter() - start

    total_prompt_tokens = 0
    total_completion_tokens = 0
    ttfts = []
    tpots = []

    for output in outputs:
        total_prompt_tokens += len(output.prompt_token_ids)
        comp = output.outputs[0]
        n_tokens = len(comp.token_ids)
        total_completion_tokens += n_tokens

    mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    mem_current = torch.cuda.memory_allocated() / (1024 * 1024)

    throughput = total_completion_tokens / wall_time if wall_time > 0 else 0

    result = {
        "kv_dtype": kv_dtype,
        "model": model_id,
        "total_prompts": len(prompts),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "wall_time_s": round(wall_time, 2),
        "throughput_tok_s": round(throughput, 1),
        "gpu_memory_peak_mb": round(mem_peak, 1),
        "gpu_memory_allocated_mb": round(mem_current, 1),
    }

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    scaledown_window=60,
)
def run_kv_quant_comparison():
    """Run FP16 and FP8 KV cache benchmarks back-to-back."""
    print("=== KV Cache Quantization Benchmark ===")
    print(f"Model: {MODEL_ID}")
    print(f"Prompts: {len(PROMPTS)}")

    print("\n--- FP16 KV Cache ---")
    fp16_result = _run_benchmark(MODEL_ID, "auto", PROMPTS)
    print(f"  Peak memory: {fp16_result['gpu_memory_peak_mb']:.0f} MB")
    print(f"  Throughput: {fp16_result['throughput_tok_s']:.1f} tok/s")

    print("\n--- FP8 KV Cache ---")
    fp8_result = _run_benchmark(MODEL_ID, "fp8", PROMPTS)
    print(f"  Peak memory: {fp8_result['gpu_memory_peak_mb']:.0f} MB")
    print(f"  Throughput: {fp8_result['throughput_tok_s']:.1f} tok/s")

    # Compute deltas
    mem_savings = 1.0 - (fp8_result["gpu_memory_peak_mb"] / fp16_result["gpu_memory_peak_mb"])
    throughput_gain = fp8_result["throughput_tok_s"] / fp16_result["throughput_tok_s"] if fp16_result["throughput_tok_s"] > 0 else 0

    comparison = {
        "fp16": fp16_result,
        "fp8": fp8_result,
        "deltas": {
            "memory_savings_pct": round(mem_savings * 100, 1),
            "throughput_ratio": round(throughput_gain, 2),
            "wall_time_delta_s": round(fp8_result["wall_time_s"] - fp16_result["wall_time_s"], 2),
        },
    }

    print(f"\n=== Results ===")
    print(f"Memory savings: {mem_savings*100:.1f}%")
    print(f"Throughput ratio: {throughput_gain:.2f}x")

    return comparison


@app.local_entrypoint()
def main():
    """Run on Modal, save results locally."""
    print("Running KV quantization benchmark on Modal H100...")
    results = run_kv_quant_comparison.remote()

    fp16_path = "/tmp/kv_bench_fp16.json"
    fp8_path = "/tmp/kv_bench_fp8.json"
    comparison_path = "/tmp/kv_bench_comparison.json"

    with open(fp16_path, "w") as f:
        json.dump(results["fp16"], f, indent=2)
    with open(fp8_path, "w") as f:
        json.dump(results["fp8"], f, indent=2)
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFP16 result: {fp16_path}")
    print(f"FP8 result:  {fp8_path}")
    print(f"Comparison:  {comparison_path}")

    deltas = results["deltas"]
    print(f"\n--- Summary ---")
    print(f"Memory savings: {deltas['memory_savings_pct']}%")
    print(f"Throughput ratio: {deltas['throughput_ratio']}x")
    print(f"Wall time delta: {deltas['wall_time_delta_s']}s")
