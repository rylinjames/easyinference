---
title: Wedge Integration Plan
version: 1.0.0
date: 2026-04-12
status: draft
---

# Wedge Integration Plan

How the top 3 wedges from the research catalog plug into EasyInference's existing architecture, what to build, how to test on Modal, and in what order.

## What exists today

- **ISB-1**: 4 workload families (chat, agent, rag, coding), vLLM harness, sweep/replay/analysis pipeline
- **InferScope**: 12 MCP tools, 16 CLI commands, Prometheus normalizer, 31 audit checks, benchmark artifact comparison, 5 engine adapters
- **Modal scaffolds**: `demo/modal_vllm.py`, `demo/modal_dynamo_kimi.py` (skeletons, not turnkey)
- **Active profile**: `modal profile activate romirj` (working, $0 spent this month)

## Wedge → product mapping

### Wedge 1: TurboQuant FP8 KV Cache Quantization

**What it is:** FP8 quantization of KV cache tensors during inference, cutting KV memory ~50%. vLLM #38171 has 113 upvotes — most-requested KV feature in the repo.

**Where it lands in EasyInference:**

| Component | Change |
|-----------|--------|
| `tools/kv_cache.py` | Add `compare_kv_quantization()` — benchmark FP16 vs FP8 KV at matched context lengths, report memory savings + quality delta |
| `cli.py` | New command `inferscope kv-quant-bench` wired to above |
| `telemetry/normalizer.py` | Add `kv_quantization_mode` field to `NormalizedMetrics` (FP16/FP8/FP4) |
| `optimization/checks.py` | New audit check: "KV cache using FP16 when FP8 available on Hopper+" |
| `benchmarks/_resources/experiments/` | New YAML: `kv_fp8_quant_sweep.yaml` — A/B experiment spec |
| ISB-1 `configs/` | New sweep config targeting vLLM with `--kv-cache-dtype fp8` flag |

**Kernel work (separate from EasyInference):**
- Triton FP8 KV quantize/dequantize kernel in standalone `kernels/` dir
- Reference implementation only — the product contribution is the benchmark, not the kernel

**Modal test plan:**
```
# 1. Spin up vLLM with FP16 KV on H100, run ISB-1 rag workload
modal run bench_kv.py --kv-dtype fp16 --model meta-llama/Llama-3.1-70B-Instruct --workload rag

# 2. Same but FP8 KV
modal run bench_kv.py --kv-dtype fp8 --model meta-llama/Llama-3.1-70B-Instruct --workload rag

# 3. Compare artifacts
inferscope benchmark-compare artifact_fp16.json artifact_fp8.json
```

**Expected result:** ~45-50% KV memory reduction, <1% quality regression on RULER, 1.3-1.8x throughput gain at 32k+ context.

**Estimated cost:** 2-4 hrs H100 = $7-14 on Modal.

---

### Wedge 2: Rollout-Diff CLI (Training↔Inference Log-Prob Mismatch)

**What it is:** RL training (veRL/OpenRLHF) generates rollouts using an internal inference engine. When the serving engine (vLLM) produces different log-probs for the same prompt+completion, RL training silently degrades — causing the "!!!!" collapse documented in veRL #891/#747/#751.

**Where it lands in EasyInference:**

| Component | Change |
|-----------|--------|
| `benchmarks/rollout_diff.py` | **NEW** — core diffing logic: takes two JSONL logs (training rollout vs serving replay), computes per-token log-prob divergence, flags drift above threshold |
| `benchmarks/models.py` | New `RolloutDiffArtifact` pydantic model (extends artifact pattern) |
| `cli.py` | New command `inferscope rollout-diff` |
| `cli_benchmarks.py` | Wire rollout-diff subcommand with `--training-log`, `--serving-log`, `--threshold` args |
| `server_benchmarks.py` | New MCP tool `tool_compare_rollouts` (tool #13 — first MCP surface expansion) |
| ISB-1 `workloads/` | New workload family: `rl_rollout.py` — generates GRPO-style prompt batches for rollout replay |

**Modal test plan:**
```
# 1. Run veRL GRPO training for 50 steps on A100, capture rollout logs
modal run verl_rollout.py --steps 50 --model Qwen/Qwen2.5-7B-Instruct --save-logprobs

# 2. Replay same prompts+completions through vLLM, capture serving log-probs
modal run vllm_replay.py --model Qwen/Qwen2.5-7B-Instruct --input rollout_log.jsonl

# 3. Diff
inferscope rollout-diff --training-log rollout_log.jsonl --serving-log serving_log.jsonl --threshold 0.01
```

**Expected result:** Identifies tokens where log-prob divergence > threshold, clusters by position (early vs late tokens), flags numerical parity bugs (FP32 vs BF16 accumulation).

**Estimated cost:** 1-2 hrs A100 = $3-7 on Modal. Analysis is CPU-only.

**Why this is the best first wedge:** No GPU kernel work. Pure Python. Ships in 3 days. The 5 verified veRL issues are your launch customers.

---

### Wedge 3: P_max Adaptive Scheduling for RL Rollouts

**What it is:** In GRPO/PPO training, each rollout has a `max_tokens` budget (P_max). Fixed P_max wastes GPU on easy prompts and truncates hard ones. Adaptive P_max adjusts per-batch based on reward variance from previous round.

**Where it lands in EasyInference:**

| Component | Change |
|-----------|--------|
| ISB-1 `workloads/rl_rollout.py` | Extend with P_max distribution modeling — generate rollout traces with variable-length completions matching real GRPO distributions |
| `benchmarks/experiments.py` | New experiment type: `rl_scheduling_sweep` — test fixed vs adaptive P_max across reward variance bins |
| `benchmarks/_resources/experiments/` | New YAML: `pmax_adaptive_sweep.yaml` |
| `optimization/checks.py` | New audit check: "rollout P_max is fixed when reward variance is high" |
| `tools/recommend.py` | Add `recommend_pmax_schedule()` — given reward history, output optimal P_max curve |
| `cli.py` | New command `inferscope pmax-recommend` |

**Modal test plan:**
```
# 1. Run fixed P_max=2048 GRPO training, capture per-rollout token counts + rewards
modal run verl_pmax.py --pmax 2048 --steps 200

# 2. Run adaptive P_max (our scheduler), same config
modal run verl_pmax.py --pmax adaptive --steps 200

# 3. Compare reward curves + GPU utilization
inferscope benchmark-compare fixed_pmax.json adaptive_pmax.json
```

**Expected cost:** 4-8 hrs A100-80GB = $14-28 on Modal.

**Depends on:** Wedge 2's `rl_rollout` workload family.

---

## Execution order

```
Week 1 (days 1-5):  Wedge 2 — Rollout-diff CLI
                    - Day 1-2: rollout_diff.py + RolloutDiffArtifact model
                    - Day 3: CLI wiring + basic tests
                    - Day 4: Modal scripts (verl_rollout.py, vllm_replay.py)
                    - Day 5: Run on Modal, validate against veRL #891 repro

Week 2 (days 6-10): Wedge 1 — TurboQuant KV benchmark
                    - Day 6-7: kv_fp8_quant_sweep.yaml + ISB-1 config
                    - Day 8: compare_kv_quantization() + CLI command
                    - Day 9: Modal benchmark runs (FP16 vs FP8)
                    - Day 10: Audit check + write-up with numbers

Week 3 (days 11-15): Wedge 3 — P_max scheduling
                    - Day 11-12: rl_rollout workload family in ISB-1
                    - Day 13: pmax_adaptive_sweep experiment spec
                    - Day 14: Modal runs (fixed vs adaptive)
                    - Day 15: recommend_pmax_schedule() + CLI
```

## Modal infrastructure setup (day 0)

```python
# demo/modal_verl.py — veRL training environment
import modal

app = modal.App("inferscope-verl")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("verl[vllm]", "torch", "transformers", "wandb")
)

@app.function(gpu="A100-80GB", image=image, timeout=3600)
def run_grpo_rollout(model: str, steps: int, pmax: int):
    ...

# demo/modal_kv_bench.py — vLLM KV quantization benchmark
@app.function(gpu="H100", image=vllm_image, timeout=3600)
def run_kv_quant_bench(model: str, kv_dtype: str, workload: str):
    ...
```

## New files created (total)

| File | Product | Purpose |
|------|---------|---------|
| `benchmarks/rollout_diff.py` | inferscope | Core log-prob diff logic |
| `benchmarks/models.py` (extend) | inferscope | RolloutDiffArtifact |
| `cli_benchmarks.py` (extend) | inferscope | rollout-diff command |
| `server_benchmarks.py` (extend) | inferscope | tool_compare_rollouts MCP tool |
| `tools/kv_cache.py` (extend) | inferscope | compare_kv_quantization() |
| `optimization/checks.py` (extend) | inferscope | 2 new audit checks |
| `tools/recommend.py` (extend) | inferscope | recommend_pmax_schedule() |
| `workloads/rl_rollout.py` | isb1 | 5th workload family |
| `configs/workloads/rl_rollout.yaml` | isb1 | Workload config |
| `experiments/kv_fp8_quant_sweep.yaml` | inferscope | KV quant A/B spec |
| `experiments/pmax_adaptive_sweep.yaml` | inferscope | P_max A/B spec |
| `demo/modal_verl.py` | demo | veRL on Modal |
| `demo/modal_kv_bench.py` | demo | KV bench on Modal |

## Success criteria

1. `inferscope rollout-diff` reproduces the "!!!!" collapse from veRL #891 on Modal with real numbers
2. `inferscope kv-quant-bench` shows >40% memory savings with <1% RULER regression
3. `inferscope pmax-recommend` outputs a schedule that beats fixed P_max by >15% GPU utilization
4. All three produce `BenchmarkArtifact` JSONs that feed into existing `benchmark-compare`
5. Blog post with real numbers from each → post to veRL/vLLM GitHub discussions

## What this does NOT include

- Production kernel authoring (TurboQuant kernel is reference-only, not production Triton)
- MoE expert routing changes
- Reasoning/thinking budget control
- VLA or multimodal wedges (separate phase)
- Dynamo-specific integration (stays vLLM-first for wedge validation)
