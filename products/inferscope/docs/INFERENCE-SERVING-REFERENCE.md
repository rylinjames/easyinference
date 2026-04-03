# LLM Inference Serving Reference

End-to-end pipeline: model loading, weight sharding, KV cache management, scheduling, prefill/decode, offloading, and disaggregated serving — across Ampere, Hopper, and Blackwell GPUs.

All details verified against vLLM/SGLang docs, NVIDIA tech blogs, and LMCache documentation (March 2026).

---

## 1. Model Loading Pipeline

### Weight Formats

| Format | Description | vLLM Support | Use Case |
|--------|------------|-------------|----------|
| **safetensors** | Tensor-only, fast mmap loading | Primary format | Default for HuggingFace models |
| **GPTQ** | INT4 weight-only quantization | `--quantization gptq` | Ampere (no FP8) |
| **AWQ** | INT4 weight-only (Marlin kernels) | `--quantization awq` | Ampere, budget inference |
| **FP8** (dynamic) | On-the-fly BF16->FP8 quantization | `--quantization fp8` | Hopper/Blackwell, no calibration needed |
| **FP8** (static) | Pre-calibrated FP8 weights in safetensors | `--quantization fp8` | Best accuracy, requires calibration |
| **NVFP4** | 4-bit with 2-level scaling (E4M3 + FP32) | `--quantization nvfp4` | **Blackwell only** (SM100/SM103) |
| **GGUF** | Self-contained (weights + metadata) | `repo_id:quant_type` | CPU inference, llama.cpp ecosystem |

### Loading Process (vLLM)

```
1. Download weights       HuggingFace Hub -> local cache (safetensors/bin)
        |
2. Weight sharding        Split by TP: column parallel (QKV, gate) / row parallel (O, down)
        |                 Split by EP: expert weights distributed across EP ranks
        |
3. Quantization           Dynamic FP8: BF16 -> FP8 E4M3 on-the-fly (no calibration)
        |                 Static FP8/NVFP4: load pre-quantized weights directly
        |
4. GPU memory allocation  Weights loaded to each GPU's HBM
        |                 Remaining HBM allocated to KV cache blocks
        |
5. KV cache profiling     Run dummy forward pass to measure actual memory usage
        |                 Compute max KV cache blocks that fit
        |
6. CUDA graph capture     Capture optimized execution graphs for common batch sizes
        |                 Reduces kernel launch overhead
        |
7. Ready to serve         API server starts accepting requests
```

### Memory Layout (per GPU)

```
┌─────────────────────────────────────────┐
│              GPU HBM                     │
├─────────────────────────────────────────┤
│  Model Weights (sharded by TP)          │  <- Fixed after loading
│  e.g., 70B FP8 / TP=2 = ~35 GB/GPU    │
├─────────────────────────────────────────┤
│  Activation Memory (~1-3 GB)            │  <- Overhead for forward pass
├─────────────────────────────────────────┤
│  KV Cache Blocks (PagedAttention)       │  <- Dynamic, grows/shrinks
│  Block size: typically 16 tokens        │
│  e.g., H100 80GB: ~40 GB for KV        │
│  e.g., H200 141GB: ~100 GB for KV      │
│  e.g., B200 192GB: ~140 GB for KV      │
├─────────────────────────────────────────┤
│  CUDA Graphs + Misc (~2-5 GB)          │
└─────────────────────────────────────────┘
```

### Per-GPU Memory by Generation (70B FP8, TP=2)

| GPU | HBM | Usable (0.95) | Weights/GPU | KV Budget/GPU | Max Sequences (4K avg) |
|-----|-----|--------------|------------|--------------|----------------------|
| A100 80GB | 80 GB | 76 GB | ~35 GB | ~38 GB | ~4,000 |
| H100 SXM | 80 GB | 76 GB | ~35 GB | ~38 GB | ~4,000 |
| H200 SXM | 141 GB | 134 GB | ~35 GB | ~96 GB | ~10,000 |
| B200 | 192 GB | 182 GB | ~35 GB | ~144 GB | ~15,000 |
| B300 | 288 GB | 274 GB | ~35 GB | ~236 GB | ~25,000 |

Note: With NVFP4 on Blackwell, 70B model weights drop to ~17 GB/GPU, leaving even more for KV.

---

## 2. KV Cache Management

### PagedAttention (Core Mechanism)

```
Traditional:  Allocate contiguous memory per sequence → 60-80% waste (fragmentation)

PagedAttention:  Allocate fixed-size blocks on demand → <4% waste

┌──────────────────────────────────────────────────────┐
│  Logical KV Cache (per sequence)                      │
│  [token 0..15] [token 16..31] [token 32..47] ...     │
│       ↓              ↓              ↓                 │
│  Block Table: maps logical blocks → physical blocks   │
│       ↓              ↓              ↓                 │
│  Physical Blocks (anywhere in GPU memory):            │
│  [Block 42]    [Block 7]      [Block 128]  ...       │
└──────────────────────────────────────────────────────┘

Key properties:
- Blocks are non-contiguous (like OS virtual memory pages)
- Block size: typically 16 tokens (configurable)
- Allocated on demand as tokens are generated
- Freed immediately when sequence completes
- Prefix sharing: multiple sequences can point to same physical blocks
```

### KV Cache Per Token (bytes, all layers)

| Model | FP16 KV | FP8 KV | Savings |
|-------|---------|--------|---------|
| Llama-3-8B (32 layers, 8 KV heads) | 512 B/tok | 256 B/tok | 2x |
| Llama-3-70B (80 layers, 8 KV heads) | 1,280 B/tok | 640 B/tok | 2x |
| DeepSeek-R1 (MLA, 61 layers) | ~160 B/tok | ~80 B/tok | 2x (MLA compresses ~32x vs GQA) |
| Qwen3.5-72B (80 layers, 8 KV heads) | 1,280 B/tok | 640 B/tok | 2x |

### FP8 KV Cache (Hopper + Blackwell)

FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) is the **single highest-impact optimization** for long-context inference:
- 2x memory savings → 2x more concurrent sequences or 2x longer context
- Native on all Hopper (sm_90/90a) and Blackwell (sm_100/103)
- Negligible accuracy impact for inference
- NOT available on Ampere (A100 uses FP16 KV)

### Prefix Caching

**vLLM V1:** Zero-overhead automatic prefix caching (always on). Hash-based lookup. Shared system prompts, tool schemas cached across requests.

**SGLang RadixAttention:** Tree-based prefix cache using radix tree data structure.
- Cache hit rates: 50-99% depending on workload
- Coding workloads: 85-95% hit rate (system prompt + repo context reuse)
- Chat workloads: 30-50% hit rate (shared system prompt)
- Each node in radix tree = token sequence + associated KV cache pages
- LRU eviction when memory pressure

---

## 3. Scheduling (vLLM V1)

### Unified Scheduler

vLLM V1 removes the traditional prefill/decode distinction. Each step is represented as:

```python
schedule = {request_id: num_tokens}  # tokens to process per request this step
```

This enables mixing prefill and decode in the same batch (chunked prefill).

### Three-Stage Execution Loop

```
┌──────────────────────────────────────────────┐
│  Step N                                       │
│                                               │
│  1. SCHEDULE                                  │
│     Select requests: decode and/or prefill    │
│     Respect token budget (max_num_batched_tokens) │
│     Apply chunked prefill limits              │
│                                               │
│  2. EXECUTE (GPU forward pass)                │
│     Flatten batch into single sequence        │
│     FlashAttention kernel handles variable lengths │
│     Output: logits for each token position    │
│                                               │
│  3. POSTPROCESS                               │
│     Sample next tokens                        │
│     Append to request state                   │
│     Detokenize + check stop conditions        │
│     Free KV blocks for completed sequences    │
└──────────────────────────────────────────────┘
```

### Key Scheduling Parameters

| Parameter | Default (online) | Effect |
|-----------|-----------------|--------|
| `max_num_batched_tokens` | 8192 | Max tokens per step. Lower = better ITL, higher = better TTFT |
| `max_num_seqs` | 256 | Max concurrent sequences. Limits KV cache consumption |
| `chunked_prefill` | ON (V1) | Split large prefills into chunks, interleave with decode |
| `enable_chunked_prefill` | True (Hopper) | Hopper handles compute/memory overlap well |

### Chunked Prefill: Prefill vs Decode Tradeoff

```
WITHOUT chunked prefill:
  Step 1: [========== PREFILL 32K tokens ==========]  ← all other requests wait
  Step 2: [d][d][d][d][d][d]  ← decode resumes

WITH chunked prefill (budget=8192):
  Step 1: [== PREFILL chunk 1 ==][d][d][d][d]  ← decode interleaved
  Step 2: [== PREFILL chunk 2 ==][d][d][d][d]  ← decode continues
  Step 3: [== PREFILL chunk 3 ==][d][d][d][d]
  Step 4: [== PREFILL chunk 4 ==][d][d][d][d]

Result: TTFT slightly worse, but ITL (decode latency) stays stable.
```

**When to disable chunked prefill:**
- AMD CDNA + long context (KV staging overhead)
- TTFT SLO < 500ms with model context > 8K
- Model context > 32K for RAG workloads (contiguous prefill avoids KV fragmentation)

**When chunked prefill is best:**
- Hopper/Blackwell + Chat/Agent workloads (Tier 1)
- High concurrency (many users, short prompts)
- wgmma/TMA handle compute-memory overlap efficiently on Hopper

---

## 4. Prefill vs Decode: The Fundamental Asymmetry

```
PREFILL (prompt processing):
  - Process ALL prompt tokens in parallel
  - COMPUTE-BOUND: limited by TFLOPS (matrix multiply)
  - Duration: proportional to prompt length
  - Metric: TTFT (Time to First Token)
  - Hardware bottleneck: Tensor Cores

DECODE (token generation):
  - Generate ONE token at a time, autoregressively
  - MEMORY-BOUND: limited by memory bandwidth (KV cache reads)
  - Duration: proportional to output length
  - Metric: ITL (Inter-Token Latency) / TPS (Tokens Per Second)
  - Hardware bottleneck: HBM bandwidth
```

### GPU Suitability by Phase

| GPU | Prefill (Compute) | Decode (Memory BW) | Best For |
|-----|-------------------|-------------------|----------|
| A100 (19.5 TFLOPS FP32, 2.0 TB/s) | Moderate | Moderate | Balanced |
| H100 (1,979 TFLOPS FP8, 3.35 TB/s) | Excellent | Good | Prefill-heavy |
| H200 (1,979 TFLOPS FP8, 4.8 TB/s) | Excellent | **Excellent** | Decode-heavy (2x BW) |
| B200 (4,500 TFLOPS FP8, 8.0 TB/s) | Outstanding | Outstanding | Everything |
| B300 (7,000 TFLOPS FP8, 8.0 TB/s) | Peak | Outstanding | Max throughput |

### H200 vs H100: The Memory Bandwidth Story

Same compute die. Same TFLOPS. But H200's 4.8 TB/s (vs 3.35 TB/s) means:
- ~43% more decode throughput at large batch sizes
- Decode is memory-bound → more bandwidth = more tokens/sec
- This is why H200 is the best "value" GPU for inference

### B200/B300: Everything Gets Better

- 8.0 TB/s = 2.4x H100's bandwidth → decode throughput scales proportionally
- 4,500+ FP8 TFLOPS = 2.3x H100 → prefill scales too
- NVFP4 doubles throughput again (9,000 TFLOPS FP4 on B200)

---

## 5. KV Cache Offloading and Tiering

### The Problem

KV cache grows linearly with: `num_sequences × context_length × kv_bytes_per_token`

At scale (e.g., 1000 concurrent users × 32K context × Llama-70B FP8 = ~20 GB KV), it exceeds GPU memory.

### Tiering Architecture

```
┌─────────────────────────────────────────────────┐
│  Tier 0: GPU HBM (fastest, smallest)            │
│  - Active decode sequences                       │
│  - Hot prefix cache                              │
│  - H100: 40 GB, H200: 100 GB, B200: 140 GB     │
├─────────────────────────────────────────────────┤
│  Tier 1: CPU DRAM (via PCIe or NVLink-C2C)      │
│  - Paused/idle sessions                          │
│  - Warm prefix cache overflow                    │
│  - Typical: 256-1024 GB available                │
│  - PCIe Gen5: ~64 GB/s (bottleneck!)            │
│  - NVLink-C2C (Grace): ~900 GB/s (no bottleneck)│
├─────────────────────────────────────────────────┤
│  Tier 2: Local SSD / Remote Storage              │
│  - Cold prefix cache                             │
│  - Long-horizon agent session persistence        │
│  - NVMe: ~7 GB/s                                │
│  - Ceph/S3: network-bound                       │
└─────────────────────────────────────────────────┘
```

### Offloading Mechanisms in vLLM

| Mechanism | How It Works | When to Use |
|-----------|-------------|-------------|
| **OffloadingConnector** | Async GPU→CPU block transfer via `--kv-transfer-config` | Simple offloading, single-node |
| **LMCacheConnectorV1** | Content-addressed KV store, cross-session sharing, NIXL transport | Production disaggregated + offloading |
| **CPU prefix cache** | Evicted GPU prefix blocks stored in CPU DRAM for potential reuse | High prefix reuse workloads |

### OffloadingConnector (vLLM native)

```bash
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":2048}}'
```

- Loads KV from CPU reduces TTFT by 2-22x vs recomputation (depends on prompt size)
- **CRITICAL:** Do NOT offload during active decode — PCIe transfer dominates latency
- Only offload cold/idle sessions

### GPU-Specific Offloading Strategy

| GPU | KV Budget | Offload Needed? | Strategy | Why |
|-----|----------|-----------------|----------|-----|
| **A100 80GB** | ~38 GB | Often | PCIe offload, aggressive eviction | Small KV budget, no FP8 KV |
| **H100 SXM** | ~38 GB | Often | Cold-only offload (60s idle), NVLink cap 0.8 | Same budget as A100 but FP8 KV doubles it |
| **H100 PCIe** | ~38 GB | Often | Cold-only (30s), PCIe cap 0.5 | **No NVLink — PCIe bottleneck** |
| **H200 SXM** | ~96 GB | Rarely | Disabled for chat/agent, cold-only for RAG | 141GB = massive KV headroom |
| **GH200** | ~55 GB + 480 GB Grace | For long-context | GPU→Grace via C2C (900 GB/s) | Grace LPDDR5X as Tier 1, ~7x faster than PCIe |
| **B200** | ~140 GB | Rarely | Disabled for most workloads | 192GB = fits almost everything |
| **B300** | ~236 GB | Almost never | Disabled | 288GB = enormous headroom |
| **GB200** | ~140 GB + 480 GB Grace | For extreme long-ctx | GPU→Grace via C2C (900 GB/s) | Grace eliminates PCIe bottleneck |
| **GB300** | ~236 GB + 480 GB Grace | For extreme long-ctx | GPU→Grace via C2C (900 GB/s) | 288GB + 480GB = 768GB per die |

### The PCIe Bottleneck (Critical)

```
PCIe Gen5 x16:  ~64 GB/s bidirectional (128 GB/s theoretical)
NVLink4:         900 GB/s  (14x PCIe)
NVLink5:        1800 GB/s  (28x PCIe)
NVLink-C2C:      900 GB/s  (~7x PCIe, dedicated Grace<->GPU)

Rule: If KV offloading goes through PCIe, it WILL bottleneck decode.
      Use NVLink-connected systems or Grace Superchips for offloading.
      On PCIe-only systems (H100 PCIe), prefer disaggregation over offloading.
```

---

## 6. Disaggregated Prefill/Decode Serving

### Architecture

```
                    ┌──────────────┐
     requests ───→  │   Router     │
                    │ (vLLM Router)│
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌────────────────┐       ┌────────────────┐
     │ Prefill Worker  │       │ Decode Worker   │
     │ (compute-bound) │──KV──→│ (memory-bound)  │
     │ GPU 0, 1        │ xfer │ GPU 2, 3, 4, 5  │
     └────────────────┘       └────────────────┘
```

**Why disaggregate:**
- Prefill = compute-bound → wants maximum TFLOPS
- Decode = memory-bound → wants maximum bandwidth
- Co-locating them wastes resources: prefill starves decode, or decode idles compute

### KV Transfer Connectors

| Connector | Transport | Bandwidth | Use Case |
|-----------|-----------|-----------|----------|
| **NixlConnector** | UCX (RDMA/InfiniBand/EFA) | Network-speed | Multi-node disaggregation |
| **P2pNcclConnector** | NVLink (same-node NCCL) | NVLink speed | Same-node, no RDMA |
| **LMCacheConnectorV1** | NIXL + content-addressed store | NVLink/RDMA | Cross-session KV sharing + disagg |
| **MooncakeConnector** | RDMA (AMD-optimized) | Network-speed | AMD MI300X/MI355X |

### GPU Generation Impact on Disaggregation

| | Hopper (H100/H200) | Blackwell (B200/B300/GB200/GB300) |
|---|---|---|
| **KV transfer BW** | NVLink4: 900 GB/s | NVLink5: 1,800 GB/s (2x) |
| **Suggested P:D ratio (8 GPU)** | 2P : 6D | 1P : 7D (faster KV transfer) |
| **KV staging** | HBM only | HBM + Grace LPDDR5X (GB200/GB300) |
| **nvCOMP decompression** | No | Hardware-accelerated data I/O |
| **Helix parallelism** | No | Multi-node overlapped transfer + compute |

### When to Disaggregate

| Condition | Recommendation |
|-----------|---------------|
| Short prompts (<4K tokens) | **Do NOT** — disagg overhead degrades 20-30% |
| Low request rate (<10/sec) | **Do NOT** — overhead outweighs benefit |
| Single GPU | **Impossible** — need >=2 GPUs |
| No RDMA/NVLink | **Do NOT** — PCIe-only KV transfer kills benefit |
| Long prompts (>8K) + high rate + RDMA | **Do it** — disagg improves TTFT significantly |
| H100 PCIe + long context | **Prefer disagg over offloading** (PCIe bottleneck) |

### Layer-wise Async KV Transfer (Latest)

Instead of transferring the entire KV cache after prefill completes, transfer each layer's KV as soon as it's computed:

```
Prefill GPU:  Layer 0 → Layer 1 → Layer 2 → ... → Layer N
                 ↓          ↓          ↓
                KV 0       KV 1       KV 2      (async transfer)
                 ↓          ↓          ↓
Decode GPU:   [recv KV0] [recv KV1] [recv KV2]  ... ready to decode

Benefit: Communication hidden behind computation of subsequent layers.
```

---

## 7. Attention Kernels by GPU Generation

| GPU Generation | Primary Kernel | Key Innovation | SM Utilization |
|---------------|---------------|----------------|---------------|
| **Ampere** (A100) | FlashAttention-2 | Tiled attention, memory-efficient | ~35% |
| **Hopper** (H100/H200) | FlashAttention-3 | wgmma + TMA + async pipeline | **~75%** |
| **Blackwell** (B200/B300) | FlashAttention-4 | RoPE+KV cache fusion, TMEM | ~85%+ (est) |
| **Blackwell Ultra** (B300) | FlashAttention-4 + accel softmax | 10.7 TeraExp/s SFU | ~90%+ (est) |

### Hopper FlashAttention-3 Advantage

- **wgmma:** Warp Group Matrix Multiply-Accumulate — operates on larger tiles
- **TMA:** Tensor Memory Accelerator — zero-overhead address computation for KV cache loads
- **Async pipeline:** 2-3 stage software pipelining overlaps compute with memory access
- Result: 2x+ throughput over FA2 at same hardware

### Blackwell FlashAttention-4 Advantage

- **RoPE fusion:** Rotary position embedding computed inside attention kernel (not separate pass)
- **KV cache fusion:** KV cache loads fused with attention computation
- **TMEM:** 256 KB per-SM Tensor Memory — dedicated fast storage for attention tiles
- **tcgen05:** New tensor core generation instructions (SM100/SM103 only)
- Result: 4.5x speedup over FA2

---

## 8. Quantization Pipeline by GPU

| Precision | Ampere | Hopper | Blackwell | Model Size (70B) | KV Cache |
|-----------|--------|--------|-----------|-------------------|----------|
| **BF16/FP16** | Native | Native | Native | ~140 GB | FP16: 1,280 B/tok |
| **FP8** (dynamic) | W8A16 Marlin | **Native** | **Native** | ~70 GB | FP8: 640 B/tok |
| **NVFP4** | Not supported | Not supported | **Native** | **~35 GB** | FP8: 640 B/tok |
| **AWQ/GPTQ** (INT4) | Marlin kernels | Marlin kernels | Marlin kernels | ~35 GB | FP16: 1,280 B/tok |
| **INT8** | Native | Native | Native | ~70 GB | FP16: 1,280 B/tok |

### NVFP4 Details (Blackwell Only)

```
Format: 4-bit floating point with 2-level scaling
  - Level 1: E4M3 scale factor per 16-element block
  - Level 2: FP32 per-tensor scalar

Memory: 3.5x reduction vs FP16, 1.8x vs FP8
Accuracy: <1% degradation on most benchmarks
Throughput: 2x vs FP8 on Blackwell tensor cores

Available models (March 2026):
  nvidia/Llama-3.3-70B-Instruct-NVFP4
  nvidia/DeepSeek-R1-NVFP4
  nvidia/DeepSeek-V3.2-NVFP4
  nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4
  nvidia/Llama-3.1-405B-Instruct-NVFP4

vLLM command:
  vllm serve nvidia/Llama-3.3-70B-Instruct-NVFP4 --quantization nvfp4
```

---

## 9. End-to-End: Life of an Inference Request

```
1. HTTP Request arrives at API server
   POST /v1/chat/completions {"messages": [...], "max_tokens": 512}
        │
2. Tokenization (API server process)
   messages → token IDs (prompt_tokens)
        │
3. Scheduler picks up request
   Checks: KV cache budget, active sequences, token budget
   Decision: {request_id: len(prompt_tokens)}  (prefill this request)
        │
4. Prefix cache lookup
   Hash prompt tokens → check if KV blocks already cached
   HIT:  Reuse cached KV blocks, only compute new tokens
   MISS: Full prefill needed
        │
5. PREFILL (GPU forward pass)
   All prompt tokens processed in parallel
   Attention kernel: FA2 (Ampere), FA3 (Hopper), FA4 (Blackwell)
   Output: KV cache blocks allocated + first token logits
        │
6. Sample first token → stream to client (TTFT measured here)
        │
7. DECODE loop (autoregressive)
   Each step:
     a. Scheduler: {request_id: 1}  (one new token)
     b. GPU forward: read KV cache, compute attention, generate logit
     c. Sample token → stream to client
     d. Allocate new KV block if needed
     e. Check stop condition (EOS, max_tokens)
   Repeat until done (ITL measured per token)
        │
8. Completion
   Free KV cache blocks (or retain for prefix cache)
   Return final response + usage stats
```

---

## Sources

- [vLLM V1 Architecture (vLLM Blog)](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System (vLLM Blog)](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [Life of an Inference Request (Ubicloud)](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1)
- [PagedAttention Paper (arXiv)](https://arxiv.org/abs/2309.06180)
- [vLLM Optimization and Tuning Docs](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM KV Cache Offloading (vLLM Blog)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [LMCache: Offload KV Cache to CPU](https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html)
- [NIXL-based P/D Disaggregation in vLLM V1 (LMCache Blog)](https://blog.lmcache.ai/en/2025/04/11/shaping-nixl-based-pd-disaggregation-in-vllm-v1/)
- [NixlConnector Usage Guide (vLLM Docs)](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
- [SGLang RadixAttention (LMSYS)](https://lmsys.org/blog/2024-01-17-sglang/)
- [SGLang HiCache (LMSYS)](https://lmsys.org/blog/2025-09-10-sglang-hicache/)
- [vLLM FP8 Quantization Docs](https://docs.vllm.ai/en/latest/features/quantization/fp8/)
- [NVFP4 for LLM Inference (NVIDIA Tech Blog)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [vLLM NVFP4 Quantization (llm-compressor Docs)](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)
- [vLLM Parallelism and Scaling Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [KV Cache Transfer in Disaggregated Serving (NVIDIA Dynamo Docs)](https://docs.nvidia.com/dynamo/latest/backends/trtllm/kv-cache-transfer.html)
- [KV Caching with vLLM, LMCache, and Ceph](https://ceph.io/en/news/blog/2025/vllm-kv-caching/)
