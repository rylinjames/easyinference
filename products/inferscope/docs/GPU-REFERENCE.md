# NVIDIA Hopper & Blackwell GPU Reference

All specs verified against NVIDIA datasheets and technical blogs (March 2026).

---

## Complete GPU Lineup

### Hopper Family (8 variants)

| GPU | CC | Die | HBM | BW | NVLink | PCIe | TDP | Form Factor |
|-----|-----|-----|-----|-----|--------|------|-----|-------------|
| **H100 SXM** | sm_90a | GH100 | 80 GB HBM3 | 3.35 TB/s | v4 @ 900 GB/s | Gen5 | 700W | SXM baseboard |
| **H100 NVL** | sm_90a | GH100 (dual) | 94 GB HBM3 | 3.9 TB/s | 600 GB/s (bridge) | Gen5 | 400W | Dual-GPU PCIe card |
| **H100 PCIe** | sm_90 | GH100 | 80 GB HBM2e | 2.0 TB/s | None | Gen5 | 350W | Single PCIe card |
| **H200 SXM** | sm_90a | GH100 | 141 GB HBM3e | 4.8 TB/s | v4 @ 900 GB/s | Gen5 | 700W | SXM baseboard |
| **H200 NVL** | sm_90a | GH100 | 141 GB HBM3e | 4.8 TB/s | v4 @ 900 GB/s | Gen5 | 700W | Air-cooled rack |
| **GH200** | sm_90a | GH200 Superchip | 96 GB HBM3 | 4.0 TB/s | v4 @ 900 GB/s + C2C 900 | Gen5 | 1000W | Grace+Hopper |
| **H800** | sm_90a | GH100 | 80 GB HBM3 | 3.35 TB/s | v4 @ **400 GB/s** | Gen5 | 700W | **EXPORT-RESTRICTED** |
| **H20** | sm_90a | GH100 (cut) | 96 GB HBM3 | 4.0 TB/s | v4 @ 900 GB/s | Gen5 | 350W | **EXPORT-RESTRICTED** (78 SMs) |

### Blackwell Family (6 variants)

| GPU | CC | Die | HBM | BW | NVLink | PCIe | TDP | Form Factor |
|-----|-----|-----|-----|-----|--------|------|-----|-------------|
| **B100** | sm_100 | GB100 (dual-die) | 192 GB HBM3e | 8.0 TB/s | v5 @ 1800 GB/s | Gen5 | 700W | Lower-power, mostly skipped |
| **B200** | sm_100 | GB100 (dual-die) | 192 GB HBM3e | 8.0 TB/s | v5 @ 1800 GB/s | Gen5 | 1000W | Primary Blackwell |
| **B300** | sm_103 | GB100 Ultra | 288 GB HBM3e | 8.0 TB/s | v5 @ 1800 GB/s | Gen6 | 1400W | Blackwell Ultra |
| **GB200** | sm_100 | 2x GB100 + Grace | 192 GB HBM3e /die | 8.0 TB/s | v5 @ 1800 + C2C 900 | Gen5 | ~2700W | Superchip |
| **GB300** | sm_103 | 2x GB100U + Grace | 288 GB HBM3e /die | 8.0 TB/s | v5 @ 1800 + C2C 900 | Gen6 | ~3800W | Superchip |

---

## Hopper Detailed Specs

### H100 SXM

```
Architecture:    Hopper (sm_90a)
Die:             GH100, TSMC 4N
Compute:         132 SMs / 528 Tensor Cores / 16,896 CUDA Cores
Memory:          80 GB HBM3 @ 3.35 TB/s
L2 Cache:        50 MB
NVLink:          v4, 900 GB/s bidirectional
TDP:             700W
FP8 TC:          1,979 TFLOPS (3,958 sparse)
FP16 TC:         990 TFLOPS (1,979 sparse)
FP32:            60 TFLOPS
```

**ISA Features:** wgmma (FlashAttention-3 at 75%+ util), TMA (zero-overhead address compute), Thread Block Clusters, Transformer Engine (auto FP8/FP16 mixed precision)

**Inference:** FP8 E4M3 default for weights+activations. `--kv-cache-dtype fp8_e4m3`. `gpu_memory_utilization` safe at 0.95.

---

### H100 NVL

```
Architecture:    Hopper (sm_90a)
Die:             GH100 dual-GPU PCIe card
Compute:         132 SMs / 528 TC / 16,896 CUDA (per die)
Memory:          94 GB HBM3 @ 3.9 TB/s (per die)
NVLink:          600 GB/s bridge between the two dies
PCIe:            Gen5 x16
TDP:             400W (passive cooled)
```

**Note:** Dual-GPU PCIe card. 94 GB per die (vs 80 GB on SXM). NVLink bridge at 600 GB/s is lower than SXM's 900 GB/s. Designed for LLM inference in PCIe servers.

---

### H100 PCIe

```
Architecture:    Hopper (sm_90, NOT sm_90a)
Compute:         114 SMs / 456 TC / 14,592 CUDA
Memory:          80 GB HBM2e @ 2.0 TB/s
NVLink:          NONE
PCIe:            Gen5 x16
TDP:             350W
FP8 TC:          1,600 TFLOPS
```

**Warnings:** No async wgmma (sm_90, not sm_90a). HBM2e not HBM3 (40% less BW). No NVLink — TP across GPUs is PCIe-only. KV offloading is PCIe-bound.

---

### H200 SXM

```
Architecture:    Hopper (sm_90a)
Die:             GH100 (same compute as H100 SXM)
Compute:         132 SMs / 528 TC / 16,896 CUDA
Memory:          141 GB HBM3e @ 4.8 TB/s
L2 Cache:        50 MB
NVLink:          v4, 900 GB/s
TDP:             700W
```

**Key:** Same compute die as H100 — all gains are memory. 76% more HBM, 43% more bandwidth. 141 GB fits Llama-70B FP8 on TP=1 with 71 GB for KV cache. Best single-GPU value for 70B-class models.

---

### H200 NVL

```
Architecture:    Hopper (sm_90a)
Memory:          141 GB HBM3e @ 4.8 TB/s
NVLink:          v4, 900 GB/s
TDP:             700W
```

**Note:** H200 in air-cooled NVL form factor. Same 141 GB HBM3e as H200 SXM. Up to 1.7x LLM inference over H100 NVL.

---

### GH200 (Grace Hopper Superchip)

```
Architecture:    Hopper (sm_90a) + Grace CPU
Configuration:   1x H100 GPU + 1x Grace CPU
GPU Memory:      96 GB HBM3 @ 4.0 TB/s
Grace CPU:       72 Arm Neoverse V2 cores
Grace Memory:    480 GB LPDDR5X @ ~500 GB/s sustained
NVLink-C2C:      900 GB/s (Grace <-> GPU, coherent)
Total Memory:    96 + 480 = 576 GB coherent
TDP:             1000W
```

**Key:** NVLink-C2C is ~7x faster than PCIe Gen5 for KV offloading. Grace LPDDR5X as high-bandwidth KV overflow tier. 96 GB HBM3 (16 GB more than H100 SXM).

---

### H800 (EXPORT-RESTRICTED)

```
Architecture:    Hopper (sm_90a)
Die:             GH100 (export-restricted variant)
Compute:         132 SMs / 528 TC / 16,896 CUDA (IDENTICAL to H100 SXM)
Memory:          80 GB HBM3 @ 3.35 TB/s (IDENTICAL to H100 SXM)
NVLink:          v4 @ 400 GB/s (REDUCED from H100's 900 GB/s)
TDP:             700W
FP8 TC:          1,979 TFLOPS (IDENTICAL to H100 SXM)
```

**Export compliance:** China-market variant. Same silicon die as H100 SXM with NVLink bandwidth capped to 400 GB/s (55% reduction) for U.S. export control compliance. Single-GPU inference identical to H100 SXM. Multi-GPU training/inference degraded by NVLink cap.

---

### H20 (EXPORT-RESTRICTED)

```
Architecture:    Hopper (sm_90a)
Die:             GH100 (heavily cut down for export compliance)
Compute:         78 SMs / 312 TC / 9,984 CUDA (41% FEWER than H100)
Memory:          96 GB HBM3 @ 4.0 TB/s (MORE memory than H100 SXM)
NVLink:          v4 @ 900 GB/s (NOT capped, unlike H800)
TDP:             350W
FP8 TC:          296 TFLOPS (85% LESS than H100's 1,979)
```

**Export compliance:** China-market variant with heavily reduced compute but large memory. 96 GB + 4.0 TB/s makes it memory-bandwidth-rich but compute-poor. Optimized for memory-bound inference workloads, not compute-bound training. 350W makes it highly energy-efficient.

---

## Blackwell Detailed Specs

### B100

```
Architecture:    Blackwell (sm_100)
Die:             GB100, dual-die, TSMC 4NP
Compute:         148 SMs / 592 TC / 18,944 CUDA
Memory:          192 GB HBM3e @ 8.0 TB/s
NVLink:          v5, 1,800 GB/s
TDP:             700W (vs B200's 1000W)
FP4 TC:          7,000 TFLOPS (vs B200's 9,000)
FP8 TC:          3,500 TFLOPS (vs B200's 4,500)
```

**Status:** Same memory as B200 but 22% less compute at 30% less power. Most cloud providers skipped B100 and went straight to B200. Best for power-constrained Blackwell deployments.

---

### B200

```
Architecture:    Blackwell (sm_100)
Die:             GB100, dual-die, NV-HBI @ 10 TB/s, TSMC 4NP
Compute:         148 SMs / 592 Tensor Cores / 18,944 CUDA Cores
Memory:          192 GB HBM3e @ 8.0 TB/s
L2 Cache:        60 MB
NVLink:          v5, 1,800 GB/s bidirectional
TDP:             1000W
FP4 TC:          9,000 TFLOPS (18,000 sparse)
FP8 TC:          4,500 TFLOPS (9,000 sparse)
FP16 TC:         2,250 TFLOPS (4,500 sparse)
FP32:            75 TFLOPS
```

**ISA Features (SM100):** TMEM (256 KB/SM), tcgen05 instructions (SM100/SM103 only, NOT SM120 consumer), nvCOMP decompression engine (LZ4/Snappy/zstd/ANS), Helix parallelism, NVFP4 (scale granularity: 16)

**Inference:** NVFP4 delivers 2x throughput vs FP8 at <1% accuracy loss. `--quantization nvfp4`. FlashAttention-4 with RoPE+KV fusion (4.5x over FA2). 192 GB fits DeepSeek-R1 FP8 on TP=4.

**Systems:** HGX B200 (8x B200), DGX B200

---

### B300 (Blackwell Ultra)

```
Architecture:    Blackwell Ultra (sm_103)
Die:             GB100 Ultra, dual-die, TSMC 4NP
Compute:         160 SMs / 640 TC / 20,480 CUDA
Memory:          288 GB HBM3e @ 8.0 TB/s (12-Hi stacks)
L2 Cache:        60 MB
NVLink:          v5, 1,800 GB/s
PCIe:            Gen6 x16
TDP:             1400W
FP4 TC:          15,000 TFLOPS
FP8 TC:          7,000 TFLOPS
FP16 TC:         3,500 TFLOPS
FP64:            1.2 TFLOPS
```

**Key vs B200:** 50% more HBM (288 vs 192). 67% more FP4 TFLOPS. Accelerated softmax in hardware (SFU: 5 -> 10.7 TeraExp/s). Nearly all FP64 traded for inference gains. PCIe Gen6. Shipped January 2026.

**Systems:** HGX B300 (8x B300), DGX B300

---

### GB200 (Grace Blackwell Superchip)

```
Architecture:    Blackwell (sm_100) + Grace CPU
Configuration:   2x B200 GPU dies + 1x Grace CPU
GPU Memory:      192 GB HBM3e @ 8.0 TB/s (per die)
GPU NVLink:      v5, 1,800 GB/s (GPU-to-GPU)
Grace CPU:       72 Arm Neoverse V2 cores
Grace Memory:    Up to 480 GB LPDDR5X @ 546 GB/s peak
NVLink-C2C:      900 GB/s per die (Grace <-> GPU, coherent)
Total Memory:    2x192 + 480 = 864 GB per superchip
TDP:             ~2700W total
```

**Key:** NVLink-C2C ~7x faster than PCIe Gen5. Grace LPDDR5X as KV overflow tier without PCIe bottleneck. Same GPU compute as standalone B200.

**Systems:** GB200 NVL72 (36 Grace + 72 GPUs, liquid-cooled rack)

---

### GB300 (Grace Blackwell Ultra Superchip)

```
Architecture:    Blackwell Ultra (sm_103) + Grace CPU
Configuration:   2x B300 Ultra GPU dies + 1x Grace CPU
GPU Memory:      288 GB HBM3e @ 8.0 TB/s (per die)
GPU NVLink:      v5, 1,800 GB/s
Grace CPU:       72 Arm Neoverse V2 cores
Grace Memory:    Up to 480 GB LPDDR5X @ 546 GB/s peak
NVLink-C2C:      900 GB/s per die
Total Memory:    2x288 + 480 = 1,056 GB per superchip
TDP:             ~3800W total
FP4:             15,000 TFLOPS per die
```

**Key:** Largest memory in the lineup (1+ TB per superchip). Accelerated softmax. GB300 NVL72 delivers 1.1 ExaFLOPS dense FP4.

**Systems:** GB300 NVL72 (36 Grace + 72 GPUs, liquid-cooled rack), HGX GB300 NVL16

---

## Architecture Comparison

### Compute

| | H100 SXM | H200 SXM | B200 | B300 |
|---|----------|----------|------|------|
| SMs | 132 | 132 | 148 | 160 |
| Tensor Cores | 528 | 528 | 592 | 640 |
| CUDA Cores | 16,896 | 16,896 | 18,944 | 20,480 |
| FP4 TFLOPS | - | - | 9,000 | 15,000 |
| FP8 TFLOPS | 1,979 | 1,979 | 4,500 | 7,000 |
| FP16 TFLOPS | 990 | 990 | 2,250 | 3,500 |

### Memory

| | H100 SXM | H100 PCIe | H100 NVL | H200 | GH200 | B200 | B300 | GB200/die | GB300/die |
|---|----------|-----------|----------|------|--------|------|------|-----------|-----------|
| HBM | 80 GB | 80 GB | 94 GB | 141 GB | 96 GB | 192 GB | 288 GB | 192 GB | 288 GB |
| Type | HBM3 | HBM2e | HBM3 | HBM3e | HBM3 | HBM3e | HBM3e | HBM3e | HBM3e |
| BW | 3.35 TB/s | 2.0 TB/s | 3.9 TB/s | 4.8 TB/s | 4.0 TB/s | 8.0 TB/s | 8.0 TB/s | 8.0 TB/s | 8.0 TB/s |
| Grace | - | - | - | - | 480 GB | - | - | 480 GB | 480 GB |

### Interconnect

| | H100 SXM | H100 NVL | H100 PCIe | H200 | GH200 | B200 | B300 | GB200 | GB300 |
|---|----------|----------|-----------|------|--------|------|------|-------|-------|
| NVLink | v4 900 | 600 | None | v4 900 | v4 900 | v5 1800 | v5 1800 | v5 1800 | v5 1800 |
| NVLink-C2C | - | - | - | - | 900 | - | - | 900 | 900 |
| PCIe | Gen5 | Gen5 | Gen5 | Gen5 | Gen5 | Gen5 | Gen6 | Gen5 | Gen6 |

(All NVLink values in GB/s bidirectional)

### Precision Support

| | Hopper (all) | Blackwell (all) |
|---|---|---|
| FP32 | Native | Native |
| TF32 | Tensor Core | Tensor Core |
| FP16/BF16 | Tensor Core | Tensor Core |
| FP8 | Native (OCP E4M3) | Native (OCP E4M3) |
| FP4 | **No** | **NVFP4** (scale: 16 elements) |
| INT8 | Tensor Core | Tensor Core |
| Structured Sparsity | 2:4 | 2:4 |

### ISA Features

| Feature | Hopper | Blackwell | Blackwell Ultra |
|---------|--------|-----------|-----------------|
| Attention Kernel | FlashAttention-3 | FlashAttention-4 | FlashAttention-4 |
| wgmma | Yes (sm_90a only) | Yes | Yes |
| TMA | Yes | Yes | Yes |
| TMEM | No | 256 KB/SM | 256 KB/SM |
| tcgen05 | No | SM100/SM103 only | SM100/SM103 only |
| nvCOMP Decompression | No | Hardware accelerated | Hardware accelerated |
| Helix Parallelism | No | Yes | Yes |
| Accelerated Softmax | No | No | **10.7 TeraExp/s** |
| Transformer Engine | Yes | Yes | Yes |

---

## Grace CPU (Shared by GH200, GB200, GB300)

```
CPU:             72 Arm Neoverse V2 cores @ 3.1 GHz
Memory:          Up to 480 GB LPDDR5X
Memory BW:       546 GB/s peak (~500 GB/s sustained)
NVLink-C2C:      900 GB/s per GPU die (coherent)
Process:         TSMC 4nm
Power:           ~500W
```

**Why Grace matters for inference:** NVLink-C2C at 900 GB/s is ~7x faster than PCIe Gen5 (128 GB/s bidirectional). KV cache overflow to Grace LPDDR5X avoids the PCIe bottleneck that kills offloading performance on standard GPU servers.

---

## InferScope Registry Aliases

```
# Ampere
a100, a100_sxm         -> A100 SXM 80GB
a100_pcie              -> A100 PCIe 80GB
a100_40gb              -> A100 40GB
a10g                   -> A10G

# Hopper
h100, h100_sxm         -> H100 SXM
h100_nvl               -> H100 NVL
h100_pcie              -> H100 PCIe
h200, h200_sxm         -> H200 SXM
h200_nvl               -> H200 NVL
gh200, grace_hopper    -> GH200
h800                   -> H800 (export-restricted)
h20                    -> H20 (export-restricted)

# Blackwell
b100                   -> B100
b200                   -> B200
b300                   -> B300
gb200, grace_blackwell -> GB200
gb300                  -> GB300
```

---

## Sources

- [NVIDIA H100 GPU](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA H200 GPU](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA GH200 Grace Hopper Superchip](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)
- [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)
- [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [Inside NVIDIA Blackwell Ultra (NVIDIA Tech Blog)](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [nvCOMP Decompression Engine (NVIDIA Tech Blog)](https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/)
- [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/)
- [NVIDIA Grace CPU Architecture In Depth (NVIDIA Tech Blog)](https://developer.nvidia.com/blog/nvidia-grace-cpu-superchip-architecture-in-depth/)
- [H100 NVL Product Brief (NVIDIA PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf)
- [B300 vs B200 Comparison (Verda)](https://verda.com/blog/nvidia-b300-vs-b200-complete-gpu-comparison-to-date)
- [B200 Architecture Deep Dive (Chips and Cheese)](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut)
- [Blackwell Products Decoded (Modal)](https://modal.com/blog/nvidia-blackwell)
- [GB300 NVL72 Architecture (Verda)](https://verda.com/blog/gb300-nvl72-architecture)
