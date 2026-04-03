"""GPU hardware profiles — ISA-level specs for every supported GPU.

This is the knowledge that makes InferScope different from generic advice.
Every recommendation cites the specific GPU architecture spec it's based on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GPUProfile:
    """Complete hardware profile for a GPU variant."""

    # Identity
    name: str
    vendor: str  # nvidia | amd
    architecture: str  # Ampere | Hopper | Blackwell | CDNA3 | CDNA4
    compute_capability: str  # sm_80, sm_90a, sm_100, gfx942, gfx950
    die: str = ""
    process: str = ""

    # Compute
    sms: int = 0  # NVIDIA SMs or AMD CUs
    tensor_cores: int = 0
    cuda_cores: int = 0
    peak_clock_mhz: int = 0
    warp_size: int = 32  # 32 for NVIDIA, 64 for AMD

    # Memory
    memory_type: str = ""  # HBM2e, HBM3, HBM3e, GDDR6
    memory_gb: float = 0.0
    memory_bandwidth_tb_s: float = 0.0
    l2_cache_mb: float = 0.0

    # Interconnect
    nvlink_version: int = 0
    nvlink_bandwidth_gb_s: float = 0.0
    infinity_fabric_version: int = 0
    if_bandwidth_gb_s: float = 0.0
    pcie: str = ""

    # Power
    tdp_watts: int = 0

    # Precision support
    fp8_support: bool = False
    fp8_format: str = ""  # OCP | FNUZ
    fp4_support: bool = False
    fp4_format: str = ""  # NVFP4 | MXFP4
    structured_sparsity: bool = False

    # Peak TFLOPS
    peak_tflops: dict[str, float] = field(default_factory=dict)

    # Inference-specific notes
    inference_notes: list[str] = field(default_factory=list)

    # Extra details
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "architecture": self.architecture,
            "compute_capability": self.compute_capability,
            "die": self.die,
            "process": self.process,
            "compute": {
                "sms_or_cus": self.sms,
                "tensor_cores": self.tensor_cores,
                "cuda_cores": self.cuda_cores,
                "peak_clock_mhz": self.peak_clock_mhz,
                "warp_size": self.warp_size,
            },
            "memory": {
                "type": self.memory_type,
                "capacity_gb": self.memory_gb,
                "bandwidth_tb_s": self.memory_bandwidth_tb_s,
                "l2_cache_mb": self.l2_cache_mb,
            },
            "interconnect": {
                "nvlink_version": self.nvlink_version,
                "nvlink_bandwidth_gb_s": self.nvlink_bandwidth_gb_s,
                "infinity_fabric_version": self.infinity_fabric_version,
                "if_bandwidth_gb_s": self.if_bandwidth_gb_s,
                "pcie": self.pcie,
            },
            "power": {"tdp_watts": self.tdp_watts},
            "precision": {
                "fp8_support": self.fp8_support,
                "fp8_format": self.fp8_format,
                "fp4_support": self.fp4_support,
                "fp4_format": self.fp4_format,
                "structured_sparsity": self.structured_sparsity,
            },
            "platform_features": {
                "grace_coherent_memory": bool(self.extra.get("grace_cpu_cores", 0)),
                "grace_memory_gb": self.extra.get("grace_memory_gb", 0),
                "grace_memory_bandwidth_gb_s": self.extra.get("grace_memory_bandwidth_gb_s", 0),
                "c2c_bandwidth_gb_s": self.extra.get("nvlink_c2c_bandwidth_gb_s", 0),
                "decompression_engine": bool(self.extra.get("decompression_engine")),
                "helix_parallelism": bool(self.extra.get("helix_parallelism")),
                "accelerated_softmax": bool(self.extra.get("accelerated_softmax")),
            },
            "peak_tflops": self.peak_tflops,
            "inference_notes": self.inference_notes,
        }


# =============================================================================
# NVIDIA Ampere (SM80/SM86)
# =============================================================================

A100_SXM_80GB = GPUProfile(
    name="A100 SXM 80GB",
    vendor="nvidia",
    architecture="Ampere",
    compute_capability="sm_80",
    die="GA100",
    sms=108,
    tensor_cores=432,
    cuda_cores=6912,
    warp_size=32,
    memory_type="HBM2e",
    memory_gb=80,
    memory_bandwidth_tb_s=2.039,
    l2_cache_mb=40,
    nvlink_version=3,
    nvlink_bandwidth_gb_s=600,
    pcie="Gen4 x16",
    tdp_watts=400,
    fp8_support=False,
    fp4_support=False,
    structured_sparsity=True,
    peak_tflops={
        "fp32": 19.5,
        "tf32_tc": 156,
        "tf32_tc_sparse": 312,
        "fp16_tc": 312,
        "fp16_tc_sparse": 624,
        "int8_tc": 624,
        "int8_tc_sparse": 1248,
    },
    inference_notes=[
        "No FP8 — use FP16/BF16 or INT8 quantization (AWQ/GPTQ via Marlin)",
        "FP8 models run via W8A16 Marlin (weight-only dequant), NOT native FP8",
        "gpu_memory_utilization sweet spot: 0.92-0.95",
        "MIG useful for serving multiple small models (<13B)",
        "40 MB L2 supports persistence controls for weight pinning",
        "cp.async enables 2-3 stage software pipelining for attention",
    ],
)

A100_PCIE_80GB = GPUProfile(
    name="A100 PCIe 80GB",
    vendor="nvidia",
    architecture="Ampere",
    compute_capability="sm_80",
    sms=108,
    tensor_cores=432,
    cuda_cores=6912,
    warp_size=32,
    memory_type="HBM2e",
    memory_gb=80,
    memory_bandwidth_tb_s=1.935,
    l2_cache_mb=40,
    nvlink_bandwidth_gb_s=600,
    pcie="Gen4 x16",
    tdp_watts=300,
    fp8_support=False,
    fp4_support=False,
    structured_sparsity=True,
    peak_tflops={"fp32": 19.5, "tf32_tc": 156, "fp16_tc": 312, "int8_tc": 624},
    inference_notes=["Same compute as SXM, slightly lower memory bandwidth"],
)

A100_40GB = GPUProfile(
    name="A100 40GB",
    vendor="nvidia",
    architecture="Ampere",
    compute_capability="sm_80",
    sms=108,
    tensor_cores=432,
    cuda_cores=6912,
    warp_size=32,
    memory_type="HBM2e",
    memory_gb=40,
    memory_bandwidth_tb_s=1.555,
    l2_cache_mb=40,
    fp8_support=False,
    fp4_support=False,
    peak_tflops={"fp32": 19.5, "tf32_tc": 156, "fp16_tc": 312, "int8_tc": 624},
    inference_notes=["40 GB limits to ~20B FP16 or ~40B INT4 models"],
)

A10G = GPUProfile(
    name="A10G",
    vendor="nvidia",
    architecture="Ampere",
    compute_capability="sm_86",
    die="GA102",
    sms=72,
    tensor_cores=288,
    cuda_cores=9216,
    warp_size=32,
    memory_type="GDDR6",
    memory_gb=24,
    memory_bandwidth_tb_s=0.600,
    l2_cache_mb=6,
    nvlink_bandwidth_gb_s=0,
    pcie="Gen4 x16",
    tdp_watts=150,
    fp8_support=False,
    fp4_support=False,
    peak_tflops={"fp32": 31.2, "tf32_tc": 62.5, "fp16_tc": 125, "int8_tc": 250},
    inference_notes=[
        "Budget inference GPU — good for models ≤13B",
        "No NVLink — multi-GPU requires PCIe (64 GB/s), TP is expensive",
        "24 GB GDDR6 limits to ~7B FP16 or ~13B INT4 models",
        "SM86 has 25% fewer warps than SM80 — different occupancy math",
        "6 MB L2 cache is 6.7x smaller than A100 — less weight caching",
    ],
)

# =============================================================================
# NVIDIA Hopper (SM90)
# =============================================================================

H100_SXM = GPUProfile(
    name="H100 SXM",
    vendor="nvidia",
    architecture="Hopper",
    compute_capability="sm_90a",
    die="GH100",
    process="TSMC 4N",
    sms=132,
    tensor_cores=528,
    cuda_cores=16896,
    warp_size=32,
    memory_type="HBM3",
    memory_gb=80,
    memory_bandwidth_tb_s=3.35,
    l2_cache_mb=50,
    nvlink_version=4,
    nvlink_bandwidth_gb_s=900,
    pcie="Gen5 x16",
    tdp_watts=700,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=False,
    structured_sparsity=True,
    peak_tflops={
        "fp32": 60,
        "tf32_tc": 495,
        "tf32_tc_sparse": 990,
        "fp16_tc": 990,
        "fp16_tc_sparse": 1979,
        "fp8_tc": 1979,
        "fp8_tc_sparse": 3958,
        "int8_tc": 1979,
    },
    inference_notes=[
        "FP8 E4M3 is the default for weights AND activations",
        "KV cache: use fp8_e4m3 (--kv-cache-dtype fp8_e4m3 in vLLM)",
        "wgmma enables FlashAttention-3 at 75%+ utilization (vs FA2's 35%)",
        "TMA eliminates per-thread address computation — critical for FA3",
        "gpu_memory_utilization: push to 0.95 in production",
        "50 MB L2 is 25% larger than A100 — better weight caching",
    ],
    extra={
        "wgmma_shapes": {"fp8_e4m3": {"M": 64, "K": 32, "N_range": "8-256"}},
        "tma": True,
        "thread_block_clusters": True,
        "transformer_engine": True,
    },
)

H100_NVL = GPUProfile(
    name="H100 NVL",
    vendor="nvidia",
    architecture="Hopper",
    compute_capability="sm_90a",
    die="GH100 (dual-GPU PCIe card)",
    sms=132,
    tensor_cores=528,
    cuda_cores=16896,
    warp_size=32,
    memory_type="HBM3",
    memory_gb=94,
    memory_bandwidth_tb_s=3.9,
    l2_cache_mb=50,
    nvlink_bandwidth_gb_s=600,
    pcie="Gen5 x16",
    tdp_watts=400,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=False,
    structured_sparsity=True,
    peak_tflops={"fp8_tc": 1979, "fp16_tc": 990, "fp32": 60},
    inference_notes=[
        "Dual-GPU PCIe card — 94 GB HBM3 per die (vs 80 GB on SXM)",
        "NVLink bridge between the two dies at 600 GB/s (lower than SXM's 900 GB/s)",
        "PCIe form factor, passive cooled — requires system airflow",
        "Designed for LLM inference in PCIe servers (not SXM baseboard)",
    ],
)

H100_PCIE = GPUProfile(
    name="H100 PCIe",
    vendor="nvidia",
    architecture="Hopper",
    compute_capability="sm_90",
    sms=114,
    tensor_cores=456,
    cuda_cores=14592,
    warp_size=32,
    memory_type="HBM2e",
    memory_gb=80,
    memory_bandwidth_tb_s=2.0,
    l2_cache_mb=50,
    nvlink_bandwidth_gb_s=0,
    pcie="Gen5 x16",
    tdp_watts=350,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=False,
    peak_tflops={"fp32": 48, "fp16_tc": 800, "fp8_tc": 1600},
    inference_notes=[
        "sm_90 (not sm_90a) — no async wgmma features",
        "NO NVLink — PCIe only, TP across GPUs is expensive",
        "HBM2e not HBM3 — lower bandwidth than SXM variant",
    ],
)

H200_SXM = GPUProfile(
    name="H200 SXM",
    vendor="nvidia",
    architecture="Hopper",
    compute_capability="sm_90a",
    die="GH100",
    sms=132,
    tensor_cores=528,
    cuda_cores=16896,
    warp_size=32,
    memory_type="HBM3e",
    memory_gb=141,
    memory_bandwidth_tb_s=4.8,
    l2_cache_mb=50,
    nvlink_version=4,
    nvlink_bandwidth_gb_s=900,
    pcie="Gen5 x16",
    tdp_watts=700,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=False,
    structured_sparsity=True,
    peak_tflops={
        "fp32": 60,
        "tf32_tc": 495,
        "fp16_tc": 990,
        "fp8_tc": 1979,
        "int8_tc": 1979,
    },
    inference_notes=[
        "Same compute as H100 SXM — gains are ALL memory",
        "141 GB fits Llama-70B FP8 on TP=1 with 71 GB for KV cache",
        "4.8 TB/s enables ~2x decode throughput vs H100 at large batch",
        "Best single-GPU value for 70B-class models in FP8",
    ],
)

H200_NVL = GPUProfile(
    name="H200 NVL",
    vendor="nvidia",
    architecture="Hopper",
    compute_capability="sm_90a",
    die="GH100",
    sms=132,
    tensor_cores=528,
    cuda_cores=16896,
    warp_size=32,
    memory_type="HBM3e",
    memory_gb=141,
    memory_bandwidth_tb_s=4.8,
    l2_cache_mb=50,
    nvlink_version=4,
    nvlink_bandwidth_gb_s=900,
    pcie="Gen5 x16",
    tdp_watts=700,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=False,
    structured_sparsity=True,
    peak_tflops={"fp32": 60, "fp16_tc": 990, "fp8_tc": 1979},
    inference_notes=[
        "H200 in NVL (air-cooled) form factor — same 141 GB HBM3e as H200 SXM",
        "Lower-power air-cooled enterprise rack designs",
        "Up to 1.7x LLM inference acceleration over H100 NVL",
    ],
)

H800 = GPUProfile(
    name="H800",
    vendor="nvidia",
    architecture="Hopper",
    compute_capability="sm_90a",
    die="GH100 (export-restricted variant)",
    sms=132,
    tensor_cores=528,
    cuda_cores=16896,
    warp_size=32,
    memory_type="HBM3",
    memory_gb=80,
    memory_bandwidth_tb_s=3.35,
    l2_cache_mb=50,
    nvlink_version=4,
    nvlink_bandwidth_gb_s=400,
    pcie="Gen5 x16",
    tdp_watts=700,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=False,
    structured_sparsity=True,
    peak_tflops={"fp8_tc": 1979, "fp16_tc": 990, "fp32": 60},
    inference_notes=[
        "EXPORT-RESTRICTED: China-market variant of H100 SXM",
        "Same compute as H100 SXM — identical SMs, TC, CUDA cores, FP8 TFLOPS",
        "NVLink reduced to 400 GB/s (vs H100 SXM's 900 GB/s) for export compliance",
        "Single-GPU inference performance identical to H100 SXM",
        "Multi-GPU scaling degraded ~55% due to NVLink bandwidth cap",
    ],
)

H20 = GPUProfile(
    name="H20",
    vendor="nvidia",
    architecture="Hopper",
    compute_capability="sm_90a",
    die="GH100 (export-restricted, heavily cut down)",
    sms=78,
    tensor_cores=312,
    cuda_cores=9984,
    warp_size=32,
    memory_type="HBM3",
    memory_gb=96,
    memory_bandwidth_tb_s=4.0,
    l2_cache_mb=50,
    nvlink_version=4,
    nvlink_bandwidth_gb_s=900,
    pcie="Gen5 x16",
    tdp_watts=350,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=False,
    peak_tflops={"fp8_tc": 296, "fp16_tc": 148, "tf32_tc": 74, "fp32": 44, "fp64": 1},
    inference_notes=[
        "EXPORT-RESTRICTED: China-market variant with heavily reduced compute",
        "78 SMs (vs H100's 132) — 41% fewer cores",
        "96 GB HBM3 @ 4.0 TB/s — MORE memory than H100 SXM (80 GB)",
        "NVLink 900 GB/s retained (not capped like H800)",
        "FP8 only 296 TFLOPS (vs H100's 1,979) — 85% less compute",
        "Optimized for memory-bound inference, not compute-bound training",
        "350W TDP — highly energy-efficient",
    ],
)

GH200 = GPUProfile(
    name="GH200",
    vendor="nvidia",
    architecture="Hopper",
    compute_capability="sm_90a",
    die="GH200 Superchip: 1× GH100 GPU + 1× Grace CPU (per-GPU profile)",
    process="TSMC 4N (GPU) + 4nm (Grace)",
    sms=132,
    tensor_cores=528,
    cuda_cores=16896,
    warp_size=32,
    memory_type="HBM3",
    memory_gb=96,
    memory_bandwidth_tb_s=4.0,
    l2_cache_mb=50,
    nvlink_version=4,
    nvlink_bandwidth_gb_s=900,
    pcie="Gen5 x16",
    tdp_watts=1000,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=False,
    structured_sparsity=True,
    peak_tflops={"fp32": 60, "fp16_tc": 990, "fp8_tc": 1979},
    inference_notes=[
        "GH200 Grace Hopper Superchip = 1× H100 GPU + 1× Grace CPU",
        "NVLink-C2C @ 900 GB/s between Grace and Hopper — coherent memory",
        "Grace CPU: 72 Arm Neoverse V2 cores, 480 GB LPDDR5X @ ~500 GB/s sustained",
        "Total addressable: 96 GB HBM3 + 480 GB LPDDR5X = 576 GB coherent",
        "KV offload to Grace LPDDR5X via NVLink-C2C ~7x faster than PCIe Gen5",
        "96 GB HBM3 (vs H100 SXM's 80 GB) — extra 16 GB for KV headroom",
    ],
    extra={
        "grace_cpu_cores": 72,
        "grace_memory_gb": 480,
        "grace_memory_bandwidth_gb_s": 500,
        "nvlink_c2c_bandwidth_gb_s": 900,
    },
)

# =============================================================================
# NVIDIA Blackwell (SM100/SM103)
# =============================================================================

B100 = GPUProfile(
    name="B100",
    vendor="nvidia",
    architecture="Blackwell",
    compute_capability="sm_100",
    die="GB100 (dual-die, NV-HBI @ 10 TB/s)",
    process="TSMC 4NP",
    sms=148,
    tensor_cores=592,
    cuda_cores=18944,
    warp_size=32,
    memory_type="HBM3e",
    memory_gb=192,
    memory_bandwidth_tb_s=8.0,
    l2_cache_mb=50,
    nvlink_version=5,
    nvlink_bandwidth_gb_s=1800,
    pcie="Gen5",
    tdp_watts=700,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=True,
    fp4_format="NVFP4",
    structured_sparsity=True,
    peak_tflops={
        "fp4_tc": 7000,
        "fp4_tc_sparse": 14000,
        "fp8_tc": 3500,
        "fp8_tc_sparse": 7000,
        "fp16_tc": 1750,
        "fp16_tc_sparse": 3500,
        "fp32": 60,
    },
    inference_notes=[
        "Lower-power Blackwell (700W vs B200's 1000W) — same memory, less compute",
        "Same 192 GB HBM3e @ 8.0 TB/s as B200",
        "FP4: 7 PFLOPS dense (vs B200's 9 PFLOPS) — 22% less compute",
        "Largely skipped by cloud providers in favor of B200",
        "Best for power-constrained deployments needing Blackwell memory capacity",
    ],
    extra={
        "tensor_memory_per_sm_kb": 256,
        "decompression_engine": True,
        "helix_parallelism": True,
        "nvfp4_scale_granularity": 16,
    },
)

B200 = GPUProfile(
    name="B200",
    vendor="nvidia",
    architecture="Blackwell",
    compute_capability="sm_100",
    die="GB100 (dual-die, NV-HBI @ 10 TB/s)",
    process="TSMC 4NP",
    sms=148,
    tensor_cores=592,
    cuda_cores=18944,
    warp_size=32,
    memory_type="HBM3e",
    memory_gb=192,
    memory_bandwidth_tb_s=8.0,
    l2_cache_mb=60,
    nvlink_version=5,
    nvlink_bandwidth_gb_s=1800,
    pcie="Gen5",
    tdp_watts=1000,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=True,
    fp4_format="NVFP4",
    structured_sparsity=True,
    peak_tflops={
        "fp4_tc": 9000,
        "fp4_tc_sparse": 18000,
        "fp8_tc": 4500,
        "fp8_tc_sparse": 9000,
        "fp16_tc": 2250,
        "fp16_tc_sparse": 4500,
        "tf32_tc": 1100,
        "fp32": 75,
        "fp64": 37,
    },
    inference_notes=[
        "NVFP4 delivers 2x throughput vs FP8 at <1% accuracy loss",
        "Use --quantization nvfp4 in vLLM/TRT-LLM for Blackwell",
        "nvCOMP decompression engine accelerates data I/O (LZ4, Snappy, zstd, ANS)",
        "SM120 is CONSUMER Blackwell (RTX 5090) — lacks TMEM, different ISA",
        "tcgen05 instructions are SM100/SM103 ONLY — not SM120",
        "192 GB fits DeepSeek-R1 FP8 on TP=4 with expert parallelism",
    ],
    extra={
        "tensor_memory_per_sm_kb": 256,
        "decompression_engine": True,
        "helix_parallelism": True,
        "nvfp4_scale_granularity": 16,
    },
)

B300 = GPUProfile(
    name="B300",
    vendor="nvidia",
    architecture="Blackwell",
    compute_capability="sm_103",
    die="GB100 Ultra (dual-die)",
    sms=160,
    tensor_cores=640,
    cuda_cores=20480,
    warp_size=32,
    memory_type="HBM3e",
    memory_gb=288,
    memory_bandwidth_tb_s=8.0,
    l2_cache_mb=60,
    nvlink_version=5,
    nvlink_bandwidth_gb_s=1800,
    pcie="Gen6 x16",
    tdp_watts=1400,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=True,
    fp4_format="NVFP4",
    structured_sparsity=True,
    peak_tflops={
        "fp4_tc": 15000,
        "fp8_tc": 7000,
        "fp16_tc": 3500,
        "fp64": 1.2,
    },
    inference_notes=[
        "Blackwell Ultra — accelerated softmax in hardware",
        "Doubled SFU throughput",
        "Nearly all FP64 traded for FP4 gains — inference-optimized",
        "288 GB fits DeepSeek-R1 FP4 on TP=2 with massive KV headroom",
        "NVLink5 @ 1800 GB/s — same interconnect as B200",
    ],
    extra={
        "tensor_memory_per_sm_kb": 256,
        "decompression_engine": True,
        "helix_parallelism": True,
        "nvfp4_scale_granularity": 16,
        "accelerated_softmax": True,
    },
)

GB200 = GPUProfile(
    name="GB200",
    vendor="nvidia",
    architecture="Blackwell",
    compute_capability="sm_100",
    die="GB200 Superchip: 2× GB100 GPU + 1× Grace CPU (per-GPU-die profile)",
    process="TSMC 4NP (GPU) + 4nm (Grace)",
    sms=148,
    tensor_cores=592,
    cuda_cores=18944,
    warp_size=32,
    memory_type="HBM3e",
    memory_gb=192,
    memory_bandwidth_tb_s=8.0,
    l2_cache_mb=60,
    nvlink_version=5,
    nvlink_bandwidth_gb_s=1800,
    pcie="Gen5",
    tdp_watts=1000,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=True,
    fp4_format="NVFP4",
    structured_sparsity=True,
    peak_tflops={
        "fp4_tc": 9000,
        "fp4_tc_sparse": 18000,
        "fp8_tc": 4500,
        "fp8_tc_sparse": 9000,
        "fp16_tc": 2250,
        "fp16_tc_sparse": 4500,
        "tf32_tc": 1100,
        "fp32": 75,
        "fp64": 37,
    },
    inference_notes=[
        "GB200 Superchip = 2× B200 GPUs + 1× Grace CPU (this profile = per GPU die)",
        "NVLink-C2C @ 900 GB/s between Grace and each Blackwell die — coherent memory",
        "Grace CPU: 72 Arm Neoverse V2 cores, up to 480GB LPDDR5X @ 546 GB/s peak",
        "KV offload to Grace LPDDR5X via NVLink-C2C is ~7x faster than PCIe Gen5",
        "Per die: 192 GB GPU HBM3e; Grace LPDDR5X shared across both GPU dies",
        "Same GPU compute as standalone B200 — gains are memory hierarchy + coherence",
        "Ideal for long-context inference: KV spills to Grace without PCIe bottleneck",
    ],
    extra={
        "tensor_memory_per_sm_kb": 256,
        "decompression_engine": True,
        "helix_parallelism": True,
        "nvfp4_scale_granularity": 16,
        "grace_cpu_cores": 72,
        "grace_memory_gb": 480,
        "grace_memory_bandwidth_gb_s": 546,
        "nvlink_c2c_bandwidth_gb_s": 900,
    },
)

GB300 = GPUProfile(
    name="GB300",
    vendor="nvidia",
    architecture="Blackwell",
    compute_capability="sm_103",
    die="GB300 Superchip: 2× GB100 Ultra GPU + 1× Grace CPU (per-GPU-die profile)",
    process="TSMC 4NP (GPU) + 4nm (Grace)",
    sms=160,
    tensor_cores=640,
    cuda_cores=20480,
    warp_size=32,
    memory_type="HBM3e",
    memory_gb=288,
    memory_bandwidth_tb_s=8.0,
    l2_cache_mb=60,
    nvlink_version=5,
    nvlink_bandwidth_gb_s=1800,
    pcie="Gen6 x16",
    tdp_watts=1400,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=True,
    fp4_format="NVFP4",
    structured_sparsity=True,
    peak_tflops={
        "fp4_tc": 15000,
        "fp8_tc": 7000,
        "fp16_tc": 3500,
        "fp64": 1.2,
    },
    inference_notes=[
        "GB300 Superchip = 2× B300 Ultra GPUs + 1× Grace CPU (this profile = per GPU die)",
        "NVLink-C2C @ 900 GB/s between Grace and each Blackwell Ultra die",
        "Grace CPU: 72 Arm Neoverse V2 cores, up to 480 GB LPDDR5X @ 546 GB/s",
        "288 GB HBM3e per die + Grace LPDDR5X shared — massive KV capacity",
        "Accelerated softmax (10.7 TeraExp/s) + doubled SFU throughput",
        "GB300 NVL72: 72 GPUs + 36 Grace CPUs, 1.1 ExaFLOPS FP4",
    ],
    extra={
        "tensor_memory_per_sm_kb": 256,
        "decompression_engine": True,
        "helix_parallelism": True,
        "nvfp4_scale_granularity": 16,
        "accelerated_softmax": True,
        "grace_cpu_cores": 72,
        "grace_memory_gb": 480,
        "grace_memory_bandwidth_gb_s": 546,
        "nvlink_c2c_bandwidth_gb_s": 900,
    },
)

# =============================================================================
# AMD CDNA3 (gfx942)
# =============================================================================

MI300X = GPUProfile(
    name="MI300X",
    vendor="amd",
    architecture="CDNA3",
    compute_capability="gfx942",
    die="8 XCDs on 3D chiplets",
    process="TSMC 5nm (XCD) + 6nm (IOD)",
    sms=304,  # CUs
    tensor_cores=0,  # AMD uses MFMA instead
    cuda_cores=0,
    peak_clock_mhz=2100,
    warp_size=64,
    memory_type="HBM3",
    memory_gb=192,
    memory_bandwidth_tb_s=5.3,
    l2_cache_mb=32,
    infinity_fabric_version=3,
    if_bandwidth_gb_s=896,
    tdp_watts=750,
    fp8_support=True,
    fp8_format="FNUZ",
    fp4_support=False,
    structured_sparsity=False,
    peak_tflops={
        "fp16_tc": 1307,
        "fp8_tc": 2615,
        "int8_tc": 2615,
        "fp32": 163.4,
        "fp64_tc": 163.4,
    },
    inference_notes=[
        "MUST set VLLM_ROCM_USE_AITER=1 for competitive performance",
        "MUST set HIP_FORCE_DEV_KERNARG=1",
        "MUST set TORCH_BLAS_PREFER_HIPBLASLT=1",
        "VLLM_ROCM_USE_AITER_FP8BMM=0 — CRASHES on gfx942, only works on gfx950",
        "NCCL_MIN_NCHANNELS=112 for multi-GPU",
        "192 GB HBM3 fits Llama-70B FP16 on TP=1 (140 GB weights + 52 GB KV)",
        "FP8 models from NVIDIA need automatic FNUZ conversion (vLLM handles)",
        "FP8 is FNUZ (NOT OCP) — bit-level incompatible with NVIDIA",
        "No Marlin kernels — AWQ/GPTQ use Triton backends (slower)",
        "DeepSeek models: --block-size 1 required for MLA compatibility",
    ],
    extra={
        "xcds": 8,
        "cus_per_xcd": 38,
        "lds_per_cu_kb": 64,
        "lds_banks": 32,
        "llc_mb": 256,
        "mfma_fp8_fnuz": True,
        "mxfp4_support": False,
    },
)

MI325X = GPUProfile(
    name="MI325X",
    vendor="amd",
    architecture="CDNA3",
    compute_capability="gfx942",
    sms=304,
    warp_size=64,
    memory_type="HBM3e",
    memory_gb=256,
    memory_bandwidth_tb_s=6.0,
    tdp_watts=1000,
    fp8_support=True,
    fp8_format="FNUZ",
    fp4_support=False,
    peak_tflops={"fp16_tc": 1307, "fp8_tc": 2615, "int8_tc": 2615},
    inference_notes=[
        "Same compute as MI300X — gains are memory only",
        "256 GB fits larger models or more KV cache on TP=1",
    ],
)

# =============================================================================
# AMD CDNA4 (gfx950)
# =============================================================================

MI355X = GPUProfile(
    name="MI355X",
    vendor="amd",
    architecture="CDNA4",
    compute_capability="gfx950",
    process="TSMC N3P",
    sms=256,  # CUs
    peak_clock_mhz=2400,
    warp_size=64,
    memory_type="HBM3e",
    memory_gb=288,
    memory_bandwidth_tb_s=8.0,
    l2_cache_mb=32,
    infinity_fabric_version=4,
    if_bandwidth_gb_s=1075,
    tdp_watts=1400,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=True,
    fp4_format="MXFP4",
    structured_sparsity=True,
    peak_tflops={
        "fp16_tc": 2500,
        "fp16_tc_sparse": 5000,
        "fp8_tc": 5000,
        "fp8_tc_sparse": 10100,
        "mxfp6_tc": 10066,
        "mxfp4_tc": 10066,
        "fp32": 156,
        "fp64_tc": 78.6,
    },
    inference_notes=[
        "OCP FP8 — no conversion from NVIDIA models needed (unlike MI300X)",
        "MXFP4 native via AITER: dequant+matmul fused in single MFMA instruction",
        "FP4 and FP6 same FLOPS — MXFP6 better accuracy at same compute cost",
        "Use SGLANG_USE_AITER=1 for SGLang",
        "288 GB fits Kimi K2.5 MXFP4 (~551 GB) on TP=4 (4×288=1152 GB)",
        "8 TB/s HBM3e matches Blackwell B200 bandwidth",
        "L2 writeback-retain helps stage-boundary weight reuse in MoE",
    ],
    extra={
        "xcds": 8,
        "cus_per_xcd": 32,
        "lds_per_cu_kb": 160,
        "lds_banks": 64,
        "llc_mb": 256,
        "ds_read_tr": True,
        "mxfp4_scale_granularity": 32,
        "mxfp6_support": True,
    },
)


MI350X = GPUProfile(
    name="MI350X",
    vendor="amd",
    architecture="CDNA4",
    compute_capability="gfx950",
    process="TSMC N3P",
    die="Antares",
    sms=256,  # CUs
    peak_clock_mhz=2400,
    warp_size=64,
    memory_type="HBM3e",
    memory_gb=288,
    memory_bandwidth_tb_s=8.0,
    l2_cache_mb=32,
    infinity_fabric_version=4,
    if_bandwidth_gb_s=1075,
    pcie="gen5",
    tdp_watts=750,
    fp8_support=True,
    fp8_format="OCP",
    fp4_support=True,
    fp4_format="MXFP4",
    structured_sparsity=True,
    peak_tflops={
        "fp16_tc": 2500,
        "fp16_tc_sparse": 5000,
        "fp8_tc": 5000,
        "fp8_tc_sparse": 10100,
        "mxfp6_tc": 10066,
        "mxfp4_tc": 10066,
        "fp32": 156,
        "fp64_tc": 78.6,
    },
    inference_notes=[
        "OCP FP8 — no conversion from NVIDIA models needed (unlike MI300X FNUZ)",
        "MXFP4 native: 35x inference improvement over MI300X (largely from native FP4)",
        "288 GB HBM3e at 8 TB/s matches Blackwell B200 bandwidth",
        "Available on Azure (ND-MI350X-v1), OCI, CoreWeave, Lambda",
        "Priced 25-30% below Blackwell B200 ($5.50-$7.00/GPU-hr on-demand)",
        "Same CDNA4 ISA as MI355X — lower TDP variant (750W vs 1400W)",
    ],
    extra={
        "xcds": 8,
        "cus_per_xcd": 32,
        "lds_per_cu_kb": 160,
        "lds_banks": 64,
        "llc_mb": 256,
        "ds_read_tr": True,
        "mxfp4_scale_granularity": 32,
        "mxfp6_support": True,
    },
)


# =============================================================================
# GPU Registry
# =============================================================================

GPU_REGISTRY: dict[str, GPUProfile] = {
    # NVIDIA Ampere
    "a100_sxm_80gb": A100_SXM_80GB,
    "a100_sxm": A100_SXM_80GB,
    "a100_80gb": A100_SXM_80GB,
    "a100": A100_SXM_80GB,
    "a100_pcie_80gb": A100_PCIE_80GB,
    "a100_pcie": A100_PCIE_80GB,
    "a100_40gb": A100_40GB,
    "a10g": A10G,
    # NVIDIA Hopper
    "h100_sxm": H100_SXM,
    "h100_sxm_80gb": H100_SXM,
    "h100": H100_SXM,
    "h100_nvl": H100_NVL,
    "h100_nvl_94gb": H100_NVL,
    "h100_pcie": H100_PCIE,
    "h200_sxm": H200_SXM,
    "h200_sxm_141gb": H200_SXM,
    "h200": H200_SXM,
    "h200_nvl": H200_NVL,
    "gh200": GH200,
    "gh200_superchip": GH200,
    "grace_hopper": GH200,
    # NVIDIA Hopper (export-restricted)
    "h800": H800,
    "h20": H20,
    # NVIDIA Blackwell
    "b100": B100,
    "b200": B200,
    "b300": B300,
    "b300_ultra": B300,
    "gb200": GB200,
    "gb200_superchip": GB200,
    "grace_blackwell": GB200,
    "gb300": GB300,
    "gb300_superchip": GB300,
    # AMD CDNA3
    "mi300x": MI300X,
    "mi325x": MI325X,
    # AMD CDNA4
    "mi350x": MI350X,
    "mi355x": MI355X,
}


def get_gpu_profile(gpu_name: str) -> GPUProfile | None:
    """Look up a GPU profile by name (case-insensitive, flexible matching)."""
    key = gpu_name.lower().replace(" ", "_").replace("-", "_")
    return GPU_REGISTRY.get(key)


def list_gpus() -> list[str]:
    """List all known GPU names."""
    # Deduplicate by returning only canonical names
    seen = set()
    result = []
    for name, profile in GPU_REGISTRY.items():
        if profile.name not in seen:
            seen.add(profile.name)
            result.append(name)
    return result
