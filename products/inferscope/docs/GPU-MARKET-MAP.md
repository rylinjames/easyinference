# NVIDIA Data Center GPU Market Map

GPU inventory across Ampere, Hopper, and Blackwell generations. Specs verified against NVIDIA datasheets. Market data sourced from public filings, SemiAnalysis, Epoch AI, and cloud provider pricing pages (March 2026).

---

## Every GPU by Generation

### Ampere (2020-2023) — SM80/SM86

| GPU | CC | Memory | BW | NVLink | FP8 | FP4 | TDP | Notes |
|-----|-----|--------|-----|--------|-----|-----|-----|-------|
| **A100 SXM 80GB** | sm_80 | 80 GB HBM2e | 2.039 TB/s | v3 600 GB/s | No | No | 400W | Workhorse of 2022-2024 AI |
| **A100 PCIe 80GB** | sm_80 | 80 GB HBM2e | 1.935 TB/s | 600 GB/s | No | No | 300W | Lower BW, lower TDP |
| **A100 40GB** | sm_80 | 40 GB HBM2e | 1.555 TB/s | - | No | No | 250W | Budget A100, ~20B FP16 max |
| **A10G** | sm_86 | 24 GB GDDR6 | 0.600 TB/s | None | No | No | 150W | AWS G5, models <=13B |

**Ampere inference profile:** No native FP8. Use BF16/FP16 or INT8/INT4 (AWQ/GPTQ via Marlin kernels). FP8 models run via W8A16 weight-only dequant.

---

### Hopper (2023-2025) — SM90/SM90a

| GPU | CC | Memory | BW | NVLink | FP8 | FP4 | TDP | Notes |
|-----|-----|--------|-----|--------|-----|-----|-----|-------|
| **H100 SXM** | sm_90a | 80 GB HBM3 | 3.35 TB/s | v4 900 GB/s | OCP | No | 700W | Primary AI training/inference GPU |
| **H100 NVL** | sm_90a | 94 GB HBM3 | 3.9 TB/s | 600 GB/s bridge | OCP | No | 400W | Dual-GPU PCIe card for inference |
| **H100 PCIe** | sm_90 | 80 GB HBM2e | 2.0 TB/s | None | OCP | No | 350W | No NVLink, no async wgmma |
| **H200 SXM** | sm_90a | 141 GB HBM3e | 4.8 TB/s | v4 900 GB/s | OCP | No | 700W | Memory upgrade, TP=1 for 70B |
| **H200 NVL** | sm_90a | 141 GB HBM3e | 4.8 TB/s | v4 900 GB/s | OCP | No | 700W | Air-cooled rack form factor |
| **GH200** | sm_90a | 96 GB HBM3 + 480 GB LPDDR5X | 4.0 TB/s | v4 900 GB/s + C2C 900 GB/s | OCP | No | 1000W | Grace Hopper Superchip |
| **H800** | sm_90a | 80 GB HBM3 | 3.35 TB/s | v4 **400 GB/s** | OCP | No | 700W | **EXPORT-RESTRICTED** (China) |
| **H20** | sm_90a | 96 GB HBM3 | 4.0 TB/s | v4 900 GB/s | OCP | No | 350W | **EXPORT-RESTRICTED** (China, 78 SMs) |

**Hopper inference profile:** Native FP8 (OCP E4M3). FlashAttention-3 via wgmma/TMA. `--kv-cache-dtype fp8_e4m3`. Production-safe `gpu_memory_utilization=0.95` on SXM variants.

---

### Blackwell (2025-2026) — SM100/SM103

| GPU | CC | Memory | BW | NVLink | FP8 | FP4 | TDP | Notes |
|-----|-----|--------|-----|--------|-----|-----|-----|-------|
| **B100** | sm_100 | 192 GB HBM3e | 8.0 TB/s | v5 1800 GB/s | OCP | NVFP4 | 700W | Lower-power Blackwell, mostly skipped |
| **B200** | sm_100 | 192 GB HBM3e | 8.0 TB/s | v5 1800 GB/s | OCP | NVFP4 | 1000W | Primary Blackwell |
| **B300** | sm_103 | 288 GB HBM3e | 8.0 TB/s | v5 1800 GB/s | OCP | NVFP4 | 1400W | Blackwell Ultra, accel softmax |
| **GB200** | sm_100 | 192 GB/die + 480 GB Grace | 8.0 TB/s | v5 1800 + C2C 900 | OCP | NVFP4 | ~2700W | 2xB200 + Grace Superchip |
| **GB300** | sm_103 | 288 GB/die + 480 GB Grace | 8.0 TB/s | v5 1800 + C2C 900 | OCP | NVFP4 | ~3800W | 2xB300 + Grace Superchip |

**Blackwell inference profile:** Native NVFP4 (2x throughput vs FP8). FlashAttention-4 with RoPE+KV fusion. nvCOMP decompression engine. NVLink5 2x bandwidth vs Hopper.

---

## Total: 17 GPU Variants

| Generation | Standard | High-Memory | Superchip (Grace) | Export-Restricted | Budget |
|-----------|----------|-------------|-------------------|-------------------|--------|
| **Ampere** | A100 SXM 80GB, A100 PCIe 80GB | - | - | - | A100 40GB, A10G |
| **Hopper** | H100 SXM, H100 PCIe | H100 NVL (94GB), H200 SXM, H200 NVL | GH200 | H800, H20 | - |
| **Blackwell** | B200 | B300 (288GB) | GB200, GB300 | - | B100 (700W) |

---

## Worldwide GPU Deployment (Estimated, March 2026)

### Shipment Volumes

| Period | GPUs Shipped | Primary SKUs | Source |
|--------|-------------|--------------|--------|
| 2023 | ~550K | H100 | HPCwire, NVIDIA earnings |
| 2024 | ~3.5M (all high-end) | H100, H200, H20 | TrendForce |
| 2025 | ~5.4M (est) | H200, B200, GB200 | TrendForce (+55% YoY) |
| 2025 H2 | ~2M Blackwell | B200, GB200 | NVIDIA guidance |
| 2026 (proj) | ~4.3M Blackwell | B200, B300, GB200, GB300 | SemiAnalysis |

### Cumulative Installed Base (estimated end of 2025)

| GPU Family | Est. Units Worldwide | Notes |
|------------|---------------------|-------|
| A100 (all variants) | ~1.5-2M | Still heavily deployed, approaching EOL for new orders |
| H100/H200/H800 | ~3.5M | ~1.46M H20s additionally for China market |
| H20 (export) | ~1.5M | China-only deployment |
| Blackwell (B200/GB200) | ~2M | Ramping rapidly, 80%+ of 2025 H2 shipments |
| **Total data center GPUs** | **~8-10M** | Across all generations still active |

---

## Who Has Them

### Hyperscalers (Largest Buyers)

| Company | Est. GPU Fleet | Key SKUs | CapEx (2025) | Notes |
|---------|---------------|----------|-------------|-------|
| **Microsoft Azure** | 500K+ H100/H200, ramping B200 | H100, H200, B200, GB200 | ~$80B | Largest NVIDIA customer. ND H100 v5, ND H200 v5, upcoming Blackwell instances |
| **Meta** | 600K+ H100, ramping B200 | H100, H200, B200 | ~$66-72B | Largest single H100 buyer in 2024. Building 2GW+ AI data centers |
| **Google Cloud** | 300K+ H100/H200, A4 B200 | H100, H200, B200, GB200 | ~$75B | A3 (H100), A3 Ultra (H200), A4 (B200), A4X (GB200) instances |
| **Amazon AWS** | 400K+ H100/H200, P6 B200 | H100, H200, B200, GB200 | ~$100B | P5 (H100), P5e (H200), P6 (B200), P6e (GB200) instances |
| **Oracle Cloud** | 100K+, Stargate campus | H100, H200, GB200 | ~$40B | Stargate I (Abilene, TX): planned 450K+ GB200 GPUs |

### Neoclouds (Specialized GPU Providers)

| Company | Est. GPU Fleet | Key SKUs | H100 Price/hr | Notes |
|---------|---------------|----------|--------------|-------|
| **CoreWeave** | 250K+ GPUs | H100, H200, B200, GB200 | ~$6.16 | Largest neocloud. IPO'd 2025. 40-70% cheaper than hyperscalers |
| **Lambda Labs** | 30K+ GPUs | H100, H200, A100 | ~$2.99 | Developer-focused. Good availability |
| **Voltage Park** | 20K+ GPUs | H100, B200, B300, GB200 | ~$2.50 | Enterprise-focused |
| **Together AI** | 10K+ GPUs | H100, H200 | ~$2.50 | Inference-focused, serverless endpoints |
| **RunPod** | 10K+ GPUs | H100, A100, community | ~$1.99 | Community marketplace model |
| **Vast.ai** | Marketplace | Mixed (H100, A100, consumer) | ~$1.49-1.87 | Peer-to-peer GPU marketplace |
| **Crusoe Energy** | 50K+ GPUs | H100, H200 | Market rate | Clean energy (flare gas, solar). Sending H100s to orbit (Starcloud) |

### AI Labs (Own Clusters)

| Company | Est. GPU Fleet | Key SKUs | Notes |
|---------|---------------|----------|-------|
| **OpenAI** | 100K+ H100 (via Azure) | H100, H200, Stargate GB200 | Microsoft partnership. Stargate project |
| **xAI (Elon Musk)** | 100K+ H100 | H100, ramping Blackwell | Memphis "Colossus" cluster. 100K H100 in single cluster |
| **Anthropic** | 50K+ (via AWS/GCP) | H100, H200 | Multi-cloud (AWS primary, GCP) |
| **DeepSeek** | 50K+ (est) | A100, H800, H20 | China-based, export-restricted GPUs |
| **ByteDance** | 100K+ (est) | A100, H800, H20 | China market, pre-restriction A100 stockpile |

### Enterprise (On-Premise)

| Sector | Typical GPUs | Use Case |
|--------|-------------|----------|
| **Finance** (JPMorgan, Goldman) | H100, A100 | Risk modeling, NLP, trading |
| **Healthcare** (Mayo Clinic, pharma) | A100, H100 | Drug discovery, medical imaging |
| **Automotive** (Tesla, Waymo) | A100, H100 | Autonomous driving training |
| **Telecom** (AT&T, Verizon) | A100, A10G | Network optimization, edge inference |
| **Government/Defense** | H200 (Azure Gov Secret/Top Secret) | Classified AI workloads |

---

## Cloud Pricing (March 2026)

### H100 SXM (per GPU per hour)

| Provider | On-Demand | Reserved/Spot | Notes |
|----------|----------|---------------|-------|
| AWS (P5) | ~$3.90-6.88 | ~$2.00-2.50 spot | P5 instances |
| Azure (ND H100 v5) | ~$6.98 | ~$3.50 spot | East US pricing |
| GCP (A3) | ~$4.00-5.00 | ~$2.50 spot | A3 instances |
| CoreWeave | ~$6.16 (node) | Contract pricing | 8xH100 HGX nodes |
| Lambda | ~$2.99 | - | Best general availability |
| RunPod | ~$1.99 | Community pricing | |
| Vast.ai | ~$1.49-1.87 | Marketplace | Lowest, variable quality |

### B200 (per GPU per hour, early pricing)

| Provider | On-Demand | Notes |
|----------|----------|-------|
| AWS (P6) | ~$7-10 (est) | Early availability |
| CoreWeave | ~$8-12 (est) | Ramping supply |
| Lambda | TBD | Coming soon |

### GPU Purchase Pricing (per unit)

| GPU | Approx. Purchase Price | Notes |
|-----|----------------------|-------|
| A100 80GB SXM | ~$10,000-12,000 | Secondary market, declining |
| H100 SXM | ~$25,000-30,000 | Down from $40K+ in 2023 |
| H200 SXM | ~$30,000-35,000 | Limited availability |
| B200 | ~$30,000-40,000 (est) | Supply-constrained |
| B300 | ~$40,000-50,000 (est) | Shipped Jan 2026, premium |

---

## Generation Transition Timeline

```
2020    2021    2022    2023    2024    2025    2026    2027
  |       |       |       |       |       |       |       |
  A100 launch     |       H100    H200    B200    B300    Rubin
  |               |       launch  launch  launch  launch  (HBM4)
  |               |               |       GB200   GB300
  |               |               |       GH200
  |               |               H800/H20 (export)
  |               |               B100 (skipped)
  └─── Ampere ────┘── Hopper ─────┘── Blackwell ──┘─ Vera Rubin ─
```

### What's Next: Vera Rubin (2026-2027)
- HBM4: 288 GB per GPU, 13 TB/s bandwidth
- Rubin NVL144: 144 GPUs per rack, 3.6 ExaFLOPS dense FP4
- TSMC 3nm process

---

## Sources

- [NVIDIA A100 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)
- [NVIDIA H100 GPU](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA H200 GPU](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA H100 NVL Product Brief (PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)
- [Inside NVIDIA Blackwell Ultra (NVIDIA Tech Blog)](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [CoreWeave 250K GPU Fleet (Next Platform)](https://www.nextplatform.com/2025/03/05/coreweaves-250000-strong-gpu-fleet-undercuts-the-big-clouds/)
- [H100 Rental Prices Compared (IntuitionLabs)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [Cloud GPU Pricing Comparison 2026 (Spheron)](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [Epoch AI: Computing Capacity of AI Companies](https://epoch.ai/data-insights/computing-capacity)
- [NVIDIA Hopper GPUs Expand Reach (NVIDIA Newsroom)](https://nvidianews.nvidia.com/news/nvidia-hopper-gpus-expand-reach-as-demand-for-ai-grows)
- [Hyperscaler AI CapEx 2026 (DataCenter Knowledge)](https://www.datacenterknowledge.com/hyperscalers/hyperscalers-in-2026-what-s-next-for-the-world-s-largest-data-center-operators-)
- [TrendForce GPU Shipment Projections](https://files.futurememorystorage.com/proceedings/2024/20240808_BMKT-301-1_KUNG.pdf)
- [AI GPU Rental Market Trends March 2026 (ThunderCompute)](https://www.thundercompute.com/blog/ai-gpu-rental-market-trends)
