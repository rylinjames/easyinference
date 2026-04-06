"""vLLM v0.18+ engine adapter and config compiler (updated for vLLM 2026 / GTC features)."""

from __future__ import annotations

import json
import shlex
from typing import Any, cast

import httpx

from inferscope.endpoint_auth import EndpointAuthConfig, build_auth_headers
from inferscope.engines.base import (
    ConfigCompiler,
    DeploymentInventory,
    EngineAdapter,
    EngineConfig,
)
from inferscope.logging import get_logger
from inferscope.optimization.serving_profile import ModelClass, ServingProfile

_adapter_log = get_logger(component="vllm_adapter")


class VLLMCompiler(ConfigCompiler):
    """Compiles a ServingProfile into vLLM CLI flags and env vars."""

    def engine_name(self) -> str:
        return "vllm"

    def compile(self, profile: ServingProfile, inventory: DeploymentInventory) -> EngineConfig:
        cfg = EngineConfig(engine="vllm")

        # --- Model ---
        cfg.cli_flags["model"] = profile.model

        # --- Parallelism ---
        if profile.topology.tp > 1:
            cfg.cli_flags["tensor-parallel-size"] = profile.topology.tp
        if profile.topology.pp > 1:
            cfg.cli_flags["pipeline-parallel-size"] = profile.topology.pp
        if profile.topology.dp > 1:
            cfg.cli_flags["data-parallel-size"] = profile.topology.dp
        if profile.topology.ep > 1:
            cfg.cli_flags["enable-expert-parallel"] = True

        # --- Precision / Quantization ---
        if profile.precision.weights == "fp8":
            if inventory.fp8_support:
                cfg.cli_flags["quantization"] = "fp8"
            elif inventory.gpu_arch.startswith("sm_8"):
                # Ampere: FP8 models use W8A16 Marlin, not native
                cfg.cli_flags["quantization"] = "fp8"
                cfg.notes.append("FP8 on Ampere uses W8A16 Marlin (weight-only dequant), not native FP8")
        elif profile.precision.weights == "fp4":
            if inventory.gpu_arch in ("sm_100", "sm_103"):
                cfg.cli_flags["quantization"] = "nvfp4"
                # NVFP4 uses 2-level scaling: E4M3 per 16-element block + FP32 per-tensor
                cfg.notes.append(
                    "NVFP4: 2-level scaling (E4M3 per 16 elements + FP32 per tensor). "
                    "3.5x memory reduction vs FP16, 2x throughput vs FP8."
                )
            else:
                cfg.warnings.append(f"NVFP4 requires Blackwell (SM100+), got {inventory.gpu_arch}")
        elif profile.precision.weights in ("awq", "gptq") or profile.precision.weights in (
            "int8",
            "int4",
        ):
            cfg.cli_flags["quantization"] = profile.precision.weights

        # --- KV Cache ---
        if profile.precision.kv_cache == "fp8_e4m3" and inventory.fp8_support:
            cfg.cli_flags["kv-cache-dtype"] = "fp8_e4m3"
        elif profile.precision.kv_cache == "fp8_e5m2" and inventory.fp8_support:
            cfg.cli_flags["kv-cache-dtype"] = "fp8_e5m2"

        cfg.cli_flags["gpu-memory-utilization"] = profile.cache.gpu_memory_utilization

        # --- Scheduling ---
        cfg.cli_flags["max-num-batched-tokens"] = profile.scheduler.batched_token_budget
        cfg.cli_flags["max-num-seqs"] = profile.scheduler.max_num_seqs

        # Chunked prefill (V1 default is on; explicitly control based on workload)
        cfg.cli_flags["enable-chunked-prefill"] = profile.scheduler.chunked_prefill
        if profile.scheduler.prefill_chunk_tokens > 0:
            cfg.cli_flags["max-num-prefill-tokens"] = profile.scheduler.prefill_chunk_tokens

        # --- Prefix caching (always on in V1, but explicit is fine) ---
        # V1 has zero-overhead prefix caching — no reason to disable
        cfg.notes.append("vLLM V1 enables prefix caching with zero overhead by default")

        # --- Hopper-specific optimizations (H100/H200) ---
        is_hopper_sxm = inventory.gpu_arch == "sm_90a"
        is_hopper_pcie = inventory.gpu_arch == "sm_90"
        is_h200 = is_hopper_sxm and inventory.gpu_memory_gb >= 140

        if is_hopper_sxm:
            cfg.notes.append(
                "Hopper (sm_90a): FlashAttention-3 via wgmma at 75%+ utilization "
                "(2x vs FA2 on Ampere) + TMA for zero-overhead address computation"
            )
            if is_h200:
                cfg.notes.append(
                    f"H200 {inventory.gpu_memory_gb:.0f}GB HBM3e @ "
                    f"{inventory.gpu_memory_bandwidth_tb_s} TB/s — "
                    "most workloads fit GPU-resident without KV offloading"
                )
            if profile.cache.gpu_memory_utilization >= 0.95:
                cfg.notes.append(
                    "gpu_memory_utilization=0.95: safe on Hopper HBM3/HBM3e (stable thermal envelope at 700W TDP)"
                )
        elif is_hopper_pcie:
            cfg.warnings.append(
                "H100 PCIe (sm_90): no NVLink, no async wgmma — "
                "TP>1 uses PCIe Gen5, KV offloading is PCIe-bound. "
                "Consider single-GPU or disaggregated serving."
            )

        # --- Blackwell-specific optimizations (B200/B300/GB200) ---
        is_blackwell = inventory.platform_family.startswith("blackwell") or inventory.gpu_arch in ("sm_100", "sm_103")
        is_b300 = inventory.platform_family == "blackwell_ultra"
        is_gb200 = inventory.platform_family == "blackwell_grace"
        is_gb300 = inventory.platform_family == "blackwell_ultra_grace"

        if is_blackwell:
            if is_b300:
                cfg.notes.append(
                    f"B300 Ultra {inventory.gpu_memory_gb:.0f}GB — fits most models on TP=1-2, "
                    "accelerated softmax in hardware, inference-optimized"
                )
            if is_gb200:
                cfg.notes.append(
                    "GB200 Grace Blackwell: KV cache overflow to Grace LPDDR5X (480GB) "
                    f"via NVLink-C2C @ {inventory.c2c_bandwidth_gb_s:.0f} GB/s — "
                    "~7x faster than PCIe Gen5 offloading"
                )
            if is_gb300:
                cfg.notes.append(
                    "GB300 Grace Blackwell Ultra: combines B300-class compute with Grace coherent overflow "
                    "for long-context and disaggregated deployments."
                )
            if inventory.fp4_support:
                cfg.notes.append("NVFP4 native: --quantization nvfp4 for 2x throughput vs FP8 at <1% accuracy loss")
            if profile.cache.gpu_memory_utilization >= 0.95:
                cfg.notes.append("gpu_memory_utilization=0.95: safe on Blackwell HBM3e (stable thermal envelope)")

        # --- Speculative decoding ---
        if profile.speculation.mode != "off" and profile.speculation.method:
            if profile.speculation.method == "mtp":
                cfg.cli_flags["speculative-config"] = {
                    "method": "mtp",
                    "num_speculative_tokens": profile.speculation.num_speculative_tokens,
                }
            elif profile.speculation.model:
                cfg.cli_flags["speculative-model"] = profile.speculation.model
                cfg.cli_flags["num-speculative-tokens"] = profile.speculation.num_speculative_tokens

        # --- Model class specifics ---
        if profile.model_class == ModelClass.FRONTIER_MLA_MOE:
            cfg.cli_flags["trust-remote-code"] = True
            # MLA on ROCm needs block-size 1
            if inventory.gpu_arch in ("gfx942", "gfx950"):
                cfg.cli_flags["block-size"] = 1
                cfg.notes.append("block-size=1 required for MLA compatibility on ROCm")

        # --- AMD-specific flags ---
        if inventory.gpu_arch in ("gfx942", "gfx950"):
            cfg.env_vars["VLLM_ROCM_USE_AITER"] = "1"
            cfg.env_vars["HIP_FORCE_DEV_KERNARG"] = "1"
            cfg.env_vars["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"

            # MI300X Bottleneck Fixes
            cfg.cli_flags["enable-chunked-prefill"] = profile.scheduler.chunked_prefill
            if not profile.scheduler.chunked_prefill:
                cfg.notes.append("Chunked prefill disabled: reduces TTFT for long-context (med tech/coding) on MI300X.")
            cfg.warnings.append(
                "CRITICAL: Ensure `sysctl -w kernel.numa_balancing=0` is set on the host to prevent GPU hangs."
            )

            if cfg.cli_flags.get("quantization") == "fp8":
                cfg.warnings.append(
                    "FP8 inference on MI300X MoE is currently slower than BF16 due to ROCm regressions."
                )
            if profile.topology.tp > 1 and profile.model_class not in (
                ModelClass.DENSE_GQA,
                ModelClass.FRONTIER_MLA_MOE,
            ):
                cfg.warnings.append("High TP on small models (<40B) via RCCL is slower than TP=1 replicas on AMD.")

            if inventory.gpu_arch == "gfx942":
                cfg.env_vars["VLLM_ROCM_USE_AITER_FP8BMM"] = "0"
                cfg.env_vars["NCCL_MIN_NCHANNELS"] = "112"
                cfg.notes.append("FP8BMM disabled on MI300X (gfx942) — crashes with memory faults")
            elif inventory.gpu_arch == "gfx950":
                cfg.env_vars["VLLM_ROCM_USE_AITER_FP8BMM"] = "1"

            # ATOM backend recommendation for MLA/MoE on AMD
            if profile.model_class == ModelClass.FRONTIER_MLA_MOE:
                cfg.cli_flags["model-impl"] = "atom"
                cfg.notes.append(
                    "Using ATOM model backend for MLA/MoE on AMD — best of vLLM ecosystem + AITER kernel performance"
                )

        # --- Prefill/Decode Isolation ---
        if profile.scheduler.prefill_decode_isolation == "soft_priority":
            # Soft priority: constrain prefill chunk ratio to prevent decode starvation
            effective_prefill_budget = int(
                profile.scheduler.batched_token_budget * profile.scheduler.max_prefill_chunk_ratio
            )
            if effective_prefill_budget < profile.scheduler.batched_token_budget:
                cfg.cli_flags["max-num-batched-tokens"] = effective_prefill_budget
                cfg.notes.append(
                    f"Prefill/decode soft isolation: prefill capped at {profile.scheduler.max_prefill_chunk_ratio:.0%} "
                    f"of batch budget ({effective_prefill_budget} tokens) to prevent decode starvation"
                )

        # --- KV Cache Offloading Policy ---
        if profile.cache.offload_policy != "disabled" and profile.cache.kv_tiering != "gpu_only":
            if profile.cache.offload_policy == "cold_only":
                cfg.notes.append(
                    f"Cold-only KV offload: sessions idle >{profile.cache.offload_idle_threshold_s:.0f}s "
                    f"eligible for CPU offload (PCIe cap: {profile.cache.pcie_utilization_cap:.0%})"
                )
            cfg.warnings.append(
                "CRITICAL: Do NOT offload KV during active decode loops — "
                "PCIe transfer becomes bottleneck and causes 99th percentile latency explosion"
            )
        elif profile.cache.offload_policy == "disabled":
            cfg.notes.append("KV offloading disabled — all cache GPU-resident")

        # --- KV Fragmentation Monitoring ---
        if profile.cache.fragmentation_check:
            cfg.notes.append(
                f"KV fragmentation monitoring recommended: trigger compaction when "
                f"block utilization < {profile.cache.kv_compaction_trigger:.0%}"
            )

        # --- Block Reuse Strategy ---
        if profile.cache.block_reuse_strategy == "prefix_sharing_cross_session":
            cfg.notes.append(
                "Cross-session prefix sharing: common system prompts and tool schemas "
                "shared across PagedAttention blocks (requires consistent prompt structure)"
            )

        # --- Disaggregated Prefill (GTC 2026) ---
        if profile.topology.split_prefill_decode:
            # Select connector: explicit override > auto-detect
            connector = profile.topology.disagg_connector
            if not connector:
                connector = "NixlConnector" if inventory.has_rdma else "P2pNcclConnector"

            kv_config = {
                "kv_connector": connector,
                "kv_role": "kv_both",
            }
            cfg.cli_flags["kv-transfer-config"] = kv_config

            if connector == "NixlConnector" and inventory.has_rdma:
                cfg.notes.append(
                    f"NIXL connector with {inventory.rdma_type or 'RDMA'} — "
                    "fully async KV cache transfer for disaggregated prefill"
                )
            elif connector == "P2pNcclConnector":
                cfg.notes.append(
                    "P2pNcclConnector fallback — no RDMA detected. "
                    "Consider RDMA for production disaggregated deployments"
                )
            cfg.notes.append("Route traffic via vLLM Router (github.com/vllm-project/router)")

            # Blackwell ISA advantages for disaggregated serving
            if is_blackwell:
                if inventory.interconnect_bandwidth_gb_s >= 1800:
                    cfg.notes.append(
                        f"NVLink5 @ {inventory.interconnect_bandwidth_gb_s:.0f} GB/s — "
                        "2x KV transfer bandwidth vs Hopper NVLink4, "
                        "enabling higher decode:prefill GPU ratio"
                    )
                # Decompression engine: compressed KV transfer
                cfg.notes.append(
                    "Blackwell nvCOMP decompression engine accelerates I/O — "
                    "can reduce KV cache transfer volume between prefill/decode nodes"
                )
                if is_gb200 or is_gb300:
                    cfg.notes.append(
                        "Grace Blackwell: prefill node can stage KV in Grace LPDDR5X (480GB) "
                        "via NVLink-C2C before async transfer — "
                        "eliminates HBM pressure during KV staging"
                    )

        # --- MoE Compute/Comm Overlap (GTC 2026) ---
        if profile.scheduler.enable_moe_overlap and profile.model_class in (
            ModelClass.FRONTIER_MLA_MOE,
            ModelClass.COMPACT_AGENTIC_MOE,
            ModelClass.CLASSICAL_MOE,
        ):
            cfg.cli_flags["enable-moe-dense-overlap"] = True
            cfg.notes.append("MoE comm/compute overlap — dispatch/combine overlaps with expert/attention execution")

        # --- Expert Parallel Load Balancing / EPLB (GTC 2026) ---
        if profile.topology.enable_eplb and profile.topology.ep > 1:
            cfg.cli_flags["enable-eplb"] = True
            cfg.notes.append("EPLB enabled — hot experts replicated across EP ranks for load balancing")

        # --- Hybrid Memory Allocator (GTC 2026) ---
        if profile.cache.hybrid_block_sizes:
            cfg.cli_flags["block-size"] = ",".join(str(s) for s in profile.cache.hybrid_block_sizes)
            cfg.notes.append(
                f"Hybrid block sizes {profile.cache.hybrid_block_sizes} — "
                "dynamic partitioning for mixed full/sparse attention (0-12% fragmentation)"
            )

        # --- Attention Kernel Fusion Notes (GTC 2026) ---
        if inventory.gpu_arch in ("sm_100", "sm_103"):
            cfg.notes.append("Blackwell: FlashAttention-4 with aggressive RoPE+KV cache fusion (4.5x speedup)")
        elif inventory.gpu_arch == "sm_90a":
            cfg.notes.append("Hopper: FlashAttention-2/3 auto-selected — RoPE+KV cache fusion available")

        # --- Qwen3.5 Hybrid Attention (GTC 2026) ---
        if profile.model_class == ModelClass.QWEN35_HYBRID:
            if not profile.cache.hybrid_block_sizes:
                cfg.warnings.append(
                    "Qwen3.5 uses alternating full/sparse attention layers — "
                    "set hybrid_block_sizes for optimal memory allocator performance"
                )
            cfg.notes.append(
                "Qwen3.5 hybrid attention — alternating full and sparse layers. "
                "Hybrid memory allocator recommended for minimal fragmentation"
            )

        # --- Encoder Prefill Disaggregation (GTC 2026) ---
        if inventory.has_encoder_gpu:
            cfg.notes.append(
                "Dedicated encoder GPUs detected — encoder prefill disaggregation "
                "can yield up to 2.5x throughput for multimodal workloads"
            )

        # --- Build command string ---
        # Use shlex.quote for any value that may contain shell-special characters
        # (notably: dict values serialised to JSON, which can include single quotes
        # inside string fields). Previously dict values were wrapped with literal
        # f"'{json.dumps(v)}'" which is unsafe — a JSON string containing a single
        # quote breaks the wrapping. shlex.quote produces a shell-safe quoted form.
        cmd_parts = ["vllm", "serve"]
        for k, v in cfg.cli_flags.items():
            if isinstance(v, bool):
                if v:
                    cmd_parts.append(f"--{k}")
            elif isinstance(v, dict):
                cmd_parts.append(f"--{k}")
                cmd_parts.append(shlex.quote(json.dumps(v)))
            else:
                cmd_parts.append(f"--{k}")
                cmd_parts.append(shlex.quote(str(v)))
        cfg.command = " \\\n  ".join(cmd_parts)

        return cfg


class VLLMAdapter(EngineAdapter):
    """Connects to a running vLLM instance."""

    def engine_name(self) -> str:
        return "vllm"

    async def detect_engine(self, endpoint: str) -> bool:
        """Detect vLLM by checking for vLLM-specific metrics."""
        try:
            url = self._validate_endpoint(endpoint)
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/metrics")
                return "vllm:" in resp.text
        except Exception:  # noqa: S110
            return False

    async def get_metrics(self, endpoint: str) -> dict[str, Any]:
        """Scrape Prometheus metrics from vLLM /metrics endpoint."""
        url = self._validate_endpoint(endpoint)
        metrics: dict[str, Any] = {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{url}/metrics")
                for line in resp.text.splitlines():
                    if line.startswith("#"):
                        continue
                    if "vllm:" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            name = parts[0].split("{")[0]
                            try:
                                metrics[name] = float(parts[-1])
                            except ValueError:
                                metrics[name] = parts[-1]
        except Exception:  # noqa: S110
            _adapter_log.warning("vllm_metrics_scrape_failed", endpoint=endpoint)
        return metrics

    async def get_config(
        self,
        endpoint: str,
        *,
        allow_private: bool = True,
        auth: EndpointAuthConfig | None = None,
    ) -> dict[str, Any]:
        """Get vLLM model info from /v1/models."""
        try:
            url = self._validate_endpoint(endpoint, allow_private=allow_private)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{url}/v1/models", headers=build_auth_headers(auth))
                return cast(dict[str, Any], resp.json())
        except Exception:  # noqa: S110
            return {}
