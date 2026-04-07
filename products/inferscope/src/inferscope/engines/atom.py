"""AMD ATOM engine adapter and config compiler.

ATOM (AiTer Optimized Model) is AMD's standalone inference engine built on
AITER kernels + MoRI communication. It's purpose-built for MLA+MoE models
like DeepSeek R1/V3 on MI300X/MI355X hardware.
"""

from __future__ import annotations

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

_adapter_log = get_logger(component="atom_adapter")


class ATOMCompiler(ConfigCompiler):
    """Compiles a ServingProfile into ATOM CLI flags + AITER env vars + MoRI config."""

    def engine_name(self) -> str:
        return "atom"

    def compile(self, profile: ServingProfile, inventory: DeploymentInventory) -> EngineConfig:
        cfg = EngineConfig(engine="atom")

        # Validate: ATOM only works on AMD hardware. Set support_tier
        # explicitly so callers filtering on tier alone can detect the
        # rejection — matches DynamoCompiler's contract.
        # Closes the snapshot v1.0.0 P1 bug
        # `atom_compiler_unsupported_tier_inconsistency`.
        if inventory.gpu_arch not in ("gfx942", "gfx950"):
            cfg.support_tier = "unsupported"
            cfg.support_reason = (
                f"ATOM requires AMD MI300X/MI325X/MI355X (gfx942/gfx950), got {inventory.gpu_arch}"
            )
            cfg.warnings.append(cfg.support_reason)
            return cfg

        # --- Model ---
        cfg.cli_flags["model"] = profile.model

        # --- KV cache ---
        cfg.cli_flags["kv_cache_dtype"] = "fp8"
        cfg.notes.append("FP8 KV cache is recommended for all ATOM deployments")

        # --- Parallelism ---
        if profile.topology.tp > 1:
            cfg.cli_flags["tp"] = profile.topology.tp
        if profile.topology.dp > 1:
            cfg.cli_flags["dp"] = profile.topology.dp
        if profile.topology.ep > 1:
            cfg.cli_flags["ep"] = profile.topology.ep

        # --- Memory ---
        cfg.cli_flags["gpu_memory_utilization"] = profile.cache.gpu_memory_utilization
        cfg.cli_flags["max_num_seqs"] = profile.scheduler.max_num_seqs

        # --- Compilation level ---
        # Level 3 (PIECEWISE) is default production — CUDA graph + piecewise compile
        cfg.cli_flags["compilation_level"] = 3
        cfg.notes.append("Compilation level 3 (PIECEWISE) — production default with CUDA graph acceleration")

        # --- Speculative decoding (MTP) ---
        if profile.speculation.mode != "off" and profile.speculation.method == "mtp":
            cfg.cli_flags["method"] = "mtp"
            cfg.cli_flags["num-speculative-tokens"] = profile.speculation.num_speculative_tokens
            cfg.notes.append("MTP speculative decoding via EAGLE proposer")

        # --- AITER env vars (mandatory) ---
        cfg.env_vars["VLLM_ROCM_USE_AITER"] = "1"
        cfg.env_vars["HIP_FORCE_DEV_KERNARG"] = "1"
        cfg.env_vars["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"

        if inventory.gpu_arch == "gfx942":
            cfg.env_vars["VLLM_ROCM_USE_AITER_FP8BMM"] = "0"
            cfg.notes.append("FP8BMM disabled on gfx942 (MI300X) — CRASHES with memory faults")
        elif inventory.gpu_arch == "gfx950":
            cfg.env_vars["VLLM_ROCM_USE_AITER_FP8BMM"] = "1"

        # --- Attention backend selection ---
        if profile.model_class == ModelClass.FRONTIER_MLA_MOE:
            cfg.env_vars["ROCM_AITER_MLA"] = "1"
            cfg.notes.append("ROCM_AITER_MLA backend for MLA models — up to 17x decode speedup")
        else:
            cfg.env_vars["ROCM_AITER_FA"] = "1"
            cfg.notes.append("ROCM_AITER_FA with 3-path routing: prefill→CK, extend→CK, decode→ASM")

        # --- MoRI communication for multi-GPU ---
        if profile.topology.tp > 1 or profile.topology.ep > 1:
            cfg.env_vars["NCCL_MIN_NCHANNELS"] = "112"
            if profile.model_class in (
                ModelClass.FRONTIER_MLA_MOE,
                ModelClass.COMPACT_AGENTIC_MOE,
                ModelClass.CLASSICAL_MOE,
            ):
                cfg.notes.append("MoRI all-to-all enabled for MoE expert dispatch/aggregation")

        # --- KV Cache Offloading Policy (AMD CDNA) ---
        if profile.cache.offload_policy != "disabled" and profile.cache.kv_tiering != "gpu_only":
            cfg.notes.append(
                f"Cold-only KV offload on AMD: sessions idle >{profile.cache.offload_idle_threshold_s:.0f}s "
                f"eligible for CPU offload (PCIe cap: {profile.cache.pcie_utilization_cap:.0%})"
            )
            cfg.warnings.append(
                "AMD PCIe offload: Infinity Fabric provides higher bandwidth than PCIe — "
                "prefer IF-connected CPU DRAM for KV offload targets"
            )

        # --- Chunked Prefill Control ---
        if not profile.scheduler.chunked_prefill:
            cfg.cli_flags["disable_chunked_prefill"] = True
            cfg.notes.append(
                "Chunked prefill disabled on AMD CDNA — "
                "contiguous prefill avoids KV staging overhead and TTFT degradation"
            )

        # --- Build command ---
        cmd_parts = ["python", "-m", "atom.entrypoints.openai_server"]
        for k, v in cfg.cli_flags.items():
            if isinstance(v, bool):
                if v:
                    cmd_parts.append(f"--{k}")
            else:
                cmd_parts.append(f"--{k}")
                cmd_parts.append(str(v))
        cfg.command = " \\\n  ".join(cmd_parts)

        return cfg


class ATOMAdapter(EngineAdapter):
    """Connects to a running ATOM instance."""

    def engine_name(self) -> str:
        return "atom"

    async def detect_engine(self, endpoint: str) -> bool:
        try:
            url = self._validate_endpoint(endpoint)
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/metrics")
                return "atom:" in resp.text
        except Exception:  # noqa: S110
            return False

    async def get_metrics(self, endpoint: str) -> dict[str, Any]:
        url = self._validate_endpoint(endpoint)
        metrics: dict[str, Any] = {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{url}/metrics")
                for line in resp.text.splitlines():
                    if line.startswith("#"):
                        continue
                    if "atom:" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            name = parts[0].split("{")[0]
                            try:
                                metrics[name] = float(parts[-1])
                            except ValueError:
                                metrics[name] = parts[-1]
        except Exception:  # noqa: S110
            _adapter_log.warning("atom_metrics_scrape_failed", endpoint=endpoint)
        return metrics

    async def get_config(
        self,
        endpoint: str,
        *,
        allow_private: bool = True,
        auth: EndpointAuthConfig | None = None,
    ) -> dict[str, Any]:
        try:
            url = self._validate_endpoint(endpoint, allow_private=allow_private)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{url}/v1/models", headers=build_auth_headers(auth))
                return cast(dict[str, Any], resp.json())
        except Exception:  # noqa: S110
            return {}
