"""SGLang v0.5+ engine adapter and config compiler."""

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
from inferscope.optimization.serving_profile import (
    ModelClass,
    ServingProfile,
    WorkloadMode,
)

_adapter_log = get_logger(component="sglang_adapter")


class SGLangCompiler(ConfigCompiler):
    """Compiles a ServingProfile into SGLang server args."""

    def engine_name(self) -> str:
        return "sglang"

    def compile(self, profile: ServingProfile, inventory: DeploymentInventory) -> EngineConfig:
        cfg = EngineConfig(engine="sglang")

        # --- Model ---
        cfg.cli_flags["model-path"] = profile.model

        # --- Parallelism ---
        if profile.topology.tp > 1:
            cfg.cli_flags["tp"] = profile.topology.tp
        if profile.topology.dp > 1:
            cfg.cli_flags["dp"] = profile.topology.dp

        # --- Memory ---
        cfg.cli_flags["mem-fraction-static"] = profile.cache.gpu_memory_utilization

        # --- Scheduling ---
        cfg.cli_flags["chunked-prefill-size"] = profile.scheduler.prefill_chunk_tokens

        # RadixAttention scheduling policy
        if profile.workload_mode == WorkloadMode.CODING:
            cfg.cli_flags["schedule-policy"] = "lpm"  # longest prefix match
            cfg.notes.append("RadixAttention with LPM scheduling — optimal for high prefix reuse in coding")
        elif profile.workload_mode == WorkloadMode.AGENT:
            cfg.cli_flags["schedule-policy"] = "lpm"
            cfg.notes.append("LPM scheduling for session-sticky agent workloads")
        else:
            cfg.cli_flags["schedule-policy"] = "fcfs"

        # --- Quantization ---
        if profile.precision.weights == "fp8" and inventory.fp8_support:
            cfg.cli_flags["quantization"] = "fp8"
        elif profile.precision.weights in ("awq", "gptq"):
            cfg.cli_flags["quantization"] = profile.precision.weights

        if profile.precision.kv_cache == "fp8_e4m3" and inventory.fp8_support:
            cfg.cli_flags["kv-cache-dtype"] = "fp8_e4m3"

        # --- Speculative decoding ---
        if profile.speculation.mode != "off" and profile.speculation.method == "mtp":
            cfg.cli_flags["speculative-algo"] = "NEXTN"
            cfg.cli_flags["speculative-num-draft-tokens"] = profile.speculation.num_speculative_tokens

        # --- Enable metrics (required for monitoring) ---
        cfg.cli_flags["enable-metrics"] = True

        # --- DeepSeek DP attention ---
        if profile.model_class == ModelClass.FRONTIER_MLA_MOE:
            cfg.cli_flags["trust-remote-code"] = True
            if profile.topology.dp > 1:
                cfg.cli_flags["enable-dp-attention"] = True
                cfg.notes.append("DP attention enabled for DeepSeek — up to 1.9x decode throughput")

        # --- HiCache for long-context / agent workloads ---
        if (
            profile.workload_mode in (WorkloadMode.CODING, WorkloadMode.AGENT)
            and profile.cache.kv_tiering != "gpu_only"
        ):
            cfg.cli_flags["enable-hierarchical-cache"] = True
            cfg.cli_flags["hicache-write-policy"] = "write-through"
            cfg.notes.append("HiCache enabled for multi-level KV cache (GPU → CPU → SSD)")

        # --- Cross-session prefix sharing ---
        if profile.cache.block_reuse_strategy == "prefix_sharing_cross_session":
            cfg.cli_flags["schedule-policy"] = "lpm"
            cfg.notes.append(
                "Cross-session prefix sharing via RadixAttention LPM — "
                "shared system prompts and tool schemas across sessions"
            )

        # --- KV Cache Offloading Policy ---
        if profile.cache.offload_policy != "disabled" and profile.cache.kv_tiering != "gpu_only":
            cfg.notes.append(
                f"Cold-only KV offload: sessions idle >{profile.cache.offload_idle_threshold_s:.0f}s "
                f"eligible for CPU offload (PCIe cap: {profile.cache.pcie_utilization_cap:.0%})"
            )

        # --- Prefill/Decode Isolation ---
        if profile.scheduler.prefill_decode_isolation == "soft_priority":
            cfg.notes.append(
                f"Prefill/decode soft isolation: max prefill chunk ratio "
                f"{profile.scheduler.max_prefill_chunk_ratio:.0%} to prevent decode starvation"
            )

        # --- AMD-specific ---
        if inventory.gpu_arch in ("gfx942", "gfx950"):
            cfg.env_vars["SGLANG_USE_AITER"] = "1"
            cfg.env_vars["HIP_FORCE_DEV_KERNARG"] = "1"
            cfg.env_vars["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"

        # --- Build command ---
        cmd_parts = ["python", "-m", "sglang.launch_server"]
        for k, v in cfg.cli_flags.items():
            if isinstance(v, bool):
                if v:
                    cmd_parts.append(f"--{k}")
            else:
                cmd_parts.append(f"--{k}")
                cmd_parts.append(str(v))
        cfg.command = " \\\n  ".join(cmd_parts)

        return cfg


class SGLangAdapter(EngineAdapter):
    """Connects to a running SGLang instance."""

    def engine_name(self) -> str:
        return "sglang"

    async def detect_engine(self, endpoint: str) -> bool:
        try:
            url = self._validate_endpoint(endpoint)
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/metrics")
                return "sglang:" in resp.text
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
                    if "sglang:" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            name = parts[0].split("{")[0]
                            try:
                                metrics[name] = float(parts[-1])
                            except ValueError:
                                metrics[name] = parts[-1]
        except Exception:  # noqa: S110
            _adapter_log.warning("sglang_metrics_scrape_failed", endpoint=endpoint)
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
