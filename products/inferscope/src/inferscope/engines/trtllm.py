"""TRT-LLM v1.2+ engine adapter and config compiler."""

from __future__ import annotations

import json
import shlex
from typing import Any

from inferscope.engines.base import (
    ConfigCompiler,
    DeploymentInventory,
    EngineAdapter,
    EngineConfig,
)
from inferscope.logging import get_logger
from inferscope.optimization.serving_profile import ServingProfile

_adapter_log = get_logger(component="trtllm_adapter")


class TRTLLMCompiler(ConfigCompiler):
    """Compiles a ServingProfile into TRT-LLM args."""

    def engine_name(self) -> str:
        return "trtllm"

    def compile(self, profile: ServingProfile, inventory: DeploymentInventory) -> EngineConfig:
        cfg = EngineConfig(engine="trtllm")
        cfg.support_tier = "preview"
        cfg.support_reason = (
            "TensorRT-LLM is exposed as a preview planning target in InferScope; "
            "validate manually before production use."
        )
        cfg.warnings.append(f"Preview engine: {cfg.support_reason}")
        cfg.cli_flags["model_dir"] = profile.model

        # --- Parallelism ---
        if profile.topology.tp > 1:
            cfg.cli_flags["tp_size"] = profile.topology.tp
        if profile.topology.pp > 1:
            cfg.cli_flags["pp_size"] = profile.topology.pp

        # --- Memory and Cache ---
        cfg.cli_flags["kv_cache_type"] = "paged"
        cfg.cli_flags["max_batch_size"] = profile.scheduler.max_num_seqs
        cfg.cli_flags["max_num_tokens"] = profile.scheduler.batched_token_budget

        # --- Disaggregated Serving / KV Cache Transfer ---
        if profile.topology.split_prefill_decode:
            if not inventory.has_rdma:
                cfg.warnings.append(
                    "CRITICAL: Disaggregated serving without RDMA causes severe bottleneck on TRT-LLM. "
                    "Either enable RDMA or disable prefill/decode splitting."
                )
            # Use TRT-LLM 1.1+ KV Cache Connector API
            cfg.cli_flags["enable_kv_cache_transfer"] = True

            connector = profile.topology.disagg_connector or "ucx"
            cfg.cli_flags["kv_cache_transfer_config"] = {"connector": connector, "overlap_compute": True}
            cfg.notes.append("Using TRT-LLM 1.1+ KV Cache Connector for disaggregated serving")

        # --- Build command string ---
        # Use shlex.quote so dict-valued flags (serialised to JSON) cannot
        # break shell parsing if they contain single quotes. See the
        # equivalent fix in engines/vllm.py.
        cmd_parts = ["trtllm-serve", "serve"]
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


class TRTLLMAdapter(EngineAdapter):
    """TRT-LLM engine adapter."""

    def engine_name(self) -> str:
        return "trtllm"

    async def detect_engine(self, endpoint: str) -> bool:
        _adapter_log.info("trtllm_detect_not_implemented", endpoint=endpoint)
        return False

    async def get_metrics(self, endpoint: str) -> dict[str, Any]:
        _adapter_log.warning("trtllm_metrics_not_implemented", endpoint=endpoint)
        return {}

    async def get_config(
        self,
        endpoint: str,
        *,
        allow_private: bool = True,
        auth=None,
    ) -> dict[str, Any]:
        _adapter_log.warning("trtllm_config_not_implemented", endpoint=endpoint)
        return {}
