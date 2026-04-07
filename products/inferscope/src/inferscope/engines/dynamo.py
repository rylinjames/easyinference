"""NVIDIA Dynamo engine adapter and compiler for the production InferScope lane."""

from __future__ import annotations

from typing import Any

import httpx

from inferscope.endpoint_auth import EndpointAuthConfig, build_auth_headers
from inferscope.engines.base import (
    ConfigCompiler,
    DeploymentInventory,
    EngineAdapter,
    EngineConfig,
)
from inferscope.logging import get_logger
from inferscope.optimization.serving_profile import ServingProfile
from inferscope.production_target import is_target_gpu, is_target_model

_adapter_log = get_logger(component="dynamo_adapter")

# Metric prefixes a Dynamo deployment MUST expose for InferScope to
# diagnose it. Names are taken from the authoritative Dynamo source:
# https://github.com/ai-dynamo/dynamo/blob/main/docs/observability/metrics.md
# and lib/runtime/src/metrics/prometheus_names.rs.
#
# Historical note: earlier revisions of this file used made-up prefixes
# (`dynamo_scheduler_`, `dynamo_request_`, `dynamo_prefill_`, `dynamo_decode_`)
# that don't appear anywhere in Dynamo. Those were removed — the real
# separation is frontend-vs-component, plus `dynamo_router_overhead_*`
# for routing overhead histograms. Prefill/decode roles are distinguished
# by component labels, not by metric prefix.
_REQUIRED_PRIMARY_PREFIXES = [
    "dynamo_frontend_",
    "dynamo_component_",
    "dynamo_router_overhead_",
]

_REQUIRED_CACHE_PREFIXES = [
    "lmcache:",
    # KV-stats metrics live under the broader `dynamo_component_` prefix
    # already covered by _REQUIRED_PRIMARY_PREFIXES — the real names are
    # `dynamo_component_total_blocks` and
    # `dynamo_component_gpu_cache_usage_percent`. Earlier revisions of
    # this file declared a separate `dynamo_component_kvstats_` prefix;
    # there is no such namespace in the Dynamo source. Closes the
    # snapshot v1.0.0 P0 bug `dynamo_required_metric_prefixes_fictional`.
]

# In disaggregated (split prefill/decode) topology, prefill and decode
# workers are distinguished by the `dynamo_component` label value, not
# by a dedicated metric prefix. The same `dynamo_component_*` metrics
# appear on both roles with different label values.
_REQUIRED_PREFILL_PREFIXES = [
    "dynamo_component_",
]

_REQUIRED_DECODE_PREFIXES = [
    "dynamo_component_",
]


class DynamoCompiler(ConfigCompiler):
    """Compile a normalized ServingProfile into a Dynamo deployment contract."""

    def engine_name(self) -> str:
        return "dynamo"

    def compile(self, profile: ServingProfile, inventory: DeploymentInventory) -> EngineConfig:
        cfg = EngineConfig(engine="dynamo")

        # --- Hard support gates run FIRST. On failure we return early
        # without populating cfg.command, so an unsupported config never
        # carries a misleading launch command. Closes the snapshot v1.0.0
        # P1 bug `dynamo_compiler_command_set_before_gate`.
        if not inventory.gpu_arch.startswith("sm_"):
            cfg.support_tier = "unsupported"
            cfg.support_reason = "Dynamo requires NVIDIA Hopper or Blackwell GPUs."
            cfg.warnings.append(cfg.support_reason)
            return cfg

        if not is_target_gpu(inventory.gpu_type):
            cfg.support_tier = "unsupported"
            cfg.support_reason = "Supported GPUs are limited to H100, H200, B200, and B300 variants."
            cfg.warnings.append(cfg.support_reason)
            return cfg

        if not is_target_model(profile.model):
            cfg.support_tier = "unsupported"
            cfg.support_reason = "Supported models are limited to Kimi-K2.5."
            cfg.warnings.append(cfg.support_reason)
            return cfg

        # --- Gates passed: populate command, env vars, support metadata.
        cfg.command = (
            "python -m dynamo.vllm "
            f"--model {profile.model} "
            f"--tensor-parallel-size {profile.topology.tp} "
            f"--max-num-seqs {profile.scheduler.max_num_seqs}"
        )
        cfg.env_vars["DYNAMO_CONFIG_FILE"] = "dynamo-config.yaml"
        cfg.support_tier = "supported"
        cfg.support_reason = "Dynamo + LMCache is the supported production lane for InferScope long-context coding."

        split_topology = profile.topology.split_prefill_decode
        lmcache_mode = (
            profile.cache.lmcache_mode
            if profile.cache.lmcache_mode != "disabled"
            else ("shared" if split_topology else "local")
        )
        roles = ["primary", "cache"]
        if split_topology:
            roles = ["primary", "prefill", "decode", "cache"]

        topology_mode = "prefill_decode_split" if split_topology else "single_endpoint"
        has_fast_interconnect = inventory.has_rdma or inventory.interconnect.startswith("nvlink")
        session_affinity = profile.cache.session_affinity or profile.cache.cache_backend == "lmcache"
        namespace = (
            profile.cache.lmcache_namespace
            or f"{profile.model.lower().replace('.', '-').replace(' ', '-')}-{topology_mode}"
        )

        required_metric_prefixes: dict[str, list[str]] = {
            "primary": list(_REQUIRED_PRIMARY_PREFIXES),
            "cache": list(_REQUIRED_CACHE_PREFIXES),
        }
        if split_topology:
            required_metric_prefixes["prefill"] = list(_REQUIRED_PREFILL_PREFIXES)
            required_metric_prefixes["decode"] = list(_REQUIRED_DECODE_PREFIXES)

        topology = {
            "mode": topology_mode,
            "tensor_parallel_size": profile.topology.tp,
            "pipeline_parallel_size": profile.topology.pp,
            "data_parallel_size": profile.topology.dp,
            "expert_parallel_size": profile.topology.ep,
            "split_prefill_decode": split_topology,
            "roles": roles,
        }
        backend = {
            "worker": "vllm",
            "model": profile.model,
            "quantization": profile.precision.weights,
            "kv_cache_dtype": profile.precision.kv_cache,
        }
        scheduler = {
            "max_num_batched_tokens": profile.scheduler.batched_token_budget,
            "prefill_chunk_tokens": profile.scheduler.prefill_chunk_tokens,
            "max_num_seqs": profile.scheduler.max_num_seqs,
            "decode_priority": profile.scheduler.decode_priority,
            "scheduling_policy": profile.scheduler.scheduling_policy,
            "chunked_prefill": profile.scheduler.chunked_prefill,
            "prefill_decode_isolation": profile.scheduler.prefill_decode_isolation,
        }
        router = {
            "mode": "kv_aware",
            "session_affinity": session_affinity,
            "session_header_name": "X-Session-ID",
            "request_target": "primary",
        }
        lmcache = {
            "enabled": True,
            "mode": lmcache_mode,
            "namespace": namespace,
            "session_affinity": session_affinity,
            "tiers": ["gpu_hbm"] + (["cpu_dram"] if split_topology or profile.cache.kv_tiering != "gpu_only" else []),
            "connector": profile.topology.disagg_connector or "lmcache",
            "prefix_cache_expected": profile.cache.prefix_cache,
        }
        observability = {
            "metrics_path": "/metrics",
            "health_path": "/health",
            "required_metric_prefixes": required_metric_prefixes,
            "emit_request_ids": True,
            "trace_headers": ["x-request-id", "x-session-id"],
        }
        storage = {
            "kv_tiering": profile.cache.kv_tiering,
            "gpu_memory_utilization": profile.cache.gpu_memory_utilization,
            "eviction_policy": profile.cache.eviction_policy,
        }

        cfg.cli_flags = {
            "version": 1,
            "topology": topology,
            "backend": backend,
            "scheduler": scheduler,
            "router": router,
            "lmcache": lmcache,
            "storage": storage,
            "observability": observability,
        }
        cfg.metadata = {
            "backend": backend["worker"],
            "topology": {
                "mode": topology_mode,
                "roles": roles,
                "tp": profile.topology.tp,
                "dp": profile.topology.dp,
                "ep": profile.topology.ep,
                "split_prefill_decode": split_topology,
            },
            "cache": {
                "strategy": "lmcache",
                "lmcache_mode": lmcache_mode,
                "session_affinity": session_affinity,
                "tiers": lmcache["tiers"],
                "namespace": namespace,
            },
            "observability": observability,
        }

        if split_topology and not has_fast_interconnect:
            cfg.warnings.append(
                "Disaggregated Dynamo serving without NVLink-class bandwidth or RDMA "
                "is transport-sensitive; expect higher TTFT and KV handoff variance."
            )
        if inventory.gpu_type.lower().startswith("h100 pcie") and split_topology:
            cfg.warnings.append(
                "H100 PCIe is supported for split topology, but production reliability "
                "depends on RDMA and careful prefill/decode concurrency limits."
            )
        if profile.model == "Kimi-K2.5" and inventory.gpu_arch == "sm_90a" and profile.topology.tp < 4:
            cfg.warnings.append(
                "Kimi-K2.5 on Hopper typically needs TP>=4 to retain safe KV headroom for long-context coding."
            )

        cfg.notes.extend(
            [
                "InferScope expects Dynamo to expose one stable OpenAI-compatible "
                "request endpoint plus Prometheus metrics.",
                "LMCache is the required cache backend for both single-endpoint and "
                "prefill/decode-split benchmark lanes.",
                "Observability must include frontend request metrics and cache visibility "
                "so MCP clients can explain reliability regressions.",
            ]
        )
        if split_topology:
            cfg.notes.append(
                "Split topology uses shared LMCache and should emit distinct "
                "prefill/decode metric families even when served from one aggregate "
                "metrics endpoint."
            )
        else:
            cfg.notes.append(
                "Single-endpoint topology uses local LMCache with sticky session "
                "routing to maximize long-context prefix reuse."
            )

        if inventory.has_decompression_engine:
            cfg.env_vars["DYNAMO_ENABLE_NVCOMP"] = "1"
        if inventory.has_rdma:
            cfg.env_vars["DYNAMO_KV_TRANSPORT"] = inventory.rdma_type or "rdma"
        elif split_topology:
            cfg.env_vars["DYNAMO_KV_TRANSPORT"] = "tcp"

        return cfg


class DynamoAdapter(EngineAdapter):
    """Connect to a running Dynamo deployment."""

    def engine_name(self) -> str:
        return "dynamo"

    async def detect_engine(self, endpoint: str) -> bool:
        """Detect Dynamo by checking for Dynamo-specific metrics or health response."""
        try:
            url = self._validate_endpoint(endpoint)
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/metrics")
                if "dynamo_" in resp.text or "lmcache_" in resp.text:
                    return True

                try:
                    health = await client.get(f"{url}/health")
                    if health.status_code == 200 and "dynamo" in health.text.lower():
                        return True
                except Exception:  # noqa: S110
                    pass

                try:
                    models = await client.get(f"{url}/v1/models")
                    if "x-dynamo" in {k.lower() for k in models.headers}:
                        return True
                except Exception:  # noqa: S110
                    pass

        except Exception:  # noqa: S110
            pass
        return False

    async def get_metrics(self, endpoint: str) -> dict[str, Any]:
        """Scrape Prometheus metrics from Dynamo's gateway/router."""
        url = self._validate_endpoint(endpoint)
        metrics: dict[str, Any] = {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{url}/metrics")
                for line in resp.text.splitlines():
                    if line.startswith("#"):
                        continue
                    if any(prefix in line for prefix in ("dynamo_", "lmcache_", "vllm:")):
                        parts = line.split()
                        if len(parts) >= 2:
                            name = parts[0].split("{")[0]
                            try:
                                metrics[name] = float(parts[-1])
                            except ValueError:
                                metrics[name] = parts[-1]
        except Exception:  # noqa: S110
            _adapter_log.warning("dynamo_metrics_scrape_failed", endpoint=endpoint)
        return metrics

    async def get_config(
        self,
        endpoint: str,
        *,
        allow_private: bool = True,
        auth: EndpointAuthConfig | None = None,
    ) -> dict[str, Any]:
        """Get deployment info from Dynamo's management API."""
        try:
            url = self._validate_endpoint(endpoint, allow_private=allow_private)
            async with httpx.AsyncClient(timeout=10.0) as client:
                models_resp = await client.get(f"{url}/v1/models", headers=build_auth_headers(auth))
                models_payload: dict[str, Any] = models_resp.json()

                status_payload: dict[str, Any] = {}
                try:
                    status = await client.get(f"{url}/v1/status", headers=build_auth_headers(auth))
                    if status.status_code == 200:
                        status_payload = status.json()
                except Exception:  # noqa: S110
                    pass

                observed_model = ""
                data = models_payload.get("data") if isinstance(models_payload, dict) else None
                if isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict):
                        observed_model = str(first.get("id") or "")

                topology = status_payload.get("topology") if isinstance(status_payload, dict) else {}
                lmcache = status_payload.get("lmcache") if isinstance(status_payload, dict) else {}
                observability = status_payload.get("observability") if isinstance(status_payload, dict) else {}

                return {
                    "engine": "dynamo",
                    "model": observed_model,
                    "models": models_payload,
                    "backend": status_payload.get("backend", "vllm") if isinstance(status_payload, dict) else "vllm",
                    "topology": topology if isinstance(topology, dict) else {},
                    "lmcache": lmcache if isinstance(lmcache, dict) else {},
                    "observability": observability if isinstance(observability, dict) else {},
                    "status": status_payload,
                }
        except Exception:  # noqa: S110
            _adapter_log.warning("dynamo_config_fetch_failed", endpoint=endpoint)
            return {}
