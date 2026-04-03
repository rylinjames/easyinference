"""Normalized ServingProfile — the single object InferScope optimizes.

InferScope is now productized around the Dynamo + LMCache long-context coding lane,
but the normalized profile remains the main handoff between policy and compiler code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any


class WorkloadMode(StrEnum):
    """Detected or declared workload pattern."""

    CODING = "coding"
    CHAT = "chat"
    AGENT = "agent"
    LONG_CONTEXT_RAG = "long_context_rag"


class BottleneckType(StrEnum):
    """Diagnosed performance bottleneck category."""

    PREFILL_COMPUTE = "prefill_compute_bound"
    DECODE_MEMORY = "decode_memory_bound"
    CACHE_BOUND = "cache_bound"
    INTERCONNECT_BOUND = "interconnect_bound"
    MOE_ROUTING = "moe_routing_bound"
    SCHEDULER_BOUND = "scheduler_bound"
    MISCONFIGURATION = "misconfiguration_bound"


class ModelClass(Enum):
    """Model architecture class — InferScope tunes per class, not per model name."""

    DENSE_GQA = "dense_gqa"
    QWEN35_HYBRID = "qwen35_hybrid"
    FRONTIER_MLA_MOE = "frontier_mla_moe"
    COMPACT_AGENTIC_MOE = "compact_agentic_moe"
    CLASSICAL_MOE = "classical_moe"


class EngineType(Enum):
    """Supported inference engines."""

    VLLM = "vllm"
    SGLANG = "sglang"
    ATOM = "atom"
    TRTLLM = "trtllm"
    DYNAMO = "dynamo"


@dataclass
class ObjectiveSpec:
    """SLO targets for optimization."""

    ttft_p95_ms: float = 0.0  # 0 = not constrained
    itl_p95_ms: float = 0.0
    throughput_min_tps: float = 0.0
    cost_weight: float = 0.5  # 0=perf only, 1=cost only


@dataclass
class TopologySpec:
    """Parallelism strategy."""

    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
    split_prefill_decode: bool = False
    disagg_connector: str = ""  # NixlConnector | LMCacheConnectorV1 | P2pNcclConnector | ""
    enable_eplb: bool = False  # Expert Parallel Load Balancing (hot-expert replication)


@dataclass
class SchedulerSpec:
    """Scheduling parameters."""

    batched_token_budget: int = 8192
    prefill_chunk_tokens: int = 8192
    max_num_seqs: int = 256
    decode_priority: float = 0.5  # 0=prefill-first, 1=decode-first
    scheduling_policy: str = "fcfs"
    enable_moe_overlap: bool = False  # Overlap MoE dispatch/combine with compute
    chunked_prefill: bool = True  # Enabled by default

    # Prefill/decode isolation policy
    prefill_decode_isolation: str = "colocated"  # colocated | soft_priority | hard_lane
    prefill_lane_budget: int = 0  # Max tokens for prefill per step (0 = shared budget)
    decode_lane_budget: int = 0  # Max tokens for decode per step
    co_batch_utilization_threshold: float = 0.6  # GPU util below which co-batching is permitted
    max_prefill_chunk_ratio: float = 0.5  # Max fraction of batch budget a single prefill can consume


@dataclass
class CacheSpec:
    """KV cache configuration."""

    prefix_cache: bool = True
    cache_backend: str = "lmcache"  # lmcache | none
    lmcache_mode: str = "local"  # disabled | local | shared
    lmcache_namespace: str = ""
    kv_tiering: str = "gpu_only"  # gpu_only | gpu_cpu | gpu_cpu_ssd
    eviction_policy: str = "lru"
    session_affinity: bool = False
    gpu_memory_utilization: float = 0.92
    hybrid_block_sizes: list[int] | None = None  # Dynamic block sizes for hybrid attention (e.g. Qwen3.5)

    # KV cache pressure and fragmentation management
    kv_compaction_trigger: float = 0.4  # Trigger compaction when block utilization below this
    fragmentation_check: bool = False  # Enable fragmentation monitoring recommendation

    # PCIe-aware offloading policy
    offload_policy: str = "cold_only"  # disabled | cold_only | aggressive
    offload_idle_threshold_s: float = 30.0  # Seconds of inactivity before KV is offload-eligible
    pcie_utilization_cap: float = 0.7  # Disable offloading above this PCIe utilization

    # PagedAttention block reuse
    block_reuse_strategy: str = "prefix_sharing"  # none | prefix_sharing | prefix_sharing_cross_session


@dataclass
class PrecisionSpec:
    """Quantization and numeric precision."""

    weights: str = "fp16"  # fp16 | bf16 | fp8 | fp4 | int8 | int4 | awq | gptq
    activations: str = "fp16"
    kv_cache: str = "auto"  # auto | fp16 | fp8_e4m3 | fp8_e5m2


@dataclass
class SpeculationSpec:
    """Speculative decoding config."""

    mode: str = "off"  # off | low_batch_only | always
    method: str = ""  # eagle | eagle3 | mtp | ngram
    num_speculative_tokens: int = 5
    model: str = ""


@dataclass
class ServingProfile:
    """The normalized object InferScope optimizes.

    Engine-specific compilers translate this to vLLM/SGLang/ATOM/TRT-LLM/Dynamo flags.
    All optimization logic operates on this object, never on raw engine flags.
    """

    # Context
    model: str = ""
    model_class: ModelClass = ModelClass.DENSE_GQA
    engine: EngineType = EngineType.DYNAMO
    gpu_type: str = ""
    num_gpus: int = 1

    # Optimization targets
    workload_mode: WorkloadMode = WorkloadMode.CODING
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)

    # Configuration
    topology: TopologySpec = field(default_factory=TopologySpec)
    scheduler: SchedulerSpec = field(default_factory=SchedulerSpec)
    cache: CacheSpec = field(default_factory=CacheSpec)
    precision: PrecisionSpec = field(default_factory=PrecisionSpec)
    speculation: SpeculationSpec = field(default_factory=SpeculationSpec)

    # Engine-specific overrides (populated by compiler)
    engine_flags: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # DAG Execution Trace for MCP Agent transparency
    reasoning_trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "model": self.model,
            "model_class": self.model_class.value,
            "engine": self.engine.value,
            "gpu_type": self.gpu_type,
            "num_gpus": self.num_gpus,
            "workload_mode": self.workload_mode.value,
            "objective": {
                "ttft_p95_ms": self.objective.ttft_p95_ms,
                "itl_p95_ms": self.objective.itl_p95_ms,
                "throughput_min_tps": self.objective.throughput_min_tps,
                "cost_weight": self.objective.cost_weight,
            },
            "topology": {
                "tp": self.topology.tp,
                "pp": self.topology.pp,
                "dp": self.topology.dp,
                "ep": self.topology.ep,
                "split_prefill_decode": self.topology.split_prefill_decode,
                "disagg_connector": self.topology.disagg_connector,
                "enable_eplb": self.topology.enable_eplb,
            },
            "scheduler": {
                "batched_token_budget": self.scheduler.batched_token_budget,
                "prefill_chunk_tokens": self.scheduler.prefill_chunk_tokens,
                "max_num_seqs": self.scheduler.max_num_seqs,
                "decode_priority": self.scheduler.decode_priority,
                "scheduling_policy": self.scheduler.scheduling_policy,
                "enable_moe_overlap": self.scheduler.enable_moe_overlap,
                "chunked_prefill": self.scheduler.chunked_prefill,
                "prefill_decode_isolation": self.scheduler.prefill_decode_isolation,
                "prefill_lane_budget": self.scheduler.prefill_lane_budget,
                "decode_lane_budget": self.scheduler.decode_lane_budget,
                "co_batch_utilization_threshold": self.scheduler.co_batch_utilization_threshold,
                "max_prefill_chunk_ratio": self.scheduler.max_prefill_chunk_ratio,
            },
            "cache": {
                "prefix_cache": self.cache.prefix_cache,
                "cache_backend": self.cache.cache_backend,
                "lmcache_mode": self.cache.lmcache_mode,
                "lmcache_namespace": self.cache.lmcache_namespace,
                "kv_tiering": self.cache.kv_tiering,
                "eviction_policy": self.cache.eviction_policy,
                "session_affinity": self.cache.session_affinity,
                "gpu_memory_utilization": self.cache.gpu_memory_utilization,
                "hybrid_block_sizes": self.cache.hybrid_block_sizes,
                "kv_compaction_trigger": self.cache.kv_compaction_trigger,
                "fragmentation_check": self.cache.fragmentation_check,
                "offload_policy": self.cache.offload_policy,
                "offload_idle_threshold_s": self.cache.offload_idle_threshold_s,
                "pcie_utilization_cap": self.cache.pcie_utilization_cap,
                "block_reuse_strategy": self.cache.block_reuse_strategy,
            },
            "precision": {
                "weights": self.precision.weights,
                "activations": self.precision.activations,
                "kv_cache": self.precision.kv_cache,
            },
            "speculation": {
                "mode": self.speculation.mode,
                "method": self.speculation.method,
                "num_speculative_tokens": self.speculation.num_speculative_tokens,
                "model": self.speculation.model,
            },
            "engine_flags": self.engine_flags,
            "env_vars": self.env_vars,
            "warnings": self.warnings,
            "reasoning_trace": self.reasoning_trace,
        }
