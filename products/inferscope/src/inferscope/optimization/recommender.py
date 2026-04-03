"""Recommendation engine — Dynamo-first serving profiles for the production lane."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TypeVar

from inferscope.engines.base import DeploymentInventory, EngineConfig
from inferscope.engines.registry import get_compiler
from inferscope.hardware.gpu_profiles import GPUProfile
from inferscope.logging import get_logger
from inferscope.models.registry import ModelVariant
from inferscope.optimization.memory_planner import MemoryPlan, plan_memory
from inferscope.optimization.platform_policy import (
    EngineSupportTier,
    PlatformTraits,
    resolve_engine_support,
    resolve_platform_traits,
    resolve_preferred_precision,
    resolve_preferred_tp,
)
from inferscope.optimization.serving_profile import (
    CacheSpec,
    EngineType,
    ObjectiveSpec,
    PrecisionSpec,
    SchedulerSpec,
    ServingProfile,
    SpeculationSpec,
    TopologySpec,
    WorkloadMode,
)
from inferscope.production_target import is_target_gpu, is_target_model, target_profile_summary
from inferscope.profiling import ProfilingIntent, resolve_profiling_intent

_T = TypeVar("_T")


def _require(value: _T | None, name: str) -> _T:
    """Return a required pipeline value or raise a deterministic error."""
    if value is None:
        raise ValueError(f"Internal recommender error: missing {name}.")
    return value


@dataclass
class PipelineContext:
    """State object passed along the Recommender DAG edges."""

    model: ModelVariant
    gpu: GPUProfile
    num_gpus: int
    workload: WorkloadMode
    objective: ObjectiveSpec
    forced_engine: str = "auto"

    platform_traits: PlatformTraits | None = None
    engine_type: EngineType | None = None
    precision: PrecisionSpec | None = None
    topology: TopologySpec | None = None
    scheduler: SchedulerSpec | None = None
    cache: CacheSpec | None = None
    speculation: SpeculationSpec | None = None
    profile: ServingProfile | None = None
    engine_config: EngineConfig | None = None
    memory_plan: MemoryPlan | None = None
    profiling_intent: ProfilingIntent | None = None
    reasoning_trace: list[str] = field(default_factory=list)


class DAGNode(abc.ABC):
    """Abstract base class for a recommendation pipeline node."""

    @abc.abstractmethod
    def process(self, ctx: PipelineContext) -> None:
        pass


class HardwareNode(DAGNode):
    """Validate target-scope hardware and resolve the only supported engine."""

    def process(self, ctx: PipelineContext) -> None:
        ctx.platform_traits = resolve_platform_traits(ctx.gpu)
        ctx.reasoning_trace.append(target_profile_summary())
        ctx.reasoning_trace.append(
            f"HardwareNode: Detected {ctx.gpu.name} ({ctx.gpu.compute_capability}) with {ctx.gpu.memory_gb}GB HBM."
        )

        if not is_target_model(ctx.model):
            raise ValueError("Supported models are limited to Kimi-K2.5.")
        if not is_target_gpu(ctx.gpu):
            raise ValueError("Supported GPUs are limited to H100, H200, B200, and B300 variants.")
        if ctx.workload not in {WorkloadMode.CODING, WorkloadMode.CHAT}:
            raise ValueError("Supported workloads are coding and chat.")
        if ctx.forced_engine not in {"auto", "dynamo", "vllm"}:
            raise ValueError("InferScope's production lane supports Dynamo serving plus vLLM comparison benchmarks.")

        ctx.engine_type = EngineType.VLLM if ctx.forced_engine == "vllm" else EngineType.DYNAMO
        support = resolve_engine_support(ctx.engine_type, ctx.gpu, multi_node=ctx.num_gpus > 1)
        if support.tier == EngineSupportTier.UNSUPPORTED:
            raise ValueError(support.reason)
        ctx.reasoning_trace.append(f"HardwareNode: {support.reason}")

        ctx.precision, precision_reason = resolve_preferred_precision(
            ctx.model,
            ctx.gpu,
            ctx.workload,
            num_gpus=ctx.num_gpus,
        )
        ctx.reasoning_trace.append(f"HardwareNode: {precision_reason}")


class ModelNode(DAGNode):
    """Resolve tensor parallelism and the base single-endpoint topology."""

    def process(self, ctx: PipelineContext) -> None:
        precision = _require(ctx.precision, "precision")

        tp, tp_reason = resolve_preferred_tp(
            ctx.model,
            ctx.gpu,
            ctx.num_gpus,
            precision.weights,
            ctx.workload,
        )
        if tp is None:
            raise ValueError(tp_reason or "Unable to derive a tensor-parallel plan that fits in memory.")

        dp = max(1, ctx.num_gpus // tp)
        ctx.topology = TopologySpec(
            tp=tp,
            pp=1,
            dp=dp,
            ep=1,
            split_prefill_decode=False,
            disagg_connector="lmcache",
        )
        ctx.reasoning_trace.append(f"ModelNode: {tp_reason or f'Resolved TP={tp}.'}")
        ctx.reasoning_trace.append(
            "ModelNode: Single-endpoint production default uses "
            f"TP={tp}, DP={dp}, EP=1; split topology is a benchmark override, "
            "not the base recommendation."
        )

        ctx.speculation = SpeculationSpec()


class WorkloadNode(DAGNode):
    """Apply coding-specific scheduler and LMCache defaults."""

    def process(self, ctx: PipelineContext) -> None:
        platform_traits = _require(ctx.platform_traits, "platform_traits")

        if platform_traits.is_hopper_pcie:
            high_util = 0.90
        elif platform_traits.is_blackwell or platform_traits.is_h200:
            high_util = 0.94
        else:
            high_util = 0.92
        token_budget = self._derive_token_budget(ctx.gpu.memory_bandwidth_tb_s)
        max_num_seqs = 64 if ctx.model.name == "Kimi-K2.5" else 96
        namespace = self._namespace(ctx)

        ctx.scheduler = SchedulerSpec(
            batched_token_budget=token_budget,
            prefill_chunk_tokens=token_budget,
            max_num_seqs=max_num_seqs,
            decode_priority=0.75,
            scheduling_policy="latency_guarded_fcfs",
            chunked_prefill=not platform_traits.is_hopper_pcie,
            prefill_decode_isolation="soft_priority",
            max_prefill_chunk_ratio=0.45,
        )
        ctx.cache = CacheSpec(
            prefix_cache=True,
            cache_backend="lmcache",
            lmcache_mode="local",
            lmcache_namespace=namespace,
            kv_tiering="gpu_only",
            eviction_policy="lru",
            session_affinity=True,
            gpu_memory_utilization=high_util,
            fragmentation_check=True,
            offload_policy="disabled",
            block_reuse_strategy="prefix_sharing_cross_session",
        )
        ctx.reasoning_trace.append(
            "WorkloadNode: Coding defaults set LMCache local mode, sticky sessions, "
            f"token budget={token_budget}, max_num_seqs={max_num_seqs}, util={high_util:.2f}."
        )
        if platform_traits.is_hopper_pcie:
            ctx.reasoning_trace.append(
                "WorkloadNode: H100 PCIe keeps a lower HBM utilization target "
                "to protect reliability under long-context coding spikes."
            )
        if platform_traits.is_blackwell:
            ctx.reasoning_trace.append(
                "WorkloadNode: Blackwell enables higher HBM utilization and is "
                "the preferred path for Kimi FP4 + LMCache density."
            )

    @staticmethod
    def _derive_token_budget(memory_bandwidth_tb_s: float) -> int:
        if memory_bandwidth_tb_s >= 7.5:
            return 32768
        if memory_bandwidth_tb_s >= 4.0:
            return 24576
        return 16384

    @staticmethod
    def _namespace(ctx: PipelineContext) -> str:
        model_key = ctx.model.name.lower().replace(".", "-")
        gpu_key = ctx.gpu.name.lower().replace(" ", "-")
        return f"{model_key}-{gpu_key}-coding"


class ProfilingNode(DAGNode):
    """Attach advisory profiling intent for future kernel/profiler integrations."""

    def process(self, ctx: PipelineContext) -> None:
        ctx.profiling_intent = resolve_profiling_intent(ctx.gpu.vendor)
        ctx.reasoning_trace.append(ctx.profiling_intent.summary)


class TelemetryNode(DAGNode):
    """Describe the observability bias of the production lane."""

    def process(self, ctx: PipelineContext) -> None:
        ctx.reasoning_trace.append(
            "TelemetryNode: Production recommendations assume Prometheus coverage "
            "for primary request flow plus LMCache metrics and request/session identifiers."
        )


class CompilerNode(DAGNode):
    """Finalize the profile and bind it to the selected engine compiler."""

    def process(self, ctx: PipelineContext) -> None:
        engine_type = _require(ctx.engine_type, "engine_type")
        topology = _require(ctx.topology, "topology")
        scheduler = _require(ctx.scheduler, "scheduler")
        cache = _require(ctx.cache, "cache")
        precision = _require(ctx.precision, "precision")
        speculation = _require(ctx.speculation, "speculation")

        ctx.profile = ServingProfile(
            model=ctx.model.name,
            model_class=ctx.model.model_class,
            engine=engine_type,
            gpu_type=ctx.gpu.name,
            num_gpus=ctx.num_gpus,
            workload_mode=ctx.workload,
            objective=ctx.objective,
            topology=topology,
            scheduler=scheduler,
            cache=cache,
            precision=precision,
            speculation=speculation,
            reasoning_trace=ctx.reasoning_trace,
        )

        inventory = _build_inventory(ctx)
        compiler = get_compiler(engine_type.value)
        ctx.engine_config = compiler.compile(ctx.profile, inventory)
        support = resolve_engine_support(
            engine_type,
            ctx.gpu,
            multi_node=topology.split_prefill_decode or ctx.num_gpus > 1,
        )
        ctx.engine_config.support_tier = support.tier.value
        ctx.engine_config.support_reason = support.reason
        if support.tier == EngineSupportTier.UNSUPPORTED:
            raise ValueError(support.reason)

        ctx.memory_plan = plan_memory(
            model=ctx.model,
            gpu=ctx.gpu,
            num_gpus=ctx.num_gpus,
            tp=topology.tp,
            precision=precision.weights,
            kv_precision=precision.kv_cache,
            gpu_memory_utilization=cache.gpu_memory_utilization,
        )
        if not ctx.memory_plan.fits:
            raise ValueError(
                f"{ctx.model.name} does not retain enough KV headroom on {ctx.gpu.name} "
                f"with TP={topology.tp} {precision.weights}."
            )

        ctx.profile.engine_flags = ctx.engine_config.cli_flags
        ctx.profile.env_vars = ctx.engine_config.env_vars
        ctx.profile.warnings.extend(ctx.engine_config.warnings)
        ctx.reasoning_trace.append(
            f"CompilerNode: Bound the production profile to a {engine_type.value} engine config with LMCache metadata."
        )


def _build_inventory(ctx: PipelineContext) -> DeploymentInventory:
    platform_traits = _require(ctx.platform_traits, "platform_traits")
    return DeploymentInventory(
        gpu_type=ctx.gpu.name,
        gpu_arch=ctx.gpu.compute_capability,
        gpu_count=ctx.num_gpus,
        gpu_memory_gb=ctx.gpu.memory_gb,
        gpu_memory_bandwidth_tb_s=ctx.gpu.memory_bandwidth_tb_s,
        interconnect=(
            f"nvlink{ctx.gpu.nvlink_version}"
            if ctx.gpu.nvlink_version
            else f"infinity_fabric_{ctx.gpu.infinity_fabric_version}"
            if ctx.gpu.infinity_fabric_version
            else ctx.gpu.pcie
        ),
        interconnect_bandwidth_gb_s=ctx.gpu.nvlink_bandwidth_gb_s or ctx.gpu.if_bandwidth_gb_s,
        fp8_support=ctx.gpu.fp8_support,
        fp4_support=ctx.gpu.fp4_support,
        fp8_format=ctx.gpu.fp8_format,
        platform_family=platform_traits.family.value,
        has_grace=bool(platform_traits.is_grace),
        grace_memory_gb=platform_traits.grace_memory_gb,
        grace_memory_bandwidth_gb_s=platform_traits.grace_memory_bandwidth_gb_s,
        c2c_bandwidth_gb_s=platform_traits.c2c_bandwidth_gb_s,
        has_decompression_engine=bool(platform_traits.has_decompression_engine),
        has_helix_parallelism=bool(platform_traits.has_helix_parallelism),
        has_accelerated_softmax=bool(platform_traits.has_accelerated_softmax),
        platform_features=platform_traits.to_dict(),
    )


def recommend(
    model: ModelVariant,
    gpu: GPUProfile,
    num_gpus: int = 1,
    workload: WorkloadMode = WorkloadMode.CODING,
    engine: str = "auto",
    objective: ObjectiveSpec | None = None,
) -> tuple[ServingProfile, EngineConfig, MemoryPlan]:
    """Execute the modular recommendation DAG."""
    ctx = PipelineContext(
        model=model,
        gpu=gpu,
        num_gpus=num_gpus,
        workload=workload,
        objective=objective or ObjectiveSpec(),
        forced_engine=engine,
    )

    nodes: list[DAGNode] = [
        HardwareNode(),
        ModelNode(),
        WorkloadNode(),
        ProfilingNode(),
        TelemetryNode(),
        CompilerNode(),
    ]

    for node in nodes:
        node.process(ctx)

    profile = _require(ctx.profile, "profile")
    engine_config = _require(ctx.engine_config, "engine_config")
    memory_plan = _require(ctx.memory_plan, "memory_plan")
    engine_type = _require(ctx.engine_type, "engine_type")
    topology = _require(ctx.topology, "topology")

    log = get_logger(component="recommender")
    log.info(
        "recommendation_compiled_via_dag",
        model=ctx.model.name,
        gpu=ctx.gpu.name,
        engine=engine_type.value,
        tp=topology.tp,
        trace_length=len(ctx.reasoning_trace),
    )

    return profile, engine_config, memory_plan
