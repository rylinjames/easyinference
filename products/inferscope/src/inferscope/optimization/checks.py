"""31 audit checks for live inference deployments.

Each check takes normalized metrics + deployment context and returns
an AuditFinding if the check fires. All checks are ISA-grounded —
they reference specific GPU capabilities, not generic advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferscope.telemetry.normalizer import NormalizedMetrics


@dataclass
class AuditFinding:
    """Structured finding from an audit check."""

    check_id: str
    severity: str  # "critical" | "warning" | "info"
    title: str
    description: str
    current_value: str
    recommended_value: str
    fix_command: str
    confidence: float  # 0-1
    evidence: str  # "threshold_rule" | "metric_correlation" | "hardware_constraint"
    gpu_specific: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "fix_command": self.fix_command,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "gpu_specific": self.gpu_specific,
        }


@dataclass
class DeploymentContext:
    """What we know about the deployment (from config + detection)."""

    engine: str = ""  # vllm | sglang | atom
    gpu_arch: str = ""  # sm_80, sm_90a, gfx942, gfx950, etc.
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_vendor: str = ""  # nvidia | amd
    model_name: str = ""
    model_type: str = ""  # dense | moe
    attention_type: str = ""  # GQA | MLA | MHA
    experts_total: int = 0
    tp: int = 1
    ep: int = 0
    fp8_support: bool = False
    fp8_format: str = ""  # OCP | FNUZ
    # Config flags (detected or declared)
    gpu_memory_utilization: float = 0.0
    kv_cache_dtype: str = ""
    quantization: str = ""
    block_size: int = 0
    env_vars: dict[str, str] = field(default_factory=dict)
    has_rdma: bool = False
    split_prefill_decode: bool = False
    multi_node: bool = False
    prefix_caching: bool = True
    max_num_batched_tokens: int = 0


def run_all_checks(
    metrics: NormalizedMetrics,
    ctx: DeploymentContext,
) -> list[AuditFinding]:
    """Run all 31 audit checks and return findings that fire."""
    findings = []
    for check_fn in _ALL_CHECKS:
        result = check_fn(metrics, ctx)
        if result is not None:
            findings.append(result)
    # Sort by severity: critical first, then warning, then info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f.severity, 3))
    return findings


# =============================================================================
# CHECK IMPLEMENTATIONS
# =============================================================================


def _check_kv_preemption_storm(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#3: KV_PREEMPTION_STORM — preemptions indicate KV cache thrashing.

    `preemptions_total` is a monotonic Prometheus counter, so comparing it
    against a raw threshold false-positives on any long-running deployment.
    Instead, compare against the successful-request counter to get a rate —
    the same pattern `normalizer.py:_compute_goodput` already uses.
    """
    if m.request_success_total <= 0 or m.preemptions_total <= 0:
        return None
    preemption_rate = m.preemptions_total / m.request_success_total
    if preemption_rate > 0.02:
        return AuditFinding(
            check_id="KV_PREEMPTION_STORM",
            severity="critical",
            title="KV cache preemption storm detected",
            description=(
                f"Preemption rate is {preemption_rate:.1%} "
                f"({m.preemptions_total:.0f} preemptions across {m.request_success_total:.0f} "
                "successful requests). Requests are being evicted from KV cache under memory "
                "pressure, causing recomputation and latency spikes."
            ),
            current_value=f"{preemption_rate:.1%} preemption rate",
            recommended_value="<1% preemption rate",
            fix_command=(
                "Lower gpu_memory_utilization by 2-3 points, reduce max_model_len, or add replicas to spread load"
            ),
            confidence=0.9,
            evidence="threshold_rule",
        )
    return None


def _check_missing_quantization(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#4: MISSING_QUANTIZATION — BF16/FP16 on FP8-capable GPU."""
    if ctx.fp8_support and ctx.quantization in ("bf16", "fp16", "auto", ""):
        return AuditFinding(
            check_id="MISSING_QUANTIZATION",
            severity="warning",
            title="Running BF16/FP16 on FP8-capable GPU",
            description=(
                f"GPU {ctx.gpu_name} supports native FP8 ({ctx.fp8_format}), but model is "
                f"running in {ctx.quantization or 'auto/BF16'}. FP8 halves memory and nearly "
                "doubles throughput with negligible accuracy loss."
            ),
            current_value=ctx.quantization or "bf16 (default)",
            recommended_value="fp8",
            fix_command="--quantization fp8 --kv-cache-dtype fp8_e4m3",
            confidence=0.85,
            evidence="hardware_constraint",
            gpu_specific=True,
        )
    return None


def _check_prefix_cache_disabled(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#7: PREFIX_CACHE_DISABLED — prefix reuse potential but caching off."""
    if not ctx.prefix_caching and m.prefix_cache_hit_rate == 0:
        return AuditFinding(
            check_id="PREFIX_CACHE_DISABLED",
            severity="warning",
            title="Prefix caching appears disabled",
            description=(
                "No prefix cache hits detected. vLLM V1 has zero-overhead prefix caching "
                "(always on). SGLang needs --enable-metrics to expose hit rate. "
                "Coding/agent workloads typically see 50-95% prefix reuse."
            ),
            current_value="0% hit rate (caching likely off)",
            recommended_value=">50% for coding/agent workloads",
            fix_command="Prefix caching is free in vLLM V1. For SGLang: --schedule-policy lpm",
            confidence=0.7,
            evidence="threshold_rule",
        )
    return None


def _check_batch_size_mismatch(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#8: BATCH_SIZE_MISMATCH — batch budget too low or high for workload."""
    if ctx.max_num_batched_tokens > 0 and m.requests_waiting > 20 and ctx.max_num_batched_tokens < 4096:
        return AuditFinding(
            check_id="BATCH_SIZE_MISMATCH",
            severity="warning",
            title="Batched token budget too low for queue depth",
            description=(
                f"Queue depth is {m.requests_waiting:.0f} but max_num_batched_tokens is only "
                f"{ctx.max_num_batched_tokens}. Raise the budget to improve throughput."
            ),
            current_value=f"max_num_batched_tokens={ctx.max_num_batched_tokens}",
            recommended_value="8192-16384 for throughput workloads",
            fix_command=f"--max-num-batched-tokens 8192 (currently {ctx.max_num_batched_tokens})",
            confidence=0.75,
            evidence="metric_correlation",
        )
    return None


def _check_kv_dtype_suboptimal(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#11: KV_DTYPE_SUBOPTIMAL — FP16 KV cache on FP8-capable GPU."""
    if ctx.fp8_support and ctx.kv_cache_dtype in ("auto", "fp16", ""):
        return AuditFinding(
            check_id="KV_DTYPE_SUBOPTIMAL",
            severity="warning",
            title="KV cache not using FP8 on FP8-capable GPU",
            description=(
                "FP8 KV cache halves memory footprint — the single highest-impact optimization "
                "for long context. This GPU supports it natively."
            ),
            current_value=f"kv_cache_dtype={ctx.kv_cache_dtype or 'auto (fp16)'}",
            recommended_value="fp8_e4m3",
            fix_command="--kv-cache-dtype fp8_e4m3",
            confidence=0.9,
            evidence="hardware_constraint",
            gpu_specific=True,
        )
    return None


def _check_aiter_disabled(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#12: AITER_DISABLED — AMD GPU without AITER env vars."""
    if ctx.gpu_vendor == "amd" and ctx.env_vars.get("VLLM_ROCM_USE_AITER") != "1":
        return AuditFinding(
            check_id="AITER_DISABLED",
            severity="critical",
            title="AITER disabled on AMD GPU — 2-10x performance left on table",
            description=(
                f"Running on {ctx.gpu_name} without VLLM_ROCM_USE_AITER=1. "
                "AITER provides hand-tuned ASM/CK kernels that are 2-10x faster than defaults."
            ),
            current_value="VLLM_ROCM_USE_AITER not set",
            recommended_value="VLLM_ROCM_USE_AITER=1",
            fix_command="export VLLM_ROCM_USE_AITER=1 HIP_FORCE_DEV_KERNARG=1 TORCH_BLAS_PREFER_HIPBLASLT=1",
            confidence=0.95,
            evidence="hardware_constraint",
            gpu_specific=True,
        )
    return None


def _check_block_size_wrong(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#13: BLOCK_SIZE_WRONG — MLA model without block-size 1 on ROCm."""
    if ctx.attention_type == "MLA" and ctx.gpu_vendor == "amd" and ctx.engine in ("vllm",) and ctx.block_size != 1:
        return AuditFinding(
            check_id="BLOCK_SIZE_WRONG",
            severity="critical",
            title="DeepSeek MLA model needs --block-size 1 on ROCm",
            description=(
                "MLA attention on ROCm requires block-size=1 for correct results. "
                "Without this, you get incorrect outputs or crashes."
            ),
            current_value=f"block_size={ctx.block_size or 'default (16)'}",
            recommended_value="1",
            fix_command="--block-size 1",
            confidence=0.95,
            evidence="hardware_constraint",
            gpu_specific=True,
        )
    return None


def _check_memory_util_low(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#14: MEMORY_UTIL_LOW — gpu_memory_utilization below 0.90 in production."""
    if 0 < ctx.gpu_memory_utilization < 0.90:
        return AuditFinding(
            check_id="MEMORY_UTIL_LOW",
            severity="info",
            title="GPU memory utilization is conservative",
            description=(
                f"gpu_memory_utilization={ctx.gpu_memory_utilization:.2f}. "
                "Production workloads can safely push to 0.92-0.95 for more KV cache headroom."
            ),
            current_value=f"{ctx.gpu_memory_utilization:.2f}",
            recommended_value="0.92-0.95",
            fix_command=f"--gpu-memory-utilization 0.93 (currently {ctx.gpu_memory_utilization:.2f})",
            confidence=0.7,
            evidence="threshold_rule",
        )
    return None


def _check_speculative_overhead(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#15: SPECULATIVE_OVERHEAD — acceptance rate too low at high concurrency."""
    if m.spec_acceptance_rate > 0 and m.spec_acceptance_rate < 0.55 and m.requests_running > 30:
        return AuditFinding(
            check_id="SPECULATIVE_OVERHEAD",
            severity="warning",
            title="Speculative decoding overhead at high concurrency",
            description=(
                f"Speculation acceptance rate is {m.spec_acceptance_rate:.0%} with "
                f"{m.requests_running:.0f} concurrent requests. Below 55% at high concurrency, "
                "spec decode wastes more compute than it saves."
            ),
            current_value=f"{m.spec_acceptance_rate:.0%} acceptance at {m.requests_running:.0f} concurrent",
            recommended_value=">55% or disable spec decode",
            fix_command="Remove --speculative-model or limit to low-concurrency pools",
            confidence=0.8,
            evidence="metric_correlation",
        )
    return None


def _check_moe_ep_missing(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#16: MOE_EP_MISSING — large MoE without expert parallelism."""
    if ctx.model_type == "moe" and ctx.experts_total > 64 and ctx.ep <= 1 and ctx.tp > 1:
        return AuditFinding(
            check_id="MOE_EP_MISSING",
            severity="warning",
            title="Large MoE model without Expert Parallelism",
            description=(
                f"Model has {ctx.experts_total} experts but EP is not enabled (only TP={ctx.tp}). "
                "Expert Parallelism distributes expert weights across GPUs for better utilization."
            ),
            current_value=f"TP={ctx.tp}, EP={ctx.ep or 0}",
            recommended_value=f"TP={ctx.tp // 2}, EP=2 (or similar EP>1 split)",
            fix_command="--enable-expert-parallel (vLLM) or -ep 2 (ATOM)",
            confidence=0.75,
            evidence="threshold_rule",
        )
    return None


def _check_atom_not_used(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#17: ATOM_NOT_USED — MI355X with MLA model not using ATOM."""
    if ctx.gpu_arch == "gfx950" and ctx.attention_type == "MLA" and ctx.engine not in ("atom",):
        return AuditFinding(
            check_id="ATOM_NOT_USED",
            severity="warning",
            title="MI355X deployment not using ATOM for MLA model",
            description=(
                "ATOM is purpose-built for MLA+MoE on MI355X with up to 17x MLA decode speedup. "
                f"Currently using {ctx.engine}."
            ),
            current_value=f"engine={ctx.engine}",
            recommended_value="ATOM standalone or --model-impl atom",
            fix_command="python -m atom.entrypoints.openai_server --model <MODEL> --kv_cache_dtype fp8",
            confidence=0.8,
            evidence="hardware_constraint",
            gpu_specific=True,
        )
    return None


def _check_wrong_attention_backend(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#18: WRONG_ATTENTION_BACKEND — MLA model on FA or vice versa."""
    if ctx.gpu_vendor == "amd":
        has_mla_backend = ctx.env_vars.get("ROCM_AITER_MLA") == "1"
        has_fa_backend = ctx.env_vars.get("ROCM_AITER_FA") == "1"
        if ctx.attention_type == "MLA" and not has_mla_backend and has_fa_backend:
            return AuditFinding(
                check_id="WRONG_ATTENTION_BACKEND",
                severity="critical",
                title="MLA model using MHA attention backend",
                description=(
                    "ROCM_AITER_FA is set but model uses MLA attention. "
                    "ROCM_AITER_MLA provides up to 17x decode speedup for DeepSeek/Kimi."
                ),
                current_value="ROCM_AITER_FA=1 (MHA backend)",
                recommended_value="ROCM_AITER_MLA=1 (MLA backend)",
                fix_command="export ROCM_AITER_MLA=1 (and unset ROCM_AITER_FA)",
                confidence=0.9,
                evidence="hardware_constraint",
                gpu_specific=True,
            )
        if ctx.attention_type != "MLA" and has_mla_backend and not has_fa_backend:
            return AuditFinding(
                check_id="WRONG_ATTENTION_BACKEND",
                severity="warning",
                title="Non-MLA model using MLA attention backend",
                description="ROCM_AITER_MLA is set but model uses standard MHA/GQA attention.",
                current_value="ROCM_AITER_MLA=1",
                recommended_value="ROCM_AITER_FA=1",
                fix_command="export ROCM_AITER_FA=1 (and unset ROCM_AITER_MLA)",
                confidence=0.85,
                evidence="hardware_constraint",
                gpu_specific=True,
            )
    return None


def _check_fp8bmm_crash_risk(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#19: FP8BMM_CRASH_RISK — FP8BMM enabled on MI300X."""
    if ctx.gpu_arch == "gfx942" and ctx.env_vars.get("VLLM_ROCM_USE_AITER_FP8BMM") == "1":
        return AuditFinding(
            check_id="FP8BMM_CRASH_RISK",
            severity="critical",
            title="FP8BMM enabled on MI300X — WILL CRASH",
            description=(
                "VLLM_ROCM_USE_AITER_FP8BMM=1 on gfx942 (MI300X) causes GPU memory access "
                "faults and crashes. This only works on gfx950 (MI355X)."
            ),
            current_value="VLLM_ROCM_USE_AITER_FP8BMM=1",
            recommended_value="VLLM_ROCM_USE_AITER_FP8BMM=0",
            fix_command="export VLLM_ROCM_USE_AITER_FP8BMM=0",
            confidence=0.99,
            evidence="hardware_constraint",
            gpu_specific=True,
        )
    return None


def _check_disagg_without_rdma(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """#21: DISAGG_WITHOUT_RDMA — disaggregated serving without RDMA."""
    if ctx.split_prefill_decode and not ctx.has_rdma:
        return AuditFinding(
            check_id="DISAGG_WITHOUT_RDMA",
            severity="critical",
            title="Disaggregated serving without RDMA",
            description=(
                "Prefill/decode disaggregation is enabled but no RDMA detected. "
                "PCIe-only KV transfer creates a severe bottleneck — can DEGRADE "
                "performance 20-30% vs colocated serving."
            ),
            current_value="disaggregated=true, rdma=false",
            recommended_value="Enable RDMA or disable disaggregation",
            fix_command="Disable P/D split or provision RDMA (NVLink/InfiniBand/EFA)",
            confidence=0.9,
            evidence="hardware_constraint",
        )
    return None


def _check_high_queue_depth(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """Queue depth check — requests waiting indicates under-provisioning."""
    if m.requests_waiting > 50:
        return AuditFinding(
            check_id="HIGH_QUEUE_DEPTH",
            severity="warning",
            title="High queue depth — requests starving",
            description=(
                f"{m.requests_waiting:.0f} requests waiting in queue. "
                "This indicates the deployment is under-provisioned for current load."
            ),
            current_value=f"{m.requests_waiting:.0f} waiting",
            recommended_value="<10 waiting in steady state",
            fix_command="Add replicas (DP scale-out) or increase max_num_batched_tokens",
            confidence=0.85,
            evidence="metric_correlation",
        )
    return None


def _check_kv_cache_critical(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """KV cache utilization above 95% — preemptions imminent."""
    if m.kv_cache_usage > 0.95:
        return AuditFinding(
            check_id="KV_CACHE_CRITICAL",
            severity="critical",
            title="KV cache utilization critical (>95%)",
            description=(
                f"KV cache at {m.kv_cache_usage:.0%}. Preemptions are imminent or occurring. "
                "Reduce max_model_len, lower gpu_memory_utilization, enable FP8 KV cache, "
                "or enable CPU offloading."
            ),
            current_value=f"{m.kv_cache_usage:.0%}",
            recommended_value="<85% for healthy headroom",
            fix_command="--kv-cache-dtype fp8_e4m3 (2x savings) or add CPU offloading",
            confidence=0.9,
            evidence="threshold_rule",
        )
    return None


def _check_high_itl(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """High inter-token latency — decode is memory-bound or overloaded."""
    if m.itl_avg_s is not None and m.itl_avg_s > 0.1:
        return AuditFinding(
            check_id="HIGH_ITL",
            severity="warning",
            title=f"High inter-token latency ({m.itl_avg_s * 1000:.0f}ms avg)",
            description=(
                "ITL above 100ms indicates decode is memory-bandwidth-bound or the scheduler "
                "is overloaded. For coding workloads, target ITL < 30ms."
            ),
            current_value=f"{m.itl_avg_s * 1000:.0f}ms avg ITL",
            recommended_value="<50ms for chat, <30ms for coding",
            fix_command="Reduce max_num_batched_tokens (e.g., 2048-4096 for latency-sensitive)",
            confidence=0.75,
            evidence="metric_correlation",
        )
    return None


def _check_high_ttft(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """High time-to-first-token — prefill is compute-bound or queued."""
    if m.ttft_avg_s is not None and m.ttft_avg_s > 5.0:
        return AuditFinding(
            check_id="HIGH_TTFT",
            severity="warning",
            title=f"High TTFT ({m.ttft_avg_s * 1000:.0f}ms avg)",
            description=(
                "Time to first token above 5 seconds suggests prefill compute saturation, "
                "long prompts, or queue congestion. Consider chunked prefill, disaggregation, "
                "or more TP shards."
            ),
            current_value=f"{m.ttft_avg_s * 1000:.0f}ms avg TTFT",
            recommended_value="<500ms for chat, <2s for coding",
            fix_command="Enable chunked prefill, consider P/D disaggregation, or increase TP",
            confidence=0.7,
            evidence="metric_correlation",
        )
    return None


def _check_low_prefix_hit_coding(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """Low prefix cache hits when high reuse is expected."""
    if 0 < m.prefix_cache_hit_rate < 0.3 and m.kv_cache_usage > 0.5:
        return AuditFinding(
            check_id="LOW_PREFIX_HIT_RATE",
            severity="info",
            title=f"Low prefix cache hit rate ({m.prefix_cache_hit_rate:.0%})",
            description=(
                "Prefix reuse is low. If this is a coding or agent workload, check prompt "
                "structure: remove timestamps, request IDs, and tool noise from cacheable prefixes. "
                "For SGLang: use --schedule-policy lpm for longest-prefix-match routing."
            ),
            current_value=f"{m.prefix_cache_hit_rate:.0%}",
            recommended_value=">50% for coding/agent, >30% for chat with shared system prompts",
            fix_command="Canonicalize prompts; SGLang: --schedule-policy lpm",
            confidence=0.65,
            evidence="threshold_rule",
        )
    return None


# =============================================================================
# WORKLOAD-AWARE FAILURE MODE CHECKS
# =============================================================================


def _check_kv_fragmentation_high(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """KV_FRAGMENTATION_HIGH — high KV usage but low active sequences indicates block fragmentation."""
    if m.kv_cache_usage > 0.7 and m.requests_running < 20 and m.requests_waiting == 0:
        return AuditFinding(
            check_id="KV_FRAGMENTATION_HIGH",
            severity="warning",
            title="KV cache fragmentation detected",
            description=(
                f"KV cache at {m.kv_cache_usage:.0%} but only {m.requests_running:.0f} active sequences "
                "with no queue. High memory usage with low utilization indicates PagedAttention "
                "block fragmentation — over-allocated blocks not being reclaimed."
            ),
            current_value=f"KV {m.kv_cache_usage:.0%}, {m.requests_running:.0f} running, 0 waiting",
            recommended_value="KV usage proportional to active sequences",
            fix_command=(
                "Enable block compaction, lower gpu_memory_utilization by 2-3 points, "
                "or restart to reset allocator state"
            ),
            confidence=0.75,
            evidence="metric_correlation",
        )
    return None


def _check_decode_starvation(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """DECODE_STARVATION — prefill is hogging the scheduler, starving decode tokens."""
    if m.itl_avg_s is not None and m.ttft_avg_s is not None and m.itl_avg_s > 0.1 and m.ttft_avg_s < 1.0:
        return AuditFinding(
            check_id="DECODE_STARVATION",
            severity="warning",
            title="Decode starvation — prefill hogging scheduler",
            description=(
                f"ITL is high ({m.itl_avg_s * 1000:.0f}ms) while TTFT is acceptable "
                f"({m.ttft_avg_s * 1000:.0f}ms). Prefill batches are consuming scheduler budget "
                "and starving decode tokens, causing user-visible generation lag."
            ),
            current_value=f"ITL={m.itl_avg_s * 1000:.0f}ms, TTFT={m.ttft_avg_s * 1000:.0f}ms",
            recommended_value="ITL <50ms with decode priority >=0.7",
            fix_command=(
                "Increase decode_priority, lower max_prefill_chunk_ratio, or enable prefill/decode lane isolation"
            ),
            confidence=0.8,
            evidence="metric_correlation",
        )
    return None


def _check_prefill_starvation(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """PREFILL_STARVATION — decode-heavy batch is starving new prefills."""
    if m.ttft_avg_s is not None and m.itl_avg_s is not None and m.ttft_avg_s > 5.0 and m.itl_avg_s < 0.03:
        return AuditFinding(
            check_id="PREFILL_STARVATION",
            severity="warning",
            title="Prefill starvation — decode-heavy batch blocking new requests",
            description=(
                f"TTFT is very high ({m.ttft_avg_s * 1000:.0f}ms) while ITL is fast "
                f"({m.itl_avg_s * 1000:.0f}ms). Active decode sequences are consuming scheduler "
                "budget and blocking new prefills from starting."
            ),
            current_value=f"TTFT={m.ttft_avg_s * 1000:.0f}ms, ITL={m.itl_avg_s * 1000:.0f}ms",
            recommended_value="TTFT <2s with prefill priority rebalanced",
            fix_command=(
                "Decrease decode_priority, increase prefill_lane_budget, or enable chunked prefill to interleave"
            ),
            confidence=0.8,
            evidence="metric_correlation",
        )
    return None


def _check_pcie_offload_thrash(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """PCIE_OFFLOAD_THRASH — KV offloading during active decode causes PCIe bottleneck.
    Gated on preemption rate rather than a raw counter so long-running
    deployments with healthy CPU offload don't false-positive."""
    if m.request_success_total <= 0 or m.preemptions_total <= 0:
        return None
    preemption_rate = m.preemptions_total / m.request_success_total
    if (
        m.cpu_cache_usage > 0.1
        and preemption_rate > 0.01
        and m.requests_running > 10
        and m.itl_avg_s is not None
        and m.itl_avg_s > 0.08
    ):
        return AuditFinding(
            check_id="PCIE_OFFLOAD_THRASH",
            severity="critical",
            title="PCIe offload thrashing during active decode",
            description=(
                f"CPU cache is active ({m.cpu_cache_usage:.0%}), preemption rate is "
                f"{preemption_rate:.1%}, and ITL is elevated ({m.itl_avg_s * 1000:.0f}ms) "
                f"with {m.requests_running:.0f} running sequences. KV blocks are being shuttled "
                "between GPU and CPU during decode — PCIe transfer dominates latency."
            ),
            current_value=(
                f"CPU cache {m.cpu_cache_usage:.0%}, {preemption_rate:.1%} preemption rate, "
                f"ITL {m.itl_avg_s * 1000:.0f}ms"
            ),
            recommended_value="Disable offload during active decode or offload only cold sessions",
            fix_command=(
                "Set offload_policy='cold_only', increase offload_idle_threshold, "
                "or disable offloading entirely and add GPU replicas"
            ),
            confidence=0.85,
            evidence="metric_correlation",
        )
    return None


def _check_gpu_underutilization(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """GPU_UNDERUTILIZATION — GPU has room but requests are queuing (scheduler misconfiguration)."""
    if m.kv_cache_usage < 0.3 and m.requests_waiting > 5:
        return AuditFinding(
            check_id="GPU_UNDERUTILIZATION",
            severity="warning",
            title="GPU underutilized while requests queue",
            description=(
                f"KV cache at only {m.kv_cache_usage:.0%} but {m.requests_waiting:.0f} requests "
                "are waiting. The GPU has capacity but the scheduler is not admitting new requests. "
                "This is typically a max_num_seqs or batched_token_budget misconfiguration."
            ),
            current_value=f"KV {m.kv_cache_usage:.0%}, {m.requests_waiting:.0f} waiting",
            recommended_value="Queue <5 when KV cache has headroom",
            fix_command=(
                "Increase max_num_seqs (e.g., 256-512), raise max_num_batched_tokens, "
                "or lower co_batch_utilization_threshold"
            ),
            confidence=0.85,
            evidence="metric_correlation",
        )
    return None


def _check_oom_despite_free(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """OOM_DESPITE_FREE — preemptions happening despite available KV cache
    (fragmentation-induced OOM). Uses a preemption rate rather than a raw
    counter so long-running deployments don't false-positive."""
    if m.request_success_total <= 0 or m.preemptions_total <= 0:
        return None
    preemption_rate = m.preemptions_total / m.request_success_total
    if preemption_rate > 0.01 and m.kv_cache_usage < 0.8:
        return AuditFinding(
            check_id="OOM_DESPITE_FREE",
            severity="critical",
            title="Preemptions despite available KV cache — fragmentation OOM",
            description=(
                f"Preemption rate is {preemption_rate:.1%} "
                f"({m.preemptions_total:.0f}/{m.request_success_total:.0f}) but KV cache is "
                f"only at {m.kv_cache_usage:.0%}. This indicates internal block fragmentation — "
                "the allocator cannot find contiguous blocks despite free memory. "
                "This is a known PagedAttention failure mode under mixed workloads."
            ),
            current_value=f"{preemption_rate:.1%} preemption rate at {m.kv_cache_usage:.0%} KV",
            recommended_value="<0.5% preemption rate with proper block management",
            fix_command=(
                "Enable fragmentation monitoring, lower kv_compaction_trigger, "
                "separate long/short context queues, or restart to reset allocator"
            ),
            confidence=0.85,
            evidence="metric_correlation",
        )
    return None


# =============================================================================
# BREAKPOINT-INFORMED CHECKS (derived from inferencebreakpoints knowledge base)
# =============================================================================


def _check_nixl_transfer_dominates(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """NIXL_TRANSFER_DOMINATES — KV transfer latency exceeds prefill recompute cost.

    Grounded in inferencebreakpoints/07-kv-cache/disaggregated-kv/cache-hit-vs-recompute-decision:
    In disaggregated systems, the router must decide transfer vs recompute. When NIXL transfer
    latency is high relative to context length, recompute may be cheaper.

    DORMANT BY DEFAULT. NIXL exposes its own Prometheus endpoint via
    NIXL_TELEMETRY_PROMETHEUS_PORT, separate from the Dynamo frontend
    /metrics endpoint. The real NIXL metric schema is not documented in
    the current Dynamo repo. This check will only fire when:
      (1) the operator adds the NIXL endpoint as an extra metrics_target,
      (2) the schema pinned in telemetry/prometheus.py matches the real
          NIXL metric names — which needs verification against a captured
          scrape from a real disaggregated deployment.
    Until (2) is done, nixl_transfer_latency_s stays None and this check
    no-fires cleanly.
    """
    if (
        ctx.split_prefill_decode
        and m.nixl_transfer_latency_s is not None
        and m.nixl_transfer_latency_s > 0.5
        and m.ttft_avg_s is not None
        and m.ttft_avg_s > 3.0
    ):
        return AuditFinding(
            check_id="NIXL_TRANSFER_DOMINATES",
            severity="warning",
            title="NIXL KV transfer latency exceeds recompute threshold",
            description=(
                f"NIXL transfer latency is {m.nixl_transfer_latency_s * 1000:.0f}ms with "
                f"TTFT at {m.ttft_avg_s * 1000:.0f}ms. For shorter contexts, local recompute "
                "may be faster than cross-node KV transfer. The crossover point depends on "
                "context length, network bandwidth, and GPU speed."
            ),
            current_value=f"NIXL latency={m.nixl_transfer_latency_s * 1000:.0f}ms, TTFT={m.ttft_avg_s * 1000:.0f}ms",
            recommended_value="NIXL latency <200ms or route short contexts to local recompute",
            fix_command=(
                "Tune Dynamo router to prefer local recompute for contexts <4K tokens, "
                "or validate RDMA/NVLink transport health"
            ),
            confidence=0.8,
            evidence="metric_correlation",
        )
    return None


def _check_lmcache_cold_start(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """LMCACHE_COLD_START — very low LMCache hit rate indicating cache warmup needed.

    Grounded in inferencebreakpoints/06-prefill/prompt-caching/cache-aware-routing:
    Cache-aware routing saves 50-80% of prefill compute. Very low hit rates after
    deployment indicate the cache hasn't warmed up or routing isn't cache-aware.
    """
    if 0 < m.lmcache_hit_rate < 0.05 and m.requests_running > 5:
        return AuditFinding(
            check_id="LMCACHE_COLD_START",
            severity="info",
            title="LMCache hit rate near zero — cache cold start",
            description=(
                f"LMCache hit rate is {m.lmcache_hit_rate:.1%} with {m.requests_running:.0f} "
                "active requests. This suggests a recent deployment restart or namespace change. "
                "Coding workloads should see 50-95% hit rates once warmed."
            ),
            current_value=f"{m.lmcache_hit_rate:.1%} hit rate",
            recommended_value=">50% for coding workloads after warmup",
            fix_command=(
                "Wait for cache warmup (typically 2-5 minutes under load), "
                "verify LMCache namespace matches across restarts, "
                "or pre-warm with representative prompts"
            ),
            confidence=0.7,
            evidence="threshold_rule",
        )
    return None


def _check_batch_itl_tradeoff(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """BATCH_ITL_TRADEOFF — batch size too large for target ITL.

    Grounded in inferencebreakpoints/08-decode/batch-effects/batch-size-vs-itl:
    Increasing batch size improves throughput but degrades ITL. For interactive
    applications, ITL >50-100ms causes perceptible lag.
    """
    if (
        m.itl_avg_s is not None
        and m.itl_avg_s > 0.05
        and m.requests_running > 50
        and m.gen_throughput_tps > 0
    ):
        return AuditFinding(
            check_id="BATCH_ITL_TRADEOFF",
            severity="warning",
            title="Large batch size degrading inter-token latency",
            description=(
                f"ITL is {m.itl_avg_s * 1000:.0f}ms with {m.requests_running:.0f} concurrent "
                f"requests at {m.gen_throughput_tps:.0f} tok/s. The batch-ITL tradeoff is "
                "unfavorable for interactive workloads — consider splitting into latency-optimized "
                "and throughput-optimized pools."
            ),
            current_value=(
                f"ITL={m.itl_avg_s * 1000:.0f}ms, batch={m.requests_running:.0f}, "
                f"throughput={m.gen_throughput_tps:.0f} tok/s"
            ),
            recommended_value="ITL <30ms for coding, <50ms for chat; split pools if needed",
            fix_command=(
                "Lower max_num_seqs to 32-48 for latency pool, or use SLO-tiered scheduling "
                "to prioritize interactive requests over batch workloads"
            ),
            confidence=0.75,
            evidence="metric_correlation",
        )
    return None


def _check_router_overhead_dominates(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """ROUTER_OVERHEAD_DOMINATES — Dynamo routing overhead is a large fraction of TTFT.

    The Dynamo frontend publishes per-request routing overhead histograms
    (`dynamo_router_overhead_*_ms`) separate from backend compute time.
    When the router's scheduling, block hashing, or KV indexer overhead
    becomes comparable to the backend prefill time, TTFT is limited by
    the router rather than by compute — the fix is usually router
    tuning, not adding GPU capacity.

    Fires only when router overhead metrics are present in the scrape
    (Dynamo may not emit them in all versions), so this check is a
    no-op for non-Dynamo or older Dynamo deployments.
    """
    if m.router_overhead_total_ms is None or m.ttft_avg_s is None or m.ttft_avg_s <= 0:
        return None
    ttft_ms = m.ttft_avg_s * 1000
    # Meaningful-absolute-overhead guard (50ms) plus a ratio test (>30%
    # of TTFT). Either alone would false-positive on very fast or very
    # slow deployments; together they target the "routing dominates"
    # regime specifically.
    if m.router_overhead_total_ms > 50 and m.router_overhead_total_ms / ttft_ms > 0.30:
        overhead_ratio = m.router_overhead_total_ms / ttft_ms
        return AuditFinding(
            check_id="ROUTER_OVERHEAD_DOMINATES",
            severity="warning",
            title=(
                f"Router overhead is {overhead_ratio:.0%} of TTFT "
                f"({m.router_overhead_total_ms:.0f}ms of {ttft_ms:.0f}ms)"
            ),
            description=(
                f"Dynamo router overhead ({m.router_overhead_total_ms:.0f}ms) is a large "
                f"fraction of total time-to-first-token ({ttft_ms:.0f}ms). "
                "Investigate the per-stage breakdown "
                f"(block_hashing={m.router_overhead_block_hashing_ms or 0:.0f}ms, "
                f"indexer={m.router_overhead_indexer_ms or 0:.0f}ms, "
                f"scheduling={m.router_overhead_scheduling_ms or 0:.0f}ms). "
                "Adding GPU capacity will not help if routing is the bottleneck."
            ),
            current_value=(
                f"total={m.router_overhead_total_ms:.0f}ms, "
                f"{overhead_ratio:.0%} of TTFT"
            ),
            recommended_value="<10% of TTFT and <30ms total",
            fix_command=(
                "Tune router worker count, reduce KV indexer block-hash depth, "
                "or move to a simpler routing policy if the workload doesn't "
                "benefit from KV-aware routing"
            ),
            confidence=0.8,
            evidence="metric_correlation",
        )
    return None


def _check_ttft_cpu_bound(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """TTFT_CPU_BOUND — tokenizer latency is a large fraction of TTFT.

    Completes the TTFT attribution chain. Given a TTFT value, the four
    possible bottlenecks are:
      * queue wait        -> caught by HIGH_TTFT, prefill_starvation logic
      * router scheduling -> caught by ROUTER_OVERHEAD_DOMINATES
      * prefill compute   -> implicit (remainder)
      * CPU pre-processing (tokenization) -> caught by THIS check

    `dynamo_frontend_tokenizer_latency_ms` is emitted by the HTTP
    frontend and measures per-request tokenizer overhead in
    milliseconds. When this number becomes a meaningful fraction of
    TTFT, the fix is on the CPU side (faster tokenizer, prompt
    canonicalization, process pinning), not the GPU side.

    Fires when both:
      (1) tokenizer latency is absolutely meaningful (> 30ms), and
      (2) tokenizer latency is >= 20% of total TTFT.

    No-fires when either metric is absent, when TTFT is zero, or
    when the absolute tokenizer latency is small regardless of its
    ratio — the latter case is "tokenizer is slow but overall TTFT
    is also slow", where the tokenizer isn't the main problem.
    """
    if (
        m.tokenizer_latency_ms is None
        or m.ttft_avg_s is None
        or m.ttft_avg_s <= 0
    ):
        return None
    ttft_ms = m.ttft_avg_s * 1000
    if m.tokenizer_latency_ms <= 30:
        return None
    ratio = m.tokenizer_latency_ms / ttft_ms
    if ratio < 0.20:
        return None
    return AuditFinding(
        check_id="TTFT_CPU_BOUND",
        severity="warning",
        title=(
            f"Tokenizer is {ratio:.0%} of TTFT "
            f"({m.tokenizer_latency_ms:.0f}ms of {ttft_ms:.0f}ms)"
        ),
        description=(
            f"Tokenizer latency ({m.tokenizer_latency_ms:.0f}ms) is a "
            f"significant fraction of total TTFT ({ttft_ms:.0f}ms). The "
            "bottleneck is on the CPU pre-processing path, not the GPU "
            "prefill path. Adding GPU capacity will not help — look at "
            "tokenizer backend, prompt canonicalization, and preprocessor "
            "worker count instead."
        ),
        current_value=(
            f"tokenizer={m.tokenizer_latency_ms:.0f}ms, "
            f"{ratio:.0%} of TTFT"
        ),
        recommended_value="<10% of TTFT and <20ms absolute",
        fix_command=(
            "Switch to a faster tokenizer backend (hf-tokenizers rust "
            "tokenizer), normalize prompts to remove variable-length "
            "whitespace, pin preprocessor workers, or increase the "
            "frontend worker count"
        ),
        confidence=0.8,
        evidence="metric_correlation",
    )


def _check_kvbm_tiering_ineffective(m: NormalizedMetrics, ctx: DeploymentContext) -> AuditFinding | None:
    """KVBM_TIERING_INEFFECTIVE — KVBM is configured but the host tier never hits.

    When KVBM (KV Block Manager) is enabled, it tiers KV cache across
    GPU HBM -> CPU DRAM -> NVMe -> object storage. An effective tiering
    policy should produce meaningful hit rates on the CPU (host) tier
    for bursty or long-tail workloads — tokens that couldn't fit on the
    GPU are demoted to host memory and then re-onboarded when needed.

    This check fires when:
      (1) GPU KV cache is hot (usage > 85%),
      (2) KVBM is active (any offload or onboard counter is nonzero —
          this is the "KVBM is configured" gate),
      (3) the host tier hit rate is near-zero (<5%).
    That combination means blocks are being demoted but not re-used —
    the working set exceeds tier capacity, or the demotion policy is
    evicting blocks faster than they would be re-requested.

    Silently no-fires when KVBM is not enabled, because the KVBM
    metrics only populate when the operator scrapes the separate KVBM
    /metrics endpoint (default port 6880 via DYN_KVBM_METRICS_PORT,
    behind DYN_KVBM_METRICS=true).

    This replaces the deleted GROVE_TIER_IMBALANCE check, which was
    based on a conceptual confusion (Grove is Dynamo's Kubernetes
    scheduler, not a KV tiering system).
    """
    kvbm_active = (
        m.kvbm_offload_d2h > 0
        or m.kvbm_onboard_h2d > 0
        or m.kvbm_host_hit_rate > 0
    )
    if not kvbm_active:
        return None
    if m.kv_cache_usage > 0.85 and m.kvbm_host_hit_rate < 0.05:
        return AuditFinding(
            check_id="KVBM_TIERING_INEFFECTIVE",
            severity="warning",
            title="KVBM tiering enabled but host tier hit rate is near zero",
            description=(
                f"KVBM is active ({m.kvbm_offload_d2h:.0f} offloads, "
                f"{m.kvbm_onboard_h2d:.0f} onboards) and the GPU KV cache is hot at "
                f"{m.kv_cache_usage:.0%}, but the CPU (host) tier hit rate is "
                f"only {m.kvbm_host_hit_rate:.1%}. Blocks are being demoted to "
                "host memory but not re-used — either the working set exceeds "
                "tier capacity or the demotion policy is evicting blocks faster "
                "than they would be re-requested."
            ),
            current_value=(
                f"GPU {m.kv_cache_usage:.0%}, host hit rate {m.kvbm_host_hit_rate:.1%}"
            ),
            recommended_value="Host tier hit rate >10% when GPU is under KV pressure",
            fix_command=(
                "Increase CPU DRAM allocated to the KVBM host tier, raise "
                "kvbm_promotion_threshold to keep blocks warmer before demotion, "
                "or add NVMe tier capacity for longer-lived sessions"
            ),
            confidence=0.75,
            evidence="metric_correlation",
        )
    return None


# =============================================================================
# CHECK REGISTRY
# =============================================================================

_ALL_CHECKS = [
    _check_kv_preemption_storm,  # 3
    _check_missing_quantization,  # 4
    _check_prefix_cache_disabled,  # 7
    _check_batch_size_mismatch,  # 8
    _check_kv_dtype_suboptimal,  # 11
    _check_aiter_disabled,  # 12
    _check_block_size_wrong,  # 13
    _check_memory_util_low,  # 14
    _check_speculative_overhead,  # 15
    _check_moe_ep_missing,  # 16
    _check_atom_not_used,  # 17
    _check_wrong_attention_backend,  # 18
    _check_fp8bmm_crash_risk,  # 19
    _check_disagg_without_rdma,  # 21
    _check_high_queue_depth,  # extra
    _check_kv_cache_critical,  # extra
    _check_high_itl,  # extra
    _check_high_ttft,  # extra
    _check_low_prefix_hit_coding,  # extra
    # Workload-aware failure mode checks
    _check_kv_fragmentation_high,
    _check_decode_starvation,
    _check_prefill_starvation,
    _check_pcie_offload_thrash,
    _check_gpu_underutilization,
    _check_oom_despite_free,
    # Breakpoint-informed checks (derived from inferencebreakpoints knowledge base)
    _check_nixl_transfer_dominates,
    _check_lmcache_cold_start,
    _check_batch_itl_tradeoff,
    _check_router_overhead_dominates,
    _check_ttft_cpu_bound,
    _check_kvbm_tiering_ineffective,
]
