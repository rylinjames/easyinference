"""Shared runtime profiling orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.engines.registry import get_engine_adapter
from inferscope.optimization.checks import AuditFinding, DeploymentContext, run_all_checks
from inferscope.optimization.serving_profile import BottleneckType, WorkloadMode
from inferscope.optimization.workload_classifier import WorkloadClassification, classify_workload
from inferscope.profiling.intents import ProfilingIntent, resolve_profiling_intent
from inferscope.profiling.models import (
    ProfileSourceKind,
    RuntimeBottleneck,
    RuntimeContextHints,
    RuntimeIdentity,
    RuntimeProfileReport,
    TuningPreview,
)
from inferscope.profiling.tuning import build_tuning_preview
from inferscope.telemetry.capture import capture_endpoint_telemetry
from inferscope.telemetry.models import MetricSnapshot
from inferscope.telemetry.normalizer import NormalizedMetrics
from inferscope.telemetry.prometheus import resolve_api_base_url

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

_BOTTLENECK_CHECK_MAP: dict[str, BottleneckType] = {
    "HIGH_TTFT": BottleneckType.PREFILL_COMPUTE,
    "HIGH_ITL": BottleneckType.DECODE_MEMORY,
    "KV_PREEMPTION_STORM": BottleneckType.CACHE_BOUND,
    "KV_CACHE_CRITICAL": BottleneckType.CACHE_BOUND,
    "LOW_PREFIX_HIT_RATE": BottleneckType.CACHE_BOUND,
    "KV_FRAGMENTATION_HIGH": BottleneckType.CACHE_BOUND,
    "OOM_DESPITE_FREE": BottleneckType.CACHE_BOUND,
    "PREFIX_CACHE_DISABLED": BottleneckType.CACHE_BOUND,
    "KV_DTYPE_SUBOPTIMAL": BottleneckType.CACHE_BOUND,
    "PCIE_OFFLOAD_THRASH": BottleneckType.INTERCONNECT_BOUND,
    "DISAGG_WITHOUT_RDMA": BottleneckType.INTERCONNECT_BOUND,
    "MOE_EP_MISSING": BottleneckType.MOE_ROUTING,
    "HIGH_QUEUE_DEPTH": BottleneckType.SCHEDULER_BOUND,
    "BATCH_SIZE_MISMATCH": BottleneckType.SCHEDULER_BOUND,
    "DECODE_STARVATION": BottleneckType.SCHEDULER_BOUND,
    "PREFILL_STARVATION": BottleneckType.SCHEDULER_BOUND,
    "GPU_UNDERUTILIZATION": BottleneckType.SCHEDULER_BOUND,
    "BATCH_ITL_TRADEOFF": BottleneckType.SCHEDULER_BOUND,
    "SLO_VIOLATION_RATE": BottleneckType.SCHEDULER_BOUND,
    "NIXL_TRANSFER_DOMINATES": BottleneckType.INTERCONNECT_BOUND,
    "GROVE_TIER_IMBALANCE": BottleneckType.CACHE_BOUND,
    "GROVE_EVICTION_STORM": BottleneckType.CACHE_BOUND,
    "LMCACHE_COLD_START": BottleneckType.CACHE_BOUND,
    "MISSING_QUANTIZATION": BottleneckType.MISCONFIGURATION,
    "AITER_DISABLED": BottleneckType.MISCONFIGURATION,
    "BLOCK_SIZE_WRONG": BottleneckType.MISCONFIGURATION,
    "MEMORY_UTIL_LOW": BottleneckType.MISCONFIGURATION,
    "SPECULATIVE_OVERHEAD": BottleneckType.MISCONFIGURATION,
    "ATOM_NOT_USED": BottleneckType.MISCONFIGURATION,
    "WRONG_ATTENTION_BACKEND": BottleneckType.MISCONFIGURATION,
    "FP8BMM_CRASH_RISK": BottleneckType.MISCONFIGURATION,
}

_BOTTLENECK_METRICS: dict[BottleneckType, tuple[str, ...]] = {
    BottleneckType.PREFILL_COMPUTE: ("ttft_avg_ms", "requests_waiting"),
    BottleneckType.DECODE_MEMORY: ("itl_avg_ms", "requests_running", "gen_throughput_tps"),
    BottleneckType.CACHE_BOUND: ("kv_cache_usage", "prefix_cache_hit_rate", "cpu_cache_usage", "preemptions_total"),
    BottleneckType.INTERCONNECT_BOUND: ("cpu_cache_usage", "preemptions_total", "itl_avg_ms"),
    BottleneckType.MOE_ROUTING: ("requests_running", "ttft_avg_ms"),
    BottleneckType.SCHEDULER_BOUND: ("requests_running", "requests_waiting", "ttft_avg_ms", "itl_avg_ms"),
    BottleneckType.MISCONFIGURATION: (),
}


def _severity_literal(value: str) -> Literal["critical", "warning", "info"]:
    if value in {"critical", "warning", "info"}:
        return cast(Literal["critical", "warning", "info"], value)
    return "info"


def _engine_source_literal(value: str) -> Literal["metrics", "adapter", "unknown"]:
    if value in {"metrics", "adapter", "unknown"}:
        return cast(Literal["metrics", "adapter", "unknown"], value)
    return "unknown"


@dataclass
class RuntimeAnalysisBundle:
    """Typed intermediate bundle reused by runtime analysis surfaces."""

    snapshot: MetricSnapshot
    normalized: NormalizedMetrics
    health: dict[str, Any] = field(default_factory=dict)
    memory_pressure: dict[str, Any] = field(default_factory=dict)
    cache_effectiveness: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=dict)
    workload: WorkloadClassification | None = None
    identity: RuntimeIdentity | None = None
    deployment_context: DeploymentContext | None = None
    findings: list[AuditFinding] = field(default_factory=list)
    bottlenecks: list[RuntimeBottleneck] = field(default_factory=list)
    tuning_preview: TuningPreview | None = None
    profiling_intent: ProfilingIntent | None = None
    reasoning: list[str] = field(default_factory=list)


async def analyze_runtime(
    endpoint: str,
    *,
    context_hints: RuntimeContextHints | None = None,
    current_scheduler: dict[str, Any] | None = None,
    current_cache: dict[str, Any] | None = None,
    allow_private: bool = True,
    metrics_auth: EndpointAuthConfig | None = None,
    include_workload: bool = True,
    include_identity: bool = False,
    include_findings: bool = True,
    include_tuning_preview: bool = False,
    include_raw_metrics: bool = False,
    include_samples: bool = False,
) -> RuntimeAnalysisBundle:
    """Capture and analyze a runtime endpoint once."""
    hints = context_hints or RuntimeContextHints()
    captured = await capture_endpoint_telemetry(
        endpoint,
        allow_private=allow_private,
        metrics_auth=metrics_auth,
        include_samples=include_samples,
    )

    snapshot = captured.snapshot.model_copy(
        update={"raw_metrics": captured.snapshot.raw_metrics if include_raw_metrics else {}}
    )
    bundle = RuntimeAnalysisBundle(snapshot=snapshot, normalized=captured.normalized)

    if snapshot.error:
        bundle.reasoning.append(f"Runtime analysis stopped: {snapshot.error}")
        return bundle

    bundle.health = assess_health(bundle.normalized)
    bundle.memory_pressure = build_memory_pressure_analysis(bundle.normalized)
    bundle.cache_effectiveness = build_cache_effectiveness_analysis(bundle.normalized)
    bundle.reliability = build_reliability_analysis(bundle.normalized)
    bundle.reasoning.append(f"Scraped {bundle.normalized.engine} metrics from {bundle.normalized.endpoint}")

    # Surface TTFT decomposition and goodput in reasoning
    if bundle.normalized.prefill_compute_s is not None:
        bundle.reasoning.append(
            f"TTFT decomposition: ~{bundle.normalized.queue_fraction:.0%} queue wait, "
            f"~{1 - bundle.normalized.queue_fraction:.0%} prefill compute "
            f"(prefill ≈ {bundle.normalized.prefill_compute_s * 1000:.0f}ms)"
        )
    if bundle.normalized.goodput_ratio > 0 and bundle.normalized.goodput_ratio < 1.0:
        waste_pct = (1.0 - bundle.normalized.goodput_ratio) * 100
        bundle.reasoning.append(
            f"Goodput analysis: {waste_pct:.0f}% estimated compute waste "
            f"(goodput={bundle.normalized.goodput_tps:.0f} tok/s vs "
            f"raw={bundle.normalized.gen_throughput_tps:.0f} tok/s)"
        )

    if include_workload:
        try:
            bundle.workload = classify_workload(bundle.normalized)
        except Exception:  # noqa: BLE001
            bundle.workload = WorkloadClassification(
                mode=WorkloadMode.CHAT,
                confidence=0.33,
                signals={"fallback": "workload classification degraded to chat"},
            )
        bundle.reasoning.append(
            f"Classified workload as {bundle.workload.mode.value} ({bundle.workload.confidence:.0%} confidence)"
        )

    if include_identity:
        bundle.identity = await enrich_runtime_identity(
            endpoint,
            bundle.normalized,
            hints,
            allow_private=allow_private,
            metrics_auth=metrics_auth,
        )
        if bundle.identity is not None:
            if bundle.identity.config_error:
                bundle.reasoning.append(f"Runtime identity enrichment degraded: {bundle.identity.config_error}")
            else:
                bundle.reasoning.append(
                    f"Runtime identity resolved with {len(bundle.identity.served_models)} served model(s)"
                )

    if include_findings:
        bundle.deployment_context = build_deployment_context(
            bundle.normalized,
            hints,
            identity=bundle.identity,
            current_scheduler=current_scheduler,
        )
        bundle.findings = run_all_checks(bundle.normalized, bundle.deployment_context)
        bundle.bottlenecks = derive_bottlenecks(bundle.findings, bundle.normalized)
        bundle.reasoning.append(
            f"Derived {len(bundle.findings)} audit finding(s) and {len(bundle.bottlenecks)} grouped bottleneck(s)"
        )

    if include_tuning_preview:
        bundle.tuning_preview = build_tuning_preview(
            bundle.findings,
            bundle.normalized,
            current_scheduler=current_scheduler,
            current_cache=current_cache,
        )
        bundle.reasoning.append(bundle.tuning_preview.summary)

    gpu_vendor = detect_gpu_vendor(hints)
    if gpu_vendor:
        bundle.profiling_intent = resolve_profiling_intent(gpu_vendor)
        bundle.reasoning.append(bundle.profiling_intent.summary)

    return bundle


async def build_runtime_profile(
    endpoint: str,
    *,
    context_hints: RuntimeContextHints | None = None,
    current_scheduler: dict[str, Any] | None = None,
    current_cache: dict[str, Any] | None = None,
    allow_private: bool = True,
    metrics_auth: EndpointAuthConfig | None = None,
    include_identity: bool = True,
    include_tuning_preview: bool = True,
    include_raw_metrics: bool = False,
    include_samples: bool = False,
) -> RuntimeProfileReport:
    """Build the full runtime profile report."""
    bundle = await analyze_runtime(
        endpoint,
        context_hints=context_hints,
        current_scheduler=current_scheduler,
        current_cache=current_cache,
        allow_private=allow_private,
        metrics_auth=metrics_auth,
        include_workload=True,
        include_identity=include_identity,
        include_findings=True,
        include_tuning_preview=include_tuning_preview,
        include_raw_metrics=include_raw_metrics,
        include_samples=include_samples,
    )
    if bundle.snapshot.error:
        raise RuntimeError(bundle.snapshot.error)

    workload_payload = bundle.workload.to_dict() if bundle.workload is not None else {}
    audit_payload = build_audit_payload(bundle.findings)
    confidence = calculate_profile_confidence(bundle, context_hints or RuntimeContextHints())
    summary = build_runtime_profile_summary(bundle, audit_payload)
    return RuntimeProfileReport(
        source_kind=ProfileSourceKind.PROMETHEUS_RUNTIME,
        endpoint=endpoint,
        metrics_snapshot=bundle.snapshot,
        metrics=bundle.normalized.to_dict(),
        health=bundle.health,
        memory_pressure=bundle.memory_pressure,
        cache_effectiveness=bundle.cache_effectiveness,
        reliability=bundle.reliability,
        workload=workload_payload,
        identity=bundle.identity if include_identity else None,
        audit=audit_payload,
        bottlenecks=bundle.bottlenecks,
        tuning_preview=bundle.tuning_preview if include_tuning_preview else None,
        profiling_intent=(bundle.profiling_intent.to_dict() if bundle.profiling_intent is not None else None),
        reasoning=bundle.reasoning,
        summary=summary,
        confidence=confidence,
    )


def assess_health(metrics: NormalizedMetrics) -> dict[str, Any]:
    """Derive health indicators from normalized metrics."""
    issues = []
    status = "healthy"

    if metrics.requests_waiting > 10:
        issues.append(
            {
                "indicator": "queue_buildup",
                "severity": "warning" if metrics.requests_waiting < 50 else "critical",
                "detail": f"{metrics.requests_waiting:.0f} requests waiting in queue",
            }
        )
        status = "degraded"

    if metrics.kv_cache_usage > 0.95:
        issues.append(
            {
                "indicator": "kv_cache_critical",
                "severity": "critical",
                "detail": f"KV cache at {metrics.kv_cache_usage:.0%} — preemptions likely",
            }
        )
        status = "critical"
    elif metrics.kv_cache_usage > 0.85:
        issues.append(
            {
                "indicator": "kv_cache_high",
                "severity": "warning",
                "detail": f"KV cache at {metrics.kv_cache_usage:.0%}",
            }
        )
        if status == "healthy":
            status = "degraded"

    if metrics.ttft_avg_s and metrics.ttft_avg_s > 5.0:
        issues.append(
            {
                "indicator": "high_ttft",
                "severity": "warning",
                "detail": f"Average TTFT is {metrics.ttft_avg_s * 1000:.0f}ms",
            }
        )

    if metrics.itl_avg_s and metrics.itl_avg_s > 0.1:
        issues.append(
            {
                "indicator": "high_itl",
                "severity": "warning",
                "detail": f"Average ITL is {metrics.itl_avg_s * 1000:.0f}ms",
            }
        )

    if metrics.spec_acceptance_rate > 0 and metrics.spec_acceptance_rate < 0.55:
        issues.append(
            {
                "indicator": "low_spec_acceptance",
                "severity": "warning",
                "detail": (
                    f"Speculative decode acceptance at {metrics.spec_acceptance_rate:.0%} — "
                    "consider disabling at high concurrency"
                ),
            }
        )

    return {
        "status": status,
        "issues": issues,
        "issue_count": len(issues),
    }


def build_health_summary(metrics: NormalizedMetrics, health: dict[str, Any]) -> str:
    """Build a human-readable health summary."""
    parts = [f"{metrics.engine.upper()} at {metrics.endpoint}"]
    parts.append(f"Status: {health['status']}")
    parts.append(f"{metrics.requests_running:.0f} running, {metrics.requests_waiting:.0f} waiting")
    parts.append(f"KV cache: {metrics.kv_cache_usage:.0%}")

    if metrics.ttft_avg_s:
        parts.append(f"TTFT avg: {metrics.ttft_avg_s * 1000:.0f}ms")
    if metrics.itl_avg_s:
        parts.append(f"ITL avg: {metrics.itl_avg_s * 1000:.0f}ms")
    if metrics.gen_throughput_tps > 0:
        parts.append(f"Throughput: {metrics.gen_throughput_tps:.0f} tok/s")

    if health["issue_count"] > 0:
        parts.append(f"⚠ {health['issue_count']} issue(s) detected")

    return " | ".join(parts)


def build_memory_pressure_analysis(metrics: NormalizedMetrics) -> dict[str, Any]:
    """Analyze KV cache utilization and preemption rates."""
    kv_usage = metrics.kv_cache_usage
    preemptions = metrics.preemptions_total
    prefix_hits = metrics.prefix_cache_hit_rate

    pressure = "low"
    findings: list[str] = []

    if kv_usage > 0.95:
        pressure = "critical"
        findings.append(
            f"KV cache at {kv_usage:.0%} — preemptions imminent. "
            "Reduce gpu_memory_utilization, lower max_model_len, or add replicas."
        )
    elif kv_usage > 0.85:
        pressure = "high"
        findings.append(
            f"KV cache at {kv_usage:.0%} — approaching saturation. "
            "Consider FP8 KV cache (2x savings) or CPU offloading."
        )
    elif kv_usage > 0.70:
        pressure = "moderate"
        findings.append(f"KV cache at {kv_usage:.0%} — healthy utilization.")
    else:
        findings.append(
            f"KV cache at {kv_usage:.0%} — underutilized. "
            "Could increase gpu_memory_utilization or serve more concurrent requests."
        )

    if preemptions > 0:
        findings.append(
            f"⚠ {preemptions:.0f} preemptions recorded — requests being evicted from KV cache under pressure."
        )

    if prefix_hits > 0:
        findings.append(f"Prefix cache hit rate: {prefix_hits:.0%}")

    return {
        "level": pressure,
        "kv_cache_usage": round(kv_usage, 4),
        "prefix_cache_hit_rate": round(prefix_hits, 4),
        "preemptions_total": preemptions,
        "cpu_cache_usage": round(metrics.cpu_cache_usage, 4),
        "findings": findings,
        "summary": f"Memory pressure: {pressure} (KV {kv_usage:.0%}, {len(findings)} findings)",
    }


def build_reliability_analysis(metrics: NormalizedMetrics) -> dict[str, Any]:
    """Analyze request stability and observability-critical reliability counters."""
    level = "healthy"
    findings: list[str] = []

    if metrics.request_migrations_total > 0:
        level = "degraded"
        findings.append(
            f"Dynamo migrated {metrics.request_migrations_total:.0f} request(s) due to worker unavailability."
        )

    if metrics.disconnected_clients > 0:
        level = "degraded"
        findings.append(f"{metrics.disconnected_clients:.0f} disconnected client(s) observed during streaming.")

    if metrics.requests_waiting >= 10:
        level = "degraded" if level == "healthy" else level
        findings.append(f"Frontend queue depth is {metrics.requests_waiting:.0f} request(s).")

    if metrics.kv_total_blocks > 0:
        findings.append(
            f"KV block occupancy: {metrics.kv_active_blocks:.0f}/{metrics.kv_total_blocks:.0f} active blocks."
        )

    if not findings:
        findings.append("No migration, disconnect, or queue-depth reliability signals detected.")

    return {
        "level": level,
        "request_migrations_total": round(metrics.request_migrations_total, 4),
        "disconnected_clients": round(metrics.disconnected_clients, 4),
        "kv_active_blocks": round(metrics.kv_active_blocks, 4),
        "kv_total_blocks": round(metrics.kv_total_blocks, 4),
        "findings": findings,
        "summary": (
            f"Reliability: {level} (migrations={metrics.request_migrations_total:.0f}, "
            f"disconnects={metrics.disconnected_clients:.0f})"
        ),
    }


def build_cache_effectiveness_analysis(metrics: NormalizedMetrics) -> dict[str, Any]:
    """Measure prefix cache effectiveness and routing hints."""
    hit_rate = metrics.prefix_cache_hit_rate
    kv_usage = metrics.kv_cache_usage

    effectiveness = "unknown"
    recommendations: list[str] = []

    if hit_rate > 0.8:
        effectiveness = "excellent"
        recommendations.append("High cache hit rate — prefix caching is working well.")
    elif hit_rate > 0.5:
        effectiveness = "good"
        recommendations.append(
            "Moderate cache hits — consider canonicalizing prompts "
            "(remove timestamps, request IDs, tool noise from prefix)."
        )
    elif hit_rate > 0.2:
        effectiveness = "poor"
        recommendations.append(
            "Low cache hit rate — check prompt structure. "
            "Stable system prompts and tool schemas should be prefix-cached."
        )
    elif hit_rate > 0:
        effectiveness = "minimal"
        recommendations.append(
            "Very low cache hits — workload may have unique prompts. "
            "Consider if prefix caching is appropriate for this workload."
        )
    else:
        effectiveness = "disabled_or_no_data"
        recommendations.append(
            "No cache hit data — is prefix caching enabled? "
            "vLLM V1 has zero-overhead prefix caching (always on). "
            "SGLang needs --enable-metrics to expose cache_hit_rate."
        )

    if metrics.engine == "sglang" and hit_rate < 0.5:
        recommendations.append(
            "SGLang RadixAttention typically achieves 85-95% hit rates on coding workloads. "
            "Consider --schedule-policy lpm for longest-prefix-match routing."
        )

    return {
        "effectiveness": effectiveness,
        "prefix_hit_rate": round(hit_rate, 4),
        "kv_cache_usage": round(kv_usage, 4),
        "recommendations": recommendations,
        "summary": f"Cache effectiveness: {effectiveness} (hit rate: {hit_rate:.0%}, KV usage: {kv_usage:.0%})",
    }


def build_audit_payload(findings: list[AuditFinding]) -> dict[str, Any]:
    """Serialize findings into the existing audit response shape."""
    critical_count = sum(1 for finding in findings if finding.severity == "critical")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    info_count = sum(1 for finding in findings if finding.severity == "info")
    return {
        "findings": [finding.to_dict() for finding in findings],
        "total": len(findings),
        "critical": critical_count,
        "warnings": warning_count,
        "info": info_count,
    }


def build_deployment_context(
    metrics: NormalizedMetrics,
    hints: RuntimeContextHints,
    *,
    identity: RuntimeIdentity | None = None,
    current_scheduler: dict[str, Any] | None = None,
) -> DeploymentContext:
    """Build the deployment context used by audit checks."""
    gpu_arch = hints.gpu_arch
    gpu_vendor = detect_gpu_vendor(hints)
    fp8_support = gpu_arch in {"sm_90a", "sm_90", "sm_100", "sm_103", "gfx942", "gfx950"}
    fp8_format = (
        "OCP"
        if gpu_arch in {"sm_90a", "sm_90", "sm_100", "sm_103", "gfx950"}
        else ("FNUZ" if gpu_arch == "gfx942" else "")
    )
    model_name = hints.model_name
    if not model_name and identity is not None and len(identity.served_models) == 1:
        model_name = identity.served_models[0]

    engine = metrics.engine if metrics.engine != "unknown" else hints.engine
    scheduler = current_scheduler or {}
    return DeploymentContext(
        engine=engine,
        gpu_arch=gpu_arch,
        gpu_name=hints.gpu_name,
        gpu_vendor=gpu_vendor,
        model_name=model_name,
        model_type=hints.model_type,
        attention_type=hints.attention_type,
        experts_total=hints.experts_total,
        tp=hints.tp,
        ep=hints.ep,
        fp8_support=fp8_support,
        fp8_format=fp8_format,
        gpu_memory_utilization=hints.gpu_memory_utilization,
        kv_cache_dtype=hints.kv_cache_dtype,
        quantization=hints.quantization,
        block_size=hints.block_size,
        env_vars={},
        has_rdma=hints.has_rdma,
        split_prefill_decode=hints.split_prefill_decode,
        prefix_caching=True,
        max_num_batched_tokens=int(scheduler.get("batched_token_budget", 0) or 0),
    )


def detect_gpu_vendor(hints: RuntimeContextHints) -> str:
    """Best-effort GPU vendor inference from runtime hints."""
    gpu_arch = hints.gpu_arch.strip().lower()
    gpu_name = hints.gpu_name.strip().lower()
    if gpu_arch.startswith("gfx") or gpu_name.startswith("mi"):
        return "amd"
    if gpu_arch.startswith("sm"):
        return "nvidia"
    return ""


def _metric_value(metrics: NormalizedMetrics, name: str) -> float | None:
    mapping: dict[str, float | None] = {
        "ttft_avg_ms": (metrics.ttft_avg_s * 1000) if metrics.ttft_avg_s is not None else None,
        "itl_avg_ms": (metrics.itl_avg_s * 1000) if metrics.itl_avg_s is not None else None,
        "requests_running": metrics.requests_running,
        "requests_waiting": metrics.requests_waiting,
        "gen_throughput_tps": metrics.gen_throughput_tps,
        "kv_cache_usage": metrics.kv_cache_usage,
        "prefix_cache_hit_rate": metrics.prefix_cache_hit_rate,
        "cpu_cache_usage": metrics.cpu_cache_usage,
        "preemptions_total": metrics.preemptions_total,
    }
    return mapping.get(name)


def derive_bottlenecks(findings: list[AuditFinding], metrics: NormalizedMetrics) -> list[RuntimeBottleneck]:
    """Group findings into a stable bottleneck taxonomy."""
    grouped: dict[BottleneckType, list[AuditFinding]] = {}
    for finding in findings:
        kind = _BOTTLENECK_CHECK_MAP.get(finding.check_id)
        if kind is None:
            continue
        grouped.setdefault(kind, []).append(finding)

    bottlenecks: list[RuntimeBottleneck] = []
    for kind, grouped_findings in grouped.items():
        grouped_findings.sort(key=lambda finding: _SEVERITY_ORDER.get(finding.severity, 3))
        top = grouped_findings[0]
        confidence = max(finding.confidence for finding in grouped_findings)
        if len(grouped_findings) > 1:
            confidence = min(0.95, confidence + 0.05)
        supporting_metrics = {name: _metric_value(metrics, name) for name in _BOTTLENECK_METRICS[kind]}
        bottlenecks.append(
            RuntimeBottleneck(
                kind=kind,
                severity=_severity_literal(top.severity),
                confidence=round(confidence, 2),
                summary=top.title,
                trigger_check_ids=[finding.check_id for finding in grouped_findings],
                evidence=[f"{finding.check_id}: {finding.current_value}" for finding in grouped_findings],
                supporting_metrics=supporting_metrics,
            )
        )

    bottlenecks.sort(key=lambda bottleneck: (_SEVERITY_ORDER.get(bottleneck.severity, 3), bottleneck.kind.value))
    return bottlenecks


async def enrich_runtime_identity(
    endpoint: str,
    metrics: NormalizedMetrics,
    hints: RuntimeContextHints,
    *,
    allow_private: bool,
    metrics_auth: EndpointAuthConfig | None,
) -> RuntimeIdentity | None:
    """Best-effort runtime identity enrichment via engine adapters."""
    engine_candidate = metrics.engine
    engine_source: Literal["metrics", "adapter", "unknown"] = "metrics"
    if engine_candidate == "unknown":
        engine_candidate = hints.engine.strip().lower()
        engine_source = "adapter" if engine_candidate else "unknown"
    if not engine_candidate:
        return None

    try:
        adapter = get_engine_adapter(engine_candidate)
    except Exception as exc:  # noqa: BLE001
        return RuntimeIdentity(
            engine=engine_candidate,
            engine_source="unknown",
            config_error=str(exc),
            notes=["No registered engine adapter available for identity enrichment."],
        )

    try:
        base_endpoint = resolve_api_base_url(endpoint)
        payload = await adapter.get_config(base_endpoint, allow_private=allow_private, auth=metrics_auth)
    except Exception as exc:  # noqa: BLE001
        return RuntimeIdentity(
            engine=engine_candidate,
            engine_source=_engine_source_literal(engine_source),
            adapter_name=adapter.engine_name(),
            config_error=str(exc),
            notes=["Adapter config lookup failed during runtime profiling."],
        )

    served_models: list[str] = []
    config_snapshot: dict[str, Any] = {}
    notes: list[str] = []
    config_error = ""

    if isinstance(payload, dict):
        config_snapshot = payload
        data = payload.get("data")
        if isinstance(data, list):
            served_models = [
                str(item.get("id", "")).strip()
                for item in data
                if isinstance(item, dict) and str(item.get("id", "")).strip()
            ]
        elif isinstance(payload.get("model"), str):
            served_models = [str(payload["model"]).strip()]
        elif payload:
            notes.append("Runtime config payload did not match OpenAI-style /v1/models schema.")
        else:
            config_error = "Runtime config unavailable from engine adapter."
    else:
        config_error = "Engine adapter returned a non-dict config payload."

    return RuntimeIdentity(
        engine=engine_candidate,
        engine_source=_engine_source_literal(engine_source),
        adapter_name=adapter.engine_name(),
        served_models=served_models,
        config_snapshot=config_snapshot,
        config_error=config_error,
        notes=notes,
    )


def calculate_profile_confidence(bundle: RuntimeAnalysisBundle, hints: RuntimeContextHints) -> float:
    """Compute a coarse confidence score for the runtime profile."""
    confidence = 0.60
    if bundle.normalized.engine != "unknown":
        confidence += 0.15
    if hints.gpu_arch:
        confidence += 0.10
    if bundle.workload is not None and bundle.workload.confidence >= 0.60:
        confidence += 0.05
    if bundle.identity is not None and not bundle.identity.config_error:
        confidence += 0.05
    return min(confidence, 0.95)


def build_runtime_profile_summary(bundle: RuntimeAnalysisBundle, audit: dict[str, Any]) -> str:
    """Build the unified runtime profile summary."""
    workload = bundle.workload.mode.value if bundle.workload is not None else "unknown"
    return (
        f"{bundle.normalized.engine.upper()} runtime profile: "
        f"{audit['total']} finding(s), {len(bundle.bottlenecks)} bottleneck(s), "
        f"workload={workload}, status={bundle.health.get('status', 'unknown')}"
    )
