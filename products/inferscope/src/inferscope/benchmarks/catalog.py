"""Packaged workload/experiment resolution plus artifact comparison helpers."""

from __future__ import annotations

import json
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from inferscope.benchmarks.experiments import BenchmarkExperimentSpec
from inferscope.benchmarks.models import BenchmarkArtifact, WorkloadPack
from inferscope.benchmarks.procedural import (
    ProceduralWorkloadOptions,
    materialize_procedural_workload,
)

_RESOURCE_EXTENSIONS = (".yaml", ".yml", ".json")
_LEGACY_RESOURCE_PREFIXES: dict[str, tuple[str, ...]] = {
    "workload": (
        "benchmarks/workloads/",
        "src/inferscope/benchmarks/workloads/",
    ),
    "experiment": (
        "benchmarks/experiment_specs/",
        "src/inferscope/benchmarks/experiment_specs/",
    ),
}


def _find_packaged_resource(package: str, builtin_name: str) -> Path | None:
    resource_root = files(package)
    for extension in _RESOURCE_EXTENSIONS:
        candidate = resource_root.joinpath(f"{builtin_name}{extension}")
        if candidate.is_file():
            with as_file(candidate) as packaged_file:
                return packaged_file.resolve()
    return None


def _normalize_reference(reference: str | Path) -> str:
    normalized = str(reference).strip().replace("\\", "/")
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def _legacy_builtin_name(reference: str | Path, *, kind: str) -> str | None:
    normalized = _normalize_reference(reference)
    for prefix in _LEGACY_RESOURCE_PREFIXES[kind]:
        if not normalized.startswith(prefix):
            continue
        candidate = normalized[len(prefix) :]
        resource_path = Path(candidate)
        if resource_path.suffix.lower() not in _RESOURCE_EXTENSIONS:
            return None
        if resource_path.parent != Path("."):
            return None
        return resource_path.stem
    return None


def _list_packaged_resources(package: str) -> list[str]:
    resource_root = files(package)
    names: set[str] = set()
    for item in resource_root.iterdir():
        if item.is_file() and item.name.endswith(_RESOURCE_EXTENSIONS):
            names.add(Path(item.name).stem)
    return sorted(names)


def _resolve_packaged_resource(package: str, reference: str | Path, kind: str) -> Path:
    path = Path(reference)
    if path.exists():
        return path.resolve()

    builtin_name = str(reference).strip()
    if not builtin_name:
        raise ValueError(f"{kind} reference must not be empty")

    packaged = _find_packaged_resource(package, builtin_name)
    if packaged is not None:
        return packaged

    legacy_name = _legacy_builtin_name(reference, kind=kind)
    if legacy_name is not None:
        packaged = _find_packaged_resource(package, legacy_name)
        if packaged is not None:
            return packaged

    available = ", ".join(_list_packaged_resources(package))
    raise ValueError(f"Unknown {kind} reference '{reference}'. Available built-ins: {available}")


def resolve_workload_reference(reference: str | Path) -> Path:
    """Resolve a workload reference from a file path or packaged built-in name."""
    return _resolve_packaged_resource("inferscope.benchmarks.workloads", reference, "workload")


def load_workload(reference: str | Path) -> WorkloadPack:
    """Load a workload pack from a file path or built-in name."""
    return WorkloadPack.from_file(resolve_workload_reference(reference))


def materialize_workload(
    reference: str | Path,
    *,
    options: ProceduralWorkloadOptions | None = None,
) -> WorkloadPack:
    """Resolve a workload reference and optionally expand it procedurally."""
    if options is None or not options.enabled:
        return load_workload(reference)
    if Path(reference).exists():
        raise ValueError("Procedural generation is supported only for packaged built-in workloads, not explicit files")
    seed_pack = load_workload(reference)
    return materialize_procedural_workload(seed_pack, options)


def resolve_experiment_reference(reference: str | Path) -> Path:
    """Resolve an experiment reference from a file path or packaged built-in name."""
    return _resolve_packaged_resource("inferscope.benchmarks.experiment_specs", reference, "experiment")


def load_experiment(reference: str | Path) -> BenchmarkExperimentSpec:
    """Load a benchmark experiment spec from a file path or built-in name."""
    return BenchmarkExperimentSpec.from_file(resolve_experiment_reference(reference))


def load_benchmark_artifact(path: str | Path) -> BenchmarkArtifact:
    """Load a benchmark artifact from JSON."""
    file_path = Path(path)
    data = json.loads(file_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Benchmark artifact JSON must contain an object at the top level")
    return BenchmarkArtifact.model_validate(data)


def _delta(new_value: float | None, base_value: float | None) -> float | None:
    if new_value is None or base_value is None:
        return None
    return new_value - base_value


def _ratio(new_value: float | None, base_value: float | None) -> float | None:
    if new_value is None or base_value is None or base_value == 0:
        return None
    return new_value / base_value


def _run_plan_field(artifact: BenchmarkArtifact, field: str, default: Any) -> Any:
    if not artifact.run_plan:
        return default
    return artifact.run_plan.get(field, default)


def _observed_runtime(artifact: BenchmarkArtifact) -> dict[str, Any]:
    observed = _run_plan_field(artifact, "observed_runtime", {})
    return observed if isinstance(observed, dict) else {}


def _runtime_metric(artifact: BenchmarkArtifact, *path: str) -> float | None:
    current: Any = _observed_runtime(artifact)
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    if isinstance(current, bool):
        return None
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _cache_effectiveness_metric(artifact: BenchmarkArtifact, metric: str) -> float | None:
    nested_metric = _runtime_metric(artifact, "cache_effectiveness", metric)
    if nested_metric is not None:
        return nested_metric
    return _runtime_metric(artifact, metric)


def _topology_mode(artifact: BenchmarkArtifact) -> str:
    topology = _run_plan_field(artifact, "topology", {})
    if isinstance(topology, dict):
        return str(topology.get("mode", "single_endpoint"))
    return "single_endpoint"


def _cache_strategy(artifact: BenchmarkArtifact) -> str:
    cache = _run_plan_field(artifact, "cache", {})
    if isinstance(cache, dict):
        return str(cache.get("strategy", "unknown"))
    return "unknown"


def _metrics_roles(artifact: BenchmarkArtifact) -> list[str]:
    if artifact.run_plan and isinstance(artifact.run_plan.get("metrics_targets"), list):
        role_list = [
            str(target.get("role", "primary"))
            for target in artifact.run_plan["metrics_targets"]
            if isinstance(target, dict)
        ]
        return sorted(role_list) if role_list else ["primary"]
    if artifact.metrics_before_targets or artifact.metrics_after_targets:
        combined_snapshots = artifact.metrics_before_targets + artifact.metrics_after_targets
        role_set = {snapshot.target_role for snapshot in combined_snapshots}
        return sorted(role_set) if role_set else ["primary"]
    return ["primary"]


def compare_benchmark_artifacts(
    baseline: BenchmarkArtifact,
    candidate: BenchmarkArtifact,
) -> dict[str, Any]:
    """Compare two benchmark artifacts."""
    baseline_summary = baseline.summary
    candidate_summary = candidate.summary

    compatibility_warnings: list[str] = []
    differing_fields: list[str] = []

    comparable_fields = {
        "pack_name": (baseline.pack_name, candidate.pack_name),
        "workload_class": (baseline.workload_class, candidate.workload_class),
        "model": (baseline.model, candidate.model),
        "concurrency": (baseline.concurrency, candidate.concurrency),
        "topology_mode": (_topology_mode(baseline), _topology_mode(candidate)),
        "cache_strategy": (_cache_strategy(baseline), _cache_strategy(candidate)),
        "metrics_roles": (_metrics_roles(baseline), _metrics_roles(candidate)),
    }

    for field, (baseline_value, candidate_value) in comparable_fields.items():
        if baseline_value != candidate_value:
            differing_fields.append(field)
            compatibility_warnings.append(
                f"Different {field}: baseline={baseline_value!r} candidate={candidate_value!r}"
            )

    comparison: dict[str, Any] = {
        "baseline": {
            "path": baseline.default_filename,
            "pack_name": baseline.pack_name,
            "endpoint": baseline.endpoint,
            "model": baseline.model,
            "summary": baseline_summary.model_dump(mode="json"),
        },
        "candidate": {
            "path": candidate.default_filename,
            "pack_name": candidate.pack_name,
            "endpoint": candidate.endpoint,
            "model": candidate.model,
            "summary": candidate_summary.model_dump(mode="json"),
        },
        "compatibility": {
            "comparable": not differing_fields,
            "warnings": compatibility_warnings,
            "differing_fields": differing_fields,
        },
        "deltas": {
            "latency_p95_ms": _delta(candidate_summary.latency_p95_ms, baseline_summary.latency_p95_ms),
            "ttft_p90_ms": _delta(candidate_summary.ttft_p90_ms, baseline_summary.ttft_p90_ms),
            "ttft_p95_ms": _delta(candidate_summary.ttft_p95_ms, baseline_summary.ttft_p95_ms),
            "ttft_p99_ms": _delta(candidate_summary.ttft_p99_ms, baseline_summary.ttft_p99_ms),
            "latency_avg_ms": _delta(candidate_summary.latency_avg_ms, baseline_summary.latency_avg_ms),
            "wall_time_ms": _delta(candidate_summary.wall_time_ms, baseline_summary.wall_time_ms),
            "succeeded": candidate_summary.succeeded - baseline_summary.succeeded,
            "failed": candidate_summary.failed - baseline_summary.failed,
            "total_tokens": candidate_summary.total_tokens - baseline_summary.total_tokens,
            "request_throughput_rps": _delta(
                _runtime_metric(candidate, "request_throughput_rps"),
                _runtime_metric(baseline, "request_throughput_rps"),
            ),
            "output_throughput_tps": _delta(
                _runtime_metric(candidate, "output_throughput_tps"),
                _runtime_metric(baseline, "output_throughput_tps"),
            ),
            "goodput_rps": _delta(
                _runtime_metric(candidate, "goodput_rps"),
                _runtime_metric(baseline, "goodput_rps"),
            ),
            "tpot_p95_ms": _delta(
                _runtime_metric(candidate, "tpot_ms", "p95"),
                _runtime_metric(baseline, "tpot_ms", "p95"),
            ),
            "itl_p95_ms": _delta(
                _runtime_metric(candidate, "itl_ms", "p95"),
                _runtime_metric(baseline, "itl_ms", "p95"),
            ),
            "tool_parse_success_rate": _delta(
                _runtime_metric(candidate, "tool_parse_success_rate"),
                _runtime_metric(baseline, "tool_parse_success_rate"),
            ),
            "prefix_cache_hit_rate": _delta(
                _cache_effectiveness_metric(candidate, "prefix_cache_hit_rate"),
                _cache_effectiveness_metric(baseline, "prefix_cache_hit_rate"),
            ),
            "prefix_cache_hits": _delta(
                _cache_effectiveness_metric(candidate, "prefix_cache_hits"),
                _cache_effectiveness_metric(baseline, "prefix_cache_hits"),
            ),
        },
        "ratios": {
            "latency_p95": _ratio(candidate_summary.latency_p95_ms, baseline_summary.latency_p95_ms),
            "ttft_p90": _ratio(candidate_summary.ttft_p90_ms, baseline_summary.ttft_p90_ms),
            "ttft_p95": _ratio(candidate_summary.ttft_p95_ms, baseline_summary.ttft_p95_ms),
            "ttft_p99": _ratio(candidate_summary.ttft_p99_ms, baseline_summary.ttft_p99_ms),
            "wall_time": _ratio(candidate_summary.wall_time_ms, baseline_summary.wall_time_ms),
            "request_throughput": _ratio(
                _runtime_metric(candidate, "request_throughput_rps"),
                _runtime_metric(baseline, "request_throughput_rps"),
            ),
            "output_throughput": _ratio(
                _runtime_metric(candidate, "output_throughput_tps"),
                _runtime_metric(baseline, "output_throughput_tps"),
            ),
            "goodput": _ratio(
                _runtime_metric(candidate, "goodput_rps"),
                _runtime_metric(baseline, "goodput_rps"),
            ),
        },
    }

    latency_delta = comparison["deltas"]["latency_p95_ms"]
    ttft_delta = comparison["deltas"]["ttft_p95_ms"]
    summary_parts = []
    if latency_delta is not None:
        direction = "faster" if latency_delta < 0 else "slower" if latency_delta > 0 else "unchanged"
        summary_parts.append(f"p95 latency {direction} by {abs(latency_delta):.1f} ms")
    if ttft_delta is not None:
        direction = "lower" if ttft_delta < 0 else "higher" if ttft_delta > 0 else "unchanged"
        summary_parts.append(f"p95 TTFT {direction} by {abs(ttft_delta):.1f} ms")
    if compatibility_warnings:
        summary_parts.append(f"compatibility warnings: {len(compatibility_warnings)}")
    if not summary_parts:
        summary_parts.append("comparison computed")
    comparison["summary"] = " | ".join(summary_parts)
    return comparison
