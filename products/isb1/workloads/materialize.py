"""Shared workload materialization helpers for ISB-1.

These helpers turn canonical workload configs into concrete ``Request`` traces.
They are used both by the offline trace-generation script and by the runtime
benchmark harness, so benchmark execution and pre-generated traces stay aligned.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import yaml

from workloads.base import Request, WorkloadGenerator

_PRODUCT_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_GENERATORS: dict[str, tuple[str, str]] = {
    "chat": ("workloads.chat", "ChatWorkloadGenerator"),
    "agent": ("workloads.agent", "AgentTraceGenerator"),
    "rag": ("workloads.rag", "RAGTraceGenerator"),
    "coding": ("workloads.coding", "CodingTraceGenerator"),
}

_DEFAULT_NUM_REQUESTS = 1000


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_workload_config(
    workload_name: str,
    config_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load one workload config and return ``(config, path)``."""
    root = Path(config_root) if config_root is not None else (_PRODUCT_ROOT / "configs")
    path = root / "workloads" / f"{workload_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Workload config '{workload_name}' not found at {path}")
    return _load_yaml(path), path


def resolve_generator_class(workload_cfg: dict[str, Any], workload_name: str) -> type[WorkloadGenerator]:
    """Resolve the generator class from config or default mapping."""
    trace_cfg = workload_cfg.get("trace", {})
    generator_path = str(trace_cfg.get("generator", "")).strip()

    if generator_path:
        parts = generator_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid generator path '{generator_path}' for workload '{workload_name}'"
            )
        module_path, class_name = parts
    elif workload_name in _DEFAULT_GENERATORS:
        module_path, class_name = _DEFAULT_GENERATORS[workload_name]
    else:
        raise ValueError(
            f"No generator mapping found for workload '{workload_name}'. "
            "Set trace.generator in the workload config."
        )

    module = importlib.import_module(module_path)
    generator_cls = getattr(module, class_name)
    return generator_cls


def _filter_constructor_kwargs(
    generator_cls: type[WorkloadGenerator],
    raw_kwargs: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(generator_cls.__init__)
    supported = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {key: value for key, value in raw_kwargs.items() if key in supported}


def build_generator_kwargs(workload_cfg: dict[str, Any], workload_name: str) -> dict[str, Any]:
    """Extract generator kwargs from the workload config.

    The benchmark configs are richer than the current generator constructors, so
    this function builds a superset and then filters it against the generator
    signature before instantiation.
    """
    trace_cfg = workload_cfg.get("trace", {})
    raw_kwargs: dict[str, Any] = {"seed": int(trace_cfg.get("seed", 42))}

    if workload_name == "chat":
        trace_filter = trace_cfg.get("filter", {})
        source = str(trace_cfg.get("source", "")).strip()
        if source:
            raw_kwargs["sharegpt_path"] = source
        if "min_turns" in trace_filter:
            raw_kwargs["min_turns"] = int(trace_filter["min_turns"])
        if "max_turns" in trace_filter:
            raw_kwargs["max_turns"] = int(trace_filter["max_turns"])

    if workload_name == "agent":
        turns_cfg = trace_cfg.get("turns", {})
        schemas = trace_cfg.get("tool_schemas", [])
        if "min" in turns_cfg:
            raw_kwargs["min_turns"] = int(turns_cfg["min"])
        if "max" in turns_cfg:
            raw_kwargs["max_turns"] = int(turns_cfg["max"])
        if schemas:
            # The generator accepts a directory rather than individual file names.
            # The canonical schema files already live under workloads/schemas/.
            raw_kwargs["schemas_dir"] = Path(__file__).resolve().parent / "schemas"

    if workload_name == "coding":
        num_files_cfg = trace_cfg.get("num_files", {})
        turns_cfg = trace_cfg.get("conversation_turns", {})
        if "min" in num_files_cfg:
            raw_kwargs["min_files"] = int(num_files_cfg["min"])
        if "max" in num_files_cfg:
            raw_kwargs["max_files"] = int(num_files_cfg["max"])
        if "min" in turns_cfg:
            raw_kwargs["min_turns"] = int(turns_cfg["min"])
        if "max" in turns_cfg:
            raw_kwargs["max_turns"] = int(turns_cfg["max"])

    generator_cls = resolve_generator_class(workload_cfg, workload_name)
    return _filter_constructor_kwargs(generator_cls, raw_kwargs)


def default_request_count(workload_cfg: dict[str, Any]) -> int:
    """Return the configured request-pool size for a workload."""
    configured = workload_cfg.get("trace", {}).get("num_requests")
    if configured is None:
        return _DEFAULT_NUM_REQUESTS
    value = int(configured)
    if value < 1:
        raise ValueError("trace.num_requests must be >= 1")
    return value


def materialize_requests(
    workload_name: str,
    *,
    config_root: str | Path | None = None,
    num_requests: int | None = None,
) -> list[Request]:
    """Generate concrete requests for one workload."""
    workload_cfg, _ = load_workload_config(workload_name, config_root=config_root)
    generator_cls = resolve_generator_class(workload_cfg, workload_name)
    generator_kwargs = build_generator_kwargs(workload_cfg, workload_name)
    generator = generator_cls(**generator_kwargs)
    requested = num_requests if num_requests is not None else default_request_count(workload_cfg)
    return generator.generate(int(requested))


def default_trace_path(workload_name: str, traces_root: str | Path | None = None) -> Path:
    """Return the default saved trace path for one workload."""
    root = Path(traces_root) if traces_root is not None else (_PRODUCT_ROOT / "traces")
    return root / f"{workload_name}.jsonl"


def save_requests(requests: list[Request], path: str | Path) -> Path:
    """Persist a materialized request trace to JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for request in requests:
            fh.write(json.dumps(request.to_dict(), ensure_ascii=False) + "\n")
    return output_path
