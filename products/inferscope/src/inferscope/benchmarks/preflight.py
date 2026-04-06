"""Benchmark preflight validation for local model artifacts and runtime fit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.models.registry import get_model_variant
from inferscope.optimization.memory_planner import plan_memory
from inferscope.optimization.platform_policy import resolve_platform_traits

ArtifactKind = Literal["huggingface_weights", "compiled_engine"]
ArtifactEngine = Literal["vllm", "sglang", "trtllm", "dynamo", "atom"]

_HF_WEIGHT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
)
_HF_WEIGHT_GLOBS = ("*.safetensors", "*.safetensors.index.json", "*.bin", "*.gguf")
_COMPILED_ENGINE_FILES = ("config.json",)
_COMPILED_ENGINE_GLOBS = ("*.engine", "*.plan")


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _default_quantization(model_name: str, gpu_name: str) -> str:
    model = get_model_variant(model_name)
    gpu = get_gpu_profile(gpu_name)
    if model is None or gpu is None:
        return "bf16"
    if gpu.fp4_support:
        name = gpu.name.lower()
        if "b200" in name and isinstance(model.serving.get("tp_fp4_b200"), int):
            return "fp4"
        if "b300" in name and isinstance(model.serving.get("tp_fp4_b300"), int):
            return "fp4"
    if gpu.fp8_support:
        return "fp8"
    return "bf16"


def _normalize_precision(value: str | None, *, model_name: str, gpu_name: str) -> str:
    quantization = (value or "").strip().lower()
    if not quantization or quantization == "auto":
        return _default_quantization(model_name, gpu_name)
    if quantization in {"fp8_e4m3", "fp8_e5m2"}:
        return "fp8"
    if quantization in {"nvfp4", "mxfp4"}:
        return quantization
    return quantization


def _infer_artifact_kind(engine_name: str) -> ArtifactKind:
    return "compiled_engine" if _normalize(engine_name) == "trtllm" else "huggingface_weights"


def _effective_tensor_parallelism(num_gpus: int, topology_mode: str) -> int:
    normalized = _normalize(topology_mode)
    if normalized == "prefill_decode_split":
        return max(num_gpus // 2, 1)
    if normalized == "router_prefill_decode":
        return max((num_gpus - 1) // 2, 1) if num_gpus > 2 else 1
    return max(num_gpus, 1)


def _has_any_match(path: Path, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if any(True for _ in path.glob(pattern)):
            return True
    return False


def _manifest_file_data(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("artifact manifest must contain a top-level mapping/object")
    return payload


class BenchmarkArtifactManifest(BaseModel):
    """Optional local artifact manifest used for stricter compatibility checks."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    artifact_kind: ArtifactKind = "huggingface_weights"
    model: str
    engine: ArtifactEngine | None = None
    quantization: str | None = None
    tensor_parallel_size: int | None = Field(default=None, ge=1)
    gpu_family: str | None = None
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> BenchmarkArtifactManifest:
        return cls.model_validate(_manifest_file_data(Path(path)))


class BenchmarkPreflightValidation(BaseModel):
    """Preflight validation bundle attached to a resolved benchmark plan."""

    model_config = ConfigDict(extra="forbid")

    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    info: list[str] = Field(default_factory=list)
    model_artifact_path: str | None = None
    artifact_manifest_path: str | None = None
    manifest: BenchmarkArtifactManifest | None = None
    memory_plan: dict[str, Any] | None = None


def _validate_quantization(
    *,
    quantization: str,
    gpu_name: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    gpu = get_gpu_profile(gpu_name)
    if gpu is None:
        return
    if quantization in {"fp8", "fp8_e4m3", "fp8_e5m2"} and not gpu.fp8_support:
        errors.append(f"{gpu.name} does not support FP8 artifacts or dynamic FP8 quantization.")
    if quantization in {"fp4", "nvfp4", "mxfp4"} and not gpu.fp4_support:
        errors.append(f"{gpu.name} does not support FP4 artifacts or FP4 runtime quantization.")
    if quantization == "nvfp4" and gpu.fp4_support and gpu.fp4_format != "NVFP4":
        errors.append(f"{gpu.name} supports {gpu.fp4_format}, not NVFP4.")
    if quantization == "mxfp4" and gpu.fp4_support and gpu.fp4_format != "MXFP4":
        errors.append(f"{gpu.name} supports {gpu.fp4_format}, not MXFP4.")
    if quantization in {"awq", "gptq"} and gpu.fp8_support:
        warnings.append(
            f"{gpu.name} supports native FP8; AWQ/GPTQ is usually a budget path rather than the preferred artifact format."
        )


def _validate_manifest(
    *,
    manifest: BenchmarkArtifactManifest,
    model_name: str,
    engine_name: str,
    gpu_name: str,
    num_gpus: int,
    errors: list[str],
    warnings: list[str],
    info: list[str],
) -> None:
    resolved_model = get_model_variant(model_name)
    manifest_model = get_model_variant(manifest.model)
    if manifest_model is None:
        errors.append(f"Artifact manifest model '{manifest.model}' does not resolve to a known model variant.")
    elif resolved_model is not None and manifest_model.name != resolved_model.name:
        errors.append(
            f"Artifact manifest model '{manifest_model.name}' does not match the selected benchmark model '{resolved_model.name}'."
        )

    selected_engine = _normalize(engine_name)
    manifest_engine = _normalize(manifest.engine or "")
    if manifest_engine:
        if selected_engine == "dynamo" and manifest.artifact_kind == "huggingface_weights":
            if manifest_engine not in {"dynamo", "vllm", "atom"}:
                errors.append(
                    "Dynamo weight artifacts must be declared as a Dynamo-compatible weight source "
                    f"(got engine '{manifest.engine}')."
                )
        elif selected_engine and manifest_engine != selected_engine:
            errors.append(
                f"Artifact manifest engine '{manifest.engine}' does not match the requested engine '{engine_name}'."
            )

    if manifest.tensor_parallel_size is not None:
        if manifest.artifact_kind == "compiled_engine" and manifest.tensor_parallel_size != num_gpus:
            errors.append(
                f"Compiled engine manifest expects TP={manifest.tensor_parallel_size}, but the requested run uses {num_gpus} GPU(s)."
            )
        elif manifest.artifact_kind == "huggingface_weights" and manifest.tensor_parallel_size != num_gpus:
            warnings.append(
                f"Weight manifest was recorded with TP={manifest.tensor_parallel_size}; verify the requested {num_gpus}-GPU topology matches your sharding plan."
            )

    if manifest.gpu_family and gpu_name:
        gpu = get_gpu_profile(gpu_name)
        if gpu is not None:
            gpu_family = _normalize(resolve_platform_traits(gpu).family.value)
            manifest_family = _normalize(manifest.gpu_family)
            if manifest_family != gpu_family:
                message = (
                    f"Artifact manifest targets GPU family '{manifest.gpu_family}', but '{gpu.name}' resolves to '{gpu_family}'."
                )
                if manifest.artifact_kind == "compiled_engine":
                    errors.append(message)
                else:
                    warnings.append(message)

    info.append(
        "Artifact manifest loaded"
        + (
            f" ({manifest.artifact_kind}, engine={manifest.engine or 'unspecified'})"
            if manifest.engine or manifest.artifact_kind
            else ""
        )
    )


def _validate_artifact_directory(
    *,
    path: Path,
    artifact_kind: ArtifactKind,
    engine_name: str,
    errors: list[str],
    warnings: list[str],
    info: list[str],
) -> None:
    if not path.exists():
        errors.append(f"Model artifact path '{path}' does not exist.")
        return
    if not path.is_dir():
        errors.append(f"Model artifact path '{path}' must be a directory.")
        return

    required_files = _COMPILED_ENGINE_FILES if artifact_kind == "compiled_engine" else ("config.json",)
    missing_files = [name for name in required_files if not (path / name).exists()]
    if missing_files:
        errors.append(
            f"Artifact directory '{path}' is missing required files: {', '.join(sorted(missing_files))}."
        )

    if artifact_kind == "compiled_engine":
        has_engine_blob = _has_any_match(path, _COMPILED_ENGINE_GLOBS)
        if not has_engine_blob:
            errors.append(
                f"Compiled engine directory '{path}' must contain at least one TensorRT-LLM engine blob (*.engine or *.plan)."
            )
    else:
        has_tokenizer = any((path / name).exists() for name in _HF_WEIGHT_FILES[1:])
        has_weights = _has_any_match(path, _HF_WEIGHT_GLOBS)
        if not has_tokenizer:
            errors.append(
                f"Model artifact directory '{path}' is missing tokenizer metadata (tokenizer.json, tokenizer_config.json, or tokenizer.model)."
            )
        if not has_weights:
            errors.append(
                f"Model artifact directory '{path}' is missing weights (*.safetensors, *.bin, or *.gguf)."
            )

    selected_engine = _normalize(engine_name)
    if selected_engine == "trtllm" and artifact_kind != "compiled_engine":
        errors.append("TensorRT-LLM runs require a compiled engine directory, not raw HuggingFace weights.")
    elif selected_engine in {"vllm", "sglang", "atom"} and artifact_kind == "compiled_engine":
        warnings.append(
            f"{engine_name} usually serves raw model weights rather than a compiled engine directory; verify this path is intentional."
        )
    elif selected_engine == "dynamo" and artifact_kind == "compiled_engine":
        warnings.append(
            "Dynamo usually manages backend loading from model weights; compiled-engine artifacts are only valid if your deployment path explicitly expects them."
        )

    info.append(f"Artifact directory checked as {artifact_kind}: {path}")


def validate_benchmark_preflight(
    *,
    model_name: str,
    gpu_name: str = "",
    num_gpus: int | None = None,
    engine_name: str = "",
    topology_mode: str = "single_endpoint",
    model_artifact_path: str = "",
    artifact_manifest: str = "",
) -> BenchmarkPreflightValidation:
    """Validate benchmark inputs before attempting live replay."""

    report = BenchmarkPreflightValidation(
        model_artifact_path=(str(Path(model_artifact_path).expanduser()) if model_artifact_path else None),
        artifact_manifest_path=(str(Path(artifact_manifest).expanduser()) if artifact_manifest else None),
    )
    selected_num_gpus = max(num_gpus or 1, 1)
    effective_tp = _effective_tensor_parallelism(selected_num_gpus, topology_mode)

    manifest: BenchmarkArtifactManifest | None = None
    if artifact_manifest:
        manifest_path = Path(artifact_manifest).expanduser()
        try:
            manifest = BenchmarkArtifactManifest.from_file(manifest_path)
            report.manifest = manifest
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"Failed to load artifact manifest '{manifest_path}': {exc}")

    model = get_model_variant(model_name)
    if model is None:
        report.errors.append(f"Unknown model '{model_name}' for benchmark preflight validation.")

    gpu = get_gpu_profile(gpu_name) if gpu_name else None
    if gpu_name and gpu is None:
        report.errors.append(f"Unknown GPU '{gpu_name}' for benchmark preflight validation.")
    elif gpu is None:
        report.warnings.append("No GPU identity supplied; memory-fit and quantization validation is advisory only.")

    quantization = _normalize_precision(
        manifest.quantization if manifest is not None else None,
        model_name=model_name,
        gpu_name=gpu_name,
    )
    if gpu_name:
        _validate_quantization(
            quantization=quantization,
            gpu_name=gpu_name,
            errors=report.errors,
            warnings=report.warnings,
        )

    if manifest is not None:
        _validate_manifest(
            manifest=manifest,
            model_name=model_name,
            engine_name=engine_name,
            gpu_name=gpu_name,
            num_gpus=effective_tp,
            errors=report.errors,
            warnings=report.warnings,
            info=report.info,
        )

    if model_artifact_path:
        artifact_path = Path(model_artifact_path).expanduser()
        artifact_kind = manifest.artifact_kind if manifest is not None else _infer_artifact_kind(engine_name)
        _validate_artifact_directory(
            path=artifact_path,
            artifact_kind=artifact_kind,
            engine_name=engine_name,
            errors=report.errors,
            warnings=report.warnings,
            info=report.info,
        )

    if model is not None and gpu is not None:
        memory_plan = plan_memory(
            model=model,
            gpu=gpu,
            num_gpus=effective_tp,
            tp=effective_tp,
            precision=quantization,
        )
        report.memory_plan = memory_plan.to_dict()
        if not memory_plan.fits:
            report.errors.append(
                f"Selected configuration does not fit in GPU memory: {model.name} needs {memory_plan.weight_gb:.1f} GB/GPU "
                f"at {quantization}, but only {memory_plan.usable_memory_gb / effective_tp:.1f} GB usable/GPU is available on {gpu.name}."
            )
        else:
            report.info.append(
                f"Memory fit OK: {model.name} on {gpu.name} with TP={effective_tp} leaves "
                f"{memory_plan.kv_cache_budget_gb:.1f} GB KV budget."
            )
        if effective_tp != selected_num_gpus:
            report.info.append(
                f"Memory fit evaluated per worker group: total_gpus={selected_num_gpus}, topology={topology_mode}, effective_tp={effective_tp}."
            )

    report.valid = not report.errors
    return report
