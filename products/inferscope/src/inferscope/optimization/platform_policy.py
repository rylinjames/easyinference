"""Shared platform policy for engine selection and Hopper/Blackwell tuning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from inferscope.hardware.gpu_profiles import GPUProfile
from inferscope.models.registry import ModelVariant
from inferscope.optimization.serving_profile import EngineType, PrecisionSpec, WorkloadMode
from inferscope.production_target import is_target_engine, is_target_gpu, is_target_model


class PlatformFamily(StrEnum):
    """Normalized hardware platform families used across the optimizer."""

    OTHER = "other"
    AMPERE = "ampere"
    HOPPER = "hopper"
    HOPPER_PCIE = "hopper_pcie"
    HOPPER_GRACE = "hopper_grace"
    BLACKWELL = "blackwell"
    BLACKWELL_ULTRA = "blackwell_ultra"
    BLACKWELL_GRACE = "blackwell_grace"
    BLACKWELL_ULTRA_GRACE = "blackwell_ultra_grace"
    CDNA3 = "cdna3"
    CDNA4 = "cdna4"


class EngineSupportTier(StrEnum):
    """Public support tier for an engine on a platform."""

    RECOMMENDED = "recommended"
    SUPPORTED = "supported"
    PREVIEW = "preview"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class PlatformTraits:
    """Resolved platform capabilities derived from a GPU profile."""

    family: PlatformFamily
    gpu_type: str
    vendor: str
    architecture: str
    is_nvidia: bool
    is_amd: bool
    is_ampere: bool
    is_hopper: bool
    is_hopper_pcie: bool
    is_h200: bool
    is_blackwell: bool
    is_b300: bool
    is_grace: bool
    is_gh200: bool
    is_gb200: bool
    is_gb300: bool
    has_high_speed_interconnect: bool
    has_nvlink5: bool
    has_decompression_engine: bool
    has_helix_parallelism: bool
    has_accelerated_softmax: bool
    grace_memory_gb: float
    grace_memory_bandwidth_gb_s: float
    c2c_bandwidth_gb_s: float
    overflow_tier: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "gpu_type": self.gpu_type,
            "vendor": self.vendor,
            "architecture": self.architecture,
            "is_ampere": self.is_ampere,
            "is_hopper": self.is_hopper,
            "is_hopper_pcie": self.is_hopper_pcie,
            "is_h200": self.is_h200,
            "is_blackwell": self.is_blackwell,
            "is_b300": self.is_b300,
            "is_grace": self.is_grace,
            "is_gh200": self.is_gh200,
            "is_gb200": self.is_gb200,
            "is_gb300": self.is_gb300,
            "has_high_speed_interconnect": self.has_high_speed_interconnect,
            "has_nvlink5": self.has_nvlink5,
            "has_decompression_engine": self.has_decompression_engine,
            "has_helix_parallelism": self.has_helix_parallelism,
            "has_accelerated_softmax": self.has_accelerated_softmax,
            "grace_memory_gb": self.grace_memory_gb,
            "grace_memory_bandwidth_gb_s": self.grace_memory_bandwidth_gb_s,
            "c2c_bandwidth_gb_s": self.c2c_bandwidth_gb_s,
            "overflow_tier": self.overflow_tier,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EngineSupport:
    """Support status for a specific engine on a specific platform."""

    engine: str
    tier: EngineSupportTier
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "engine": self.engine,
            "support_tier": self.tier.value,
            "support_reason": self.reason,
        }


def resolve_platform_traits(gpu: GPUProfile) -> PlatformTraits:
    """Resolve normalized platform traits from a GPU profile."""

    extra = gpu.extra or {}
    is_nvidia = gpu.vendor == "nvidia"
    is_amd = gpu.vendor == "amd"
    is_ampere = gpu.architecture == "Ampere"
    is_hopper = gpu.architecture == "Hopper"
    is_blackwell = gpu.architecture == "Blackwell"
    is_hopper_pcie = is_hopper and gpu.compute_capability == "sm_90"
    is_h200 = is_hopper and gpu.memory_gb >= 140
    is_grace = extra.get("grace_cpu_cores", 0) > 0
    is_gh200 = is_hopper and is_grace
    is_gb200 = is_blackwell and is_grace and gpu.compute_capability == "sm_100"
    is_gb300 = is_blackwell and is_grace and gpu.compute_capability == "sm_103"
    is_b300 = is_blackwell and gpu.compute_capability == "sm_103" and not is_grace
    has_high_speed_interconnect = bool(gpu.nvlink_bandwidth_gb_s or gpu.if_bandwidth_gb_s)
    has_nvlink5 = gpu.nvlink_version >= 5

    if is_ampere:
        family = PlatformFamily.AMPERE
    elif is_hopper and is_grace:
        family = PlatformFamily.HOPPER_GRACE
    elif is_hopper_pcie:
        family = PlatformFamily.HOPPER_PCIE
    elif is_hopper:
        family = PlatformFamily.HOPPER
    elif is_gb300:
        family = PlatformFamily.BLACKWELL_ULTRA_GRACE
    elif is_b300:
        family = PlatformFamily.BLACKWELL_ULTRA
    elif is_gb200:
        family = PlatformFamily.BLACKWELL_GRACE
    elif is_blackwell:
        family = PlatformFamily.BLACKWELL
    elif gpu.compute_capability == "gfx942":
        family = PlatformFamily.CDNA3
    elif gpu.compute_capability == "gfx950":
        family = PlatformFamily.CDNA4
    else:
        family = PlatformFamily.OTHER

    notes: list[str] = []
    if is_gh200:
        notes.append("Grace Hopper coherent overflow is available for KV spill and staging.")
    if is_gb200:
        notes.append("GB200 exposes Grace LPDDR5X over NVLink-C2C for coherent overflow.")
    if is_gb300:
        notes.append("GB300 combines Blackwell Ultra compute with Grace coherent overflow.")
    if is_b300:
        notes.append("B300 adds accelerated softmax and larger HBM headroom without Grace.")

    return PlatformTraits(
        family=family,
        gpu_type=gpu.name,
        vendor=gpu.vendor,
        architecture=gpu.architecture,
        is_nvidia=is_nvidia,
        is_amd=is_amd,
        is_ampere=is_ampere,
        is_hopper=is_hopper,
        is_hopper_pcie=is_hopper_pcie,
        is_h200=is_h200,
        is_blackwell=is_blackwell,
        is_b300=is_b300,
        is_grace=is_grace,
        is_gh200=is_gh200,
        is_gb200=is_gb200,
        is_gb300=is_gb300,
        has_high_speed_interconnect=has_high_speed_interconnect,
        has_nvlink5=has_nvlink5,
        has_decompression_engine=bool(extra.get("decompression_engine")),
        has_helix_parallelism=bool(extra.get("helix_parallelism")),
        has_accelerated_softmax=bool(extra.get("accelerated_softmax")),
        grace_memory_gb=float(extra.get("grace_memory_gb", 0.0) or 0.0),
        grace_memory_bandwidth_gb_s=float(extra.get("grace_memory_bandwidth_gb_s", 0.0) or 0.0),
        c2c_bandwidth_gb_s=float(extra.get("nvlink_c2c_bandwidth_gb_s", 0.0) or 0.0),
        overflow_tier="gpu_grace_coherent" if is_grace else "gpu_only",
        notes=notes,
    )


def resolve_engine_support(engine: EngineType | str, gpu: GPUProfile, multi_node: bool = False) -> EngineSupport:
    """Return support tier metadata for an engine on a platform."""

    engine_name = engine.value if isinstance(engine, EngineType) else str(engine).lower().strip()
    traits = resolve_platform_traits(gpu)

    if engine_name == "vllm":
        if not traits.is_nvidia:
            return EngineSupport(
                engine="vllm",
                tier=EngineSupportTier.UNSUPPORTED,
                reason="The supported vLLM comparison lane is limited to NVIDIA Hopper and Blackwell GPUs.",
            )
        if not is_target_gpu(gpu):
            return EngineSupport(
                engine="vllm",
                tier=EngineSupportTier.UNSUPPORTED,
                reason="Supported GPUs are limited to H100, H200, B200, and B300 variants.",
            )
        return EngineSupport(
            engine="vllm",
            tier=EngineSupportTier.PREVIEW,
            reason=(
                "vLLM is supported as a benchmark comparison lane for the Kimi long-context "
                "coding target, but Dynamo remains the primary serving engine."
            ),
        )

    if not is_target_engine(engine_name):
        return EngineSupport(
            engine=engine_name,
            tier=EngineSupportTier.UNSUPPORTED,
            reason="InferScope is productized as Dynamo-only for the production MCP and benchmark lane.",
        )

    if not traits.is_nvidia:
        return EngineSupport(
            engine="dynamo",
            tier=EngineSupportTier.UNSUPPORTED,
            reason="NVIDIA Dynamo requires NVIDIA Hopper or Blackwell GPUs.",
        )

    if not is_target_gpu(gpu):
        return EngineSupport(
            engine="dynamo",
            tier=EngineSupportTier.UNSUPPORTED,
            reason="Supported GPUs are limited to H100, H200, B200, and B300 variants.",
        )

    if traits.is_hopper_pcie and multi_node:
        return EngineSupport(
            engine="dynamo",
            tier=EngineSupportTier.SUPPORTED,
            reason=(
                "Dynamo is supported on H100 PCIe, but disaggregated LMCache lanes are transport-sensitive "
                "without NVLink-class bandwidth or RDMA."
            ),
        )

    return EngineSupport(
        engine="dynamo",
        tier=EngineSupportTier.RECOMMENDED,
        reason=(
            "Dynamo + LMCache is the supported production lane for long-context coding on "
            "H100/H200/B200/B300, including single-endpoint and prefill/decode-split topologies."
        ),
    )


def resolve_preferred_precision(
    model: ModelVariant,
    gpu: GPUProfile,
    workload: WorkloadMode,
    num_gpus: int = 1,
) -> tuple[PrecisionSpec, str]:
    """Return the preferred precision policy for a model/GPU/workload."""

    traits = resolve_platform_traits(gpu)
    kv_cache = "fp8_e4m3" if gpu.fp8_support else "auto"

    if not is_target_gpu(gpu) or not is_target_model(model):
        if gpu.fp8_support:
            return (
                PrecisionSpec(weights="fp8", activations="fp8", kv_cache=kv_cache),
                "Outside the production target scope; defaulting to conservative FP8 on capable NVIDIA hardware.",
            )
        return (
            PrecisionSpec(weights="bf16", activations="bf16", kv_cache="auto"),
            "Outside the production target scope; defaulting to BF16.",
        )

    if workload != WorkloadMode.CODING:
        return (
            PrecisionSpec(weights="fp8", activations="fp8", kv_cache=kv_cache),
            "InferScope's production lane is tuned for coding; using FP8 as the conservative fallback.",
        )

    if (
        model.name == "Kimi-K2.5"
        and traits.is_blackwell
        and gpu.fp4_support
        and _precision_fits_target_lane(model, gpu, max(num_gpus, 1), "fp4")
    ):
        return (
            PrecisionSpec(weights="fp4", activations="fp8", kv_cache=kv_cache),
            "Blackwell NVFP4 is preferred for Kimi-K2.5 when it preserves long-context KV headroom.",
        )

    return (
        PrecisionSpec(weights="fp8", activations="fp8", kv_cache=kv_cache),
        "FP8 is the default production precision for Dynamo long-context coding on Hopper and Blackwell.",
    )


def resolve_preferred_tp(
    model: ModelVariant,
    gpu: GPUProfile,
    num_gpus: int,
    precision: str,
    workload: WorkloadMode,
) -> tuple[int | None, str | None]:
    """Resolve TP from target-model hints and KV-headroom heuristics."""

    valid_tps = _valid_tps(model, num_gpus)
    if not valid_tps:
        return None, None

    traits = resolve_platform_traits(gpu)
    normalized_precision = precision.lower()
    if not is_target_gpu(gpu) or not is_target_model(model):
        for candidate in valid_tps:
            if _tp_fits_memory(model, gpu, candidate, normalized_precision):
                return candidate, "Selected the smallest memory-valid TP outside the production target scope."
        return None, "No valid TP fits in GPU memory."

    hints: list[tuple[int, str]] = []
    if normalized_precision == "fp8":
        if traits.is_h200 and isinstance(model.serving.get("tp_fp8_h200"), int):
            hints.append((model.serving["tp_fp8_h200"], "Using model-specific H200 FP8 tensor-parallel hint."))
        elif traits.is_hopper and isinstance(model.serving.get("tp_fp8_h100"), int):
            hints.append((model.serving["tp_fp8_h100"], "Using model-specific H100 FP8 tensor-parallel hint."))
        elif traits.is_b300 and isinstance(model.serving.get("tp_fp8_b300"), int):
            hints.append((model.serving["tp_fp8_b300"], "Using model-specific B300 FP8 tensor-parallel hint."))
        elif traits.is_blackwell and isinstance(model.serving.get("tp_fp8_b200"), int):
            hints.append((model.serving["tp_fp8_b200"], "Using model-specific B200 FP8 tensor-parallel hint."))
        elif isinstance(model.serving.get("tp_fp8"), int):
            hints.append((model.serving["tp_fp8"], "Using model-specific FP8 tensor-parallel hint."))
    elif normalized_precision in {"fp4", "nvfp4", "mxfp4"}:
        if traits.is_b300 and isinstance(model.serving.get("tp_fp4_b300"), int):
            hints.append((model.serving["tp_fp4_b300"], "Using model-specific B300 FP4 tensor-parallel hint."))
        elif traits.is_blackwell and isinstance(model.serving.get("tp_fp4_b200"), int):
            hints.append((model.serving["tp_fp4_b200"], "Using model-specific B200 FP4 tensor-parallel hint."))
        else:
            fp4_hint = _extract_tp_from_command(
                str(model.serving.get("nvidia_nvfp4") or model.serving.get("nvidia_fp4") or "")
            )
            if fp4_hint is not None:
                hints.append((fp4_hint, "Using Blackwell FP4 launch hint published with the model."))
    elif normalized_precision in {"bf16", "fp16"}:
        fp_hint = model.serving.get("tp_bf16") or model.serving.get("tp_fp16")
        if isinstance(fp_hint, int):
            hints.append((fp_hint, "Using model-specific BF16/FP16 tensor-parallel hint."))

    tp_min = model.serving.get("tp_min")
    if isinstance(tp_min, int) and tp_min > 1:
        hints.append((tp_min, f"Respecting model minimum TP requirement (tp_min={tp_min})."))

    seen: set[int] = set()
    ordered_candidates: list[tuple[int, str]] = []
    for hint_tp, reason in hints:
        resolved = _fit_hint_to_valid_tps(hint_tp, valid_tps)
        if resolved is not None and resolved not in seen:
            seen.add(resolved)
            ordered_candidates.append((resolved, reason))
    for candidate in valid_tps:
        if candidate not in seen:
            seen.add(candidate)
            ordered_candidates.append((candidate, f"Using the smallest KV-headroom-valid TP candidate ({candidate})."))

    for candidate, reason in ordered_candidates:
        if _tp_fits_memory(model, gpu, candidate, normalized_precision):
            return candidate, reason

    return None, "No tensor-parallel choice preserves minimum KV headroom for long-context coding."


def _valid_tps(model: ModelVariant, num_gpus: int) -> list[int]:
    valid: list[int] = []
    for tp in (1, 2, 4, 8, 16, 32):
        if tp > num_gpus:
            break
        if model.kv_heads > 0 and model.kv_heads % tp != 0:
            continue
        valid.append(tp)
    return valid or [1]


def _fit_hint_to_valid_tps(hint_tp: int, valid_tps: list[int]) -> int | None:
    if hint_tp in valid_tps:
        return hint_tp
    larger = [tp for tp in valid_tps if tp >= hint_tp]
    if larger:
        return min(larger)
    return None


def _extract_tp_from_command(value: str) -> int | None:
    if not value:
        return None
    match = re.search(r"(?:^|\s)-tp\s+(\d+)(?:\s|$)", value)
    if match is None:
        return None
    return int(match.group(1))


def _tp_fits_memory(model: ModelVariant, gpu: GPUProfile, tp: int, precision: str) -> bool:
    normalized_precision = "fp4" if precision in {"nvfp4", "mxfp4"} else precision
    per_gpu_usable = gpu.memory_gb * 0.92
    activation_overhead = min(3.0, max(1.0, model.params_total_b / 100))
    per_gpu_weight = model.weight_gb(normalized_precision) / max(tp, 1)
    kv_headroom = per_gpu_usable - per_gpu_weight - activation_overhead
    minimum_headroom = max(per_gpu_usable * 0.10, 16.0)
    return kv_headroom >= minimum_headroom


def _precision_fits_target_lane(model: ModelVariant, gpu: GPUProfile, num_gpus: int, precision: str) -> bool:
    return any(_tp_fits_memory(model, gpu, tp, precision) for tp in _valid_tps(model, num_gpus))
