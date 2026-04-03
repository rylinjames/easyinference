"""ConfigValidator — loads and validates all ISB-1 YAML configuration files."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from harness.paths import default_config_root, resolve_existing_path

# ── Bytes-per-parameter for rough VRAM estimation ───────────────────────
_BYTES_PER_PARAM: dict[str, float] = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "nvfp4": 0.5,
}

# ── KV-cache overhead multiplier (rough) ────────────────────────────────
_KV_OVERHEAD_FACTOR = 1.25  # 25 % overhead for KV cache / activations


@dataclass
class ValidationError:
    """A single validation problem."""

    level: str  # "error" or "warning"
    message: str


@dataclass
class ValidationResult:
    """Aggregated result of a validation pass."""

    errors: list[ValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(e.level == "error" for e in self.errors)

    def error(self, msg: str) -> None:
        self.errors.append(ValidationError("error", msg))

    def warn(self, msg: str) -> None:
        self.errors.append(ValidationError("warning", msg))

    def summary(self) -> str:
        lines = []
        for e in self.errors:
            prefix = "ERROR" if e.level == "error" else "WARN "
            lines.append(f"  [{prefix}] {e.message}")
        return "\n".join(lines) if lines else "  (no issues)"


class ConfigValidator:
    """Loads, parses, and validates ISB-1 configuration files."""

    def __init__(self, config_root: str | Path | None = None) -> None:
        root = default_config_root() if config_root is None else Path(config_root)
        self.config_root = root.resolve()
        self._gpu_cache: dict[str, dict] = {}
        self._model_cache: dict[str, dict] = {}
        self._workload_cache: dict[str, dict] = {}
        self._mode_dirs: list[str] = []

    # ── YAML helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    @staticmethod
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    # ── Loaders ─────────────────────────────────────────────────────────

    def load_gpu(self, gpu_short: str) -> dict:
        if gpu_short in self._gpu_cache:
            return self._gpu_cache[gpu_short]
        path = self.config_root / "gpus" / f"{gpu_short}.yaml"
        # Try matching by file name first, then scan
        if not path.exists():
            for p in (self.config_root / "gpus").glob("*.yaml"):
                data = self._load_yaml(p)
                if data.get("gpu_short") == gpu_short:
                    self._gpu_cache[gpu_short] = data
                    return data
            raise FileNotFoundError(f"GPU config for '{gpu_short}' not found")
        # Some GPU yaml use different file names (e.g. h100_sxm.yaml for "h100")
        data = self._load_yaml(path)
        self._gpu_cache[gpu_short] = data
        return data

    def load_gpu_by_short(self, gpu_short: str) -> dict:
        """Scan all GPU yamls for one matching gpu_short."""
        if gpu_short in self._gpu_cache:
            return self._gpu_cache[gpu_short]
        for p in (self.config_root / "gpus").glob("*.yaml"):
            data = self._load_yaml(p)
            if data.get("gpu_short") == gpu_short:
                self._gpu_cache[gpu_short] = data
                return data
        raise FileNotFoundError(f"GPU config for short name '{gpu_short}' not found")

    def load_model(self, model_short: str) -> dict:
        if model_short in self._model_cache:
            return self._model_cache[model_short]
        for p in (self.config_root / "models").glob("*.yaml"):
            data = self._load_yaml(p)
            if data.get("model_short") == model_short:
                self._model_cache[model_short] = data
                return data
        raise FileNotFoundError(f"Model config for '{model_short}' not found")

    def load_workload(self, workload_name: str) -> dict:
        if workload_name in self._workload_cache:
            return self._workload_cache[workload_name]
        path = self.config_root / "workloads" / f"{workload_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Workload config '{workload_name}' not found")
        data = self._load_yaml(path)
        self._workload_cache[workload_name] = data
        return data

    def load_sweep(self, sweep_path: str | Path) -> dict:
        return self._load_yaml(resolve_existing_path(sweep_path))

    # ── Validation primitives ───────────────────────────────────────────

    def validate_gpu(self, gpu_short: str) -> ValidationResult:
        result = ValidationResult()
        try:
            cfg = self.load_gpu_by_short(gpu_short)
        except FileNotFoundError as exc:
            result.error(str(exc))
            return result
        for key in ("gpu_name", "gpu_short", "hbm_capacity_gb", "fp_formats"):
            if key not in cfg:
                result.error(f"GPU config '{gpu_short}' missing required key: {key}")
        return result

    def validate_model(self, model_short: str) -> ValidationResult:
        result = ValidationResult()
        try:
            cfg = self.load_model(model_short)
        except FileNotFoundError as exc:
            result.error(str(exc))
            return result
        for key in ("model_name", "hf_model_id", "total_params_b"):
            if key not in cfg:
                result.error(f"Model config '{model_short}' missing required key: {key}")
        return result

    def validate_workload(self, workload_name: str) -> ValidationResult:
        result = ValidationResult()
        try:
            cfg = self.load_workload(workload_name)
        except FileNotFoundError as exc:
            result.error(str(exc))
            return result
        for key in ("workload_name", "workload_id"):
            if key not in cfg:
                result.error(
                    f"Workload config '{workload_name}' missing required key: {key}"
                )

        trace_cfg = cfg.get("trace", {})
        num_requests = trace_cfg.get("num_requests")
        if num_requests is not None:
            try:
                if int(num_requests) < 1:
                    result.error(
                        f"Workload config '{workload_name}' has invalid trace.num_requests={num_requests!r}; expected >= 1"
                    )
            except (TypeError, ValueError):
                result.error(
                    f"Workload config '{workload_name}' has non-integer trace.num_requests={num_requests!r}"
                )
        return result

    # ── Cross-config checks ─────────────────────────────────────────────

    def check_memory_fit(
        self,
        gpu_short: str,
        model_short: str,
        quantization: str,
        gpu_count: int,
    ) -> ValidationResult:
        """Check model fits in GPU HBM.

        Uses the model config's ``min_gpus`` table when available (these values
        are hand-verified by the benchmark authors).  Falls back to a rough
        bytes-per-parameter estimate only when ``min_gpus`` is missing.
        """
        result = ValidationResult()
        try:
            gpu_cfg = self.load_gpu_by_short(gpu_short)
            model_cfg = self.load_model(model_short)
        except FileNotFoundError as exc:
            result.error(str(exc))
            return result

        # Prefer min_gpus from model config (accounts for MoE sparsity, etc.)
        min_gpus = model_cfg.get("min_gpus", {})
        quant_key = "bf16" if quantization in ("bf16", "fp16") else quantization
        gpu_min_map = min_gpus.get(quant_key, {})
        if gpu_short in gpu_min_map:
            required_gpus = gpu_min_map[gpu_short]
            if gpu_count < required_gpus:
                result.error(
                    f"Model '{model_short}' at {quantization} requires at least "
                    f"{required_gpus}x {gpu_short} but only {gpu_count} provided"
                )
            return result

        # Fallback: rough estimate using total_params_b
        total_hbm_gb = gpu_cfg.get("hbm_capacity_gb", 0) * gpu_count
        params_b = model_cfg.get("total_params_b", 0)
        bpp = _BYTES_PER_PARAM.get(quantization, 2.0)

        model_gb = (params_b * 1e9 * bpp) / (1024**3)
        required_gb = model_gb * _KV_OVERHEAD_FACTOR

        if required_gb > total_hbm_gb:
            result.warn(
                f"Model '{model_short}' at {quantization} may require ~{required_gb:.1f} GB "
                f"but {gpu_count}x {gpu_short} provides {total_hbm_gb:.0f} GB HBM "
                f"(estimate only — no min_gpus entry for this GPU)"
            )
        return result

    def check_quantization_support(
        self,
        gpu_short: str,
        quantization: str,
    ) -> ValidationResult:
        """Verify the GPU supports the requested quantization format."""
        result = ValidationResult()
        try:
            gpu_cfg = self.load_gpu_by_short(gpu_short)
        except FileNotFoundError as exc:
            result.error(str(exc))
            return result

        fp_formats: list[str] = gpu_cfg.get("fp_formats", [])
        # Normalise: "fp8" matches any fp8 variant
        if quantization == "fp8":
            if not any("fp8" in f for f in fp_formats):
                result.error(
                    f"GPU '{gpu_short}' does not support fp8 (formats: {fp_formats})"
                )
        elif quantization == "nvfp4":
            if not gpu_cfg.get("nvfp4_support", False):
                result.error(f"GPU '{gpu_short}' does not support nvfp4")
        elif quantization not in fp_formats and quantization not in ("bf16", "fp16"):
            result.warn(
                f"Quantization '{quantization}' not explicitly in GPU fp_formats: {fp_formats}"
            )
        return result

    def check_min_gpus(
        self,
        gpu_short: str,
        model_short: str,
        quantization: str,
        gpu_count: int,
    ) -> ValidationResult:
        """Check min_gpus requirements from the model config."""
        result = ValidationResult()
        try:
            model_cfg = self.load_model(model_short)
        except FileNotFoundError as exc:
            result.error(str(exc))
            return result

        min_gpus = model_cfg.get("min_gpus", {})
        quant_key = quantization
        # Normalise fp8 variants
        if quantization.startswith("fp8"):
            quant_key = "fp8"
        quant_map = min_gpus.get(quant_key, min_gpus.get("bf16", {}))
        required = quant_map.get(gpu_short, 1)

        if gpu_count < required:
            result.error(
                f"Model '{model_short}' at {quantization} on {gpu_short} requires "
                f"min {required} GPUs, but got {gpu_count}"
            )
        return result

    # ── Full sweep validation ───────────────────────────────────────────

    def validate_sweep(self, sweep_path: str | Path) -> ValidationResult:
        """Validate an entire sweep matrix YAML."""
        result = ValidationResult()
        try:
            sweep = self.load_sweep(sweep_path)
        except Exception as exc:
            result.error(f"Failed to parse sweep YAML: {exc}")
            return result

        gpus = sweep.get("gpus", [])
        models = sweep.get("models", [])
        workloads = sweep.get("workloads", [])
        modes = sweep.get("modes", [])
        quant_cfg = sweep.get("quantizations", {})
        default_quants = quant_cfg.get("default", ["fp8"])

        # Validate individual configs exist and parse
        for g in gpus:
            sub = self.validate_gpu(g)
            result.errors.extend(sub.errors)

        model_shorts = []
        for m in models:
            ms = m if isinstance(m, str) else m.get("model", "")
            model_shorts.append(ms)
            sub = self.validate_model(ms)
            result.errors.extend(sub.errors)

        for w in workloads:
            sub = self.validate_workload(w)
            result.errors.extend(sub.errors)

        # Cross-validate each cell
        for g, ms, w, mode in itertools.product(gpus, model_shorts, workloads, modes):
            for q in default_quants:
                sub = self.check_quantization_support(g, q)
                result.errors.extend(sub.errors)

                # Use min_gpus from model config to determine gpu_count
                try:
                    model_cfg = self.load_model(ms)
                    min_gpus = model_cfg.get("min_gpus", {})
                    quant_key = "fp8" if q.startswith("fp8") else q
                    quant_map = min_gpus.get(quant_key, min_gpus.get("bf16", {}))
                    gpu_count = quant_map.get(g, 1)
                except FileNotFoundError:
                    gpu_count = 1

                sub = self.check_memory_fit(g, ms, q, gpu_count)
                result.errors.extend(sub.errors)
                sub = self.check_min_gpus(g, ms, q, gpu_count)
                result.errors.extend(sub.errors)

            # bf16 reference
            if quant_cfg.get("bf16_reference"):
                sub = self.check_quantization_support(g, "bf16")
                result.errors.extend(sub.errors)

        # nvfp4 special cells
        nvfp4_cells = quant_cfg.get("nvfp4", [])
        if isinstance(nvfp4_cells, list):
            for cell in nvfp4_cells:
                g = cell.get("gpu", "")
                ms = cell.get("model", "")
                sub = self.check_quantization_support(g, "nvfp4")
                result.errors.extend(sub.errors)

        return result

    # ── Convenience ─────────────────────────────────────────────────────

    def validate_all_yamls(self) -> ValidationResult:
        """Parse every YAML under config_root and report parse failures."""
        result = ValidationResult()
        for p in self.config_root.rglob("*.yaml"):
            try:
                self._load_yaml(p)
            except yaml.YAMLError as exc:
                result.error(f"YAML parse error in {p}: {exc}")
        return result


# ── CLI entry point ─────────────────────────────────────────────────────

def _cli() -> None:
    import click

    def _resolve_existing_click_path(
        _ctx: click.Context, _param: click.Parameter, value: Path | None
    ) -> Path | None:
        if value is None:
            return None
        try:
            return resolve_existing_path(value)
        except FileNotFoundError as exc:
            raise click.BadParameter(str(exc)) from exc

    @click.command("config-validator")
    @click.option(
        "--sweep",
        "sweep_path",
        type=click.Path(path_type=Path),
        callback=_resolve_existing_click_path,
        help="Path to sweep config YAML to validate.",
    )
    @click.option(
        "--config-root",
        default=default_config_root(),
        show_default=False,
        type=click.Path(path_type=Path),
        callback=_resolve_existing_click_path,
        help="Root directory for config files. Defaults to the product-local configs/ tree.",
    )
    @click.option("--all-yaml", is_flag=True, help="Parse-check every YAML under config root.")
    def main(sweep_path: str | None, config_root: str, all_yaml: bool) -> None:
        """Validate ISB-1 configuration files."""
        validator = ConfigValidator(config_root)

        if all_yaml:
            click.echo("Checking all YAML files...")
            res = validator.validate_all_yamls()
            click.echo(res.summary())

        if sweep_path:
            click.echo(f"Validating sweep: {sweep_path}")
            res = validator.validate_sweep(sweep_path)
            click.echo(res.summary())
            if not res.ok:
                raise SystemExit(1)
            else:
                click.echo("Sweep validation passed.")

        if not sweep_path and not all_yaml:
            click.echo("No action specified. Use --sweep or --all-yaml.")
            raise SystemExit(1)

    main()


if __name__ == "__main__":
    _cli()
