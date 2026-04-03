#!/usr/bin/env python3
"""generate_mode_a_configs.py — Generate Mode A configuration YAML files.

Reads all GPU configs, model configs, and sweep configs, then generates a
Mode A YAML for every valid (gpu, model, quantization) triple.

Mode A uses vLLM defaults:
  - gpu_memory_utilization = 0.90
  - max_num_seqs = 256
  - tensor_parallel_size from min_gpus in model config
  - Does NOT set V0-era flags (no enable_chunked_prefill, no
    enforce_eager, no max_num_batched_tokens, etc.)
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import click
import yaml

_PRODUCT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_ROOT = _PRODUCT_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_all_yamls(directory: Path) -> list[dict[str, Any]]:
    """Load every *.yaml file in a directory."""
    configs = []
    for p in sorted(directory.glob("*.yaml")):
        data = load_yaml(p)
        if data:
            configs.append(data)
    return configs


def load_configs_by_key(
    directory: Path, key_field: str
) -> dict[str, dict[str, Any]]:
    """Load YAMLs and index them by a specific field (e.g. 'gpu_short')."""
    result: dict[str, dict[str, Any]] = {}
    for cfg in load_all_yamls(directory):
        key_val = cfg.get(key_field)
        if key_val is not None:
            result[str(key_val)] = cfg
    return result


def get_quantizations_for_cell(
    sweep: dict[str, Any],
    gpu_short: str,
    model_short: str,
    gpu_cfg: dict[str, Any],
) -> list[str]:
    """Determine which quantizations apply for a (gpu, model) pair."""
    quant_cfg = sweep.get("quantizations", {})
    quants: list[str] = list(quant_cfg.get("default", ["fp8"]))

    # bf16 reference
    if quant_cfg.get("bf16_reference", False):
        if "bf16" not in quants:
            quants.append("bf16")

    # nvfp4 special cells
    nvfp4_cells = quant_cfg.get("nvfp4", [])
    if isinstance(nvfp4_cells, list):
        for cell in nvfp4_cells:
            cell_gpu = cell.get("gpu", "")
            cell_model = cell.get("model", "")
            # Match if gpu matches and (model matches or model unspecified)
            if cell_gpu == gpu_short and (not cell_model or cell_model == model_short):
                if gpu_cfg.get("nvfp4_support", False) and "nvfp4" not in quants:
                    quants.append("nvfp4")

    return quants


def get_tensor_parallel_size(
    model_cfg: dict[str, Any],
    gpu_short: str,
    quantization: str,
) -> int:
    """Look up min_gpus for the (gpu, quantization) pair — this is the TP size for Mode A."""
    min_gpus = model_cfg.get("min_gpus", {})

    # Normalise quantization key
    quant_key = quantization
    if quantization.startswith("fp8"):
        quant_key = "fp8"

    quant_map = min_gpus.get(quant_key, {})
    if isinstance(quant_map, dict):
        tp = quant_map.get(gpu_short)
        if tp is not None:
            return int(tp)

    # Fallback: try bf16 map
    bf16_map = min_gpus.get("bf16", {})
    if isinstance(bf16_map, dict):
        tp = bf16_map.get(gpu_short)
        if tp is not None:
            return int(tp)

    return 1


def build_mode_a_config(
    gpu_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    quantization: str,
    tensor_parallel_size: int,
) -> dict[str, Any]:
    """Build a single Mode A configuration dict."""
    gpu_short = gpu_cfg["gpu_short"]
    model_short = model_cfg["model_short"]

    config: dict[str, Any] = {
        "mode": "mode_a",
        "description": (
            f"Mode A (vLLM defaults) — {model_cfg['model_name']} on "
            f"{gpu_cfg['gpu_name']} at {quantization}"
        ),
        # Identity
        "gpu": gpu_short,
        "gpu_name": gpu_cfg["gpu_name"],
        "model": model_short,
        "model_name": model_cfg["model_name"],
        "hf_model_id": model_cfg["hf_model_id"],
        "quantization": quantization,
        # Parallelism — from min_gpus
        "tensor_parallel_size": tensor_parallel_size,
        # vLLM defaults — Mode A does NOT tune these
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 256,
        # Model settings
        "max_model_len": model_cfg.get("default_max_model_len", 32768),
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "tokenizer_mode": model_cfg.get("tokenizer_mode", "auto"),
    }

    # KV cache dtype: use fp8 when the GPU supports it and we are at fp8 quant
    if quantization.startswith("fp8"):
        kv_dtypes = gpu_cfg.get("kv_cache_dtypes", ["auto"])
        if "fp8_e5m2" in kv_dtypes:
            config["kv_cache_dtype"] = "fp8_e5m2"
        else:
            config["kv_cache_dtype"] = "auto"
    else:
        config["kv_cache_dtype"] = "auto"

    # dtype
    if quantization in ("bf16", "fp16"):
        config["dtype"] = quantization
    elif quantization.startswith("fp8") or quantization == "nvfp4":
        config["dtype"] = "auto"
    else:
        config["dtype"] = "auto"

    return config


def config_filename(gpu_short: str, model_short: str, quantization: str) -> str:
    """Generate a deterministic filename for a Mode A config."""
    return f"mode_a__{gpu_short}__{model_short}__{quantization}.yaml"


@click.command("generate-mode-a-configs")
@click.option(
    "--config-root",
    type=click.Path(exists=True),
    default=str(_DEFAULT_CONFIG_ROOT),
    show_default=False,
    help="Root directory for ISB-1 config files. Defaults to the product-local configs/ tree.",
)
@click.option(
    "--sweep",
    "sweep_files",
    type=click.Path(exists=True),
    multiple=True,
    default=(),
    help="Sweep config(s) to process. If empty, uses all sweeps.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for generated configs. Default: configs/modes/mode_a/",
)
@click.option("--dry-run", is_flag=True, help="Print what would be generated without writing.")
def main(
    config_root: str,
    sweep_files: tuple[str, ...],
    output_dir: str | None,
    dry_run: bool,
) -> None:
    """Generate Mode A YAML configs for every (gpu, model, quantization) triple."""
    root = Path(config_root).resolve()
    out = Path(output_dir) if output_dir else root / "modes" / "mode_a"

    # Load all GPU and model configs indexed by short name
    gpu_configs = load_configs_by_key(root / "gpus", "gpu_short")
    model_configs = load_configs_by_key(root / "models", "model_short")

    click.echo(f"Loaded {len(gpu_configs)} GPU config(s): {list(gpu_configs.keys())}")
    click.echo(f"Loaded {len(model_configs)} model config(s): {list(model_configs.keys())}")

    # Determine sweep files
    if not sweep_files:
        sweep_dir = root / "sweep"
        sweep_files = tuple(str(p) for p in sorted(sweep_dir.glob("*.yaml")))  # type: ignore[assignment]

    if not sweep_files:
        click.echo("No sweep configs found. Nothing to generate.", err=True)
        raise SystemExit(1)

    # Collect all unique (gpu, model, quant) triples across sweeps
    triples: set[tuple[str, str, str]] = set()

    for sf in sweep_files:
        sweep = load_yaml(Path(sf))
        sweep_name = sweep.get("sweep_name", Path(sf).stem)
        click.echo(f"Processing sweep: {sweep_name} ({sf})")

        gpus = sweep.get("gpus", [])
        models = sweep.get("models", [])
        model_shorts = [
            m if isinstance(m, str) else m.get("model", "")
            for m in models
        ]

        for gpu_short, model_short in itertools.product(gpus, model_shorts):
            if gpu_short not in gpu_configs:
                click.echo(f"  WARN: GPU '{gpu_short}' not in configs — skipping.", err=True)
                continue
            if model_short not in model_configs:
                click.echo(
                    f"  WARN: Model '{model_short}' not in configs — skipping.",
                    err=True,
                )
                continue

            gpu_cfg = gpu_configs[gpu_short]
            quants = get_quantizations_for_cell(sweep, gpu_short, model_short, gpu_cfg)
            for q in quants:
                triples.add((gpu_short, model_short, q))

    click.echo(f"\nTotal unique (gpu, model, quant) triples: {len(triples)}")

    if dry_run:
        click.echo("\n[DRY RUN] Would generate:")
        for gpu_short, model_short, quant in sorted(triples):
            fname = config_filename(gpu_short, model_short, quant)
            tp = get_tensor_parallel_size(
                model_configs[model_short], gpu_short, quant
            )
            click.echo(f"  {fname}  (tp={tp})")
        return

    # Generate and write configs
    out.mkdir(parents=True, exist_ok=True)
    generated = 0

    for gpu_short, model_short, quant in sorted(triples):
        gpu_cfg = gpu_configs[gpu_short]
        model_cfg = model_configs[model_short]
        tp = get_tensor_parallel_size(model_cfg, gpu_short, quant)

        config = build_mode_a_config(gpu_cfg, model_cfg, quant, tp)
        fname = config_filename(gpu_short, model_short, quant)
        outpath = out / fname

        with open(outpath, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

        generated += 1

    click.echo(f"\nGenerated {generated} Mode A config(s) in {out}")


if __name__ == "__main__":
    main()
