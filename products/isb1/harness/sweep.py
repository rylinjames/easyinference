"""SweepOrchestrator — generates and executes the full ISB-1 benchmark matrix."""

from __future__ import annotations

import itertools
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from harness.config_validator import ConfigValidator
from harness.paths import (
    default_config_root,
    default_results_root,
    resolve_existing_path,
    resolve_path,
)
from harness.runner import BenchmarkRunner, CellConfig, RunResult

logger = logging.getLogger(__name__)


@dataclass
class SweepSummary:
    """Summary statistics for a completed sweep."""

    total_cells: int = 0
    completed: int = 0
    failed: int = 0
    unstable: int = 0
    skipped: int = 0
    results: list[RunResult] = field(default_factory=list)


class SweepOrchestrator:
    """Read a sweep config and execute every cell in the benchmark matrix.

    Parameters
    ----------
    sweep_path : str | Path
        Path to the sweep YAML config (e.g. ``configs/sweep/core.yaml``).
    output_dir : str | Path
        Root output directory for all results.
    config_root : str | Path
        Root directory for ISB-1 configuration files.
    dry_run : bool
        If True, generate the plan but do not execute.
    """

    def __init__(
        self,
        sweep_path: str | Path,
        output_dir: str | Path | None = None,
        config_root: str | Path | None = None,
        dry_run: bool = False,
    ) -> None:
        self.sweep_path = resolve_existing_path(sweep_path)
        self.output_dir = resolve_path(output_dir) if output_dir is not None else default_results_root()
        self.config_root = (
            resolve_existing_path(config_root) if config_root is not None else default_config_root()
        )
        self.dry_run = dry_run

        self._sweep: dict[str, Any] = {}
        self._cells: list[CellConfig] = []
        self._validator = ConfigValidator(self.config_root)

    # ── Loading ──────────────────────────────────────────────────────────

    def _load_sweep(self) -> dict[str, Any]:
        with open(self.sweep_path, "r", encoding="utf-8") as fh:
            self._sweep = yaml.safe_load(fh) or {}
        return self._sweep

    # ── Cell generation ──────────────────────────────────────────────────

    def generate_cells(self) -> list[CellConfig]:
        """Parse the sweep config and produce the full list of CellConfig objects."""
        sweep = self._load_sweep()

        gpus: list[str] = sweep.get("gpus", [])
        models_raw: list[Any] = sweep.get("models", [])
        workloads: list[str] = sweep.get("workloads", [])
        modes: list[str] = sweep.get("modes", [])
        prefix_caching_variants: list[bool] = sweep.get("prefix_caching", [True])
        batched_tokens_sweep: list[int | None] = sweep.get("max_num_batched_tokens", [None])

        quant_cfg = sweep.get("quantizations", {})
        default_quants: list[str] = quant_cfg.get("default", ["fp8"])
        bf16_reference: bool = quant_cfg.get("bf16_reference", False)
        nvfp4_cells: list[dict] = quant_cfg.get("nvfp4", [])

        trials_cfg = sweep.get("trials", {})
        default_trials: int = trials_cfg.get("default", 3)
        high_variance_max: int = trials_cfg.get("high_variance_max", 5)  # noqa: F841 — used in execute()
        bf16_trials: int = trials_cfg.get("bf16_reference", 1)

        measurement_cfg = sweep.get("measurement", {})
        warmup_requests: int = measurement_cfg.get("warmup_requests", 100)
        warmup_seconds: float = measurement_cfg.get("warmup_seconds", 60)
        warmup_max_ext: int = measurement_cfg.get("warmup_max_extensions", 3)
        variance_thresh: float = measurement_cfg.get(
            "steady_state_variance_threshold", 0.20
        )
        duration: float = measurement_cfg.get("measurement_duration_seconds", 600)
        cooldown: float = measurement_cfg.get("cooldown_seconds", 30)

        # Parse models
        models: list[dict[str, str]] = []
        for m in models_raw:
            if isinstance(m, str):
                models.append({"model": m})
            elif isinstance(m, dict):
                models.append(m)

        cells: list[CellConfig] = []

        # Main matrix: gpus x models x workloads x modes x quantizations x prefix_caching x batched_tokens x trials
        for gpu, model_info, workload, mode, prefix_caching, max_batched_tokens in itertools.product(
            gpus, models, workloads, modes, prefix_caching_variants, batched_tokens_sweep
        ):
            model_short = model_info["model"]

            # Resolve model config
            try:
                model_cfg = self._validator.load_model(model_short)
            except FileNotFoundError:
                logger.warning("Model config not found for '%s', skipping", model_short)
                continue

            model_hf_id = model_cfg.get("hf_model_id", "")

            # Resolve workload config for rate sweep
            try:
                wl_cfg = self._validator.load_workload(workload)
            except FileNotFoundError:
                logger.warning("Workload config not found for '%s'", workload)
                wl_cfg = {}

            rate_sweep = wl_cfg.get("arrival", {}).get("rate_sweep", [1.0])
            num_prompts = int(wl_cfg.get("trace", {}).get("num_requests", 1000))
            arrival_cfg = wl_cfg.get("arrival", {})
            arrival_model = str(arrival_cfg.get("model", "poisson"))
            arrival_shape = (
                float(arrival_cfg["shape"]) if "shape" in arrival_cfg else None
            )
            goodput_slo = wl_cfg.get("slo") or None

            # Default quantizations
            for quant in default_quants:
                gpu_count, topology = self._resolve_gpu_topology(
                    gpu, model_short, model_cfg, quant
                )
                config_paths = self._collect_config_paths(
                    gpu, model_short, workload, mode
                )

                for trial in range(1, default_trials + 1):
                    cells.append(
                        CellConfig(
                            gpu=gpu,
                            gpu_count=gpu_count,
                            model=model_short,
                            model_hf_id=model_hf_id,
                            workload=workload,
                            mode=mode,
                            quantization=quant,
                            topology=topology,
                            prefix_caching=prefix_caching,
                            max_num_batched_tokens=max_batched_tokens,
                            trial_number=trial,
                            num_prompts=num_prompts,
                            rate_sweep=rate_sweep,
                            seed=42 + trial,
                            arrival_model=arrival_model,
                            arrival_shape=arrival_shape,
                            goodput_slo=goodput_slo,
                            warmup_requests=warmup_requests,
                            warmup_seconds=warmup_seconds,
                            warmup_max_extensions=warmup_max_ext,
                            warmup_variance_threshold=variance_thresh,
                            measurement_duration_seconds=duration,
                            cooldown_seconds=cooldown,
                            output_dir=str(self.output_dir),
                            config_root=str(self.config_root),
                            config_paths=config_paths,
                        )
                    )

            # bf16 reference
            if bf16_reference:
                gpu_count, topology = self._resolve_gpu_topology(
                    gpu, model_short, model_cfg, "bf16"
                )
                config_paths = self._collect_config_paths(
                    gpu, model_short, workload, mode
                )
                for trial in range(1, bf16_trials + 1):
                    cells.append(
                        CellConfig(
                            gpu=gpu,
                            gpu_count=gpu_count,
                            model=model_short,
                            model_hf_id=model_hf_id,
                            workload=workload,
                            mode=mode,
                            quantization="bf16",
                            topology=topology,
                            trial_number=trial,
                            num_prompts=num_prompts,
                            rate_sweep=rate_sweep,
                            seed=42 + trial,
                            arrival_model=arrival_model,
                            arrival_shape=arrival_shape,
                            goodput_slo=goodput_slo,
                            warmup_requests=warmup_requests,
                            warmup_seconds=warmup_seconds,
                            warmup_max_extensions=warmup_max_ext,
                            warmup_variance_threshold=variance_thresh,
                            measurement_duration_seconds=duration,
                            cooldown_seconds=cooldown,
                            output_dir=str(self.output_dir),
                            config_root=str(self.config_root),
                            config_paths=config_paths,
                        )
                    )

        # nvfp4 special cells
        if isinstance(nvfp4_cells, list):
            for nvfp4_cell in nvfp4_cells:
                gpu = nvfp4_cell.get("gpu", "")
                model_short = nvfp4_cell.get("model", "")
                try:
                    model_cfg = self._validator.load_model(model_short)
                except FileNotFoundError:
                    continue
                model_hf_id = model_cfg.get("hf_model_id", "")

                gpu_count, topology = self._resolve_gpu_topology(
                    gpu, model_short, model_cfg, "nvfp4"
                )

                for workload, mode in itertools.product(workloads, modes):
                    try:
                        wl_cfg = self._validator.load_workload(workload)
                    except FileNotFoundError:
                        wl_cfg = {}
                    rate_sweep = wl_cfg.get("arrival", {}).get("rate_sweep", [1.0])
                    num_prompts = int(wl_cfg.get("trace", {}).get("num_requests", 1000))
                    arrival_cfg = wl_cfg.get("arrival", {})
                    arrival_model = str(arrival_cfg.get("model", "poisson"))
                    arrival_shape = (
                        float(arrival_cfg["shape"]) if "shape" in arrival_cfg else None
                    )
                    goodput_slo = wl_cfg.get("slo") or None

                    config_paths = self._collect_config_paths(
                        gpu, model_short, workload, mode
                    )

                    for trial in range(1, default_trials + 1):
                        cells.append(
                            CellConfig(
                                gpu=gpu,
                                gpu_count=gpu_count,
                                model=model_short,
                                model_hf_id=model_hf_id,
                                workload=workload,
                                mode=mode,
                                quantization="nvfp4",
                                topology=topology,
                                trial_number=trial,
                                num_prompts=num_prompts,
                                rate_sweep=rate_sweep,
                                seed=42 + trial,
                                arrival_model=arrival_model,
                                arrival_shape=arrival_shape,
                                goodput_slo=goodput_slo,
                                warmup_requests=warmup_requests,
                                warmup_seconds=warmup_seconds,
                                warmup_max_extensions=warmup_max_ext,
                                warmup_variance_threshold=variance_thresh,
                                measurement_duration_seconds=duration,
                                cooldown_seconds=cooldown,
                                output_dir=str(self.output_dir),
                                config_root=str(self.config_root),
                                config_paths=config_paths,
                            )
                        )

        self._cells = cells
        return cells

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_gpu_topology(
        self,
        gpu: str,
        model_short: str,
        model_cfg: dict,
        quant: str,
    ) -> tuple[int, str]:
        """Return (gpu_count, topology_string) for a given cell."""
        quant_key = "fp8" if quant.startswith("fp8") else quant
        min_gpus = model_cfg.get("min_gpus", {})
        quant_map = min_gpus.get(quant_key, min_gpus.get("bf16", {}))
        gpu_count = quant_map.get(gpu, 1)

        rec = model_cfg.get("recommended_topology", {})
        topology = rec.get(quant_key, rec.get("bf16", {})).get(gpu, f"tp{gpu_count}")

        return gpu_count, topology

    def _collect_config_paths(
        self, gpu: str, model: str, workload: str, mode: str
    ) -> list[str | Path]:
        """Gather all config file paths relevant to a cell."""
        root = self.config_root
        paths: list[str | Path] = [self.sweep_path]

        # GPU config (scan for matching file)
        for p in (root / "gpus").glob("*.yaml"):
            try:
                data = self._validator._load_yaml(p)
                if data.get("gpu_short") == gpu:
                    paths.append(p)
                    break
            except Exception:
                logger.warning("Failed to parse GPU config %s, skipping", p, exc_info=True)

        # Model config
        for p in (root / "models").glob("*.yaml"):
            try:
                data = self._validator._load_yaml(p)
                if data.get("model_short") == model:
                    paths.append(p)
                    break
            except Exception:
                logger.warning("Failed to parse model config %s, skipping", p, exc_info=True)

        # Workload config
        wl_path = root / "workloads" / f"{workload}.yaml"
        if wl_path.exists():
            paths.append(wl_path)

        # Mode config directory
        mode_dir = root / "modes" / mode
        if mode_dir.is_dir():
            for p in mode_dir.glob("*.yaml"):
                paths.append(p)

        return paths

    # ── Plan ─────────────────────────────────────────────────────────────

    def plan(self) -> str:
        """Generate and return a human-readable summary of the sweep matrix."""
        if not self._cells:
            self.generate_cells()

        lines: list[str] = []
        lines.append(f"Sweep: {self._sweep.get('sweep_name', 'unnamed')}")
        lines.append(f"Description: {self._sweep.get('description', '')}")
        lines.append(f"Total cells: {len(self._cells)}")
        lines.append("")

        # Group by gpu x model x workload x mode x quant
        seen: dict[str, int] = {}
        for c in self._cells:
            key = f"{c.gpu}/{c.model}/{c.workload}/{c.mode}/{c.quantization}"
            seen[key] = seen.get(key, 0) + 1

        lines.append(f"Unique configurations: {len(seen)}")
        lines.append("")
        lines.append(f"{'Configuration':<60} {'Trials':>6}")
        lines.append("-" * 68)
        for key, count in sorted(seen.items()):
            lines.append(f"{key:<60} {count:>6}")

        return "\n".join(lines)

    # ── Execution ────────────────────────────────────────────────────────

    def execute(self) -> SweepSummary:
        """Run all cells in the sweep matrix.

        After completing the default trials for a configuration, checks the
        coefficient of variation (CV).  If CV exceeds the threshold, extends
        to ``high_variance_max`` trials.
        """
        if not self._cells:
            self.generate_cells()

        summary = SweepSummary(total_cells=len(self._cells))

        if self.dry_run:
            logger.info("Dry run — skipping execution of %d cells", len(self._cells))
            summary.skipped = len(self._cells)
            return summary

        trials_cfg = self._sweep.get("trials", {})
        high_variance_max = trials_cfg.get("high_variance_max", 5)
        variance_cfg = self._sweep.get("variance", {})
        cv_threshold = variance_cfg.get("cv_threshold", 0.10)
        parallel_cells: int = self._sweep.get("parallel_cells", 1)

        # Load composite-hash result cache — skip unchanged cells
        result_cache = self._load_result_cache()

        # Group cells by configuration key (all trials for same config)
        config_groups: dict[str, list[CellConfig]] = {}
        for cell in self._cells:
            key = (
                f"{cell.gpu}/{cell.model}/{cell.workload}/{cell.mode}/"
                f"{cell.quantization}/apc-{'on' if cell.prefix_caching else 'off'}/"
                f"mbt-{cell.max_num_batched_tokens or 'default'}"
            )
            config_groups.setdefault(key, []).append(cell)

        def _run_config_group(config_key: str, group: list[CellConfig]) -> list[RunResult]:
            """Run all trials for one configuration, extend on high variance."""
            cache_key = self._cell_cache_key(group[0])
            cached_trials = result_cache.get(cache_key, 0)
            if cached_trials >= len(group):
                logger.info(
                    "Cache hit for %s (key=%s, %d trials cached) — skipping",
                    config_key, cache_key, cached_trials,
                )
                return []

            logger.info("Running configuration: %s (%d trials)", config_key, len(group))
            trial_results: list[RunResult] = []
            for cell in group:
                runner = BenchmarkRunner(cell)
                trial_results.append(runner.run())

            completed = [r for r in trial_results if r.status == "completed"]
            if len(completed) >= 2:
                cv = self._compute_trial_cv(completed)
                logger.info("CV for %s: %.4f (threshold: %.4f)", config_key, cv, cv_threshold)
                if cv > cv_threshold:
                    extra_needed = high_variance_max - len(group)
                    if extra_needed > 0:
                        logger.info("High variance — running %d extra trials", extra_needed)
                        base_cell = group[0]
                        for t in range(len(group) + 1, high_variance_max + 1):
                            ext_cell = CellConfig(**{**base_cell.__dict__, "trial_number": t, "seed": 42 + t})
                            trial_results.append(BenchmarkRunner(ext_cell).run())

            return trial_results

        if parallel_cells > 1:
            logger.info("Parallel sweep: %d concurrent configurations", parallel_cells)
            with ThreadPoolExecutor(max_workers=parallel_cells) as pool:
                futures = {
                    pool.submit(_run_config_group, k, g): (k, g)
                    for k, g in config_groups.items()
                }
                for future in as_completed(futures):
                    config_key, group = futures[future]
                    try:
                        group_results = future.result()
                    except Exception:
                        logger.exception("Configuration %s raised an exception", config_key)
                        summary.failed += len(group)
                        continue
                    if not group_results:
                        summary.skipped += len(group)
                    for result in group_results:
                        summary.results.append(result)
                        summary.total_cells += 1
                        if result.status == "completed":
                            summary.completed += 1
                        elif result.status == "failed":
                            summary.failed += 1
        else:
            for config_key, group in config_groups.items():
                group_results = _run_config_group(config_key, group)
                if not group_results:
                    summary.skipped += len(group)
                for result in group_results:
                    summary.results.append(result)
                    summary.total_cells += 1
                    if result.status == "completed":
                        summary.completed += 1
                    elif result.status == "failed":
                        summary.failed += 1

        return summary

    # ── Resume ───────────────────────────────────────────────────────────

    def resume(self) -> SweepSummary:
        """Resume a sweep, skipping cells that already have completed results."""
        if not self._cells:
            self.generate_cells()

        completed_ids = self._find_completed_runs()
        remaining: list[CellConfig] = []

        for cell in self._cells:
            # Check if any completed run matches this cell
            matched = False
            for cid in completed_ids:
                parts = cid.split("-")
                if len(parts) >= 8:
                    # Compare gpu, model, workload, mode, quant, trial
                    if (
                        parts[2] == cell.gpu
                        and parts[3] == cell.model
                        and parts[4] == cell.workload
                        and parts[5] == cell.mode
                        and parts[6] == cell.quantization
                        and parts[7] == f"{cell.trial_number:03d}"
                    ):
                        matched = True
                        break

            if not matched:
                remaining.append(cell)

        skipped = len(self._cells) - len(remaining)
        logger.info(
            "Resume: %d cells total, %d completed, %d remaining",
            len(self._cells),
            skipped,
            len(remaining),
        )

        # Replace cell list with remaining and execute
        original = self._cells
        self._cells = remaining
        summary = self.execute()
        summary.skipped = skipped
        summary.total_cells = len(original)
        self._cells = original
        return summary

    @staticmethod
    def _cell_cache_key(cell: CellConfig) -> str:
        """Composite hash for result-cache invalidation.

        Keyed on (model_hf_id, workload, mode, quantization, gpu, gpu_count,
        prefix_caching, max_num_batched_tokens, config_file_hashes).
        Changes to any of these invalidate the cache and force a re-run.
        Changing only trial_number does NOT invalidate — we re-use cached trials.
        """
        import hashlib, json as _json
        from harness.lockfile import LockfileGenerator
        config_hashes = LockfileGenerator.hash_config_files(cell.config_paths)
        canonical = _json.dumps(
            {
                "model_hf_id": cell.model_hf_id,
                "workload": cell.workload,
                "mode": cell.mode,
                "quantization": cell.quantization,
                "gpu": cell.gpu,
                "gpu_count": cell.gpu_count,
                "prefix_caching": cell.prefix_caching,
                "max_num_batched_tokens": cell.max_num_batched_tokens,
                "config_hashes": config_hashes,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _load_result_cache(self) -> dict[str, int]:
        """Return {cache_key: completed_trial_count} from existing manifests."""
        cache: dict[str, int] = {}
        for manifest_path in self.output_dir.rglob("manifest.json"):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if data.get("status") == "completed":
                    key = data.get("cache_key", "")
                    if key:
                        cache[key] = cache.get(key, 0) + 1
            except Exception:
                pass
        return cache

    def _find_completed_runs(self) -> set[str]:
        """Scan output_dir for completed run manifests."""
        completed: set[str] = set()
        for manifest_path in self.output_dir.rglob("manifest.json"):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if data.get("status") == "completed":
                    completed.add(data.get("run_id", ""))
            except Exception:
                logger.warning("Failed to read manifest %s, skipping", manifest_path, exc_info=True)
        return completed

    @staticmethod
    def _compute_trial_cv(results: list[RunResult]) -> float:
        """Compute CV of throughput across trial results."""
        throughputs: list[float] = []
        for r in results:
            if r.benchmark_results:
                try:
                    data = json.loads(r.benchmark_results[-1].read_text(encoding="utf-8"))
                    # Look for output throughput
                    tp = data.get("output_throughput", data.get("generation_throughput", 0))
                    if tp:
                        throughputs.append(float(tp))
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Failed to read throughput from %s", r.benchmark_results[-1], exc_info=True,
                    )

        if len(throughputs) < 2:
            return 0.0

        arr = np.array(throughputs, dtype=np.float64)
        mean = np.mean(arr)
        if mean == 0:
            return 0.0
        return float(np.std(arr, ddof=1) / mean)

    def __repr__(self) -> str:
        return (
            f"SweepOrchestrator(sweep={self.sweep_path!r}, "
            f"cells={len(self._cells)}, dry_run={self.dry_run})"
        )


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

    def _resolve_click_path(
        _ctx: click.Context, _param: click.Parameter, value: Path | None
    ) -> Path | None:
        if value is None:
            return None
        return resolve_path(value)

    @click.command("sweep")
    @click.option(
        "--config",
        "--sweep",
        "config_path",
        required=True,
        type=click.Path(path_type=Path),
        callback=_resolve_existing_click_path,
        help="Path to sweep config YAML.",
    )
    @click.option(
        "--output",
        "output_dir",
        default=default_results_root(),
        show_default=False,
        type=click.Path(path_type=Path),
        callback=_resolve_click_path,
        help="Output directory for results. Defaults to the product-local results/ tree.",
    )
    @click.option(
        "--config-root",
        default=default_config_root(),
        show_default=False,
        type=click.Path(path_type=Path),
        callback=_resolve_existing_click_path,
        help="Root directory for config files. Defaults to the product-local configs/ tree.",
    )
    @click.option("--dry-run", is_flag=True, help="Print plan without executing.")
    @click.option("--resume", "do_resume", is_flag=True, help="Resume from previous run.")
    def main(
        config_path: str,
        output_dir: str,
        config_root: str,
        dry_run: bool,
        do_resume: bool,
    ) -> None:
        """Execute an ISB-1 benchmark sweep."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        orchestrator = SweepOrchestrator(
            sweep_path=config_path,
            output_dir=output_dir,
            config_root=config_root,
            dry_run=dry_run,
        )

        if dry_run:
            click.echo(orchestrator.plan())
            return

        if do_resume:
            summary = orchestrator.resume()
        else:
            summary = orchestrator.execute()

        click.echo("\nSweep complete:")
        click.echo(f"  Total cells:  {summary.total_cells}")
        click.echo(f"  Completed:    {summary.completed}")
        click.echo(f"  Failed:       {summary.failed}")
        click.echo(f"  Skipped:      {summary.skipped}")

        if summary.failed > 0:
            raise SystemExit(1)

    main()


if __name__ == "__main__":
    _cli()
