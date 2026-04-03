"""BenchmarkRunner — single-cell executor for ISB-1 benchmarks."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.client import BenchmarkClient
from harness.config_validator import ConfigValidator
from harness.engine_metrics import EngineMetricsCollector
from harness.lockfile import LockfileGenerator
from harness.manifest import RunManifest
from harness.paths import default_config_root, default_results_root, resolve_path
from harness.server import VLLMServer
from harness.telemetry import TelemetryCollector
from harness.warmup import WarmupValidator
from workloads.materialize import materialize_requests, save_requests

logger = logging.getLogger(__name__)


@dataclass
class CellConfig:
    """All parameters needed to execute a single benchmark cell."""

    gpu: str = ""
    gpu_count: int = 1
    model: str = ""
    model_hf_id: str = ""
    model_revision: str = ""
    workload: str = ""
    mode: str = ""
    quantization: str = "fp8"
    topology: str = "tp1"
    kv_cache_dtype: str = "auto"
    prefix_caching: bool = True
    max_num_batched_tokens: int | None = None  # None = vLLM default
    trial_number: int = 1
    port: int = 8000
    startup_timeout: int = 600
    vllm_extra_args: list[str] = field(default_factory=list)
    num_prompts: int = 1000
    rate_sweep: list[float] = field(default_factory=lambda: [1.0])
    seed: int = 42
    arrival_model: str = "poisson"
    arrival_shape: float | None = None
    goodput_slo: dict[str, Any] | None = None
    warmup_requests: int = 100
    warmup_seconds: float = 60.0
    warmup_max_extensions: int = 3
    warmup_variance_threshold: float = 0.20
    measurement_duration_seconds: float = 600.0
    cooldown_seconds: float = 30.0
    output_dir: str | Path = field(default_factory=default_results_root)
    config_root: str | Path = field(default_factory=default_config_root)
    config_paths: list[str | Path] = field(default_factory=list)
    external_endpoint: str | None = None


@dataclass
class RunResult:
    """Paths and status from a completed benchmark cell."""

    run_id: str = ""
    status: str = "completed"
    error_message: str | None = None
    manifest_path: Path | None = None
    lockfile_path: Path | None = None
    benchmark_results: list[Path] = field(default_factory=list)
    telemetry_path: Path | None = None
    engine_metrics_path: Path | None = None
    server_log_path: Path | None = None
    trace_path: Path | None = None
    warmup_stable: bool = False
    warmup_extensions: int = 0
    startup_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "status": self.status,
            "error_message": self.error_message,
            "warmup_stable": self.warmup_stable,
            "warmup_extensions": self.warmup_extensions,
            "startup_time_seconds": self.startup_time_seconds,
        }
        for key in (
            "manifest_path",
            "lockfile_path",
            "telemetry_path",
            "engine_metrics_path",
            "server_log_path",
            "trace_path",
        ):
            value = getattr(self, key)
            payload[key] = str(value) if value else None
        payload["benchmark_results"] = [str(path) for path in self.benchmark_results]
        return payload


class BenchmarkRunner:
    """Orchestrate a single ISB-1 benchmark cell execution."""

    def __init__(self, cell: CellConfig) -> None:
        self.cell = cell

    @staticmethod
    def _generate_run_id(cell: CellConfig) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        return "-".join(
            [
                "isb1",
                date_str,
                cell.gpu,
                cell.model,
                cell.workload,
                cell.mode,
                cell.quantization,
                f"{cell.trial_number:03d}",
            ]
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _compute_config_hash(cell: CellConfig) -> str:
        canonical = json.dumps(
            {
                "gpu": cell.gpu,
                "gpu_count": cell.gpu_count,
                "model": cell.model,
                "model_hf_id": cell.model_hf_id,
                "workload": cell.workload,
                "mode": cell.mode,
                "quantization": cell.quantization,
                "topology": cell.topology,
                "kv_cache_dtype": cell.kv_cache_dtype,
                "prefix_caching": cell.prefix_caching,
                "max_num_batched_tokens": cell.max_num_batched_tokens,
                "num_prompts": cell.num_prompts,
                "rate_sweep": cell.rate_sweep,
                "seed": cell.seed,
                "arrival_model": cell.arrival_model,
                "arrival_shape": cell.arrival_shape,
                "goodput_slo": cell.goodput_slo,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_vllm_args(self) -> list[str]:
        args: list[str] = []
        if self.cell.topology.startswith("tp"):
            tp = self.cell.topology[2:]
            if tp.isdigit():
                args.extend(["--tensor-parallel-size", tp])
        if self.cell.quantization and self.cell.quantization not in {"bf16", "fp16"}:
            args.extend(["--quantization", self.cell.quantization])
        if self.cell.quantization in {"bf16", "fp16"}:
            args.extend(["--dtype", self.cell.quantization])
        if self.cell.kv_cache_dtype and self.cell.kv_cache_dtype != "auto":
            args.extend(["--kv-cache-dtype", self.cell.kv_cache_dtype])
        args.extend(["--gpu-memory-utilization", "0.90"])
        if self.cell.prefix_caching:
            args.append("--enable-prefix-caching")
        if self.cell.max_num_batched_tokens is not None:
            args.extend(["--max-num-batched-tokens", str(self.cell.max_num_batched_tokens)])
        args.extend(self.cell.vllm_extra_args)
        return args

    def run(self) -> RunResult:
        cell = self.cell
        run_id = self._generate_run_id(cell)
        run_dir = resolve_path(cell.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        result = RunResult(run_id=run_id)
        manifest = RunManifest(
            run_id=run_id,
            gpu=cell.gpu,
            gpu_count=cell.gpu_count,
            model=cell.model,
            model_hf_id=cell.model_hf_id,
            model_revision=cell.model_revision,
            workload=cell.workload,
            mode=cell.mode,
            quantization=cell.quantization,
            topology=cell.topology,
            kv_cache_dtype=cell.kv_cache_dtype,
            config_hash=self._compute_config_hash(cell),
            trial_number=cell.trial_number,
            total_requests=cell.num_prompts,
            warmup_requests=cell.warmup_requests,
            benchmark_runner="isb1_openai_replay",
        )
        manifest.stamp_start()

        server: VLLMServer | None = None
        telemetry: TelemetryCollector | None = None
        engine_metrics: EngineMetricsCollector | None = None
        trace_hash: str | None = None
        base_url: str = f"http://localhost:{cell.port}"

        try:
            if not cell.external_endpoint:
                logger.info("[%s] Validating configuration", run_id)
                validator = ConfigValidator(cell.config_root)
                memory_result = validator.check_memory_fit(
                    cell.gpu, cell.model, cell.quantization, cell.gpu_count
                )
                if not memory_result.ok:
                    raise RuntimeError(f"Config validation failed:\n{memory_result.summary()}")
            else:
                logger.info("[%s] Skipping config validation (external endpoint)", run_id)

            logger.info("[%s] Materializing workload trace", run_id)
            requests = materialize_requests(
                cell.workload,
                config_root=cell.config_root,
                num_requests=cell.num_prompts,
            )
            trace_path = run_dir / "trace.jsonl"
            save_requests(requests, trace_path)
            result.trace_path = trace_path
            trace_hash = self._sha256_file(trace_path)
            manifest.trace_path = str(trace_path)
            manifest.trace_request_count = len(requests)
            manifest.trace_sha256 = trace_hash

            if cell.external_endpoint:
                base_url = cell.external_endpoint.rstrip("/")
                logger.info("[%s] Using external endpoint: %s", run_id, base_url)
                manifest.benchmark_runner = "isb1_openai_replay_external"
            else:
                logger.info("[%s] Starting vLLM server", run_id)
                server = VLLMServer(
                    model=cell.model_hf_id,
                    port=cell.port,
                    extra_args=self._build_vllm_args(),
                    log_dir=run_dir / "logs",
                    startup_timeout=cell.startup_timeout,
                )
                server.start()
                result.startup_time_seconds = server.startup_time_seconds or 0.0
                result.server_log_path = run_dir / "logs" / "vllm_server.log"
                base_url = server.base_url

            if not cell.external_endpoint:
                logger.info("[%s] Starting telemetry collection", run_id)
                telemetry = TelemetryCollector(output_path=run_dir / "telemetry.csv")
                telemetry.start()
                result.telemetry_path = run_dir / "telemetry.csv"

            metrics_url = f"{base_url}/metrics"
            logger.info("[%s] Starting engine metrics collection", run_id)
            engine_metrics = EngineMetricsCollector(
                metrics_url=metrics_url,
                output_path=run_dir / "engine_metrics.jsonl",
            )
            engine_metrics.start()
            result.engine_metrics_path = run_dir / "engine_metrics.jsonl"

            logger.info("[%s] Running benchmark replay", run_id)
            client = BenchmarkClient(
                base_url=base_url,
                model=cell.model_hf_id,
                result_dir=run_dir / "benchmark",
                requests=requests,
                arrival_model=cell.arrival_model,
                arrival_shape=cell.arrival_shape,
                goodput_slo=cell.goodput_slo,
                seed=cell.seed,
            )
            benchmark_paths = client.run_sweep(
                rate_sweep=cell.rate_sweep,
                request_pool_size=len(requests),
                measurement_duration_seconds=cell.measurement_duration_seconds,
            )
            result.benchmark_results = benchmark_paths

            logger.info("[%s] Validating warmup", run_id)
            if benchmark_paths:
                raw_data = client.parse_results(benchmark_paths[-1])
                per_request = raw_data.get("per_request", [])
                if per_request:
                    warmup = WarmupValidator(
                        warmup_requests=cell.warmup_requests,
                        warmup_seconds=cell.warmup_seconds,
                        max_extensions=cell.warmup_max_extensions,
                        variance_threshold=cell.warmup_variance_threshold,
                    )
                    warmup_result = warmup.validate(per_request)
                    manifest.warmup_stable = warmup_result.is_stable
                    result.warmup_stable = warmup_result.is_stable
                    result.warmup_extensions = warmup_result.warmup_extensions

            manifest.status = "completed"
            result.status = "completed"

        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] Benchmark failed: %s", run_id, exc)
            manifest.status = "failed"
            manifest.error_message = str(exc)
            result.status = "failed"
            result.error_message = str(exc)

        finally:
            if telemetry is not None:
                try:
                    telemetry.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to stop telemetry: %s", exc)

            if engine_metrics is not None:
                try:
                    engine_metrics.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to stop engine metrics: %s", exc)

            if server is not None:
                try:
                    server.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to stop server: %s", exc)

            if cell.cooldown_seconds > 0:
                logger.info("[%s] Cooldown for %.0fs", run_id, cell.cooldown_seconds)
                time.sleep(cell.cooldown_seconds)

            manifest.stamp_end()
            elapsed = 0.0
            if manifest.timestamp_start and manifest.timestamp_end:
                try:
                    t_start = datetime.fromisoformat(manifest.timestamp_start)
                    t_end = datetime.fromisoformat(manifest.timestamp_end)
                    elapsed = (t_end - t_start).total_seconds()
                except ValueError:
                    elapsed = 0.0
            manifest.duration_seconds = elapsed

            manifest_path = run_dir / "manifest.json"
            manifest.save(manifest_path)
            result.manifest_path = manifest_path

            lockgen = LockfileGenerator()
            lockgen.generate(
                model_hf_id=cell.model_hf_id,
                engine_args={
                    "model": cell.model_hf_id,
                    "port": cell.port,
                    "extra_args": self._build_vllm_args(),
                },
                config_paths=cell.config_paths,
                random_seeds={"benchmark": cell.seed},
                benchmark_runner="isb1_openai_replay",
                trace_info=(
                    {
                        "path": str(result.trace_path),
                        "sha256": trace_hash,
                    }
                    if result.trace_path and trace_hash
                    else None
                ),
            )
            lockfile_path = run_dir / "lockfile.json"
            lockgen.save(lockfile_path)
            result.lockfile_path = lockfile_path

            summary_path = run_dir / "run_result.json"
            summary_path.write_text(
                json.dumps(result.to_dict(), indent=2, default=str) + "\n",
                encoding="utf-8",
            )

        logger.info("[%s] Run complete: status=%s dir=%s", run_id, result.status, run_dir)
        return result

    def __repr__(self) -> str:
        return (
            f"BenchmarkRunner(model={self.cell.model!r}, gpu={self.cell.gpu!r}, "
            f"workload={self.cell.workload!r})"
        )
