"""TelemetryCollector — background GPU metrics collection via DCGM or nvidia-smi."""

from __future__ import annotations

import csv
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# CSV column order
_COLUMNS = [
    "timestamp",
    "gpu_index",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "memory_used_mb",
    "power_draw_watts",
    "gpu_clock_mhz",
    "temperature_gpu_celsius",
]

_COLLECTION_INTERVAL = 1.0  # seconds


class TelemetryCollector:
    """Collect GPU telemetry at 1-second intervals in a background thread.

    Prefers DCGM (``dcgmi``) when available, falls back to ``nvidia-smi dmon``.

    Parameters
    ----------
    output_path : str | Path
        CSV file to write telemetry samples.
    gpu_indices : list[int] | None
        GPU indices to monitor.  ``None`` monitors all.
    interval : float
        Collection interval in seconds (default 1.0).
    """

    def __init__(
        self,
        output_path: str | Path,
        gpu_indices: list[int] | None = None,
        interval: float = _COLLECTION_INTERVAL,
    ) -> None:
        self.output_path = Path(output_path)
        self.gpu_indices = gpu_indices
        self.interval = interval

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None

    # ── Backend detection ────────────────────────────────────────────────

    @staticmethod
    def _has_dcgm() -> bool:
        """Check if DCGM is available and functional."""
        try:
            result = subprocess.run(
                ["dcgmi", "discovery", "-l"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _has_nvidia_smi() -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _detect_backend(self) -> str:
        if self._has_dcgm():
            return "dcgm"
        if self._has_nvidia_smi():
            return "nvidia-smi"
        return "none"

    # ── Collection loops ─────────────────────────────────────────────────

    def _collect_nvidia_smi_loop(self, writer: csv.writer) -> None:
        """Poll nvidia-smi at the configured interval."""
        query = (
            "index,utilization.gpu,utilization.memory,"
            "memory.used,power.draw,clocks.current.graphics,temperature.gpu"
        )
        gpu_flag = []
        if self.gpu_indices:
            gpu_flag = ["--id=" + ",".join(str(i) for i in self.gpu_indices)]

        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                        *gpu_flag,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    ts = time.time()
                    for line in result.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 7:
                            row = [
                                f"{ts:.3f}",
                                parts[0],                                    # gpu_index
                                self._safe_float(parts[1]),                  # gpu_util
                                self._safe_float(parts[2]),                  # mem_util
                                self._safe_float(parts[3]),                  # mem_used_mb
                                self._safe_float(parts[4]),                  # power
                                self._safe_float(parts[5]),                  # clock
                                self._safe_float(parts[6]),                  # temp
                            ]
                            writer.writerow(row)
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.warning("nvidia-smi poll failed: %s", exc)

            self._stop_event.wait(self.interval)

    def _collect_dcgm_loop(self, writer: csv.writer) -> None:
        """Use dcgmi dmon for continuous collection."""
        field_ids = "203,204,252,155,156,150"  # gpu_util, mem_util, mem_used, power, clock, temp
        cmd = ["dcgmi", "dmon", "-e", field_ids, "-d", str(int(self.interval * 1000))]
        if self.gpu_indices:
            cmd.extend(["-i", ",".join(str(i) for i in self.gpu_indices)])

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            logger.error("dcgmi not found, falling back to nvidia-smi")
            self._backend = "nvidia-smi"
            self._collect_nvidia_smi_loop(writer)
            return

        assert self._process.stdout is not None
        for line in self._process.stdout:
            if self._stop_event.is_set():
                break

            line = line.strip()
            # Skip header / comment lines
            if not line or line.startswith("#") or line.startswith("Entity"):
                continue

            parts = line.split()
            if len(parts) >= 7:
                ts = time.time()
                row = [
                    f"{ts:.3f}",
                    parts[1],                                    # gpu_index
                    self._safe_float(parts[2]),                  # gpu_util
                    self._safe_float(parts[3]),                  # mem_util
                    self._safe_float(parts[4]),                  # mem_used_mb
                    self._safe_float(parts[5]),                  # power
                    self._safe_float(parts[6]),                  # clock
                    self._safe_float(parts[7]) if len(parts) > 7 else "",  # temp
                ]
                writer.writerow(row)

    # ── Thread entry point ───────────────────────────────────────────────

    def _run(self) -> None:
        """Background thread main loop."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(_COLUMNS)
            fh.flush()

            if self._backend == "dcgm":
                self._collect_dcgm_loop(writer)
            elif self._backend == "nvidia-smi":
                self._collect_nvidia_smi_loop(writer)
            else:
                logger.warning(
                    "No GPU telemetry backend available; writing header-only CSV"
                )

    # ── Public API ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background telemetry collection."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("TelemetryCollector already running")

        self._backend = self._detect_backend()
        logger.info("Telemetry backend: %s", self._backend)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry")
        self._thread.start()

    def stop(self) -> Path:
        """Stop collection and return the path to the CSV file.

        Blocks until the collection thread has finished writing.
        """
        self._stop_event.set()

        # Terminate dcgm subprocess if running
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

        logger.info("Telemetry stopped, data at %s", self.output_path)
        return self.output_path

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(val: str) -> str:
        """Convert to float-string, returning empty string on failure."""
        val = val.strip()
        if val in ("-", "N/A", "[N/A]", ""):
            return ""
        try:
            return str(float(val))
        except ValueError:
            return val

    @staticmethod
    def load_csv(path: str | Path) -> list[dict[str, Any]]:
        """Load a telemetry CSV as a list of dicts."""
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                parsed: dict[str, Any] = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v) if v else None
                    except ValueError:
                        parsed[k] = v
                rows.append(parsed)
        return rows

    def __repr__(self) -> str:
        state = "running" if (self._thread and self._thread.is_alive()) else "stopped"
        return f"TelemetryCollector(output={self.output_path!r}, backend={self._backend!r}, {state})"
