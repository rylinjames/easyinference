"""VLLMServer — manages a vLLM serving process lifecycle."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000
_HEALTH_POLL_INTERVAL = 5  # seconds
_DEFAULT_STARTUP_TIMEOUT = 600  # seconds
_GRACEFUL_SHUTDOWN_TIMEOUT = 30  # seconds


class VLLMServer:
    """Spawn, monitor, and stop a ``vllm serve`` process.

    Parameters
    ----------
    model : str
        HuggingFace model ID or local path.
    port : int
        Port number for the HTTP server.
    extra_args : list[str] | None
        Additional CLI arguments forwarded to ``vllm serve``.
    log_dir : str | Path | None
        Directory to write server stdout/stderr logs.  When *None* logs are
        still captured in memory but not persisted.
    startup_timeout : int
        Maximum seconds to wait for the /health endpoint to become ready.
    """

    def __init__(
        self,
        model: str,
        port: int = _DEFAULT_PORT,
        extra_args: list[str] | None = None,
        log_dir: str | Path | None = None,
        startup_timeout: int = _DEFAULT_STARTUP_TIMEOUT,
    ) -> None:
        self.model = model
        self.port = port
        self.extra_args = extra_args or []
        self.log_dir = Path(log_dir) if log_dir else None
        self.startup_timeout = startup_timeout

        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[Any] = None
        self._startup_time: Optional[float] = None

    # ── URLs ─────────────────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    def get_health_url(self) -> str:
        return f"{self.base_url}/health"

    def get_metrics_url(self) -> str:
        """Return the Prometheus metrics endpoint URL."""
        return f"{self.base_url}/metrics"

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def startup_time_seconds(self) -> Optional[float]:
        """Seconds elapsed from process spawn to first healthy response."""
        return self._startup_time

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def pid(self) -> Optional[int]:
        if self._process is None:
            return None
        return self._process.pid

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn vLLM and block until the /health endpoint responds 200.

        Raises
        ------
        RuntimeError
            If the server fails to start within *startup_timeout* seconds or
            the process exits prematurely.
        """
        if self.is_running:
            raise RuntimeError(f"Server already running (pid={self.pid})")

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            *self.extra_args,
        ]
        logger.info("Starting vLLM: %s", " ".join(cmd))

        # Prepare log handles
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / "vllm_server.log"
            self._log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
            stdout_dest = self._log_file
            stderr_dest = subprocess.STDOUT
        else:
            stdout_dest = subprocess.PIPE
            stderr_dest = subprocess.STDOUT

        env = os.environ.copy()
        self._process = subprocess.Popen(
            cmd,
            stdout=stdout_dest,
            stderr=stderr_dest,
            env=env,
            preexec_fn=os.setsid,  # own process group for clean teardown
        )

        self._wait_healthy()

    def _wait_healthy(self) -> None:
        """Poll the health endpoint until it responds or timeout."""
        t0 = time.monotonic()
        url = self.get_health_url()

        while True:
            elapsed = time.monotonic() - t0

            # Check process is still alive
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM process exited with code {self._process.returncode} "
                    f"after {elapsed:.1f}s"
                )

            if elapsed > self.startup_timeout:
                self.stop()
                raise RuntimeError(
                    f"vLLM did not become healthy within {self.startup_timeout}s"
                )

            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    self._startup_time = time.monotonic() - t0
                    logger.info(
                        "vLLM healthy after %.1fs (pid=%s)",
                        self._startup_time,
                        self.pid,
                    )
                    return
            except requests.ConnectionError:
                pass
            except requests.Timeout:
                pass

            time.sleep(_HEALTH_POLL_INTERVAL)

    def stop(self) -> None:
        """Gracefully stop the vLLM process (SIGTERM then SIGKILL)."""
        if self._process is None:
            return

        pid = self._process.pid
        logger.info("Stopping vLLM (pid=%s)", pid)

        # Send SIGTERM to the process group
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            self._process.wait(timeout=_GRACEFUL_SHUTDOWN_TIMEOUT)
            logger.info("vLLM stopped gracefully (pid=%s)", pid)
        except subprocess.TimeoutExpired:
            logger.warning(
                "vLLM did not stop within %ds, sending SIGKILL (pid=%s)",
                _GRACEFUL_SHUTDOWN_TIMEOUT,
                pid,
            )
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self._process.wait(timeout=10)

        # Clean up log file handle
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        self._process = None

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self.is_running else "stopped"
        return f"VLLMServer(model={self.model!r}, port={self.port}, {state})"
