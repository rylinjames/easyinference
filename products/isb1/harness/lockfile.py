"""LockfileGenerator — captures a complete reproducibility snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class LockfileGenerator:
    """Captures software, hardware, and configuration state for reproducibility."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    # ── Subprocess helper ───────────────────────────────────────────────

    @staticmethod
    def _run_cmd(cmd: list[str], timeout: int = 30) -> Optional[str]:
        """Run a command and return stdout, or None if it fails."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    # ── Individual collectors ───────────────────────────────────────────

    def _collect_vllm_version(self) -> dict[str, Optional[str]]:
        info: dict[str, Optional[str]] = {"version": None, "git_hash": None}
        try:
            import vllm
            info["version"] = getattr(vllm, "__version__", None)
        except ImportError:
            pass
        # Try git hash from vllm install location
        try:
            import vllm
            vllm_path = Path(vllm.__file__).resolve().parent.parent
            git_hash = self._run_cmd(["git", "-C", str(vllm_path), "rev-parse", "HEAD"])
            info["git_hash"] = git_hash
        except Exception:
            pass
        return info

    def _collect_cuda_version(self) -> Optional[str]:
        out = self._run_cmd(["nvcc", "--version"])
        if out:
            for line in out.splitlines():
                if "release" in line.lower():
                    return line.strip()
        # Fallback: try torch
        try:
            import torch
            return torch.version.cuda
        except Exception:
            return None

    def _collect_pytorch_version(self) -> Optional[str]:
        try:
            import torch
            return torch.__version__
        except ImportError:
            return None

    def _collect_nvidia_smi_q(self) -> Optional[str]:
        return self._run_cmd(["nvidia-smi", "-q"], timeout=60)

    def _collect_nvidia_smi_topo(self) -> Optional[str]:
        return self._run_cmd(["nvidia-smi", "topo", "-m"], timeout=30)

    def _collect_uname(self) -> Optional[str]:
        return self._run_cmd(["uname", "-a"])

    def _collect_pip_freeze(self) -> Optional[list[str]]:
        out = self._run_cmd([sys.executable, "-m", "pip", "freeze"], timeout=60)
        if out:
            return out.splitlines()
        return None

    # ── Public API ──────────────────────────────────────────────────────

    def collect_system_info(self) -> dict[str, Any]:
        """Gather all system-level information."""
        return {
            "vllm": self._collect_vllm_version(),
            "cuda_version": self._collect_cuda_version(),
            "pytorch_version": self._collect_pytorch_version(),
            "nvidia_smi_q": self._collect_nvidia_smi_q(),
            "nvidia_smi_topo": self._collect_nvidia_smi_topo(),
            "uname": self._collect_uname(),
            "pip_freeze": self._collect_pip_freeze(),
        }

    def collect_model_info(self, model_hf_id: str) -> dict[str, Any]:
        """Capture the HuggingFace model revision."""
        revision: Optional[str] = None
        try:
            from huggingface_hub import model_info
            info = model_info(model_hf_id)
            revision = info.sha
        except Exception:
            pass
        return {
            "hf_model_id": model_hf_id,
            "revision": revision,
        }

    def collect_engine_args(self, engine_args: dict[str, Any]) -> dict[str, Any]:
        """Store the complete vLLM engine argument set."""
        return {"vllm_engine_args": engine_args}

    @staticmethod
    def hash_config_files(paths: list[str | Path]) -> dict[str, str]:
        """Compute SHA-256 for each config file."""
        hashes: dict[str, str] = {}
        for p in paths:
            p = Path(p)
            if p.exists():
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                hashes[str(p)] = h
        return hashes

    def generate(
        self,
        *,
        model_hf_id: str = "",
        engine_args: dict[str, Any] | None = None,
        config_paths: list[str | Path] | None = None,
        random_seeds: dict[str, int] | None = None,
        benchmark_runner: str = "",
        trace_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the complete lockfile dict."""
        lockfile: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system": self.collect_system_info(),
        }

        if model_hf_id:
            lockfile["model"] = self.collect_model_info(model_hf_id)

        if engine_args is not None:
            lockfile["engine"] = self.collect_engine_args(engine_args)

        if config_paths:
            lockfile["config_hashes"] = self.hash_config_files(config_paths)

        if random_seeds:
            lockfile["random_seeds"] = random_seeds

        if benchmark_runner:
            lockfile["benchmark_runner"] = benchmark_runner

        if trace_info:
            lockfile["trace"] = trace_info

        self._data = lockfile
        return lockfile

    def save(self, path: str | Path) -> Path:
        """Write the lockfile to disk as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._data, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load a previously saved lockfile."""
        return json.loads(Path(path).read_text(encoding="utf-8"))
