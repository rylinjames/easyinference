"""Global configuration and runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
KNOWLEDGE_DIR = PACKAGE_ROOT / "knowledge"

DCGM_DEFAULT_PORT = 9400
AMD_DME_DEFAULT_PORT = 5000
ENGINE_METRICS_DEFAULT_PORT = 8000


class Settings:
    """Runtime settings, overridable via environment variables."""

    def __init__(self) -> None:
        self.debug = os.getenv("INFERSCOPE_DEBUG", "0") == "1"
        self.cache_dir = Path(os.getenv("INFERSCOPE_CACHE_DIR", str(Path.home() / ".inferscope")))
        self.benchmark_dir = Path(os.getenv("INFERSCOPE_BENCHMARK_DIR", str(self.cache_dir / "benchmarks")))
        self.default_gpu_util = float(os.getenv("INFERSCOPE_DEFAULT_GPU_UTIL", "0.92"))
        self.max_tools = 25

    @property
    def knowledge_dir(self) -> Path:
        return KNOWLEDGE_DIR


settings = Settings()
