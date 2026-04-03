"""RunManifest — metadata record for a single benchmark run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class RunManifest:
    """Immutable metadata for one ISB-1 benchmark cell execution."""

    # ── identity ────────────────────────────────────────────────────────
    run_id: str = ""
    benchmark_version: str = "1.0.0"

    # ── timestamps (ISO 8601) ───────────────────────────────────────────
    timestamp_start: str = ""
    timestamp_end: str = ""

    # ── hardware ────────────────────────────────────────────────────────
    gpu: str = ""
    gpu_count: int = 1

    # ── model ───────────────────────────────────────────────────────────
    model: str = ""
    model_hf_id: str = ""
    model_revision: str = ""

    # ── benchmark parameters ────────────────────────────────────────────
    workload: str = ""
    mode: str = ""
    quantization: str = ""
    topology: str = ""
    kv_cache_dtype: str = "auto"
    prefix_caching: bool = True
    max_num_batched_tokens: int | None = None

    # ── config integrity ────────────────────────────────────────────────
    config_hash: str = ""    # SHA-256 of serving config files
    cache_key: str = ""      # composite hash for result-cache invalidation

    # ── trial ───────────────────────────────────────────────────────────
    trial_number: int = 1

    # ── measurement ─────────────────────────────────────────────────────
    total_requests: int = 0
    duration_seconds: float = 0.0
    warmup_requests: int = 0
    warmup_stable: bool = False

    # ── runner / trace provenance ───────────────────────────────────────
    benchmark_runner: str = ""
    trace_path: str = ""
    trace_request_count: int = 0
    trace_sha256: str = ""

    # ── outcome ─────────────────────────────────────────────────────────
    status: str = "completed"  # "completed" | "failed" | "unstable"
    error_message: Optional[str] = None

    # ── helpers ─────────────────────────────────────────────────────────

    def stamp_start(self) -> None:
        """Record the start timestamp in UTC ISO 8601."""
        self.timestamp_start = datetime.now(timezone.utc).isoformat()

    def stamp_end(self) -> None:
        """Record the end timestamp in UTC ISO 8601."""
        self.timestamp_end = datetime.now(timezone.utc).isoformat()

    # ── serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        """Construct a RunManifest from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save(self, path: str | Path) -> Path:
        """Write the manifest as a JSON file.  Returns the resolved path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        """Load a manifest from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RunManifest(run_id={self.run_id!r}, model={self.model!r}, "
            f"gpu={self.gpu!r}, status={self.status!r})"
        )
