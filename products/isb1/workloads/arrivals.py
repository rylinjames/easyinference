"""Arrival-time generators for ISB-1 benchmark workloads.

Provides stochastic inter-arrival time models that produce absolute
timestamps (in seconds) for when each request should be dispatched.

Includes BurstGPT trace replay for production-realistic arrival patterns
from 213 days of Azure OpenAI serving data (KDD 2025).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BurstGPT auto-download
# ---------------------------------------------------------------------------

_BURSTGPT_HF_URL = (
    "https://huggingface.co/datasets/lzzmm/BurstGPT/resolve/main/"
    "data/BurstGPT_without_fails_1.csv"
)
_BURSTGPT_FILENAME = "BurstGPT_without_fails_1.csv"


class PoissonArrival:
    """Generate request arrival times from a Poisson process.

    Inter-arrival times are drawn from an exponential distribution with
    rate parameter ``rate`` (requests per second).

    Parameters:
        rate: Mean number of requests per second (lambda).
        seed: Random seed for reproducibility.
    """

    def __init__(self, rate: float, seed: int = 42) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, num_requests: int) -> np.ndarray:
        """Return *num_requests* absolute arrival timestamps in seconds.

        The first arrival occurs at a time drawn from the same exponential
        distribution (i.e. the process starts at time 0, and the first
        event follows).

        Args:
            num_requests: Number of timestamps to produce.

        Returns:
            A 1-D ``numpy.ndarray`` of monotonically increasing floats
            representing arrival times in seconds.
        """
        if num_requests <= 0:
            return np.array([], dtype=np.float64)

        inter_arrivals = self.rng.exponential(
            scale=1.0 / self.rate, size=num_requests
        )
        return np.cumsum(inter_arrivals)


class GammaArrival:
    """Generate bursty request arrival times using a Gamma distribution.

    The Gamma distribution generalises the exponential: when the shape
    parameter *k* equals 1 it reduces to a Poisson process; values of
    *k* < 1 produce burstier traffic (higher variance relative to the
    mean), while *k* > 1 produces more regular traffic.

    Parameters:
        rate: Mean number of requests per second.
        shape: Gamma shape parameter *k*.  Smaller values create burstier
            arrival patterns (default 0.5).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self, rate: float, shape: float = 0.5, seed: int = 42
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if shape <= 0:
            raise ValueError(f"shape must be positive, got {shape}")
        self.rate = rate
        self.shape = shape
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, num_requests: int) -> np.ndarray:
        """Return *num_requests* absolute arrival timestamps in seconds.

        Inter-arrival times are drawn from a Gamma distribution whose mean
        equals ``1 / rate`` so the long-run average throughput matches the
        requested rate.

        Args:
            num_requests: Number of timestamps to produce.

        Returns:
            A 1-D ``numpy.ndarray`` of monotonically increasing floats.
        """
        if num_requests <= 0:
            return np.array([], dtype=np.float64)

        # mean of Gamma(shape, scale) = shape * scale
        # We want mean = 1/rate  =>  scale = 1 / (rate * shape)
        scale = 1.0 / (self.rate * self.shape)
        inter_arrivals = self.rng.gamma(
            shape=self.shape, scale=scale, size=num_requests
        )
        return np.cumsum(inter_arrivals)


def _resolve_burstgpt_path(path: Path | None = None) -> Path | None:
    """Return path to BurstGPT CSV, downloading if necessary."""
    if path is not None and path.is_file():
        return path

    cache_dir = Path.home() / ".cache" / "isb1"
    cached = cache_dir / _BURSTGPT_FILENAME
    if cached.is_file():
        return cached

    logger.info("Downloading BurstGPT dataset (~250 MB)...")
    tmp = cached.with_suffix(".tmp")
    try:
        import urllib.request

        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_BURSTGPT_HF_URL, tmp)  # noqa: S310
        tmp.rename(cached)
        logger.info("BurstGPT dataset saved to %s", cached)
        return cached
    except Exception:
        logger.warning("Failed to download BurstGPT dataset", exc_info=True)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None


def _load_burstgpt_timestamps(
    path: Path,
    *,
    service_type: str = "",
    model_filter: str = "",
) -> np.ndarray:
    """Load absolute timestamps from a BurstGPT CSV.

    Parameters
    ----------
    path:
        Path to ``BurstGPT_without_fails.csv``.
    service_type:
        Filter by ``Log Type`` column (e.g. ``"Conversation"``, ``"API"``).
        Empty string means no filter.
    model_filter:
        Filter by ``Model`` column substring (e.g. ``"GPT-4"``).
        Empty string means no filter.

    Returns
    -------
    Sorted array of timestamps in seconds (relative to the first request).
    """
    timestamps: list[float] = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if service_type and row.get("Log Type", "") != service_type:
                continue
            if model_filter and model_filter.lower() not in row.get("Model", "").lower():
                continue
            try:
                timestamps.append(float(row["Timestamp"]))
            except (KeyError, ValueError):
                continue

    if not timestamps:
        return np.array([], dtype=np.float64)

    arr = np.array(timestamps, dtype=np.float64)
    arr.sort()
    arr -= arr[0]  # relative to first request
    return arr


class BurstGPTArrival:
    """Replay real arrival patterns from BurstGPT production traces.

    BurstGPT contains 10.3M traces over 213 days from Azure OpenAI with
    real burstiness patterns: weekly periodicity for conversation services,
    aperiodic bursts for API services, and Gamma-distributed concurrency.

    The trace timestamps are scaled to match the target rate while preserving
    the relative burstiness pattern.

    Parameters:
        rate: Target mean requests per second.
        seed: Random seed for selecting the trace window.
        trace_path: Path to BurstGPT CSV. Auto-downloaded if None.
        service_type: Filter by log type (``"Conversation"`` or ``"API"``).
        model_filter: Filter by model name substring (e.g. ``"GPT-4"``).
        window_seconds: Length of trace window to sample (default 3600 = 1 hour).
    """

    def __init__(
        self,
        rate: float,
        seed: int = 42,
        trace_path: str | Path | None = None,
        service_type: str = "",
        model_filter: str = "",
        window_seconds: float = 3600.0,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self.seed = seed
        self.window_seconds = window_seconds
        self.rng = np.random.default_rng(seed)

        resolved = _resolve_burstgpt_path(
            Path(trace_path) if trace_path else None
        )
        if resolved is not None:
            self._raw_timestamps = _load_burstgpt_timestamps(
                resolved,
                service_type=service_type,
                model_filter=model_filter,
            )
        else:
            self._raw_timestamps = np.array([], dtype=np.float64)

    def generate(self, num_requests: int) -> np.ndarray:
        """Return *num_requests* arrival timestamps preserving real burstiness.

        If BurstGPT data is unavailable, falls back to Poisson arrivals.
        """
        if num_requests <= 0:
            return np.array([], dtype=np.float64)

        if len(self._raw_timestamps) < 10:
            logger.warning("BurstGPT data unavailable or too small, falling back to Poisson")
            return PoissonArrival(self.rate, self.seed).generate(num_requests)

        # Sample a random window from the trace
        total_duration = self._raw_timestamps[-1]
        if total_duration <= self.window_seconds:
            window_start = 0.0
        else:
            window_start = float(self.rng.uniform(0, total_duration - self.window_seconds))

        window_end = window_start + self.window_seconds
        mask = (self._raw_timestamps >= window_start) & (self._raw_timestamps < window_end)
        window = self._raw_timestamps[mask] - window_start

        if len(window) < 2:
            return PoissonArrival(self.rate, self.seed).generate(num_requests)

        # Compute inter-arrival times from the real trace
        inter_arrivals = np.diff(window)
        inter_arrivals = inter_arrivals[inter_arrivals > 0]

        if len(inter_arrivals) == 0:
            return PoissonArrival(self.rate, self.seed).generate(num_requests)

        # Scale to target rate while preserving burstiness pattern
        # Original rate = len(window) / window_duration
        original_rate = len(window) / self.window_seconds
        scale_factor = original_rate / self.rate if self.rate > 0 else 1.0

        # Resample inter-arrivals to produce exactly num_requests
        indices = self.rng.integers(0, len(inter_arrivals), size=num_requests)
        sampled = inter_arrivals[indices] * scale_factor

        return np.cumsum(sampled)
