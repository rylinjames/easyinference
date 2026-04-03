"""WarmupValidator — detect and extend warmup phase for steady-state measurement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_SECONDS = 30.0
_DEFAULT_VARIANCE_THRESHOLD = 0.20  # 20% CV
_DEFAULT_EXTENSION_SECONDS = 60.0
_DEFAULT_MAX_EXTENSIONS = 3


@dataclass
class WarmupResult:
    """Outcome of warmup validation."""

    warmup_end_timestamp: float
    """Epoch timestamp marking the end of warmup / start of steady state."""

    is_stable: bool
    """Whether steady-state throughput was reached within allowed extensions."""

    warmup_extensions: int
    """Number of 60-second warmup extensions applied."""

    steady_state_start_index: int
    """Index into the request list where steady-state measurement begins."""

    window_throughputs: list[float]
    """Throughput (tokens/s) observed in each consecutive window."""

    window_cv: float
    """Coefficient of variation across the final set of windows."""


class WarmupValidator:
    """Determine when the inference engine has reached steady state.

    Strategy
    --------
    1.  Exclude the first *warmup_requests* requests (or *warmup_seconds*,
        whichever covers more).
    2.  Divide remaining data into consecutive windows of *window_seconds*.
    3.  Compute throughput (output tokens / wall-clock) per window.
    4.  If the coefficient of variation (CV) across windows exceeds
        *variance_threshold*, extend the warmup by *extension_seconds* and
        repeat from step 2.
    5.  Repeat up to *max_extensions* times.

    Parameters
    ----------
    warmup_requests : int
        Minimum number of initial requests to exclude.
    warmup_seconds : float
        Minimum initial seconds to exclude.
    window_seconds : float
        Size of each throughput measurement window.
    variance_threshold : float
        Maximum acceptable CV across windows.
    extension_seconds : float
        How many seconds to extend warmup on each iteration.
    max_extensions : int
        Maximum number of warmup extensions before declaring "unstable".
    """

    def __init__(
        self,
        warmup_requests: int = 100,
        warmup_seconds: float = 60.0,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
        variance_threshold: float = _DEFAULT_VARIANCE_THRESHOLD,
        extension_seconds: float = _DEFAULT_EXTENSION_SECONDS,
        max_extensions: int = _DEFAULT_MAX_EXTENSIONS,
    ) -> None:
        self.warmup_requests = warmup_requests
        self.warmup_seconds = warmup_seconds
        self.window_seconds = window_seconds
        self.variance_threshold = variance_threshold
        self.extension_seconds = extension_seconds
        self.max_extensions = max_extensions

    # ── Public API ───────────────────────────────────────────────────────

    def validate(self, requests: Sequence[dict[str, Any]]) -> WarmupResult:
        """Analyse request data and determine the warmup boundary.

        Parameters
        ----------
        requests : sequence of dict
            Per-request records, each containing at minimum:
            - ``timestamp`` (float): request start epoch
            - ``output_tokens`` (int): tokens generated
            - ``e2e_latency`` (float): total request latency in seconds

        Returns
        -------
        WarmupResult
        """
        if not requests:
            return WarmupResult(
                warmup_end_timestamp=0.0,
                is_stable=False,
                warmup_extensions=0,
                steady_state_start_index=0,
                window_throughputs=[],
                window_cv=0.0,
            )

        # Sort by timestamp
        sorted_reqs = sorted(requests, key=lambda r: r.get("timestamp", 0.0))
        t0 = sorted_reqs[0].get("timestamp", 0.0)

        # Initial exclusion: skip the larger of warmup_requests or warmup_seconds
        skip_index = self._compute_initial_skip(sorted_reqs, t0)
        extensions = 0

        while extensions <= self.max_extensions:
            steady = sorted_reqs[skip_index:]
            if not steady:
                break

            windows = self._compute_windows(steady)
            cv = self._cv(windows)

            logger.debug(
                "Warmup check: skip_index=%d, windows=%d, CV=%.4f (threshold=%.4f)",
                skip_index,
                len(windows),
                cv,
                self.variance_threshold,
            )

            if cv <= self.variance_threshold and len(windows) >= 2:
                warmup_end = steady[0].get("timestamp", t0)
                return WarmupResult(
                    warmup_end_timestamp=warmup_end,
                    is_stable=True,
                    warmup_extensions=extensions,
                    steady_state_start_index=skip_index,
                    window_throughputs=windows,
                    window_cv=cv,
                )

            # Extend warmup
            extensions += 1
            extra_skip = self._requests_in_duration(
                sorted_reqs, skip_index, self.extension_seconds
            )
            skip_index += max(extra_skip, 1)  # advance at least 1 to avoid loops

            if skip_index >= len(sorted_reqs):
                break

        # Could not stabilise
        best_steady = sorted_reqs[skip_index:] if skip_index < len(sorted_reqs) else []
        windows = self._compute_windows(best_steady) if best_steady else []
        cv = self._cv(windows)
        warmup_end = best_steady[0].get("timestamp", t0) if best_steady else t0

        return WarmupResult(
            warmup_end_timestamp=warmup_end,
            is_stable=False,
            warmup_extensions=extensions,
            steady_state_start_index=skip_index,
            window_throughputs=windows,
            window_cv=cv,
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _compute_initial_skip(
        self, reqs: list[dict[str, Any]], t0: float
    ) -> int:
        """Determine the index after the initial warmup period."""
        skip_by_count = min(self.warmup_requests, len(reqs))

        # Find index after warmup_seconds
        cutoff = t0 + self.warmup_seconds
        skip_by_time = 0
        for i, r in enumerate(reqs):
            if r.get("timestamp", 0.0) >= cutoff:
                skip_by_time = i
                break
        else:
            skip_by_time = len(reqs)

        return max(skip_by_count, skip_by_time)

    def _compute_windows(self, reqs: Sequence[dict[str, Any]]) -> list[float]:
        """Divide requests into consecutive windows, return throughput per window."""
        if not reqs:
            return []

        t_start = reqs[0].get("timestamp", 0.0)
        windows: list[float] = []
        window_tokens = 0
        window_begin = t_start

        for r in reqs:
            ts = r.get("timestamp", 0.0)
            # Check if we crossed into a new window
            while ts >= window_begin + self.window_seconds:
                elapsed = self.window_seconds
                throughput = window_tokens / elapsed if elapsed > 0 else 0.0
                windows.append(throughput)
                window_begin += self.window_seconds
                window_tokens = 0

            window_tokens += r.get("output_tokens", 0)

        # Final partial window (only if significant)
        last_end = reqs[-1].get("timestamp", 0.0) + reqs[-1].get("e2e_latency", 0.0)
        remaining = last_end - window_begin
        if remaining > self.window_seconds * 0.5 and window_tokens > 0:
            throughput = window_tokens / remaining if remaining > 0 else 0.0
            windows.append(throughput)

        return windows

    def _requests_in_duration(
        self, reqs: list[dict[str, Any]], start_index: int, duration: float
    ) -> int:
        """Count how many requests fall within *duration* seconds from start_index."""
        if start_index >= len(reqs):
            return 0
        t0 = reqs[start_index].get("timestamp", 0.0)
        cutoff = t0 + duration
        count = 0
        for i in range(start_index, len(reqs)):
            if reqs[i].get("timestamp", 0.0) >= cutoff:
                break
            count += 1
        return count

    @staticmethod
    def _cv(values: list[float]) -> float:
        """Coefficient of variation.  Returns 0.0 for empty/zero-mean data."""
        if len(values) < 2:
            return 0.0
        arr = np.array(values, dtype=np.float64)
        mean = np.mean(arr)
        if mean == 0:
            return 0.0
        return float(np.std(arr, ddof=1) / mean)

    def __repr__(self) -> str:
        return (
            f"WarmupValidator(warmup_requests={self.warmup_requests}, "
            f"warmup_seconds={self.warmup_seconds}, "
            f"variance_threshold={self.variance_threshold})"
        )
