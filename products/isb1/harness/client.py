"""BenchmarkClient — internal replay facade for ISB-1 runs."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

from harness.replay_client import run_rate
from workloads.base import Request

logger = logging.getLogger(__name__)

_DEFAULT_SESSION_HEADER = "X-Session-ID"


class BenchmarkClient:
    """Execute ISB-1 request traces against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        result_dir: str | Path,
        *,
        requests: list[Request],
        arrival_model: str,
        arrival_shape: float | None,
        goodput_slo: dict[str, Any] | None,
        seed: int = 42,
        session_header_name: str = _DEFAULT_SESSION_HEADER,
    ) -> None:
        if not requests:
            raise ValueError("BenchmarkClient requires at least one request in the pool")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.result_dir = Path(result_dir)
        self.requests = list(requests)
        self.arrival_model = arrival_model
        self.arrival_shape = arrival_shape
        self.goodput_slo = goodput_slo
        self.seed = seed
        self.session_header_name = session_header_name

    def _request_count_for_rate(
        self,
        request_rate: float,
        request_pool_size: int,
        measurement_duration_seconds: float,
    ) -> int:
        if not math.isfinite(request_rate) or request_rate <= 0 or measurement_duration_seconds <= 0:
            return max(1, request_pool_size)
        return max(request_pool_size, math.ceil(request_rate * measurement_duration_seconds))

    def run(
        self,
        request_rate: float,
        request_pool_size: int,
        *,
        measurement_duration_seconds: float,
        rate_index: int,
        timeout: int = 7200,
    ) -> Path:
        """Run one rate point and write a raw JSON result file."""
        self.result_dir.mkdir(parents=True, exist_ok=True)
        expanded_request_count = self._request_count_for_rate(
            request_rate,
            request_pool_size,
            measurement_duration_seconds,
        )
        logger.info(
            "Running internal replay benchmark: rate=%s req/s request_pool=%d expanded=%d duration=%.1fs",
            request_rate,
            request_pool_size,
            expanded_request_count,
            measurement_duration_seconds,
        )
        result = asyncio.run(
            run_rate(
                base_url=self.base_url,
                model=self.model,
                request_pool=self.requests,
                request_count=expanded_request_count,
                request_rate=request_rate,
                arrival_model=self.arrival_model,
                arrival_shape=self.arrival_shape,
                seed=self.seed + rate_index,
                request_timeout_seconds=min(timeout, 600),
                total_timeout_seconds=timeout,
                session_header_name=self.session_header_name,
                goodput_slo=self.goodput_slo,
            )
        )
        rate_label = "closed-loop" if not math.isfinite(request_rate) else f"{request_rate:g}rps"
        result_path = self.result_dir / f"{rate_index:03d}-{rate_label}.json"
        result_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        logger.info("Benchmark results written to %s", result_path)
        if result.completed < 1:
            raise RuntimeError(
                f"Benchmark replay at rate {rate_label} completed zero requests; see {result_path}"
            )
        return result_path

    def run_sweep(
        self,
        rate_sweep: list[float],
        request_pool_size: int,
        *,
        measurement_duration_seconds: float,
        timeout: int = 7200,
    ) -> list[Path]:
        """Run the benchmark at each configured rate sequentially."""
        results: list[Path] = []
        for rate_index, rate in enumerate(rate_sweep):
            path = self.run(
                request_rate=rate,
                request_pool_size=request_pool_size,
                measurement_duration_seconds=measurement_duration_seconds,
                rate_index=rate_index,
                timeout=timeout,
            )
            results.append(path)
        return results

    @staticmethod
    def parse_results(result_path: str | Path) -> dict[str, Any]:
        """Parse a raw benchmark JSON result file."""
        path = Path(result_path)
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            data = json.loads(text)
            return data[-1] if data else {}
        if text.startswith("{"):
            return json.loads(text)
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            return json.loads(lines[-1])
        return {}

    def __repr__(self) -> str:
        return (
            f"BenchmarkClient(base_url={self.base_url!r}, model={self.model!r}, "
            f"result_dir={self.result_dir!r})"
        )
