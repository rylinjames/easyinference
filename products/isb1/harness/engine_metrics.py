"""EngineMetricsCollector — scrapes vLLM Prometheus /metrics endpoint."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_SCRAPE_INTERVAL = 5.0  # seconds

# Prometheus metrics we care about (metric name -> friendly key)
# Updated for vLLM v0.18+ which renamed several metrics.
_METRIC_MAP: dict[str, str] = {
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    "vllm:num_requests_swapped": "num_requests_swapped",
    # vLLM v0.18 renamed gpu_cache_usage_perc -> kv_cache_usage_perc
    "vllm:kv_cache_usage_perc": "kv_cache_utilization",
    "vllm:gpu_cache_usage_perc": "kv_cache_utilization",  # legacy fallback
    "vllm:cpu_cache_usage_perc": "cpu_cache_usage_perc",
    "vllm:num_preemptions_total": "preemptions",
    "vllm:avg_generation_throughput_toks_per_s": "avg_generation_throughput",
    "vllm:request_success_total": "request_success_total",
    "vllm:request_failure_total": "request_failure_total",
    # vLLM v0.18 replaced prefix_cache_hit_rate gauge with separate counters
    "vllm:prefix_cache_hit_rate": "prefix_cache_hit_rate",  # legacy
    "vllm:prefix_cache_hits_total": "_prefix_cache_hits",
    "vllm:prefix_cache_queries_total": "_prefix_cache_queries",
}

# Also map num_requests_waiting -> queue_depth for MetricComputer compatibility
_DERIVED_KEYS = {
    "num_requests_waiting": "queue_depth",
}

# Regex to parse Prometheus text exposition format
_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[^\s]+)"
    r"(?:\s+(?P<timestamp>\d+))?$"
)


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text format into a flat {metric_name: value} dict.

    For metrics with labels, the full ``name{labels}`` is stored as key.
    For label-less metrics, just the name is stored.
    """
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if m:
            name = m.group("name")
            try:
                val = float(m.group("value"))
            except ValueError:
                continue
            # Use the full key with labels for disambiguation
            labels = m.group("labels")
            if labels:
                full_key = f"{name}{{{labels}}}"
                metrics[full_key] = val
            # Always store the bare name (last value wins for counters)
            metrics[name] = val
    return metrics


class EngineMetricsCollector:
    """Periodically scrape vLLM Prometheus metrics in a background thread.

    Parameters
    ----------
    metrics_url : str
        URL of the Prometheus ``/metrics`` endpoint.
    output_path : str | Path
        Path to JSONL file for persisting scraped snapshots.
    interval : float
        Scrape interval in seconds (default 5.0).
    """

    def __init__(
        self,
        metrics_url: str,
        output_path: str | Path,
        interval: float = _SCRAPE_INTERVAL,
    ) -> None:
        self.metrics_url = metrics_url
        self.output_path = Path(output_path)
        self.interval = interval

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshots: list[dict[str, Any]] = []

    # ── Scraping ─────────────────────────────────────────────────────────

    def _scrape_once(self) -> Optional[dict[str, Any]]:
        """Fetch and parse one snapshot from the metrics endpoint."""
        try:
            resp = requests.get(self.metrics_url, timeout=10)
            if resp.status_code != 200:
                logger.debug("Metrics endpoint returned %d", resp.status_code)
                return None
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.debug("Metrics scrape failed: %s", exc)
            return None

        raw = _parse_prometheus_text(resp.text)

        snapshot: dict[str, Any] = {"timestamp": time.time()}
        for prom_name, friendly_key in _METRIC_MAP.items():
            # Try exact match first, then colon-replaced variant
            val = raw.get(prom_name)
            if val is None:
                alt_name = prom_name.replace(":", "_")
                val = raw.get(alt_name)
            if val is not None:
                snapshot[friendly_key] = val

        # Derive prefix_cache_hit_rate from counters if not directly available
        if "prefix_cache_hit_rate" not in snapshot:
            hits = snapshot.pop("_prefix_cache_hits", 0)
            queries = snapshot.pop("_prefix_cache_queries", 0)
            if queries > 0:
                snapshot["prefix_cache_hit_rate"] = hits / queries
        else:
            snapshot.pop("_prefix_cache_hits", None)
            snapshot.pop("_prefix_cache_queries", None)

        # Derive queue_depth from num_requests_waiting
        for src_key, dst_key in _DERIVED_KEYS.items():
            if src_key in snapshot and dst_key not in snapshot:
                snapshot[dst_key] = snapshot[src_key]

        return snapshot

    # ── Thread loop ──────────────────────────────────────────────────────

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as fh:
            while not self._stop_event.is_set():
                snapshot = self._scrape_once()
                if snapshot:
                    self._snapshots.append(snapshot)
                    fh.write(json.dumps(snapshot, default=str) + "\n")
                    fh.flush()
                self._stop_event.wait(self.interval)

    # ── Public API ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background metrics collection."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("EngineMetricsCollector already running")

        self._stop_event.clear()
        self._snapshots.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="engine-metrics"
        )
        self._thread.start()
        logger.info("Engine metrics collection started (%s)", self.metrics_url)

    def stop(self) -> Path:
        """Stop collection and return path to the JSONL file."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        logger.info(
            "Engine metrics stopped, %d snapshots at %s",
            len(self._snapshots),
            self.output_path,
        )
        return self.output_path

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        """In-memory copy of all collected snapshots."""
        return list(self._snapshots)

    # ── Loading persisted data ───────────────────────────────────────────

    @staticmethod
    def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
        """Load a JSONL engine metrics file."""
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __repr__(self) -> str:
        state = "running" if (self._thread and self._thread.is_alive()) else "stopped"
        return (
            f"EngineMetricsCollector(url={self.metrics_url!r}, "
            f"snapshots={len(self._snapshots)}, {state})"
        )
