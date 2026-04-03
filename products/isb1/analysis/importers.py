"""Importers for external benchmark formats into ISB-1's analysis schema.

Supports:
- vLLM benchmark_serving.py JSON output
- GenAI-Perf / AIPerf CSV output
- Raw JSONL with per-request timing data

This lets operators use ISB-1's analysis, comparison, and leaderboard tools
without re-running benchmarks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def import_vllm_benchmark(path: str | Path) -> list[dict[str, Any]]:
    """Import vLLM benchmark_serving.py JSON output into ISB-1 per-request format.

    vLLM benchmark output has fields like:
    - ttft (seconds)
    - tpot (seconds)
    - latency (seconds) = e2e
    - prompt_len, output_len
    - success
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    # vLLM outputs can be a list of dicts or a dict with nested keys
    if isinstance(raw, dict):
        entries = raw.get("per_request", raw.get("results", raw.get("requests", [raw])))
    elif isinstance(raw, list):
        entries = raw
    else:
        return []

    results: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        result: dict[str, Any] = {
            "request_id": entry.get("request_id", f"vllm-import-{i:06d}"),
            "session_id": None,
            "timestamp": entry.get("timestamp", float(i)),
            "ttft": entry.get("ttft", entry.get("time_to_first_token")),
            "e2e_latency": entry.get("latency", entry.get("e2e_latency", entry.get("request_latency", 0))),
            "output_tokens": entry.get("output_len", entry.get("output_tokens", entry.get("completion_tokens", 0))),
            "prompt_tokens": entry.get("prompt_len", entry.get("prompt_tokens", entry.get("input_tokens"))),
            "total_tokens": None,
            "token_timestamps": [],
            "error": not entry.get("success", True),
            "error_message": entry.get("error", None),
        }
        if result["prompt_tokens"] is not None and result["output_tokens"]:
            result["total_tokens"] = result["prompt_tokens"] + result["output_tokens"]
        results.append(result)

    return results


def import_genai_perf_csv(path: str | Path) -> list[dict[str, Any]]:
    """Import GenAI-Perf / AIPerf CSV output into ISB-1 per-request format.

    GenAI-Perf CSV typically has columns:
    - time_to_first_token, inter_token_latency, output_token_throughput
    - request_latency, output_sequence_length, input_sequence_length
    """
    results: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            ttft = _float_or_none(row.get("time_to_first_token"))
            e2e = _float_or_none(row.get("request_latency", row.get("end_to_end_latency")))
            output_tokens = _int_or_none(row.get("output_sequence_length", row.get("num_output_token")))
            prompt_tokens = _int_or_none(row.get("input_sequence_length", row.get("num_input_token")))

            # GenAI-Perf sometimes reports in milliseconds
            if ttft is not None and ttft > 100:
                ttft /= 1000.0
            if e2e is not None and e2e > 100:
                e2e /= 1000.0

            result: dict[str, Any] = {
                "request_id": f"genai-perf-import-{i:06d}",
                "session_id": None,
                "timestamp": float(i),
                "ttft": ttft,
                "e2e_latency": e2e or 0.0,
                "output_tokens": output_tokens or 0,
                "prompt_tokens": prompt_tokens,
                "total_tokens": (prompt_tokens or 0) + (output_tokens or 0) if prompt_tokens else None,
                "token_timestamps": [],
                "error": False,
            }
            results.append(result)

    return results


def import_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Import raw JSONL with per-request timing data.

    Each line should be a JSON object with at minimum:
    - ttft (seconds)
    - e2e_latency or latency (seconds)
    - output_tokens or output_len (int)

    Extra fields are preserved in the output.
    """
    results: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            result: dict[str, Any] = {
                "request_id": entry.get("request_id", f"jsonl-import-{i:06d}"),
                "session_id": entry.get("session_id"),
                "timestamp": entry.get("timestamp", float(i)),
                "ttft": entry.get("ttft", entry.get("time_to_first_token")),
                "e2e_latency": entry.get("e2e_latency", entry.get("latency", entry.get("request_latency", 0))),
                "output_tokens": entry.get("output_tokens", entry.get("output_len", entry.get("completion_tokens", 0))),
                "prompt_tokens": entry.get("prompt_tokens", entry.get("prompt_len", entry.get("input_tokens"))),
                "total_tokens": entry.get("total_tokens"),
                "token_timestamps": entry.get("token_timestamps", []),
                "error": entry.get("error", False),
                "error_message": entry.get("error_message"),
            }
            results.append(result)

    return results


def detect_format(path: str | Path) -> str:
    """Auto-detect the format of a benchmark results file.

    Returns one of: 'vllm_json', 'genai_perf_csv', 'jsonl', 'unknown'.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return "genai_perf_csv"

    if suffix == ".jsonl":
        return "jsonl"

    if suffix == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and ("results" in raw or "requests" in raw):
                return "vllm_json"
            if isinstance(raw, list) and raw and "ttft" in raw[0]:
                return "vllm_json"
        except (json.JSONDecodeError, IndexError, KeyError):
            pass
        return "vllm_json"  # best guess for .json

    # Try JSONL
    try:
        first_line = path.read_text(encoding="utf-8").split("\n")[0]
        json.loads(first_line)
        return "jsonl"
    except (json.JSONDecodeError, IndexError):
        pass

    return "unknown"


def auto_import(path: str | Path) -> list[dict[str, Any]]:
    """Auto-detect format and import benchmark results."""
    fmt = detect_format(path)
    if fmt == "vllm_json":
        return import_vllm_benchmark(path)
    if fmt == "genai_perf_csv":
        return import_genai_perf_csv(path)
    if fmt == "jsonl":
        return import_jsonl(path)
    raise ValueError(f"Cannot detect format of {path}. Supported: .json (vLLM), .csv (GenAI-Perf), .jsonl")


def _float_or_none(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _int_or_none(val: Any) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None
