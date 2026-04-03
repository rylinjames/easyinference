"""InferenceX-style serving benchmark runtime for packaged InferScope workloads."""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

import httpx

from inferscope.benchmarks.experiments import BenchmarkExecutionProfile, BenchmarkRunPlan
from inferscope.benchmarks.models import (
    BenchmarkArtifact,
    BenchmarkRequestResult,
    BenchmarkSummary,
    MetricSnapshot,
    WorkloadPack,
    WorkloadRequest,
    slugify,
    utc_now_iso,
)
from inferscope.benchmarks.prometheus_capture import capture_metrics_targets
from inferscope.endpoint_auth import EndpointAuthConfig, build_auth_headers
from inferscope.logging import get_logger, sanitize_log_text
from inferscope.security import validate_endpoint

log = get_logger(component="benchmarks.runtime")


def _request_metadata_int(request: WorkloadRequest, key: str) -> int | None:
    value = request.metadata.get(key)
    return value if isinstance(value, int) else None


@dataclass(slots=True)
class RuntimeRequestResult:
    name: str
    session_id: str | None
    status: Literal["ok", "error"]
    started_at: str
    completed_at: str
    elapsed_ms: float
    ttft_ms: float | None
    status_code: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    error: str = ""
    generated_text: str = ""
    output_event_timestamps_ms: list[float] | None = None

    def to_benchmark_result(self) -> BenchmarkRequestResult:
        return BenchmarkRequestResult(
            name=self.name,
            session_id=self.session_id,
            status=self.status,
            started_at=self.started_at,
            completed_at=self.completed_at,
            elapsed_ms=self.elapsed_ms,
            ttft_ms=self.ttft_ms,
            status_code=self.status_code,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            error=self.error,
        )


@dataclass(slots=True)
class RuntimeExecutionResult:
    artifact: BenchmarkArtifact
    observed_runtime: dict[str, Any]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _rollup(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _extract_usage(payload: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    return (
        _coerce_int(usage.get("prompt_tokens")),
        _coerce_int(usage.get("completion_tokens")),
        _coerce_int(usage.get("total_tokens")),
    )


def _approx_token_count_from_text(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, math.ceil(len(value) / 4))
    if isinstance(value, list):
        return sum(_approx_token_count_from_text(item) for item in value)
    if isinstance(value, dict):
        return sum(_approx_token_count_from_text(item) for item in value.values())
    return max(1, math.ceil(len(str(value)) / 4))


def _approx_prompt_tokens(request: WorkloadRequest) -> int:
    approx = request.metadata.get("approx_context_tokens")
    if isinstance(approx, int) and approx > 0:
        return approx
    total = 0
    for message in request.messages:
        total += _approx_token_count_from_text(message.role)
        total += _approx_token_count_from_text(message.content)
    return max(1, total)


def _extract_output_payload(payload: dict[str, Any]) -> object | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    if isinstance(delta, dict):
        for key in ("content", "tool_calls"):
            value = delta.get(key)
            if value not in (None, "", [], {}):
                return value
    text = first.get("text")
    if text not in (None, ""):
        return text
    message = first.get("message")
    if isinstance(message, dict):
        for key in ("content", "tool_calls"):
            value = message.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _append_output_text(buffer: list[str], value: object) -> None:
    if value in (None, "", [], {}):
        return
    if isinstance(value, str):
        buffer.append(value)
        return
    buffer.append(json.dumps(value, sort_keys=True))


def _flatten_messages_to_prompt(request: WorkloadRequest) -> str:
    lines: list[str] = []
    for message in request.messages:
        content = message.content
        rendered = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
        lines.append(f"{message.role}: {rendered}")
    return "\n\n".join(lines)


def _build_headers(
    request_auth: EndpointAuthConfig | None,
    request: WorkloadRequest,
    extra_headers: dict[str, str] | None,
    *,
    session_header_name: str,
) -> dict[str, str]:
    headers = build_auth_headers(request_auth, include={"Content-Type": "application/json"})
    if request.session_id:
        headers[session_header_name] = request.session_id
    if extra_headers:
        headers.update(extra_headers)
    headers.update(request.headers)
    return headers


def _build_payload(pack: WorkloadPack, request: WorkloadRequest, model: str, backend: str) -> dict[str, Any]:
    if backend == "openai-completions":
        payload: dict[str, Any] = {
            "model": model,
            "prompt": _flatten_messages_to_prompt(request),
            "max_tokens": request.max_tokens,
            "stream": pack.stream,
        }
    elif backend == "trtllm-generate-stream":
        payload = {
            "text_input": _flatten_messages_to_prompt(request),
            "max_tokens": request.max_tokens,
            "stream": True,
            "accumulate_tokens": True,
        }
    else:
        payload = {
            "model": model,
            "messages": [message.model_dump(mode="json", exclude_none=True) for message in request.messages],
            "max_tokens": request.max_tokens,
            "stream": pack.stream,
        }
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    metadata = dict(request.metadata)
    if request.session_id and "session_id" not in metadata:
        metadata["session_id"] = request.session_id
    if metadata and backend != "trtllm-generate-stream":
        payload["metadata"] = metadata
    if pack.stream and backend != "trtllm-generate-stream":
        payload["stream_options"] = {"include_usage": True}
    payload.update(request.extra_body)
    return payload


def _detect_backend(run_plan: BenchmarkRunPlan, workload: WorkloadPack) -> str:
    configured = run_plan.execution.backend
    if configured != "auto":
        return configured
    endpoint_path = workload.endpoint_path.lower()
    if endpoint_path.endswith("/chat/completions"):
        return "openai-chat"
    if endpoint_path.endswith("/completions"):
        return "openai-completions"
    if endpoint_path.endswith("/generate_stream"):
        return "trtllm-generate-stream"
    return "openai-chat"


def _group_requests(
    requests: list[WorkloadRequest], arrival_offsets_ms: list[float]
) -> list[list[tuple[int, WorkloadRequest, float]]]:
    grouped: dict[str, list[tuple[int, WorkloadRequest, float]]] = {}
    order: list[str] = []
    for index, (request, arrival_offset_ms) in enumerate(zip(requests, arrival_offsets_ms, strict=True)):
        key = request.session_id or f"__request_{index}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((index, request, arrival_offset_ms))
    return [grouped[key] for key in order]


def _arrival_offsets_ms(request_count: int, execution: BenchmarkExecutionProfile, seed: int) -> list[float]:
    request_rate = execution.request_rate_rps
    if request_count < 1 or request_rate is None or not math.isfinite(request_rate) or request_rate <= 0:
        return [0.0 for _ in range(request_count)]
    rng = random.Random(seed)  # noqa: S311 - deterministic benchmark scheduling, not security-sensitive
    offsets: list[float] = []
    current = 0.0
    if execution.arrival_model == "gamma":
        shape = execution.arrival_shape or 1.0
        scale = 1.0 / (request_rate * shape)
        for _ in range(request_count):
            offsets.append(current * 1000.0)
            current += rng.gammavariate(shape, scale)
        return offsets
    for _ in range(request_count):
        offsets.append(current * 1000.0)
        current += rng.expovariate(request_rate)
    return offsets


def _threshold_from_value(value: Any, prompt_tokens: int | None) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, dict) or not value:
        return None
    if prompt_tokens is None or prompt_tokens < 1:
        numeric_keys = []
        for key in value:
            try:
                numeric_keys.append(int(str(key).rstrip("kK")) * (1000 if str(key).lower().endswith("k") else 1))
            except ValueError:
                continue
        prompt_tokens = max(numeric_keys) if numeric_keys else 0
    bucket_pairs: list[tuple[int, float]] = []
    for key, threshold in value.items():
        try:
            token_bucket = int(str(key).rstrip("kK")) * (1000 if str(key).lower().endswith("k") else 1)
        except ValueError:
            continue
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            continue
        bucket_pairs.append((token_bucket, float(threshold)))
    if not bucket_pairs:
        return None
    bucket_pairs.sort(key=lambda item: item[0])
    for token_bucket, threshold in bucket_pairs:
        if prompt_tokens <= token_bucket:
            return threshold
    return bucket_pairs[-1][1]


def _request_slo(request: WorkloadRequest, execution: BenchmarkExecutionProfile) -> tuple[float | None, float | None]:
    prompt_tokens = request.metadata.get("approx_context_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens < 1:
        prompt_tokens = _approx_prompt_tokens(request)
    ttft_threshold = _threshold_from_value(execution.goodput_slo.ttft_p95_ms, prompt_tokens)
    tpot_threshold = _threshold_from_value(execution.goodput_slo.tpot_p95_ms, prompt_tokens)
    return ttft_threshold, tpot_threshold


def _compute_tpot_ms(result: RuntimeRequestResult) -> float | None:
    if result.ttft_ms is None or (result.completion_tokens or 0) < 2:
        return None
    decode_ms = result.elapsed_ms - result.ttft_ms
    if decode_ms < 0:
        return None
    return decode_ms / ((result.completion_tokens or 0) - 1)


def _compute_itl_gaps_ms(result: RuntimeRequestResult) -> list[float]:
    timestamps = result.output_event_timestamps_ms or []
    if len(timestamps) < 3:
        return []
    return [timestamps[index] - timestamps[index - 1] for index in range(2, len(timestamps))]


def _requires_output_capture(workload: WorkloadPack) -> bool:
    return workload.name == "tool-agent" or any(
        request.metadata.get("bridge_source") == "mcp_tool_call" for request in workload.requests
    )


def _looks_like_tool_call(text: str) -> bool:
    patterns = [
        r'"name"\s*:',
        r'"arguments"\s*:',
        r'"tool_use"',
        r'"function"\s*:',
        r'"type"\s*:\s*"function"',
        r"<tool_call>",
        r"<function=",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _tool_parse_success_rate(workload: WorkloadPack, results: list[RuntimeRequestResult]) -> float | None:
    is_tool_workload = _requires_output_capture(workload)
    if not is_tool_workload:
        return None
    successful = [result for result in results if result.status == "ok"]
    if not successful:
        return 0.0
    parseable = sum(1 for result in successful if _looks_like_tool_call(result.generated_text))
    return parseable / len(successful)


def _metrics_capture_status(
    before: list[MetricSnapshot],
    after: list[MetricSnapshot],
    enabled: bool,
) -> tuple[int, int, bool]:
    if not enabled:
        return 0, 0, True
    total = len(before) if before else len(after)
    targets_with_errors = {snapshot.target_name for snapshot in before + after if snapshot.error}
    return total, len(targets_with_errors), not targets_with_errors


def _select_primary_snapshot(snapshots: list[MetricSnapshot]) -> MetricSnapshot | None:
    for snapshot in snapshots:
        if snapshot.target_role == "primary":
            return snapshot
    return snapshots[0] if snapshots else None


async def _run_request(
    client: httpx.AsyncClient,
    endpoint: str,
    pack: WorkloadPack,
    model: str,
    request: WorkloadRequest,
    request_auth: EndpointAuthConfig | None,
    extra_headers: dict[str, str] | None,
    *,
    session_header_name: str,
    backend: str,
    request_timeout_seconds: int,
    keep_output: bool,
) -> RuntimeRequestResult:
    if request.think_time_ms:
        await asyncio.sleep(request.think_time_ms / 1000.0)

    url = f"{endpoint}{pack.endpoint_path}"
    headers = _build_headers(request_auth, request, extra_headers, session_header_name=session_header_name)
    payload = _build_payload(pack, request, model, backend)

    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    generated_chunks: list[str] = []
    output_event_timestamps_ms: list[float] = []

    async def do_request() -> tuple[int, float | None, int | None, int | None, int | None, str, list[float]]:
        nonlocal ttft_ms, prompt_tokens, completion_tokens, total_tokens
        if backend == "trtllm-generate-stream":
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    line = raw_line.rstrip("\r")
                    if not line:
                        continue
                    payload_text = line[5:].lstrip() if line.startswith("data:") else line
                    try:
                        event = json.loads(payload_text)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    text_output = event.get("text_output")
                    if text_output not in (None, ""):
                        emitted_at_ms = (time.monotonic() - started_monotonic) * 1000.0
                        output_event_timestamps_ms.append(emitted_at_ms)
                        if ttft_ms is None:
                            ttft_ms = emitted_at_ms
                        _append_output_text(generated_chunks, text_output)
                if completion_tokens is None:
                    completion_tokens = _approx_token_count_from_text(generated_chunks)
                return (
                    response.status_code,
                    ttft_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    "".join(generated_chunks),
                    output_event_timestamps_ms,
                )

        if backend == "openai-completions" and not pack.stream:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            parsed = response.json()
            if isinstance(parsed, dict):
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(parsed)
                _append_output_text(generated_chunks, _extract_output_payload(parsed))
            return (
                response.status_code,
                None,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                "".join(generated_chunks),
                output_event_timestamps_ms,
            )

        if backend == "openai-chat" and not pack.stream:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            parsed = response.json()
            if isinstance(parsed, dict):
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(parsed)
                _append_output_text(generated_chunks, _extract_output_payload(parsed))
            return (
                response.status_code,
                None,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                "".join(generated_chunks),
                output_event_timestamps_ms,
            )

        async with client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            event_lines: list[str] = []
            fallback_lines: list[str] = []
            saw_stream_event = False

            def process_payload(payload_text: str) -> bool:
                nonlocal ttft_ms, prompt_tokens, completion_tokens, total_tokens, saw_stream_event
                if payload_text == "[DONE]":
                    return True
                try:
                    event = json.loads(payload_text)
                except json.JSONDecodeError:
                    return False
                if not isinstance(event, dict):
                    return False
                saw_stream_event = True
                output_payload = _extract_output_payload(event)
                if output_payload not in (None, "", [], {}):
                    emitted_at_ms = (time.monotonic() - started_monotonic) * 1000.0
                    output_event_timestamps_ms.append(emitted_at_ms)
                    if ttft_ms is None:
                        ttft_ms = emitted_at_ms
                    _append_output_text(generated_chunks, output_payload)
                event_prompt, event_completion, event_total = _extract_usage(event)
                if event_prompt is not None:
                    prompt_tokens = event_prompt
                if event_completion is not None:
                    completion_tokens = event_completion
                if event_total is not None:
                    total_tokens = event_total
                return False

            async for raw_line in response.aiter_lines():
                line = raw_line.rstrip("\r")
                if not line:
                    if event_lines:
                        should_stop = process_payload("\n".join(event_lines))
                        event_lines = []
                        if should_stop:
                            break
                    continue
                if line.startswith("data:"):
                    event_lines.append(line[5:].lstrip())
                else:
                    fallback_lines.append(line)
            if event_lines:
                process_payload("\n".join(event_lines))
            if not saw_stream_event and fallback_lines:
                try:
                    fallback_payload = json.loads("\n".join(fallback_lines))
                except json.JSONDecodeError:
                    fallback_payload = {}
                if isinstance(fallback_payload, dict):
                    prompt_tokens, completion_tokens, total_tokens = _extract_usage(fallback_payload)
                    _append_output_text(generated_chunks, _extract_output_payload(fallback_payload))
                    if generated_chunks and ttft_ms is None:
                        ttft_ms = (time.monotonic() - started_monotonic) * 1000.0
                        output_event_timestamps_ms.append(ttft_ms)
            return (
                response.status_code,
                ttft_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                "".join(generated_chunks),
                output_event_timestamps_ms,
            )

    try:
        (
            status_code,
            ttft_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            generated_text,
            output_event_timestamps_ms,
        ) = await asyncio.wait_for(do_request(), timeout=request_timeout_seconds)
        if prompt_tokens is None:
            prompt_tokens = _approx_prompt_tokens(request)
        if completion_tokens is None:
            completion_tokens = _approx_token_count_from_text(generated_text)
        if total_tokens is None and prompt_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        elapsed_ms = (time.monotonic() - started_monotonic) * 1000.0
        return RuntimeRequestResult(
            name=request.name,
            session_id=request.session_id,
            status="ok",
            started_at=started_at,
            completed_at=utc_now_iso(),
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            status_code=status_code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            generated_text=generated_text if keep_output else "",
            output_event_timestamps_ms=output_event_timestamps_ms,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.monotonic() - started_monotonic) * 1000.0
        return RuntimeRequestResult(
            name=request.name,
            session_id=request.session_id,
            status="error",
            started_at=started_at,
            completed_at=utc_now_iso(),
            elapsed_ms=elapsed_ms,
            ttft_ms=None,
            status_code=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            error=sanitize_log_text(str(exc)),
        )


def _build_summary(
    results: list[RuntimeRequestResult],
    concurrency: int,
    wall_time_ms: float,
    *,
    metrics_targets_total: int = 0,
    metrics_targets_with_errors: int = 0,
    metrics_capture_complete: bool = True,
) -> BenchmarkSummary:
    successful_results = [result for result in results if result.status == "ok"]
    latencies = [result.elapsed_ms for result in successful_results]
    ttfts = [result.ttft_ms for result in successful_results if result.ttft_ms is not None]
    prompt_tokens = sum(result.prompt_tokens or 0 for result in successful_results)
    completion_tokens = sum(result.completion_tokens or 0 for result in successful_results)
    total_tokens = sum(result.total_tokens or 0 for result in successful_results)
    succeeded = len(successful_results)
    failed = sum(1 for result in results if result.status == "error")
    return BenchmarkSummary(
        total_requests=len(results),
        succeeded=succeeded,
        failed=failed,
        concurrency=concurrency,
        wall_time_ms=wall_time_ms,
        latency_avg_ms=(sum(latencies) / len(latencies) if latencies else None),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        latency_p99_ms=_percentile(latencies, 0.99),
        ttft_avg_ms=(sum(ttfts) / len(ttfts) if ttfts else None),
        ttft_p90_ms=_percentile(ttfts, 0.90),
        ttft_p95_ms=_percentile(ttfts, 0.95),
        ttft_p99_ms=_percentile(ttfts, 0.99),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        metrics_targets_total=metrics_targets_total,
        metrics_targets_with_errors=metrics_targets_with_errors,
        metrics_capture_complete=metrics_capture_complete,
    )


def _observed_runtime(
    workload: WorkloadPack,
    execution: BenchmarkExecutionProfile,
    results: list[RuntimeRequestResult],
    wall_time_ms: float,
) -> dict[str, Any]:
    succeeded = [result for result in results if result.status == "ok"]
    failed = [result for result in results if result.status == "error"]
    ttfts = [result.ttft_ms for result in succeeded if result.ttft_ms is not None]
    tpots = [value for result in succeeded if (value := _compute_tpot_ms(result)) is not None]
    itls = [gap for result in succeeded for gap in _compute_itl_gaps_ms(result)]
    e2es = [result.elapsed_ms for result in succeeded]
    prompt_tokens = [float(result.prompt_tokens) for result in succeeded if result.prompt_tokens is not None]
    completion_tokens = [
        float(result.completion_tokens) for result in succeeded if result.completion_tokens is not None
    ]
    output_tokens_total = sum(result.completion_tokens or 0 for result in succeeded)
    duration_seconds = wall_time_ms / 1000.0 if wall_time_ms > 0 else 0.0

    good_thresholds = [_request_slo(request, execution) for request in workload.requests]
    good_defined = any(ttft is not None or tpot is not None for ttft, tpot in good_thresholds)
    good_count = 0
    if good_defined:
        for request, result in zip(workload.requests, results, strict=True):
            if result.status != "ok":
                continue
            ttft_threshold, tpot_threshold = _request_slo(request, execution)
            ttft_ok = ttft_threshold is None or (result.ttft_ms is not None and result.ttft_ms <= ttft_threshold)
            tpot_value = _compute_tpot_ms(result)
            tpot_ok = tpot_threshold is None or (tpot_value is not None and tpot_value <= tpot_threshold)
            if ttft_ok and tpot_ok:
                good_count += 1

    return {
        "benchmark_duration_ms": wall_time_ms,
        "request_throughput_rps": (len(succeeded) / duration_seconds if duration_seconds > 0 else 0.0),
        "output_throughput_tps": (output_tokens_total / duration_seconds if duration_seconds > 0 else 0.0),
        "generation_throughput_tps": (output_tokens_total / duration_seconds if duration_seconds > 0 else 0.0),
        "goodput_rps": (good_count / duration_seconds if good_defined and duration_seconds > 0 else None),
        "slo_attainment": (good_count / len(succeeded) if good_defined and succeeded else None),
        "error_rate": (len(failed) / len(results) if results else 0.0),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "ttft_ms": _rollup([float(value) for value in ttfts]),
        "tpot_ms": _rollup([float(value) for value in tpots]),
        "itl_ms": _rollup([float(value) for value in itls]),
        "e2e_ms": _rollup([float(value) for value in e2es]),
        "avg_prompt_tokens": (sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None),
        "avg_completion_tokens": (sum(completion_tokens) / len(completion_tokens) if completion_tokens else None),
        "tool_parse_success_rate": _tool_parse_success_rate(workload, results),
        "scenario_tags": sorted(
            {*workload.tags, *(str(request.metadata.get("bridge_source", "")) for request in workload.requests)} - {""}
        ),
        "timing_granularity": "stream_chunk" if workload.stream else "e2e_only",
        "warnings": [
            "ITL is approximated from streamed output chunks rather than token-level traces."
            if workload.stream
            else "ITL is unavailable for non-streaming runs."
        ],
    }


async def run_benchmark_runtime(
    workload: WorkloadPack,
    endpoint: str,
    *,
    run_plan: BenchmarkRunPlan,
    request_auth: EndpointAuthConfig | None,
    metrics_auth: EndpointAuthConfig | None,
    allow_private: bool,
    capture_metrics: bool,
    extra_headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> RuntimeExecutionResult:
    """Execute a packaged benchmark run with request scheduling and rich serving metrics."""

    validated_endpoint = validate_endpoint(endpoint, allow_private=allow_private)
    benchmark_id = f"{slugify(workload.name)}-{uuid4().hex[:12]}"
    backend = _detect_backend(run_plan, workload)
    run_log = log.bind(
        benchmark_id=benchmark_id,
        endpoint=validated_endpoint,
        workload=workload.name,
        model=run_plan.model,
        backend=backend,
        concurrency=run_plan.concurrency,
        topology_mode=run_plan.topology.mode,
        cache_strategy=run_plan.cache.strategy,
    )

    metrics_before_targets: list[MetricSnapshot] = []
    if capture_metrics:
        metrics_before_targets = await capture_metrics_targets(
            run_plan.metrics_targets,
            allow_private=allow_private,
            metrics_auth=metrics_auth,
        )

    execution = run_plan.execution
    keep_output = execution.capture_outputs or _requires_output_capture(workload)
    created_client = client is None
    active_client = client or httpx.AsyncClient(timeout=execution.request_timeout_seconds + 5)
    started_at = utc_now_iso()
    start_monotonic = time.monotonic()
    seed = next(
        (
            value
            for request in workload.requests
            if (value := _request_metadata_int(request, "synthetic_seed")) is not None
        ),
        42,
    )

    async def run_warmups() -> None:
        if execution.warmup_requests < 1:
            return
        for index in range(execution.warmup_requests):
            request = workload.requests[index % len(workload.requests)]
            result = await _run_request(
                active_client,
                validated_endpoint,
                workload,
                run_plan.model,
                request,
                request_auth,
                extra_headers,
                session_header_name=run_plan.topology.session_header_name,
                backend=backend,
                request_timeout_seconds=execution.request_timeout_seconds,
                keep_output=keep_output,
            )
            if result.status != "ok":
                raise RuntimeError(f"Warmup request '{request.name}' failed: {result.error}")

    arrival_offsets_ms = _arrival_offsets_ms(len(workload.requests), execution, seed)
    request_groups = _group_requests(workload.requests, arrival_offsets_ms)
    semaphore = asyncio.Semaphore(run_plan.concurrency)

    async def run_group(group: list[tuple[int, WorkloadRequest, float]]) -> list[tuple[int, RuntimeRequestResult]]:
        async with semaphore:
            group_results: list[tuple[int, RuntimeRequestResult]] = []
            prior_failed = False
            for index, request, arrival_offset_ms in group:
                if prior_failed:
                    now = utc_now_iso()
                    group_results.append(
                        (
                            index,
                            RuntimeRequestResult(
                                name=request.name,
                                session_id=request.session_id,
                                status="error",
                                started_at=now,
                                completed_at=now,
                                elapsed_ms=0.0,
                                ttft_ms=None,
                                status_code=None,
                                prompt_tokens=None,
                                completion_tokens=None,
                                total_tokens=None,
                                error="Skipped because a prior request in the same session failed",
                            ),
                        )
                    )
                    continue
                target_time = start_monotonic + (arrival_offset_ms / 1000.0)
                now_monotonic = time.monotonic()
                if now_monotonic < target_time:
                    await asyncio.sleep(target_time - now_monotonic)
                result = await _run_request(
                    active_client,
                    validated_endpoint,
                    workload,
                    run_plan.model,
                    request,
                    request_auth,
                    extra_headers,
                    session_header_name=run_plan.topology.session_header_name,
                    backend=backend,
                    request_timeout_seconds=execution.request_timeout_seconds,
                    keep_output=keep_output,
                )
                group_results.append((index, result))
                if result.status != "ok":
                    prior_failed = True
            return group_results

    try:
        await run_warmups()
        grouped_results = await asyncio.wait_for(
            asyncio.gather(*(run_group(group) for group in request_groups)),
            timeout=execution.total_timeout_seconds,
        )
    finally:
        if created_client:
            await active_client.aclose()

    indexed_results = [item for group in grouped_results for item in group]
    indexed_results.sort(key=lambda item: item[0])
    runtime_results = [result for _, result in indexed_results]
    completed_at = utc_now_iso()
    wall_time_ms = (time.monotonic() - start_monotonic) * 1000.0

    metrics_after_targets: list[MetricSnapshot] = []
    if capture_metrics:
        metrics_after_targets = await capture_metrics_targets(
            run_plan.metrics_targets,
            allow_private=allow_private,
            metrics_auth=metrics_auth,
        )

    metrics_targets_total, metrics_targets_with_errors, metrics_capture_complete = _metrics_capture_status(
        metrics_before_targets,
        metrics_after_targets,
        capture_metrics,
    )
    summary = _build_summary(
        runtime_results,
        run_plan.concurrency,
        wall_time_ms,
        metrics_targets_total=metrics_targets_total,
        metrics_targets_with_errors=metrics_targets_with_errors,
        metrics_capture_complete=metrics_capture_complete,
    )
    if summary.succeeded == 0:
        raise RuntimeError("Benchmark produced zero successful requests")

    observed_runtime = _observed_runtime(workload, execution, runtime_results, wall_time_ms)
    run_plan_payload = run_plan.model_dump(mode="json")
    run_plan_payload["observed_runtime"] = observed_runtime

    primary_before = _select_primary_snapshot(metrics_before_targets)
    primary_after = _select_primary_snapshot(metrics_after_targets)
    artifact = BenchmarkArtifact(
        benchmark_id=benchmark_id,
        pack_name=workload.name,
        workload_class=workload.workload_class,
        endpoint=validated_endpoint,
        metrics_endpoint=(
            primary_after.endpoint if primary_after else primary_before.endpoint if primary_before else None
        ),
        model=run_plan.model,
        concurrency=run_plan.concurrency,
        started_at=started_at,
        completed_at=completed_at,
        run_plan=run_plan_payload,
        metrics_before=primary_before,
        metrics_after=primary_after,
        metrics_before_targets=metrics_before_targets,
        metrics_after_targets=metrics_after_targets,
        results=[result.to_benchmark_result() for result in runtime_results],
        summary=summary,
    )
    run_log.info(
        "benchmark_completed",
        succeeded=summary.succeeded,
        failed=summary.failed,
        wall_time_ms=round(summary.wall_time_ms, 2),
        request_throughput_rps=round(float(observed_runtime["request_throughput_rps"]), 2),
        output_throughput_tps=round(float(observed_runtime["output_throughput_tps"]), 2),
        ttft_p95_ms=(
            round(float(observed_runtime["ttft_ms"]["p95"]), 2)
            if observed_runtime["ttft_ms"]["p95"] is not None
            else None
        ),
    )
    return RuntimeExecutionResult(artifact=artifact, observed_runtime=observed_runtime)
