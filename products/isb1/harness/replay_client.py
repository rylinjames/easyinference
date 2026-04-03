"""Internal OpenAI-compatible replay client for ISB-1.

This replaces the previous dependency on ``vllm.benchmarks.benchmark_serving``
for synthetic workload execution. The replay client consumes the benchmark's own
``Request`` traces directly and writes raw JSON in the shape expected by the
existing analysis pipeline.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import logging

import aiohttp

from workloads.arrivals import BurstGPTArrival, GammaArrival, PoissonArrival
from workloads.base import Request

logger = logging.getLogger(__name__)

_REQUEST_ENDPOINT = "/v1/chat/completions"
_SESSION_HEADER = "X-Session-ID"


def _normalize_base_url(url: str) -> str:
    """Strip trailing slash and any trailing /v1 path segment.

    Callers may pass either ``https://host`` or ``https://host/v1``.
    This normalizes both to ``https://host`` so that appending
    ``/v1/...`` paths never produces ``/v1/v1/...``.
    """
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


@dataclass(slots=True)
class ReplayRequestResult:
    request_id: str
    session_id: str | None
    status: str
    timestamp: float
    ttft: float | None
    e2e_latency: float
    output_tokens: int
    prompt_tokens: int | None
    total_tokens: int | None
    token_timestamps: list[float]
    error: bool
    error_message: str | None = None
    ttft_slo_seconds: float | None = None
    tpot_slo_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "ttft": self.ttft,
            "e2e_latency": self.e2e_latency,
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "token_timestamps": self.token_timestamps,
            "error": self.error,
        }
        if self.error_message:
            payload["error_message"] = self.error_message
        if self.ttft_slo_seconds is not None:
            payload["ttft_slo_seconds"] = self.ttft_slo_seconds
        if self.tpot_slo_seconds is not None:
            payload["tpot_slo_seconds"] = self.tpot_slo_seconds
        return payload


@dataclass(slots=True)
class ReplayRunResult:
    completed: int
    failed: int
    duration: float
    request_throughput: float
    output_throughput: float
    generation_throughput: float
    total_input_tokens: int
    total_output_tokens: int
    error_rate: float
    goodput: float
    slo_attainment: float
    per_request: list[ReplayRequestResult]
    runner_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "failed": self.failed,
            "duration": self.duration,
            "request_throughput": self.request_throughput,
            "output_throughput": self.output_throughput,
            "generation_throughput": self.generation_throughput,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "error_rate": self.error_rate,
            "goodput": self.goodput,
            "slo_attainment": self.slo_attainment,
            "per_request": [result.to_dict() for result in self.per_request],
            "runner_metadata": self.runner_metadata,
        }


def _approx_token_count_from_text(text: Any) -> int:
    if text is None:
        return 0
    if isinstance(text, str):
        return max(1, math.ceil(len(text) / 4))
    if isinstance(text, list):
        return sum(_approx_token_count_from_text(item) for item in text)
    if isinstance(text, dict):
        return sum(_approx_token_count_from_text(item) for item in text.values())
    return max(1, math.ceil(len(str(text)) / 4))


def _approx_prompt_tokens(request: Request) -> int:
    approx_from_metadata = request.metadata.get("approx_context_tokens")
    if isinstance(approx_from_metadata, int) and approx_from_metadata > 0:
        return approx_from_metadata
    total = 0
    for message in request.messages:
        total += _approx_token_count_from_text(message.get("role", ""))
        total += _approx_token_count_from_text(message.get("content"))
    return max(1, total)


def _usage_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
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
        _usage_int(usage, "prompt_tokens"),
        _usage_int(usage, "completion_tokens"),
        _usage_int(usage, "total_tokens"),
    )


def _event_has_output(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    delta = first.get("delta")
    if isinstance(delta, dict):
        for key, value in delta.items():
            if key == "role":
                continue
            if value not in (None, "", [], {}):
                return True
    message = first.get("message")
    if isinstance(message, dict):
        return bool(message.get("content")) or bool(message.get("tool_calls"))
    return False


def _extract_output_payload(payload: dict[str, Any]) -> Any:
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
    message = first.get("message")
    if isinstance(message, dict):
        for key in ("content", "tool_calls"):
            value = message.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _bucket_to_tokens(bucket: str) -> int | None:
    normalized = bucket.strip().lower().replace("_", "")
    if normalized.endswith("k"):
        base = normalized[:-1]
        if base.isdigit():
            return int(base) * 1000
    if normalized.isdigit():
        return int(normalized)
    return None


def _threshold_from_value(value: Any, observed_prompt_tokens: int | None) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) / 1000.0
    if isinstance(value, dict) and value:
        if observed_prompt_tokens is None or observed_prompt_tokens < 1:
            observed_prompt_tokens = max(
                (_bucket_to_tokens(str(key)) or 0) for key in value
            )
        bucket_pairs = []
        for bucket, threshold in value.items():
            bucket_tokens = _bucket_to_tokens(str(bucket))
            if bucket_tokens is None:
                continue
            if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
                continue
            bucket_pairs.append((bucket_tokens, float(threshold) / 1000.0))
        if not bucket_pairs:
            return None
        bucket_pairs.sort(key=lambda item: item[0])
        for bucket_tokens, threshold_seconds in bucket_pairs:
            if observed_prompt_tokens <= bucket_tokens:
                return threshold_seconds
        return bucket_pairs[-1][1]
    return None


def _resolve_request_slo(
    request: Request,
    *,
    prompt_tokens: int | None,
    goodput_slo: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    if not goodput_slo:
        return None, None
    observed_prompt_tokens = request.metadata.get("approx_context_tokens")
    if not isinstance(observed_prompt_tokens, int) or observed_prompt_tokens < 1:
        observed_prompt_tokens = prompt_tokens or _approx_prompt_tokens(request)
    ttft_threshold = _threshold_from_value(goodput_slo.get("ttft_p95_ms"), observed_prompt_tokens)
    tpot_threshold = _threshold_from_value(goodput_slo.get("tpot_p95_ms"), observed_prompt_tokens)
    return ttft_threshold, tpot_threshold


def _compute_tpot(result: ReplayRequestResult) -> float | None:
    if result.ttft is None or result.output_tokens < 2:
        return None
    decode_time = result.e2e_latency - result.ttft
    if decode_time < 0:
        return None
    return decode_time / (result.output_tokens - 1)


def _count_good_requests(results: list[ReplayRequestResult]) -> int:
    good = 0
    for result in results:
        if result.error:
            continue
        ttft_threshold = result.ttft_slo_seconds
        tpot_threshold = result.tpot_slo_seconds
        ttft_ok = ttft_threshold is None or (
            result.ttft is not None and result.ttft <= ttft_threshold
        )
        tpot_value = _compute_tpot(result)
        tpot_ok = tpot_threshold is None or (
            tpot_value is not None and tpot_value <= tpot_threshold
        )
        if ttft_ok and tpot_ok:
            good += 1
    return good


def _group_requests(
    requests: list[Request],
    arrival_offsets: list[float],
) -> list[list[tuple[int, Request, float]]]:
    grouped: dict[str, list[tuple[int, Request, float]]] = {}
    ordered_keys: list[str] = []
    for index, (request, arrival_offset) in enumerate(zip(requests, arrival_offsets, strict=True)):
        key = request.session_id or f"__request_{index}"
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append((index, request, arrival_offset))
    return [grouped[key] for key in ordered_keys]


def _clone_request(request: Request, cycle_index: int, ordinal: int) -> Request:
    session_id = request.session_id
    if session_id is not None:
        session_id = f"{session_id}::cycle-{cycle_index:04d}"
    metadata = dict(request.metadata)
    metadata["replay_cycle"] = cycle_index
    metadata["replay_ordinal"] = ordinal
    return Request(
        request_id=f"{request.request_id}-r{ordinal:06d}",
        messages=list(request.messages),
        expected_output_tokens=request.expected_output_tokens,
        session_id=session_id,
        metadata=metadata,
    )


def expand_request_pool(requests: list[Request], total_requests: int) -> list[Request]:
    """Expand a request pool to the target request count by cycling deterministically."""
    if total_requests < 1:
        raise ValueError("total_requests must be >= 1")
    if not requests:
        raise ValueError("request pool must not be empty")
    expanded: list[Request] = []
    cycle_index = 0
    ordinal = 0
    while len(expanded) < total_requests:
        for request in requests:
            expanded.append(_clone_request(request, cycle_index, ordinal))
            ordinal += 1
            if len(expanded) >= total_requests:
                break
        cycle_index += 1
    return expanded


def _arrival_offsets(
    num_requests: int,
    request_rate: float,
    arrival_model: str,
    arrival_shape: float | None,
    seed: int,
) -> list[float]:
    if not math.isfinite(request_rate) or request_rate <= 0:
        return [0.0 for _ in range(num_requests)]
    normalized = arrival_model.strip().lower()
    if normalized == "gamma":
        generator = GammaArrival(rate=request_rate, shape=arrival_shape or 0.5, seed=seed)
    elif normalized == "burstgpt":
        generator = BurstGPTArrival(rate=request_rate, seed=seed)
    else:
        generator = PoissonArrival(rate=request_rate, seed=seed)
    return generator.generate(num_requests).tolist()


async def _run_stream_request(
    session: aiohttp.ClientSession,
    url: str,
    request: Request,
    model: str,
    *,
    session_header_name: str,
    request_timeout_seconds: int,
    goodput_slo: dict[str, Any] | None,
    extra_headers: dict[str, str] | None = None,
) -> ReplayRequestResult:
    started_at_epoch = time.time()
    started_monotonic = time.monotonic()
    token_timestamps: list[float] = []
    first_output_timestamp: float | None = None
    completion_payloads: list[Any] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if request.session_id:
        headers[session_header_name] = request.session_id
    payload: dict[str, Any] = {
        "model": model,
        "messages": request.messages,
        "max_tokens": request.expected_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)

    try:
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
            response.raise_for_status()

            event_lines: list[str] = []
            fallback_lines: list[str] = []
            saw_stream_event = False

            def process_event(payload_text: str) -> bool:
                nonlocal first_output_timestamp, prompt_tokens, completion_tokens, total_tokens, saw_stream_event
                if payload_text == "[DONE]":
                    return True
                try:
                    event = json.loads(payload_text)
                except json.JSONDecodeError:
                    return False
                if isinstance(event, dict):
                    saw_stream_event = True
                    output_payload = _extract_output_payload(event)
                    if output_payload not in (None, "", [], {}):
                        emitted_at = time.monotonic() - started_monotonic
                        if first_output_timestamp is None:
                            first_output_timestamp = emitted_at
                        token_timestamps.append(emitted_at)
                        completion_payloads.append(output_payload)
                    event_prompt, event_completion, event_total = _extract_usage(event)
                    if event_prompt is not None:
                        prompt_tokens = event_prompt
                    if event_completion is not None:
                        completion_tokens = event_completion
                    if event_total is not None:
                        total_tokens = event_total
                return False

            while True:
                raw_line = await response.content.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
                if not line:
                    if event_lines:
                        should_stop = process_event("\n".join(event_lines))
                        event_lines = []
                        if should_stop:
                            break
                    continue
                if line.startswith("data:"):
                    event_lines.append(line[5:].lstrip())
                else:
                    fallback_lines.append(line)

            if event_lines:
                process_event("\n".join(event_lines))

            if not saw_stream_event and fallback_lines:
                try:
                    parsed = json.loads("\n".join(fallback_lines))
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, dict):
                    prompt_tokens, completion_tokens, total_tokens = _extract_usage(parsed)
                    output_payload = _extract_output_payload(parsed)
                    if output_payload not in (None, "", [], {}):
                        emitted_at = time.monotonic() - started_monotonic
                        if first_output_timestamp is None:
                            first_output_timestamp = emitted_at
                        token_timestamps.append(emitted_at)
                        completion_payloads.append(output_payload)

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Request %s failed: %s: %s",
            request.request_id,
            type(exc).__name__,
            exc,
        )
        return ReplayRequestResult(
            request_id=request.request_id,
            session_id=request.session_id,
            status="error",
            timestamp=started_at_epoch,
            ttft=None,
            e2e_latency=time.monotonic() - started_monotonic,
            output_tokens=0,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            token_timestamps=[],
            error=True,
            error_message=str(exc),
        )

    if prompt_tokens is None:
        prompt_tokens = _approx_prompt_tokens(request)
    if completion_tokens is None:
        completion_tokens = _approx_token_count_from_text(completion_payloads) if completion_payloads else 0
    if total_tokens is None and prompt_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    ttft_seconds = first_output_timestamp
    ttft_threshold, tpot_threshold = _resolve_request_slo(
        request,
        prompt_tokens=prompt_tokens,
        goodput_slo=goodput_slo,
    )
    return ReplayRequestResult(
        request_id=request.request_id,
        session_id=request.session_id,
        status="ok",
        timestamp=started_at_epoch,
        ttft=ttft_seconds,
        e2e_latency=time.monotonic() - started_monotonic,
        output_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        token_timestamps=token_timestamps,
        error=False,
        ttft_slo_seconds=ttft_threshold,
        tpot_slo_seconds=tpot_threshold,
    )


async def run_rate(
    *,
    base_url: str,
    model: str,
    request_pool: list[Request],
    request_count: int,
    request_rate: float,
    arrival_model: str,
    arrival_shape: float | None,
    seed: int,
    concurrency: int | None = None,
    session_header_name: str = _SESSION_HEADER,
    request_timeout_seconds: int = 600,
    total_timeout_seconds: int = 7200,
    goodput_slo: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> ReplayRunResult:
    """Replay one rate point against an OpenAI-compatible endpoint."""
    requests = expand_request_pool(request_pool, request_count)
    arrival_offsets = _arrival_offsets(
        len(requests),
        request_rate,
        arrival_model,
        arrival_shape,
        seed,
    )
    grouped = _group_requests(requests, arrival_offsets)
    start_monotonic = time.monotonic()
    endpoint = f"{_normalize_base_url(base_url)}{_REQUEST_ENDPOINT}"
    max_concurrency = concurrency if concurrency is not None else max(1, len(grouped))
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def run_group(group: list[tuple[int, Request, float]]) -> list[tuple[int, ReplayRequestResult]]:
            async with semaphore:
                group_results: list[tuple[int, ReplayRequestResult]] = []
                for index, request, arrival_offset in group:
                    target_time = start_monotonic + arrival_offset
                    now = time.monotonic()
                    if now < target_time:
                        await asyncio.sleep(target_time - now)
                    result = await _run_stream_request(
                        session,
                        endpoint,
                        request,
                        model,
                        session_header_name=session_header_name,
                        request_timeout_seconds=request_timeout_seconds,
                        goodput_slo=goodput_slo,
                        extra_headers=extra_headers,
                    )
                    group_results.append((index, result))
                return group_results

        grouped_results = await asyncio.wait_for(
            asyncio.gather(*(run_group(group) for group in grouped)),
            timeout=total_timeout_seconds,
        )

    indexed_results = [item for group_result in grouped_results for item in group_result]
    indexed_results.sort(key=lambda item: item[0])
    results = [result for _, result in indexed_results]

    duration = max(0.0, time.monotonic() - start_monotonic)
    completed = sum(1 for result in results if not result.error)
    failed = len(results) - completed
    total_output_tokens = sum(result.output_tokens for result in results if not result.error)
    total_input_tokens = sum((result.prompt_tokens or 0) for result in results if not result.error)
    request_throughput = completed / duration if duration > 0 else 0.0
    output_throughput = total_output_tokens / duration if duration > 0 else 0.0
    good_count = _count_good_requests(results)
    goodput = good_count / duration if duration > 0 else 0.0
    slo_attainment = good_count / completed if completed > 0 else 0.0

    return ReplayRunResult(
        completed=completed,
        failed=failed,
        duration=duration,
        request_throughput=request_throughput,
        output_throughput=output_throughput,
        generation_throughput=output_throughput,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        error_rate=failed / len(results) if results else 0.0,
        goodput=goodput,
        slo_attainment=slo_attainment,
        per_request=results,
        runner_metadata={
            "runner": "isb1_openai_replay",
            "request_rate": request_rate,
            "arrival_model": arrival_model,
            "arrival_shape": arrival_shape,
            "request_pool_size": len(request_pool),
            "expanded_request_count": len(requests),
            "session_header_name": session_header_name,
        },
    )
