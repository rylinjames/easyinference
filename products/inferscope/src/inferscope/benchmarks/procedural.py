"""Procedural workload materialization for packaged InferScope benchmarks."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from inferscope.benchmarks.models import ChatMessage, WorkloadPack, WorkloadRequest

SUPPORTED_PROCEDURAL_WORKLOADS = {"tool-agent", "coding-long-context", "kimi-k2-long-context-coding"}

_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string", "default": "utf-8"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a pattern across files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "directory": {"type": "string"},
                    "file_glob": {"type": "string", "default": "*.py"},
                    "max_results": {"type": "integer", "default": 20},
                },
                "required": ["pattern", "directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "default": "."},
                    "timeout_seconds": {"type": "integer", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_inference_config",
            "description": "Analyze an inference serving configuration and suggest optimizations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "framework": {"type": "string", "enum": ["sglang", "vllm", "trt"]},
                    "model_name": {"type": "string"},
                    "tp_size": {"type": "integer"},
                    "max_batch_size": {"type": "integer"},
                    "kv_cache_dtype": {"type": "string"},
                },
                "required": ["framework", "model_name", "tp_size"],
            },
        },
    },
]

_TOOL_AGENT_PROMPTS = [
    "Read the file at /workspace/src/kernel.cu and identify the first bottleneck.",
    "Search for all uses of 'flash_attention' in /workspace/src/ and return the best starting point.",
    "Run 'nvidia-smi --query-gpu=memory.used --format=csv' and summarize the memory picture.",
    "Analyze this vllm config: model=DeepSeek-R1, tp=8, batch=128, kv_cache=fp8.",
    "Search for 'kv_cache' in /workspace/ with max_results=50 and tell me the next patch step.",
]

_TOOL_AGENT_RESULTS = [
    '[{"file": "src/server.py", "line": 118, "match": "enable_flash_attention = True"}]',
    '{"status": "ok", "stdout": "GPU 0 memory used: 61 GiB"}',
    '{"framework": "vllm", "issue": "prefill saturation", "suggestion": "lower max_num_batched_tokens"}',
]

_TOOL_AGENT_CONTEXT_BLOCKS = [
    (
        "# Context\n"
        "The service runs a latency-sensitive coding agent on a long-lived "
        "repository session with repeated tool schemas."
    ),
    (
        "# Context\n"
        "The deployment mixes benchmark replay, prefix caching, and "
        "disaggregated prefill experiments across multiple endpoints."
    ),
    (
        "# Context\n"
        "The operator wants tool calls to stay parseable while maintaining "
        "low TTFT under concurrent sessions."
    ),
]

_CODING_CONTEXT_BLOCKS = [
    """# kernel_launcher.py
import torch

def launch_decode_kernel(hidden_states, kv_cache, block_size=128):
    if hidden_states.numel() == 0:
        raise ValueError('empty hidden state tensor')
    return torch.matmul(hidden_states, kv_cache[: hidden_states.shape[-1]])
""",
    """# server_config.yaml
parallelism:
  tensor_parallel_size: 8
memory:
  kv_cache_dtype: fp8_e4m3
  mem_fraction_static: 0.85
optimization:
  chunked_prefill_size: 32768
  enable_prefix_caching: true
""",
    """# kv_cache.py
class BlockAllocator:
    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks
        self.free_blocks = list(range(total_blocks))
        self.allocated = {}

    def utilization(self) -> float:
        return 1.0 - (len(self.free_blocks) / self.total_blocks)
""",
]

_CODING_TASKS = [
    "Implement an auto_tune_kernel_config helper that chooses better decode block sizes.",
    "Refactor the KV cache allocator to track hot blocks separately from cold blocks.",
    "Add a benchmark script that compares chunked prefill sizes and reports TTFT deltas.",
    "Write a patch plan for reducing prefix-cache misses in long-context coding sessions.",
]


class ProceduralWorkloadOptions(BaseModel):
    """Additive synthetic generation controls for packaged workloads."""

    model_config = ConfigDict(extra="forbid")

    request_count: int | None = Field(default=None, ge=1, le=4096)
    input_tokens: int | None = Field(default=None, ge=64, le=262_144)
    output_tokens: int | None = Field(default=None, ge=32, le=32_768)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    context_file: str | None = None

    @property
    def enabled(self) -> bool:
        return any(
            value is not None
            for value in (self.request_count, self.input_tokens, self.output_tokens, self.context_file)
        )


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _shape_context(blocks: list[str], target_tokens: int | None) -> str:
    if not blocks:
        return ""
    if target_tokens is None:
        return "\n\n".join(blocks)
    target = max(1, target_tokens)
    parts: list[str] = []
    total = 0
    index = 0
    while total < target:
        block = blocks[index % len(blocks)]
        parts.append(block)
        total += _approx_tokens(block)
        index += 1
    return "\n\n".join(parts)


def _load_context_file(path: str) -> str:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"Context file does not exist: {path}")
    return file_path.read_text(encoding="utf-8")


def _with_tags(seed_pack: WorkloadPack, *tags: str) -> list[str]:
    return sorted({*seed_pack.tags, *tags})


def _effective_request_count(seed_pack: WorkloadPack, options: ProceduralWorkloadOptions) -> int:
    return options.request_count or len(seed_pack.requests)


def _effective_output_tokens(seed_pack: WorkloadPack, options: ProceduralWorkloadOptions, *, default: int) -> int:
    if options.output_tokens is not None:
        return options.output_tokens
    if seed_pack.requests:
        return seed_pack.requests[0].max_tokens
    return default


def _materialize_tool_agent(
    seed_pack: WorkloadPack,
    options: ProceduralWorkloadOptions,
) -> WorkloadPack:
    if options.context_file:
        raise ValueError("context_file is supported only for coding-long-context")

    rng = random.Random(options.seed)  # noqa: S311 - deterministic benchmark materialization
    request_count = _effective_request_count(seed_pack, options)
    output_tokens = _effective_output_tokens(seed_pack, options, default=256)
    context_blocks = list(_TOOL_AGENT_CONTEXT_BLOCKS)
    rng.shuffle(context_blocks)
    context_text = _shape_context(context_blocks, options.input_tokens)
    prompt_offset = rng.randrange(len(_TOOL_AGENT_PROMPTS))
    result_offset = rng.randrange(len(_TOOL_AGENT_RESULTS))
    requests: list[WorkloadRequest] = []

    for session_index in range(math.ceil(request_count / 2)):
        prompt = _TOOL_AGENT_PROMPTS[(prompt_offset + session_index) % len(_TOOL_AGENT_PROMPTS)]
        tool_result = _TOOL_AGENT_RESULTS[(result_offset + session_index) % len(_TOOL_AGENT_RESULTS)]
        session_id = f"tool-agent-session-{session_index + 1:04d}"
        system_text = (
            "You are an MCP-oriented coding agent. Call tools when needed, keep reasoning concise, "
            "and prefer actionable next steps.\n\n"
            f"{context_text}"
        ).strip()

        requests.append(
            WorkloadRequest(
                name=f"planner-turn-{session_index + 1:04d}",
                session_id=session_id,
                max_tokens=output_tokens,
                messages=[
                    ChatMessage(role="system", content=system_text),
                    ChatMessage(role="user", content=prompt),
                ],
                tools=_TOOL_DEFINITIONS,
                tool_choice="auto",
                metadata={
                    "bridge_source": "mcp_tool_call",
                    "approx_context_tokens": max(options.input_tokens or 0, _approx_tokens(context_text)),
                    "synthetic_seed": options.seed,
                    "synthetic_index": len(requests),
                },
            )
        )
        if len(requests) >= request_count:
            break
        requests.append(
            WorkloadRequest(
                name=f"executor-turn-{session_index + 1:04d}",
                session_id=session_id,
                max_tokens=output_tokens,
                messages=[
                    ChatMessage(role="system", content=system_text),
                    ChatMessage(role="user", content=prompt),
                    ChatMessage(
                        role="assistant",
                        content=(
                            "I previously called search_code for 'kv_cache' under "
                            "/workspace. Here is the tool result summary to continue "
                            f"from:\n\n{tool_result}"
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=("Using that prior tool result, decide the next patch or validation step."),
                    ),
                ],
                tools=_TOOL_DEFINITIONS,
                tool_choice="auto",
                metadata={
                    "bridge_source": "mcp_tool_call",
                    "approx_context_tokens": max(options.input_tokens or 0, _approx_tokens(context_text)),
                    "synthetic_seed": options.seed,
                    "synthetic_index": len(requests),
                },
            )
        )
        if len(requests) >= request_count:
            break

    return WorkloadPack(
        version=seed_pack.version,
        name=seed_pack.name,
        description=(seed_pack.description + " Procedurally materialized from the MCP/tool-call benchmark bridge."),
        workload_class=seed_pack.workload_class,
        model=seed_pack.model,
        endpoint_path=seed_pack.endpoint_path,
        concurrency=seed_pack.concurrency,
        stream=seed_pack.stream,
        tags=_with_tags(seed_pack, "procedural", "mcp-bridge"),
        requests=requests[:request_count],
    )


def _materialize_coding_long_context(
    seed_pack: WorkloadPack,
    options: ProceduralWorkloadOptions,
) -> WorkloadPack:
    rng = random.Random(options.seed)  # noqa: S311 - deterministic benchmark materialization
    request_count = _effective_request_count(seed_pack, options)
    output_tokens = _effective_output_tokens(seed_pack, options, default=512)
    context_blocks = list(_CODING_CONTEXT_BLOCKS)
    rng.shuffle(context_blocks)
    context_text = (
        _load_context_file(options.context_file)
        if options.context_file
        else _shape_context(context_blocks, options.input_tokens)
    )
    if options.input_tokens is not None and options.context_file:
        context_text = _shape_context([context_text], options.input_tokens)

    task_order = list(_CODING_TASKS)
    rng.shuffle(task_order)
    approx_context_tokens = max(options.input_tokens or 0, _approx_tokens(context_text))

    requests: list[WorkloadRequest] = []
    for session_index in range(math.ceil(request_count / 2)):
        task = task_order[session_index % len(task_order)]
        session_id = f"coding-session-{session_index + 1:04d}"
        system_text = (
            "You are an expert inference optimization and coding assistant. "
            "Use the repository context below and prefer precise diffs over general advice.\n\n"
            "# === CODEBASE START ===\n"
            f"{context_text}\n"
            "# === CODEBASE END ==="
        )
        first_user = f"Review the codebase above and identify the first implementation change for this task: {task}"
        second_user = "Given the same repository context, produce the minimal patch plan and validation sequence."

        requests.append(
            WorkloadRequest(
                name=f"repo-review-{session_index + 1:04d}",
                session_id=session_id,
                max_tokens=output_tokens,
                messages=[
                    ChatMessage(role="system", content=system_text),
                    ChatMessage(role="user", content=first_user),
                ],
                metadata={
                    "bridge_source": "long_context_code",
                    "approx_context_tokens": approx_context_tokens,
                    "synthetic_seed": options.seed,
                    "synthetic_index": len(requests),
                },
            )
        )
        if len(requests) >= request_count:
            break
        requests.append(
            WorkloadRequest(
                name=f"patch-plan-{session_index + 1:04d}",
                session_id=session_id,
                max_tokens=output_tokens,
                messages=[
                    ChatMessage(role="system", content=system_text),
                    ChatMessage(role="user", content=first_user),
                    ChatMessage(
                        role="assistant",
                        content=(
                            "The main bottleneck is prefill pressure and cache churn across the shared repo context."
                        ),
                    ),
                    ChatMessage(role="user", content=second_user),
                ],
                metadata={
                    "bridge_source": "long_context_code",
                    "approx_context_tokens": approx_context_tokens,
                    "synthetic_seed": options.seed,
                    "synthetic_index": len(requests),
                },
            )
        )
        if len(requests) >= request_count:
            break

    return WorkloadPack(
        version=seed_pack.version,
        name=seed_pack.name,
        description=(
            seed_pack.description + " Procedurally materialized from the long-context coding benchmark bridge."
        ),
        workload_class=seed_pack.workload_class,
        model=seed_pack.model,
        endpoint_path=seed_pack.endpoint_path,
        concurrency=seed_pack.concurrency,
        stream=seed_pack.stream,
        tags=_with_tags(seed_pack, "procedural", "long-context-bridge"),
        requests=requests[:request_count],
    )


def materialize_procedural_workload(
    seed_pack: WorkloadPack,
    options: ProceduralWorkloadOptions,
) -> WorkloadPack:
    """Expand a supported packaged workload into a procedurally generated pack."""
    if seed_pack.name not in SUPPORTED_PROCEDURAL_WORKLOADS:
        raise ValueError(f"Procedural generation is supported only for {sorted(SUPPORTED_PROCEDURAL_WORKLOADS)}")
    if seed_pack.name == "tool-agent":
        return _materialize_tool_agent(seed_pack, options)
    return _materialize_coding_long_context(seed_pack, options)
