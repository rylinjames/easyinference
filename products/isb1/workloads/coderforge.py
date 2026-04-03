"""CoderForge agent workload generator for ISB-1 benchmarks.

Produces multi-turn agent/tool-use traces from CoderForge-Preview
trajectories (real agent trajectories with search, edit, and test tool
calls).  Each trajectory becomes a session with 4-8 turns of
tool_call → tool_result → reasoning, stressing KV cache reuse across
tool-calling rounds.
"""

from __future__ import annotations

import json
from typing import Any

from workloads.base import Request, WorkloadGenerator, _new_request_id
from workloads.datasets import load_coderforge_preview

# ---------------------------------------------------------------------------
# Tool schemas for the agent system prompt
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the repository for code matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py')",
                    },
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"},
                    "start_line": {"type": "integer", "description": "Start line (1-based)"},
                    "end_line": {"type": "integer", "description": "End line (inclusive)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff patch to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to patch"},
                    "diff": {"type": "string", "description": "Unified diff content"},
                },
                "required": ["path", "diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run test commands and return results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Test command to execute"},
                    "timeout": {"type": "integer", "default": 60},
                },
                "required": ["command"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are a coding agent with access to repository tools. "
    "Use the provided tools to search code, read files, apply patches, "
    "and run tests. Work step by step: understand the issue, locate "
    "relevant code, propose a fix, and verify with tests.\n\n"
    "Available tools:\n"
    + json.dumps([t["function"] for t in _TOOL_SCHEMAS], indent=2)
)


def _extract_coderforge_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool-calling steps from a CoderForge record.

    CoderForge records may have various formats. This function normalizes
    them into a list of step dicts with user/tool_call/tool_result/reasoning.
    """
    steps: list[dict[str, Any]] = []

    # Try 'trajectory' field (list of steps)
    trajectory = record.get("trajectory", [])
    if isinstance(trajectory, str):
        try:
            trajectory = json.loads(trajectory)
        except (json.JSONDecodeError, TypeError):
            trajectory = []

    if isinstance(trajectory, list):
        for i, step in enumerate(trajectory):
            if isinstance(step, dict):
                # Normalize various field names
                user_msg = step.get("input", step.get("user", step.get("query", "")))
                action = step.get("action", step.get("tool_call", ""))
                observation = step.get("observation", step.get("tool_result", step.get("output", "")))
                thought = step.get("thought", step.get("reasoning", step.get("response", "")))

                if user_msg or action:
                    parsed_step: dict[str, Any] = {
                        "user": str(user_msg) if user_msg else f"Continue with step {i + 1}.",
                    }
                    if action:
                        # Try to parse as tool call
                        if isinstance(action, dict):
                            parsed_step["tool_call"] = {
                                "name": action.get("name", action.get("tool", "search_code")),
                                "arguments": json.dumps(
                                    action.get("arguments", action.get("args", {}))
                                ),
                            }
                        else:
                            parsed_step["tool_call"] = {
                                "name": "search_code",
                                "arguments": json.dumps({"query": str(action)}),
                            }
                    if observation:
                        parsed_step["tool_result"] = str(observation)[:2000]
                    if thought:
                        parsed_step["reasoning"] = str(thought)[:1000]
                    steps.append(parsed_step)

    # Fallback: try 'messages' field
    if not steps:
        messages = record.get("messages", [])
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    steps.append({"user": msg.get("content", "")})

    return steps[:8]  # Cap at 8 turns


class CoderForgeAgentGenerator(WorkloadGenerator):
    """Generate agent/tool-use traces from CoderForge-Preview.

    Each trajectory becomes a multi-turn session with tool_call and
    tool_result messages, stressing KV cache reuse across tool-calling
    rounds.

    Args:
        seed: Random seed for reproducibility.
        max_sessions: Maximum number of trajectories to use.
    """

    def __init__(self, seed: int = 42, max_sessions: int = 50) -> None:
        super().__init__(seed=seed)
        self.max_sessions = max_sessions

    def generate(self, num_requests: int) -> list[Request]:
        records = load_coderforge_preview()

        # Parse trajectories
        trajectories: list[dict[str, Any]] = []
        if records:
            for rec in records:
                steps = _extract_coderforge_steps(rec)
                if steps:
                    trajectories.append({
                        "id": rec.get("id", rec.get("instance_id", f"cf-{len(trajectories)}")),
                        "steps": steps,
                    })
                if len(trajectories) >= self.max_sessions:
                    break

        if not trajectories:
            raise RuntimeError(
                "CoderForge-Preview returned no usable trajectories. "
                "Install: pip install 'isb1[datasets]' and ensure network access to HuggingFace Hub."
            )

        self.rng.shuffle(trajectories)

        requests: list[Request] = []
        traj_cycle = 0

        while len(requests) < num_requests:
            traj = trajectories[traj_cycle % len(trajectories)]
            traj_id = traj.get("id", f"traj-{traj_cycle}")
            session_id = f"agent-{traj_id}"
            steps = traj["steps"]

            # Build progressive messages (each turn adds to context)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
            ]

            for turn_idx, step in enumerate(steps):
                if len(requests) >= num_requests:
                    break

                # Add user message
                user_content = step.get("user", f"Continue with step {turn_idx + 1}.")
                messages.append({"role": "user", "content": user_content})

                # If there's a tool call, add assistant tool_call + tool result
                if "tool_call" in step:
                    tc = step["tool_call"]
                    call_id = f"call_{_new_request_id(self.rng)[:12]}"
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tc.get("name", "search_code"),
                                "arguments": tc.get("arguments", "{}"),
                            },
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": step.get("tool_result", "No output."),
                    })

                # Add reasoning as assistant response
                if "reasoning" in step:
                    messages.append({
                        "role": "assistant",
                        "content": step["reasoning"],
                    })

                # Emit a request at each turn (the model should continue)
                requests.append(
                    Request(
                        request_id=_new_request_id(self.rng),
                        messages=list(messages),  # snapshot current context
                        expected_output_tokens=min(300 + turn_idx * 100, 800),
                        session_id=session_id,
                        metadata={
                            "workload": "coderforge",
                            "trajectory_id": traj_id,
                            "turn": turn_idx,
                            "num_tools": len([s for s in steps if "tool_call" in s]),
                        },
                    )
                )

            traj_cycle += 1

        return requests[:num_requests]
