"""Deep Research Agent trace generator for ISB-1 benchmarks.

Produces long-running multi-turn research agent conversations modeled after
MiroThinker trace characteristics (see inferscope/docs/mirothinker_v1.7).

Key differences from the generic agent workload:
- Sessions are 100-300 turns (vs 3-8 for generic agent)
- Context plateaus at 50-70K tokens due to a rolling window of tool results
- Outputs are short (200-500 tokens) — action selection, not long-form generation
- High prefix cache hit rate from stable system prompt + rolling-window structure
- Heavy web tool use (search + scrape dominate)
- Occasional session restart with summary (retry_with_summary pattern)

These parameters are derived from empirical MiroThinker trace analysis:
- Per-turn input_tokens plateaus at ~60K after the first ~5 turns
- Output tokens are consistently short (~200-500 tokens per turn)
- High prefix cache hit rate due to fixed system prompt and rolling window
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workloads.base import Request, WorkloadGenerator, _new_request_id

# ---------------------------------------------------------------------------
# Constants derived from MiroThinker trace analysis
# ---------------------------------------------------------------------------

_ROLLING_WINDOW_SIZE = 5  # Keep last N tool results in context
_CONTEXT_PLATEAU_TOKENS = 60000  # Steady-state context size
_RAMP_TURNS = 5  # Turns to reach plateau

# ---------------------------------------------------------------------------
# Deep research system prompt (longer than generic agent)
# ---------------------------------------------------------------------------

_DEEP_RESEARCH_SYSTEM_PROMPT = (
    "You are a deep research agent tasked with thoroughly investigating complex "
    "questions. You have access to web search, web scraping, and code execution tools. "
    "Your workflow:\n"
    "1. Break the research question into sub-questions\n"
    "2. Search for relevant information using web search\n"
    "3. Scrape promising URLs for detailed content\n"
    "4. Synthesize findings, cross-referencing multiple sources\n"
    "5. If you find contradictory information, investigate further\n"
    "6. Provide your final answer in \\boxed{} format\n\n"
    "IMPORTANT: Be thorough. Use multiple search queries with different phrasings. "
    "Verify claims by cross-referencing at least 2 independent sources. "
    "If a search doesn't return useful results, reformulate and try again.\n\n"
    "You maintain a rolling context window of the last 5 tool results to manage "
    "context length. Earlier results are summarized.\n\n"
    "Available tools:\n"
    "- search(query: str, max_results: int): Search the web\n"
    "- scrape(url: str): Scrape a web page for content\n"
    "- code_execute(language: str, code: str): Execute code in sandbox\n"
)

# ---------------------------------------------------------------------------
# Research query templates (more complex than generic agent)
# ---------------------------------------------------------------------------

_RESEARCH_QUERIES = [
    "Research the current state of {topic}. I need a comprehensive analysis covering "
    "recent developments, key players, and future outlook.",
    "Investigate: {question}. Provide evidence from at least 3 independent sources.",
    "Deep dive into {topic}: what are the main technical challenges and how are "
    "leading organizations solving them?",
    "Compare and contrast the approaches to {topic} taken by at least 3 different "
    "organizations or research groups.",
    "What is the current scientific consensus on {question}? Are there any "
    "significant dissenting views?",
]

_RESEARCH_TOPICS = [
    "disaggregated inference serving architectures",
    "KV cache compression techniques for long-context LLMs",
    "SLO-aware scheduling for inference workloads",
    "prefix caching strategies across multi-GPU fleets",
    "FP8 vs FP4 quantization tradeoffs on Blackwell GPUs",
    "NVIDIA Dynamo vs vLLM performance characteristics",
    "RDMA-based KV cache transfer in production deployments",
    "multi-tier model weight caching for cold start elimination",
    "agentic workload scheduling for long-running sessions",
    "continuous batching vs chunked prefill performance",
    "MoE expert parallelism at scale",
    "inference cost optimization for coding agents",
]

_RESEARCH_QUESTIONS = [
    "how does cache-aware routing reduce prefill compute waste",
    "what is the optimal batch size vs ITL tradeoff for coding workloads",
    "when does prefill/decode disaggregation help vs hurt performance",
    "how do Grove-style tiered KV caches compare to single-tier approaches",
    "what SLO guarantees are feasible for sub-100ms TTFT",
    "how much compute waste comes from KV cache preemptions in production",
]

# ---------------------------------------------------------------------------
# Tool action templates (research-specific)
# ---------------------------------------------------------------------------

_SEARCH_ACTIONS = [
    "Let me search for more information about this aspect.",
    "I'll search with a different query to find additional sources.",
    "Searching for counter-arguments and alternative perspectives.",
    "Looking for recent papers or blog posts on this topic.",
    "Searching for benchmarks and empirical data.",
]

_SCRAPE_ACTIONS = [
    "This URL looks promising, let me scrape it for details.",
    "I'll scrape the official documentation for accurate information.",
    "Let me get the full content from this research paper.",
    "Scraping this blog post for the technical details.",
]

_SYNTHESIS_ACTIONS = [
    "Let me synthesize what I've found so far.",
    "Cross-referencing the information from multiple sources.",
    "I'll organize the findings into a structured analysis.",
    "Summarizing the key takeaways from my research.",
    "Let me verify the consistency of these findings.",
]

# ---------------------------------------------------------------------------
# Synthetic tool results (larger than generic agent — web scrape results)
# ---------------------------------------------------------------------------

_SEARCH_RESULT_TEMPLATES = [
    'Found {n} results for "{query}":\n'
    "1. [{title1}]({url1}) — {snippet1}\n"
    "2. [{title2}]({url2}) — {snippet2}\n"
    "3. [{title3}]({url3}) — {snippet3}\n"
    "4. [{title4}]({url4}) — {snippet4}\n"
    "5. [{title5}]({url5}) — {snippet5}",
]

_SCRAPE_RESULT_TEMPLATE = (
    "# {title}\n\n"
    "{intro_paragraph}\n\n"
    "## Key Findings\n\n"
    "{findings}\n\n"
    "## Technical Details\n\n"
    "{technical_details}\n\n"
    "## Conclusion\n\n"
    "{conclusion}"
)

_PARAGRAPHS = [
    "Recent advances in inference serving have focused on reducing latency while "
    "maintaining throughput. The key challenge is balancing batch size against "
    "per-request latency, particularly for interactive workloads like code completion.",
    "Production deployments at scale have demonstrated that KV cache management "
    "is the primary bottleneck for long-context workloads. Effective tiering "
    "strategies can reduce GPU memory pressure by 40-60%.",
    "Disaggregated serving architectures separate prefill and decode phases onto "
    "different GPU pools. This allows independent scaling but introduces KV cache "
    "transfer overhead that must be carefully managed.",
    "Benchmarking methodology significantly impacts reported performance numbers. "
    "Standardized workloads with realistic context distributions are essential "
    "for meaningful comparisons across serving engines.",
    "The economics of inference are dominated by GPU utilization efficiency. "
    "Goodput (useful tokens per second, excluding waste) is a better metric "
    "than raw throughput for capacity planning.",
]


class DeepResearchAgentGenerator(WorkloadGenerator):
    """Generate long-running deep research agent traces.

    Models the MiroThinker deep research workflow: 100-300 turn sessions with
    heavy tool use, rolling context windows, and stable steady-state context.

    Parameters:
        seed: Random seed for reproducibility.
        min_turns: Minimum turns per session (default 100).
        max_turns: Maximum turns per session (default 300).
        rolling_window_size: Number of recent tool results kept in context (default 5).
        retry_with_summary_prob: Probability of session restart with summary (default 0.05).
    """

    def __init__(
        self,
        seed: int = 42,
        min_turns: int = 100,
        max_turns: int = 300,
        rolling_window_size: int = _ROLLING_WINDOW_SIZE,
        retry_with_summary_prob: float = 0.05,
    ) -> None:
        super().__init__(seed=seed)
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.rolling_window_size = rolling_window_size
        self.retry_with_summary_prob = retry_with_summary_prob

    def _pick_research_query(self) -> str:
        template = _RESEARCH_QUERIES[int(self.rng.integers(0, len(_RESEARCH_QUERIES)))]
        return template.format(
            topic=_RESEARCH_TOPICS[int(self.rng.integers(0, len(_RESEARCH_TOPICS)))],
            question=_RESEARCH_QUESTIONS[int(self.rng.integers(0, len(_RESEARCH_QUESTIONS)))],
        )

    def _make_search_result(self) -> str:
        """Generate a synthetic web search result (~500-1000 tokens)."""
        topic = _RESEARCH_TOPICS[int(self.rng.integers(0, len(_RESEARCH_TOPICS)))]
        n_results = int(self.rng.integers(3, 8))
        lines = [f'Found {n_results} results for "{topic}":']
        for i in range(min(n_results, 5)):
            lines.append(
                f"{i+1}. [Result {i+1}: {topic}](https://example.com/{i}) — "
                f"{_PARAGRAPHS[int(self.rng.integers(0, len(_PARAGRAPHS)))][:120]}"
            )
        return "\n".join(lines)

    def _make_scrape_result(self) -> str:
        """Generate a synthetic web scrape result (~1000-5000 tokens)."""
        topic = _RESEARCH_TOPICS[int(self.rng.integers(0, len(_RESEARCH_TOPICS)))]
        n_paragraphs = int(self.rng.integers(3, 8))
        sections = []
        for _ in range(n_paragraphs):
            sections.append(_PARAGRAPHS[int(self.rng.integers(0, len(_PARAGRAPHS)))])
        return _SCRAPE_RESULT_TEMPLATE.format(
            title=f"Analysis: {topic}",
            intro_paragraph=sections[0] if sections else "",
            findings="\n\n".join(f"- {s}" for s in sections[1:3]),
            technical_details="\n\n".join(sections[3:5]) if len(sections) > 3 else "",
            conclusion=sections[-1] if sections else "",
        )

    def _make_action_message(self, turn: int) -> str:
        """Generate a short assistant action message (~200-500 tokens)."""
        if turn % 3 == 0:
            actions = _SEARCH_ACTIONS
        elif turn % 3 == 1:
            actions = _SCRAPE_ACTIONS
        else:
            actions = _SYNTHESIS_ACTIONS
        action = actions[int(self.rng.integers(0, len(actions)))]
        reasoning = _PARAGRAPHS[int(self.rng.integers(0, len(_PARAGRAPHS)))][:200]
        return f"{action}\n\n{reasoning}"

    def _make_tool_result(self, turn: int) -> str:
        """Generate a tool result based on the turn pattern."""
        if turn % 3 == 0:
            return self._make_search_result()
        elif turn % 3 == 1:
            return self._make_scrape_result()
        else:
            # Code execution or synthesis — shorter result
            return f"Execution complete. Summary of {int(self.rng.integers(3, 10))} data points processed."

    def _make_summary_for_restart(self, turn: int) -> str:
        """Generate a summary message for session restart (retry_with_summary pattern)."""
        return (
            f"[Session summary after {turn} turns]\n"
            f"Research progress: investigated {turn // 3} sub-questions, "
            f"consulted {turn // 2} sources, found {int(self.rng.integers(2, 8))} key findings.\n"
            f"Key insight so far: {_PARAGRAPHS[int(self.rng.integers(0, len(_PARAGRAPHS)))][:200]}\n"
            f"Continuing investigation from this point."
        )

    def generate(self, num_requests: int) -> list[Request]:
        """Generate num_requests deep research agent requests."""
        requests: list[Request] = []

        while len(requests) < num_requests:
            session_requests = self._generate_session()
            requests.extend(session_requests)

        return requests[:num_requests]

    def _prune_messages_for_window(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the system prompt, latest user summary/query, and recent full turn groups."""
        if len(messages) <= 2:
            return list(messages)

        system_message = messages[0]
        head_messages = [msg for msg in messages[1:] if msg.get("role") == "user"][:1]
        turn_groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = []

        for msg in messages[1:]:
            role = msg.get("role")
            if role == "assistant" and current_group:
                turn_groups.append(current_group)
                current_group = [msg]
            elif role in {"assistant", "tool", "user"}:
                current_group.append(msg)
            else:
                if current_group:
                    turn_groups.append(current_group)
                    current_group = []
                head_messages.append(msg)

        if current_group:
            turn_groups.append(current_group)

        pruned: list[dict[str, Any]] = [system_message, *head_messages]
        for group in turn_groups[-self.rolling_window_size:]:
            pruned.extend(group)
        return pruned

    def _generate_session(self) -> list[Request]:
        """Create a single long-running research session."""
        num_turns = int(self.rng.integers(self.min_turns, self.max_turns + 1))
        session_id = self.rng.bytes(6).hex()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _DEEP_RESEARCH_SYSTEM_PROMPT}
        ]
        # Track tool results for rolling window
        tool_results: list[dict[str, Any]] = []
        requests: list[Request] = []
        restart_cooldown = False

        # Initial research query
        messages.append({"role": "user", "content": self._pick_research_query()})

        for turn in range(num_turns):
            session_restarted = False
            # Check for session restart with summary
            if (
                not restart_cooldown
                and
                turn > 10
                and float(self.rng.random()) < self.retry_with_summary_prob
            ):
                summary = self._make_summary_for_restart(turn)
                messages = [
                    {"role": "system", "content": _DEEP_RESEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": summary},
                ]
                tool_results = []
                session_restarted = True
                restart_cooldown = True
            else:
                restart_cooldown = False

            # Apply rolling window: keep only last N tool results in context
            if len(tool_results) > self.rolling_window_size:
                messages = self._prune_messages_for_window(messages)
                tool_results = tool_results[-self.rolling_window_size:]

            # Estimate context size (ramp then plateau)
            if turn < _RAMP_TURNS:
                estimated_context = 2000 + (turn * (_CONTEXT_PLATEAU_TOKENS - 2000) // _RAMP_TURNS)
            else:
                estimated_context = _CONTEXT_PLATEAU_TOKENS + int(self.rng.integers(-5000, 5000))

            # Short output tokens (action selection, not long-form)
            expected_tokens = int(self.rng.integers(100, 500))

            requests.append(
                Request(
                    request_id=_new_request_id(self.rng),
                    messages=list(messages),
                    expected_output_tokens=expected_tokens,
                    session_id=session_id,
                    metadata={
                        "workload": "deep_research_agent",
                        "turn": turn,
                        "estimated_context_tokens": estimated_context,
                        "rolling_window_size": len(tool_results),
                        "session_restart": session_restarted,
                    },
                )
            )

            # Simulate assistant action + tool call + tool result
            action_msg = self._make_action_message(turn)
            messages.append({"role": "assistant", "content": action_msg})

            tool_result_content = self._make_tool_result(turn)
            tool_result_msg = {"role": "tool", "tool_call_id": f"call_{self.rng.bytes(4).hex()}", "content": tool_result_content}
            messages.append(tool_result_msg)
            tool_results.append(tool_result_msg)

            # Follow-up user message for next turn
            if turn < num_turns - 1:
                if session_restarted:
                    messages.append({"role": "user", "content": "Continue from the summary and keep investigating the highest-value open thread."})
                elif turn % 5 == 4:
                    messages.append({"role": "user", "content": "Continue investigating. What else should we check?"})
                else:
                    messages.append({"role": "user", "content": "Good, keep going."})

        return requests
