"""Agent trace generator for ISB-1 benchmarks.

Produces multi-turn agent conversations with tool calling.  Each
conversation follows the pattern: system prompt (with tool schemas) +
user query -> model generates tool_call -> tool_result -> continuation,
repeated for 3-8 turns with growing context.

Tool schemas are loaded from ``workloads/schemas/*.json`` when available;
otherwise a built-in fallback set is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from workloads.base import Request, WorkloadGenerator, _new_request_id

# ---------------------------------------------------------------------------
# Directory containing tool schemas
# ---------------------------------------------------------------------------

_SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"

# ---------------------------------------------------------------------------
# Fallback tool schemas (used when the schemas/ directory is missing)
# ---------------------------------------------------------------------------

_FALLBACK_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_execute",
            "description": "Execute code in a sandboxed environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "javascript"]},
                    "code": {"type": "string", "description": "Code to run"},
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Realistic user query templates
# ---------------------------------------------------------------------------

_AGENT_QUERIES = [
    "Find the latest quarterly revenue numbers for {company} and summarise the trends.",
    "Search for recent research papers on {tech_topic} and give me a brief overview.",
    "Look up the current weather in {city} and suggest what to wear.",
    "Query our user database to find accounts created in the last 30 days and summarise.",
    "Read the configuration file at {filepath} and check if the timeout settings are correct.",
    "Execute a Python script that calculates the first 20 Fibonacci numbers.",
    "Find the API documentation for {service} and explain how authentication works.",
    "Check our analytics database for the top 10 most visited pages this week.",
    "Search for best practices on {tech_topic} and create an action plan.",
    "Read the README in our main repository and summarise the setup instructions.",
    "Look up the current exchange rate between USD and EUR.",
    "Run a database query to find all orders with a total above $500 from last month.",
    "Search for open issues related to {tech_topic} in our internal tracker.",
    "Execute a script to validate the JSON configuration files in the project.",
    "Find recent blog posts about {tech_topic} and highlight key takeaways.",
]

_COMPANIES = [
    "Acme Corp", "GlobalTech Industries", "Vertex Solutions",
    "NovaStar Inc", "Pinnacle Systems", "BlueWave Analytics",
]

_TECH_TOPICS = [
    "transformer architecture optimisation",
    "container orchestration with Kubernetes",
    "real-time stream processing",
    "vector database indexing strategies",
    "zero-trust network security",
    "serverless function cold starts",
    "distributed consensus algorithms",
    "graph neural networks",
    "WebAssembly performance tuning",
    "incremental static regeneration",
]

_CITIES = [
    "San Francisco", "London", "Tokyo", "Berlin", "Sydney",
    "Toronto", "Singapore", "Amsterdam", "Seoul", "Austin",
]

_SERVICES = [
    "Stripe Payments", "Twilio Messaging", "SendGrid Email",
    "AWS S3", "Google Cloud Pub/Sub", "GitHub REST API",
]

_FILEPATHS = [
    "config/settings.yaml", "deploy/docker-compose.yml",
    "src/config.json", "infra/terraform.tfvars",
    ".github/workflows/ci.yml", "k8s/deployment.yaml",
]

# ---------------------------------------------------------------------------
# Follow-up templates
# ---------------------------------------------------------------------------

_FOLLOWUPS = [
    "Good, now can you drill deeper into the {aspect} part?",
    "That's useful. Can you also check {related_item} for comparison?",
    "Run a follow-up query to get more details on the second point.",
    "Search for any counter-arguments or alternative perspectives.",
    "Execute another script that formats those results as a table.",
    "Read the related configuration file and compare the values.",
    "Can you verify those numbers by cross-referencing with another source?",
    "Summarise all the findings so far into a structured report.",
]

_RELATED_ITEMS = [
    "the previous quarter's data",
    "the competitor's approach",
    "the staging environment configuration",
    "the error logs from last week",
    "the upstream dependency versions",
]

_FOLLOW_ASPECTS = [
    "the performance metrics",
    "the error handling",
    "the authentication flow",
    "the data transformation pipeline",
    "the deployment strategy",
    "the caching layer",
]

# ---------------------------------------------------------------------------
# Synthetic tool-call and tool-result templates
# ---------------------------------------------------------------------------

_SEARCH_RESULTS = [
    'Found 5 results for "{query}":\n1. "{topic}: A Comprehensive Guide" - Overview of key concepts and recent advances.\n2. "Benchmarking {topic}" - Performance comparisons across different implementations.\n3. "Production Best Practices for {topic}" - Practical recommendations from industry experts.\n4. "{topic} Case Study at Scale" - How a large organisation adopted this approach.\n5. "Common Pitfalls in {topic}" - Mistakes to avoid and how to recover from them.',
    'Search results for "{query}":\n1. Official documentation covering setup, configuration, and API reference.\n2. Community discussion with 47 replies on performance tuning.\n3. Technical blog post with benchmarks comparing three approaches.\n4. Video tutorial: "Getting Started with {topic} in 30 Minutes".\n5. Stack Overflow answer with 234 upvotes on troubleshooting common issues.',
]

_CODE_RESULTS = [
    "Execution completed successfully.\nOutput:\n{output}\nExecution time: {time_ms}ms",
    "Code executed without errors.\nStandard output:\n{output}\nMemory usage: {mem_mb}MB | Wall time: {time_ms}ms",
]

_CODE_OUTPUTS = [
    "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]",
    "{'total_records': 1247, 'valid': 1203, 'invalid': 44, 'success_rate': 0.965}",
    "Validation passed: 12 files checked, 0 errors found.",
    "| Metric    | Value  |\n|-----------|--------|\n| Mean      | 42.7   |\n| Median    | 38.2   |\n| Std Dev   | 11.4   |",
    "Processing complete. 3 warnings, 0 errors.\nWarning: deprecated field 'timeout_ms' at line 14\nWarning: unused variable 'retry_count' at line 28\nWarning: missing optional field 'description' at line 3",
]

_DB_RESULTS = [
    "Query returned {n_rows} rows.\n\n| id | name | created_at | status |\n|----|------|------------|--------|\n| 1042 | project-alpha | 2025-11-01 | active |\n| 1043 | analytics-v2 | 2025-11-03 | active |\n| 1044 | migration-tool | 2025-11-05 | pending |\n... ({n_rows} total rows)",
    "Results ({n_rows} rows):\nAggregate: sum={total}, avg={avg}, min={min_val}, max={max_val}\nMost recent entry: 2025-11-12T14:32:00Z",
]

_FILE_CONTENTS = [
    '```yaml\n# Application Configuration\nserver:\n  host: "0.0.0.0"\n  port: 8080\n  timeout_seconds: 30\n  max_connections: 500\n\ndatabase:\n  url: "postgresql://localhost:5432/appdb"\n  pool_size: 20\n  ssl: true\n\nlogging:\n  level: "INFO"\n  format: "json"\n```',
    '```json\n{{\n  "version": "2.1.0",\n  "features": {{\n    "enable_cache": true,\n    "cache_ttl_seconds": 3600,\n    "rate_limit_rpm": 1000\n  }},\n  "endpoints": [\n    {{"path": "/api/v1/users", "method": "GET", "auth": true}},\n    {{"path": "/api/v1/health", "method": "GET", "auth": false}}\n  ]\n}}\n```',
]

# ---------------------------------------------------------------------------
# Assistant reasoning sentences
# ---------------------------------------------------------------------------

_REASONING_SENTENCES = [
    "Based on the search results, there are several key findings to consider.",
    "The data indicates a clear trend that aligns with the initial hypothesis.",
    "Let me analyse the output from the tool call to extract the relevant information.",
    "Comparing these results with the previous data shows some notable differences.",
    "The configuration looks mostly correct, but there are a few items worth reviewing.",
    "I found the relevant information and will summarise the key points below.",
    "The query returned useful data that helps answer your question directly.",
    "Looking at the results, the most important takeaway is the overall pattern.",
    "I'll need to run one more check to verify these findings thoroughly.",
    "The execution completed successfully and the output confirms the expected behaviour.",
    "Cross-referencing multiple sources provides a more complete picture of the situation.",
    "The results suggest that the current approach is working well but could be optimised.",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a capable AI assistant with access to external tools. "
    "Use the provided tools when you need to search for information, "
    "execute code, query databases, or read files. Always explain your "
    "reasoning before and after using a tool. Provide clear, well-structured "
    "responses.\n\nAvailable tools:\n{tool_descriptions}"
)


def _load_tool_schemas() -> list[dict[str, Any]]:
    """Load tool schemas from disk, falling back to built-in defaults."""
    if not _SCHEMAS_DIR.is_dir():
        return list(_FALLBACK_TOOLS)

    schemas: list[dict[str, Any]] = []
    for path in sorted(_SCHEMAS_DIR.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                schemas.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue

    return schemas if schemas else list(_FALLBACK_TOOLS)


def _format_tool_descriptions(tools: list[dict[str, Any]]) -> str:
    """Create a human-readable summary of tool schemas for the system prompt."""
    parts: list[str] = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        param_list = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in params.items()
        )
        parts.append(f"- {name}({param_list}): {desc}")
    return "\n".join(parts)


class AgentTraceGenerator(WorkloadGenerator):
    """Generate multi-turn agent workload traces with tool calling.

    Each conversation consists of 3-8 turns following the pattern:
    user query -> assistant with tool_call -> tool result -> assistant
    continuation, producing progressively longer contexts.

    Parameters:
        seed: Random seed for reproducibility.
        min_turns: Minimum tool-calling rounds per conversation (default 3).
        max_turns: Maximum tool-calling rounds per conversation (default 8).
        schemas_dir: Optional override for the tool schemas directory.
    """

    def __init__(
        self,
        seed: int = 42,
        min_turns: int = 3,
        max_turns: int = 8,
        schemas_dir: str | Path | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.min_turns = min_turns
        self.max_turns = max_turns

        if schemas_dir is not None:
            global _SCHEMAS_DIR  # noqa: PLW0603
            _SCHEMAS_DIR = Path(schemas_dir)

        self._tools = _load_tool_schemas()
        tool_desc = _format_tool_descriptions(self._tools)
        self._system_prompt = _AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_desc
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_query(self) -> str:
        """Choose and fill a user query template."""
        template = _AGENT_QUERIES[int(self.rng.integers(0, len(_AGENT_QUERIES)))]
        return template.format(
            company=_COMPANIES[int(self.rng.integers(0, len(_COMPANIES)))],
            tech_topic=_TECH_TOPICS[int(self.rng.integers(0, len(_TECH_TOPICS)))],
            city=_CITIES[int(self.rng.integers(0, len(_CITIES)))],
            service=_SERVICES[int(self.rng.integers(0, len(_SERVICES)))],
            filepath=_FILEPATHS[int(self.rng.integers(0, len(_FILEPATHS)))],
        )

    def _pick_followup(self) -> str:
        """Choose and fill a follow-up query template."""
        template = _FOLLOWUPS[int(self.rng.integers(0, len(_FOLLOWUPS)))]
        return template.format(
            aspect=_FOLLOW_ASPECTS[int(self.rng.integers(0, len(_FOLLOW_ASPECTS)))],
            related_item=_RELATED_ITEMS[int(self.rng.integers(0, len(_RELATED_ITEMS)))],
        )

    def _make_tool_call(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Generate a synthetic tool_call message for a given tool schema."""
        func = tool.get("function", tool)
        name = func["name"]
        params = func.get("parameters", {}).get("properties", {})

        # Build plausible arguments
        args: dict[str, Any] = {}
        for key, schema in params.items():
            if "enum" in schema:
                args[key] = schema["enum"][
                    int(self.rng.integers(0, len(schema["enum"])))
                ]
            elif schema.get("type") == "string":
                if "query" in key:
                    args[key] = _TECH_TOPICS[
                        int(self.rng.integers(0, len(_TECH_TOPICS)))
                    ]
                elif "path" in key or "url" in key:
                    args[key] = _FILEPATHS[
                        int(self.rng.integers(0, len(_FILEPATHS)))
                    ]
                elif "code" in key:
                    args[key] = "print('Hello from synthetic agent trace')"
                else:
                    args[key] = f"sample_{key}_value"
            elif schema.get("type") == "integer":
                args[key] = int(self.rng.integers(1, 100))
            elif schema.get("type") == "object":
                args[key] = {}

        call_id = f"call_{self.rng.bytes(6).hex()}"
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
            ],
        }

    def _make_tool_result(self, tool_call_msg: dict[str, Any]) -> dict[str, Any]:
        """Generate a synthetic tool result for a preceding tool call."""
        call = tool_call_msg["tool_calls"][0]
        call_id = call["id"]
        name = call["function"]["name"]

        topic = _TECH_TOPICS[int(self.rng.integers(0, len(_TECH_TOPICS)))]
        query = topic

        if name == "search":
            template = _SEARCH_RESULTS[int(self.rng.integers(0, len(_SEARCH_RESULTS)))]
            content = template.format(query=query, topic=topic)
        elif name == "code_execute":
            template = _CODE_RESULTS[int(self.rng.integers(0, len(_CODE_RESULTS)))]
            output = _CODE_OUTPUTS[int(self.rng.integers(0, len(_CODE_OUTPUTS)))]
            content = template.format(
                output=output,
                time_ms=int(self.rng.integers(10, 2000)),
                mem_mb=round(float(self.rng.uniform(1.0, 256.0)), 1),
            )
        elif name == "database_query":
            template = _DB_RESULTS[int(self.rng.integers(0, len(_DB_RESULTS)))]
            n_rows = int(self.rng.integers(3, 500))
            content = template.format(
                n_rows=n_rows,
                total=int(self.rng.integers(1000, 100000)),
                avg=round(float(self.rng.uniform(10.0, 500.0)), 2),
                min_val=int(self.rng.integers(1, 50)),
                max_val=int(self.rng.integers(500, 10000)),
            )
        elif name == "file_read":
            content = _FILE_CONTENTS[int(self.rng.integers(0, len(_FILE_CONTENTS)))]
        elif name == "api_call":
            content = json.dumps(
                {
                    "status": 200,
                    "body": {
                        "result": "success",
                        "data": {"items": int(self.rng.integers(1, 50))},
                        "request_id": self.rng.bytes(4).hex(),
                    },
                },
                indent=2,
            )
        else:
            content = f"Tool '{name}' executed successfully. Result: OK"

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }

    def _make_reasoning(self, n_sentences: int) -> str:
        """Compose a synthetic assistant reasoning response."""
        indices = self.rng.choice(
            len(_REASONING_SENTENCES), size=min(n_sentences, len(_REASONING_SENTENCES)),
            replace=False,
        )
        return " ".join(_REASONING_SENTENCES[i] for i in indices)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, num_requests: int) -> list[Request]:
        """Generate *num_requests* agent requests across multi-turn sessions.

        Each conversation produces one :class:`Request` per user turn
        (the point at which the model must respond, potentially with a
        tool call).  Context grows across turns within a session.

        Args:
            num_requests: Target number of requests to produce.

        Returns:
            A list of :class:`Request` in session-sequential order.
        """
        requests: list[Request] = []

        while len(requests) < num_requests:
            session_requests = self._generate_session()
            requests.extend(session_requests)

        return requests[:num_requests]

    def _generate_session(self) -> list[Request]:
        """Create a single multi-turn agent conversation."""
        num_turns = int(self.rng.integers(self.min_turns, self.max_turns + 1))
        session_id = self.rng.bytes(6).hex()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]
        requests: list[Request] = []

        for turn in range(num_turns):
            # ---- User message ----
            if turn == 0:
                user_text = self._pick_query()
            else:
                user_text = self._pick_followup()

            messages.append({"role": "user", "content": user_text})

            # Emit a request at this point (model needs to respond)
            expected_tokens = int(self.rng.integers(150, 600))
            requests.append(
                Request(
                    request_id=_new_request_id(self.rng),
                    messages=list(messages),
                    expected_output_tokens=expected_tokens,
                    session_id=session_id,
                    metadata={
                        "workload": "agent",
                        "turn": turn,
                        "turn_type": "initial_query" if turn == 0 else "followup",
                        "num_tools": len(self._tools),
                    },
                )
            )

            # ---- Simulate assistant tool call ----
            tool = self._tools[int(self.rng.integers(0, len(self._tools)))]
            tool_call_msg = self._make_tool_call(tool)
            messages.append(tool_call_msg)

            # ---- Simulate tool result ----
            tool_result_msg = self._make_tool_result(tool_call_msg)
            messages.append(tool_result_msg)

            # ---- Simulate assistant reasoning after tool result ----
            n_sentences = int(self.rng.integers(2, 6))
            reasoning = self._make_reasoning(n_sentences)
            messages.append({"role": "assistant", "content": reasoning})

        return requests
