"""InferScope MCP server — the main entry point for MCP-compatible agents."""

from __future__ import annotations

from fastmcp import FastMCP

from inferscope.server_benchmarks import register_benchmark_tools
from inferscope.server_profiling import register_profiling_tools

mcp = FastMCP(
    "inferscope",
    instructions="""InferScope is scoped to one operator-facing production lane plus a small public benchmark catalog:
    Kimi-K2.5 long-context coding on Hopper/Blackwell, with Dynamo as the production target,
    plus benchmark-supported public model lanes including single-endpoint vLLM on one H100-class GPU.

    The MCP surface is intentionally narrow:
    - return the supported production contract
    - profile a live deployment from frontend and worker Prometheus metrics
    - resolve and run the supported benchmark probes
    - load and compare probe artifacts

    InferScope is not a generic hardware catalog, generic benchmark matrix, or generic MCP wrapper.
    Treat the supported contract as authoritative until the product scope expands again.
    """,
)

register_profiling_tools(mcp)
register_benchmark_tools(mcp)
