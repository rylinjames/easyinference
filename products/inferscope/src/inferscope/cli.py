"""InferScope CLI — operator tooling for runtime profiling and narrow probe execution.

Usage:
    inferscope --version
    inferscope profile-runtime http://localhost:8000
    inferscope --output-format json profile-runtime http://localhost:8000  # machine-readable
    inferscope benchmark-plan kimi-k2-long-context-coding http://localhost:8000 --gpu b200 --num-gpus 8
    inferscope benchmark kimi-k2-long-context-coding http://localhost:8000 --experiment dynamo-disagg-lmcache-kimi-k2
    inferscope serve  # Start MCP server (stdio)
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from inferscope.cli_benchmarks import register_benchmark_commands
from inferscope.cli_experiments import register_experiment_commands
from inferscope.cli_profiling import register_profiling_commands
from inferscope.endpoint_auth import parse_header_values, resolve_auth_config
from inferscope.tools.hardware_intel import compare_gpus, get_gpu_specs
from inferscope.tools.kv_cache import (
    calculate_kv_budget,
    compare_quantization,
    estimate_kv_quant_savings,
    recommend_disaggregation,
    recommend_kv_strategy,
)
from inferscope.tools.model_intel import (
    estimate_capacity,
    get_model_profile,
    validate_serving_config,
)
from inferscope.tools.pmax_scheduler import recommend_pmax_schedule
from inferscope.tools.recommend import (
    recommend_config,
    recommend_engine,
    suggest_parallelism,
)

app = typer.Typer(
    name="inferscope",
    help="Inference diagnostics and narrow probe tooling for KV cache and disaggregated serving.",
    no_args_is_help=True,
)
console = Console()

# Output mode set by the global --output-format flag.
# "pretty" (default): rich-formatted console output via _print_result.
# "json": raw JSON dump to stdout for piping into jq / CI consumption.
_OUTPUT_MODE: str = "pretty"


def _resolve_inferscope_version() -> str:
    """Return the installed inferscope package version, or a sentinel if unknown."""
    try:
        return _pkg_version("inferscope")
    except PackageNotFoundError:
        return "unknown (package metadata not found)"


def _version_callback(value: bool) -> None:
    """Eager callback for the global --version flag."""
    if value:
        typer.echo(f"inferscope {_resolve_inferscope_version()}")
        raise typer.Exit()


@app.callback()
def _global_options(
    version: bool = typer.Option(
        None,
        "--version",
        help="Show the installed InferScope version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    output_format: str = typer.Option(
        "pretty",
        "--output-format",
        help="Output format: 'pretty' (rich console, default) or 'json' (raw JSON for scripting).",
    ),
) -> None:
    """Global options that apply to every subcommand."""
    global _OUTPUT_MODE
    if output_format not in ("pretty", "json"):
        raise typer.BadParameter("--output-format must be 'pretty' or 'json'")
    _OUTPUT_MODE = output_format


def _print_result(result: dict) -> None:
    """Pretty-print a tool result, or dump raw JSON when --output-format=json."""
    if _OUTPUT_MODE == "json":
        # Machine-readable mode — dump everything to stdout as one JSON document.
        # Use typer.echo so output is unstyled and pipeable.
        typer.echo(json.dumps(result, indent=2, default=str))
        return

    summary = result.pop("summary", "")
    confidence = result.pop("confidence", None)

    if summary:
        console.print(f"\n[bold]{summary}[/bold]")

    if confidence is not None:
        level = "green" if confidence >= 0.8 else "yellow" if confidence >= 0.6 else "red"
        console.print(f"Confidence: [{level}]{confidence:.0%}[/{level}]")

    # Surface reasoning trace as readable "why" summary
    reasoning = (
        result.get("serving_profile", {}).get("reasoning_trace", [])
        or result.get("reasoning_trace", [])
        or result.get("reasoning", [])
    )
    if reasoning:
        console.print("\n[bold yellow]Why this recommendation:[/bold yellow]")
        for step in reasoning:
            node = step.split(":")[0] if ":" in step else ""
            detail = step.split(":", 1)[1].strip() if ":" in step else step
            if node:
                console.print(f"  [dim]{node}:[/dim] {detail}")
            else:
                console.print(f"  {detail}")

    launch_cmd = result.pop("launch_command", None)
    if launch_cmd:
        console.print("\n[bold cyan]Launch command:[/bold cyan]")
        console.print(Panel(Syntax(launch_cmd, "bash", theme="monokai"), border_style="cyan"))

    output = json.dumps(result, indent=2, default=str)
    console.print(Syntax(output, "json", theme="monokai"))


def _resolve_metrics_auth(
    *,
    provider: str = "",
    metrics_api_key: str = "",
    metrics_auth_scheme: str = "",
    metrics_auth_header_name: str = "",
    metrics_header: list[str] | None = None,
):
    try:
        return resolve_auth_config(
            (metrics_api_key or None),
            provider=provider,
            auth_scheme=metrics_auth_scheme,
            auth_header_name=metrics_auth_header_name,
            headers=parse_header_values(metrics_header, option_name="metrics header"),
            default_scheme="bearer",
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _build_mcp_config(
    *,
    project_dir: Path,
    server_name: str,
    transport: str,
    port: int,
) -> dict[str, object]:
    project_dir = project_dir.resolve()
    if transport == "stdio":
        return {
            "mcpServers": {
                server_name: {
                    "command": "uv",
                    "args": [
                        "run",
                        "--no-editable",
                        "--directory",
                        str(project_dir),
                        "inferscope",
                        "serve",
                    ],
                }
            }
        }

    return {
        "mcpServers": {
            server_name: {
                "transport": "streamable-http",
                "url": f"http://127.0.0.1:{port}/mcp",
            }
        }
    }


@app.command()
def profile(model: str = typer.Argument(help="Model name (e.g., DeepSeek-R1, Qwen3.5-72B)")):
    """Show serving profile for a model across all supported engines and GPUs."""
    _print_result(get_model_profile(model))


@app.command()
def validate(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type (e.g., h100, mi355x)"),
    tp: int = typer.Option(1, help="Tensor parallelism degree"),
    quantization: str = typer.Option("auto", help="Quantization (fp8, bf16, awq, etc.)"),
    engine: str = typer.Option("vllm", help="Engine (vllm, sglang, atom)"),
):
    """Validate a serving config before deployment."""
    _print_result(validate_serving_config(model, gpu, tp, quantization, engine))


@app.command(name="recommend")
def recommend_cmd(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    workload: str = typer.Option("chat", help="Workload: coding, chat, agent, long_context_rag"),
    num_gpus: int = typer.Option(1, help="Number of GPUs"),
    engine: str = typer.Option("auto", help="Engine (auto, vllm, sglang, atom, trtllm, dynamo)"),
):
    """Generate optimal ServingProfile for this deployment."""
    _print_result(recommend_config(model, gpu, workload, num_gpus, engine=engine))


@app.command()
def gpu(name: str = typer.Argument(help="GPU name (e.g., h100, a100, mi355x)")):
    """Show complete ISA-level specs for a GPU."""
    _print_result(get_gpu_specs(name))


@app.command()
def compare(
    gpu_a: str = typer.Argument(help="First GPU"),
    gpu_b: str = typer.Argument(help="Second GPU"),
):
    """Side-by-side GPU comparison with inference-relevant metrics."""
    _print_result(compare_gpus(gpu_a, gpu_b))


@app.command()
def capacity(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    num_gpus: int = typer.Option(1, help="Number of GPUs"),
    quantization: str = typer.Option("auto", help="Quantization method"),
):
    """Calculate max concurrent users and KV cache budget."""
    _print_result(estimate_capacity(model, gpu, num_gpus, quantization))


@app.command(name="engine")
def engine_cmd(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    workload: str = typer.Option("chat", help="Workload type (coding, chat, agent, long_context_rag)"),
    num_gpus: int = typer.Option(1, help="Number of GPUs"),
    multi_node: bool = typer.Option(False, help="Multi-node deployment"),
):
    """Recommend the best inference engine for this deployment."""
    _print_result(recommend_engine(model, gpu, workload, num_gpus, multi_node))


@app.command()
def parallelism(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    num_gpus: int = typer.Argument(help="Number of GPUs available"),
):
    """Recommend TP/PP/DP/EP parallelism strategy."""
    _print_result(suggest_parallelism(model, gpu, num_gpus))


@app.command(name="kv-budget")
def kv_budget(
    model: str = typer.Argument(help="Model name"),
    context_length: int = typer.Argument(help="Context length in tokens"),
    batch_size: int = typer.Option(1, help="Batch size"),
    kv_dtype: str = typer.Option("fp8", help="KV cache dtype (fp8, fp16)"),
):
    """Calculate exact KV cache memory requirement."""
    _print_result(calculate_kv_budget(model, context_length, batch_size, kv_dtype))


@app.command(name="kv-strategy")
def kv_strategy(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    workload: str = typer.Option("chat", help="Workload type (coding, chat, agent, long_context_rag)"),
    max_context: int = typer.Option(32768, help="Max context length"),
    concurrent_sessions: int = typer.Option(100, help="Concurrent sessions"),
):
    """Recommend KV cache tiering strategy."""
    _print_result(recommend_kv_strategy(model, gpu, workload, max_context, concurrent_sessions))


@app.command(name="disagg")
def disagg(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    avg_prompt: int = typer.Option(4096, help="Average prompt tokens"),
    rate: float = typer.Option(10.0, help="Requests per second"),
    rdma: bool = typer.Option(False, help="RDMA available"),
    num_gpus: int = typer.Option(2, help="Number of GPUs"),
):
    """Determine if prefill/decode disaggregation would help."""
    _print_result(recommend_disaggregation(model, gpu, 500.0, avg_prompt, rate, rdma, num_gpus))


@app.command(name="pmax-recommend")
def pmax_recommend_cmd(
    batch_size: int = typer.Option(32, help="Rollout batch size"),
    base_pmax: int = typer.Option(2048, help="Baseline max_tokens per prompt"),
    strategy: str = typer.Option("variance_scaled", help="Strategy: fixed, variance_scaled, truncation_aware, bimodal"),
    gpu_token_budget: int = typer.Option(65536, help="Total token budget across the batch"),
    reward_history_file: str = typer.Option("", help="JSON file with reward history from previous batches"),
):
    """Recommend adaptive P_max schedule for RL rollout batches."""
    history = None
    if reward_history_file:
        import json
        from pathlib import Path
        history = json.loads(Path(reward_history_file).read_text())
        if not isinstance(history, list):
            history = [history]
    _print_result(recommend_pmax_schedule(
        batch_size=batch_size,
        base_pmax=base_pmax,
        strategy=strategy,
        reward_history=history,
        gpu_token_budget=gpu_token_budget,
    ))


@app.command(name="quantization")
def quantization_cmd(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
):
    """Compare quantization options for this model + GPU."""
    _print_result(compare_quantization(model, gpu))


@app.command(name="kv-quant-estimate")
def kv_quant_estimate_cmd(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    context_length: int = typer.Option(32768, help="Context length in tokens"),
    batch_size: int = typer.Option(32, help="Concurrent batch size"),
):
    """Estimate memory savings from FP8 vs FP16 KV cache quantization."""
    _print_result(estimate_kv_quant_savings(model, gpu, context_length, batch_size))


@app.command()
def evaluate(
    benchmark_result: str = typer.Argument(help="Path to ISB-1 benchmark result JSON"),
    model: str = typer.Option(None, help="Model name (auto-detected from result if omitted)"),
    gpu: str = typer.Option(None, help="GPU type (auto-detected from result if omitted)"),
    num_gpus: int = typer.Option(1, help="Number of GPUs"),
):
    """Compare actual benchmark results against InferScope's recommendation.

    Shows whether the recommended config would have performed better or worse
    than what was actually measured. Builds trust in the recommender over time.
    """
    result_data = json.loads(Path(benchmark_result).read_text(encoding="utf-8"))
    if isinstance(result_data, list):
        result_data = result_data[0]

    actual_model = model or result_data.get("model", "")
    actual_gpu = gpu or result_data.get("gpu", "")

    if not actual_model or not actual_gpu:
        console.print("[red]Cannot determine model and GPU from result. Use --model and --gpu.[/red]")
        raise typer.Exit(1)

    # Get recommendation using the workload from the benchmark result
    actual_workload = result_data.get("workload", "coding")
    rec = recommend_config(actual_model, actual_gpu, actual_workload, num_gpus)
    if "error" in rec:
        console.print(f"[yellow]Recommendation unavailable: {rec['error']}[/yellow]")
        console.print("Showing actual results only.")
        console.print(Syntax(json.dumps(result_data, indent=2, default=str), "json", theme="monokai"))
        return

    rec_profile = rec.get("serving_profile", {})
    rec_mem = rec.get("memory_plan", {})

    # Compare
    console.print("\n[bold]Recommendation vs Actual[/bold]\n")

    rec_engine = rec_profile.get("engine", "?")
    rec_quant = rec_profile.get("precision", {}).get("weights", "?")
    rec_tp = str(rec_profile.get("topology", {}).get("tp", "?"))
    rec_util = rec_profile.get("cache", {}).get("gpu_memory_utilization", 0)
    rec_prefix = str(rec_profile.get("cache", {}).get("prefix_cache", "?"))
    rec_fits = "yes" if rec_mem.get("fits") else "no"

    act_engine = result_data.get("engine", result_data.get("mode", "?"))
    act_quant = result_data.get("quantization", "?")
    act_tp = str(result_data.get("topology", "?"))
    act_util = result_data.get("gpu_memory_utilization")
    act_util_str = f"{act_util:.0%}" if act_util else "?"
    act_status = "ran" if result_data.get("status") == "completed" else "failed"

    comparisons = [
        ("Engine", rec_engine, act_engine),
        ("Quantization", rec_quant, act_quant),
        ("TP", rec_tp, act_tp),
        ("GPU Mem Util", f"{rec_util:.0%}", act_util_str),
        ("Prefix Cache", rec_prefix, "?"),
        ("Memory Fit", rec_fits, act_status),
    ]

    for label, recommended, actual in comparisons:
        match = "[green]=[/green]" if recommended == actual else "[yellow]!=[/yellow]"
        console.print(
            f"  {label:<16} Recommended: [cyan]{recommended:<12}[/cyan] "
            f"Actual: [white]{actual:<12}[/white] {match}"
        )

    # Performance gap
    actual_throughput = result_data.get("generation_throughput", 0)
    actual_goodput = result_data.get("goodput", 0)
    actual_slo = result_data.get("slo_attainment", 0)

    if actual_throughput > 0:
        console.print("\n[bold]Measured Performance[/bold]")
        console.print(f"  Throughput:  {actual_throughput:.0f} tok/s")
        console.print(f"  Goodput:     {actual_goodput:.1f} req/s")
        console.print(f"  SLO:         {actual_slo:.0%}")
        console.print(f"  TTFT p95:    {result_data.get('ttft_p95', 0):.3f}s")
        console.print(f"  TPOT p95:    {result_data.get('tpot_p95', 0) * 1000:.1f}ms")


register_profiling_commands(app, print_result=_print_result, resolve_metrics_auth=_resolve_metrics_auth)
register_benchmark_commands(app, print_result=_print_result)
register_experiment_commands(app, print_result=_print_result, resolve_metrics_auth=_resolve_metrics_auth)


@app.command()
def connect(
    project_dir: Path = typer.Option(
        Path.cwd(),
        "--project-dir",
        help="Path to the local products/inferscope checkout.",
    ),
    server_name: str = typer.Option("InferScope", help="Server name to emit in the MCP config."),
    transport: str = typer.Option("stdio", help="Transport: stdio or streamable-http."),
    port: int = typer.Option(8765, help="Port for streamable-http transport."),
):
    """Print MCP configuration JSON for Cursor or Claude Desktop."""
    config = _build_mcp_config(
        project_dir=project_dir,
        server_name=server_name,
        transport=transport,
        port=port,
    )
    typer.echo(json.dumps(config, indent=2))


@app.command()
def serve(
    transport: str = typer.Option("stdio", help="Transport: stdio or streamable-http"),
    port: int = typer.Option(8765, help="Port for HTTP transport"),
):
    """Start the InferScope MCP server."""
    from inferscope.server import mcp

    console.print(f"[bold green]Starting InferScope MCP server ({transport})...[/bold green]")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        console.print(
            "[yellow]⚠ HTTP transport binds to 127.0.0.1 (localhost only). "
            "Use a reverse proxy with authentication for production.[/yellow]"
        )
        console.print(
            "[yellow]⚠ The SSRF guard validates endpoint URLs against literal private IPs and "
            "loopback hostnames, but it does NOT pin DNS resolution. For network-exposed "
            "deployments, add an SSRF gate (or DNS pinning) at your reverse proxy layer to block "
            "DNS-rebinding attacks via *.nip.io / *.sslip.io / similar.[/yellow]"
        )
        mcp.run(transport="streamable-http", host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
