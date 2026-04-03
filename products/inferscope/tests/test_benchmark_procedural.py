"""Tests for procedural InferScope workload materialization."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from inferscope.benchmarks import ProceduralWorkloadOptions, materialize_workload
from inferscope.cli import app

runner = CliRunner()


def test_materialize_tool_agent_produces_requested_count() -> None:
    pack = materialize_workload(
        "tool-agent",
        options=ProceduralWorkloadOptions(
            request_count=4,
            input_tokens=2048,
            output_tokens=256,
        ),
    )
    assert pack.name == "tool-agent"
    assert len(pack.requests) == 4
    assert "procedural" in pack.tags
    assert all(message.role != "tool" for request in pack.requests for message in request.messages)


def test_materialize_tool_agent_seed_changes_materialization() -> None:
    pack_a = materialize_workload(
        "tool-agent",
        options=ProceduralWorkloadOptions(
            request_count=2,
            input_tokens=2048,
            output_tokens=256,
            seed=7,
        ),
    )
    pack_b = materialize_workload(
        "tool-agent",
        options=ProceduralWorkloadOptions(
            request_count=2,
            input_tokens=2048,
            output_tokens=256,
            seed=99,
        ),
    )
    assert pack_a.requests[0].messages[1].content != pack_b.requests[0].messages[1].content


def test_materialize_coding_long_context_supports_context_file(tmp_path: Path) -> None:
    context_file = tmp_path / "context.py"
    context_file.write_text("def optimize_kernel():\n    return 'ok'\n", encoding="utf-8")
    pack = materialize_workload(
        "coding-long-context",
        options=ProceduralWorkloadOptions(
            request_count=2,
            input_tokens=1024,
            context_file=str(context_file),
        ),
    )
    assert len(pack.requests) == 2
    assert "optimize_kernel" in str(pack.requests[0].messages[0].content)


def test_procedural_generation_rejects_explicit_file_paths(tmp_path: Path) -> None:
    fake_workload = tmp_path / "tool-agent.yaml"
    fake_workload.write_text("name: tool-agent\n", encoding="utf-8")
    with pytest.raises(ValueError, match="packaged built-in workloads"):
        materialize_workload(
            str(fake_workload),
            options=ProceduralWorkloadOptions(request_count=2),
        )


def test_cli_benchmark_plan_accepts_synthetic_options() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark-plan",
            "kimi-k2-long-context-coding",
            "http://localhost:8000",
            "--gpu",
            "b200",
            "--num-gpus",
            "4",
            "--synthetic-requests",
            "4",
            "--synthetic-input-tokens",
            "2048",
            "--synthetic-output-tokens",
            "256",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Resolved probe plan for kimi-k2-long-context-coding" in result.stdout
