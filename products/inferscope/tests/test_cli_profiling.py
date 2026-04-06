"""CLI tests for runtime profiling registration and compatibility."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from inferscope.cli import app

runner = CliRunner()


def test_cli_profile_runtime_command_exists_and_runs(monkeypatch) -> None:
    async def fake_profile_runtime(*args, **kwargs):
        del args, kwargs
        return {"summary": "Runtime profile ok", "confidence": 0.9, "endpoint": "http://localhost:8000"}

    monkeypatch.setattr("inferscope.cli_profiling.profile_runtime", fake_profile_runtime)

    result = runner.invoke(app, ["profile-runtime", "http://localhost:8000"])

    assert result.exit_code == 0, result.stdout
    assert "Runtime profile ok" in result.stdout


def test_cli_profile_runtime_failure_includes_next_steps(monkeypatch) -> None:
    async def fake_profile_runtime(*args, **kwargs):
        del args, kwargs
        return {
            "error": "metrics scrape failed: 401 Unauthorized",
            "summary": "❌ Runtime profiling failed: metrics scrape failed: 401 Unauthorized",
            "confidence": 0.0,
            "endpoint": "http://localhost:8000",
            "next_steps": [
                "Check the request or metrics auth configuration and retry with the correct header scheme.",
                "See products/inferscope/docs/QUICKSTART.md for the current supported InferScope lane.",
            ],
        }

    monkeypatch.setattr("inferscope.cli_profiling.profile_runtime", fake_profile_runtime)

    result = runner.invoke(app, ["profile-runtime", "http://localhost:8000"])

    assert result.exit_code == 0, result.stdout
    assert "Runtime profiling failed" in result.stdout
    assert "next_steps" in result.stdout
    assert "auth configuration" in result.stdout


def test_cli_profile_runtime_forwards_metrics_endpoint_and_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_profile_runtime(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"summary": "Runtime profile ok", "confidence": 0.9, "endpoint": "http://localhost:8000"}

    monkeypatch.setattr("inferscope.cli_profiling.profile_runtime", fake_profile_runtime)

    result = runner.invoke(
        app,
        [
            "profile-runtime",
            "http://localhost:8000",
            "--metrics-endpoint",
            "http://localhost:9000",
            "--scrape-timeout-seconds",
            "90",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["args"] == ("http://localhost:8000",)
    assert captured["kwargs"]["metrics_endpoint"] == "http://localhost:9000"
    assert captured["kwargs"]["scrape_timeout_seconds"] == 90.0


def test_cli_profile_still_routes_to_model_intel(monkeypatch) -> None:
    monkeypatch.setattr(
        "inferscope.cli.get_model_profile",
        lambda model: {"summary": f"Model profile for {model}", "confidence": 1.0, "model": model},
    )

    result = runner.invoke(app, ["profile", "DeepSeek-R1"])

    assert result.exit_code == 0, result.stdout
    assert "Model profile for DeepSeek-R1" in result.stdout


def test_cli_help_includes_runtime_and_legacy_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.stdout
    for command_name in ("profile-runtime", "profile", "check", "memory", "cache", "audit"):
        assert command_name in result.stdout


def test_cli_connect_outputs_stdio_mcp_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "connect",
            "--project-dir",
            str(tmp_path),
            "--server-name",
            "InferScope",
        ],
    )

    assert result.exit_code == 0, result.stdout
    config = json.loads(result.stdout)
    server = config["mcpServers"]["InferScope"]
    assert server["command"] == "uv"
    assert server["args"][:3] == ["run", "--no-editable", "--directory"]
    assert server["args"][3] == str(tmp_path.resolve())
    assert server["args"][4:] == ["inferscope", "serve"]


def test_cli_connect_outputs_http_mcp_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "connect",
            "--project-dir",
            str(tmp_path),
            "--transport",
            "streamable-http",
            "--port",
            "9123",
        ],
    )

    assert result.exit_code == 0, result.stdout
    config = json.loads(result.stdout)
    server = config["mcpServers"]["InferScope"]
    assert server["transport"] == "streamable-http"
    assert server["url"] == "http://127.0.0.1:9123/mcp"
