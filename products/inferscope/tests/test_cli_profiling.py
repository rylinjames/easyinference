"""CLI tests for runtime profiling registration and compatibility."""

from __future__ import annotations

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
