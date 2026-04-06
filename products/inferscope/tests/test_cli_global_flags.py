"""Tests for the global `--version` and `--output-format` flags on the CLI.

Closes sub-bugs E and F from
`improvements/easyinference/bugs/cli_minor_correctness_pile.md`.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from inferscope.cli import app

runner = CliRunner()


# ----------------------------------------------------------------------------
# --version (sub-bug E)
# ----------------------------------------------------------------------------


def test_cli_version_flag_prints_a_version_and_exits() -> None:
    """`inferscope --version` should print a version line and exit cleanly."""
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "inferscope" in result.stdout
    # Should print SOMETHING that looks like version info, not just "inferscope"
    assert len(result.stdout.strip()) > len("inferscope")


def test_cli_version_flag_does_not_invoke_subcommand() -> None:
    """`--version` is eager — it should fire before any subcommand resolution."""
    # Combining --version with an invalid subcommand should still succeed,
    # because --version exits before subcommand parsing.
    result = runner.invoke(app, ["--version", "this-command-does-not-exist"])
    assert result.exit_code == 0
    assert "inferscope" in result.stdout


# ----------------------------------------------------------------------------
# --output-format json (sub-bug F)
# ----------------------------------------------------------------------------


def test_cli_output_format_json_emits_parseable_json(monkeypatch) -> None:
    """`--output-format json` should produce valid JSON on stdout instead of rich output."""

    async def fake_profile_runtime(*args, **kwargs):
        del args, kwargs
        return {
            "summary": "Runtime profile ok",
            "confidence": 0.9,
            "endpoint": "http://localhost:8000",
            "nested": {"key": "value", "number": 42},
        }

    monkeypatch.setattr("inferscope.cli_profiling.profile_runtime", fake_profile_runtime)

    result = runner.invoke(
        app,
        ["--output-format", "json", "profile-runtime", "http://localhost:8000"],
    )

    assert result.exit_code == 0, result.stdout
    # Parse the entire stdout as JSON
    parsed = json.loads(result.stdout)
    assert parsed["summary"] == "Runtime profile ok"
    assert parsed["confidence"] == 0.9
    assert parsed["nested"] == {"key": "value", "number": 42}


def test_cli_output_format_pretty_is_default(monkeypatch) -> None:
    """Without --output-format, the default is pretty (rich console output)."""

    async def fake_profile_runtime(*args, **kwargs):
        del args, kwargs
        return {"summary": "Runtime profile ok", "confidence": 0.9}

    monkeypatch.setattr("inferscope.cli_profiling.profile_runtime", fake_profile_runtime)

    result = runner.invoke(app, ["profile-runtime", "http://localhost:8000"])

    assert result.exit_code == 0
    # Pretty output includes the bold "Runtime profile ok" text without a JSON wrapper
    assert "Runtime profile ok" in result.stdout


def test_cli_output_format_invalid_value_rejected() -> None:
    """`--output-format yaml` (or anything not pretty/json) must error cleanly.

    Newer Typer/Click routes BadParameter errors to stderr, so we only check
    that the exit code is non-zero. The validation message text itself is
    asserted via the unit test below on the callback function.
    """
    result = runner.invoke(app, ["--output-format", "yaml", "profile-runtime", "http://localhost:8000"])
    assert result.exit_code != 0


def test_global_options_callback_rejects_unknown_format_directly() -> None:
    """Direct unit test of the validation logic — independent of Typer test runner."""
    import typer

    from inferscope.cli import _global_options

    with pytest.raises(typer.BadParameter, match="must be 'pretty' or 'json'"):
        _global_options(version=False, output_format="yaml")


# ----------------------------------------------------------------------------
# Sub-bug A regression — _parse_json_option moved to shared module
# ----------------------------------------------------------------------------


def test_parse_json_option_lives_in_shared_module() -> None:
    """`_parse_json_option` should be importable from `inferscope._cli_helpers`,
    not duplicated across the 3 CLI modules."""
    from inferscope._cli_helpers import parse_json_option

    assert parse_json_option('{"foo": "bar"}', option_name="test") == {"foo": "bar"}
    assert parse_json_option("", option_name="test") is None
    assert parse_json_option("   ", option_name="test") is None


def test_parse_json_option_rejects_invalid_json() -> None:
    """Invalid JSON should raise typer.BadParameter."""
    import typer

    from inferscope._cli_helpers import parse_json_option

    with pytest.raises(typer.BadParameter, match="must be valid JSON"):
        parse_json_option("not json", option_name="test")


def test_parse_json_option_rejects_non_object_json() -> None:
    """A JSON array or scalar should raise typer.BadParameter."""
    import typer

    from inferscope._cli_helpers import parse_json_option

    with pytest.raises(typer.BadParameter, match="must be a JSON object"):
        parse_json_option("[1, 2, 3]", option_name="test")
    with pytest.raises(typer.BadParameter, match="must be a JSON object"):
        parse_json_option('"a string"', option_name="test")
