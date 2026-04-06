"""Regression coverage for the generated InferScope console entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _resolve_console_entrypoint(repo_root: Path) -> Path | None:
    """Find the installed `inferscope` console script regardless of OS or
    venv layout (POSIX `bin/`, Windows `Scripts/`, and with or without the
    `.exe` suffix). Returns None if no matching binary is present — in
    that case the test skips, because the environment simply doesn't have
    an editable install to exercise."""
    candidates = [
        repo_root / ".venv" / "bin" / "inferscope",
        repo_root / ".venv" / "Scripts" / "inferscope",
        repo_root / ".venv" / "Scripts" / "inferscope.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def test_console_entrypoint_help_works_from_source_checkout() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cli_path = _resolve_console_entrypoint(repo_root)
    if cli_path is None:
        pytest.skip(
            "inferscope console script not found in .venv — editable "
            "install is not set up, nothing to exercise"
        )

    result = subprocess.run(
        [str(cli_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "profile-runtime" in result.stdout
    assert "serve" in result.stdout
    assert "No module named 'inferscope'" not in result.stderr
