"""Regression coverage for the generated InferScope console entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_console_entrypoint_help_works_from_source_checkout() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cli_path = repo_root / ".venv" / "bin" / "inferscope"

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
