"""Shared helpers for the InferScope CLI command modules.

Centralises small utilities that were previously duplicated across
`cli_benchmarks.py`, `cli_experiments.py`, and `cli_profiling.py`.

Closes sub-bug A from `improvements/easyinference/bugs/cli_minor_correctness_pile.md`.
"""

from __future__ import annotations

import json
from typing import Any

import typer


def parse_json_option(raw: str, *, option_name: str) -> dict[str, Any] | None:
    """Parse a JSON-string CLI option into a dict, or return None if blank.

    Raises ``typer.BadParameter`` if the value is non-empty but invalid JSON
    or not a JSON object. Previously this exact function was duplicated in
    three CLI modules.
    """
    if not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{option_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise typer.BadParameter(f"{option_name} must be a JSON object")
    return {str(key): val for key, val in value.items()}
