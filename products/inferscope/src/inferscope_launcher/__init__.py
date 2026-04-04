"""Console entrypoint for InferScope.

Kept separate from the `inferscope` package so the generated `inferscope`
launcher does not need to import a top-level module with the same name as the
launcher script itself.
"""

from __future__ import annotations

from inferscope.cli import app


def main() -> None:
    """Run the Typer application."""
    app()
