#!/usr/bin/env python3
"""validate_results.py — Validate ISB-1 benchmark result integrity.

Checks that:
  1. A manifest JSON exists for each result directory.
  2. Raw data JSONL files exist and are non-empty.
  3. A lockfile JSON exists.
  4. JSONL files are not corrupted (every line parses as valid JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import click

_PRODUCT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_ROOT = _PRODUCT_ROOT / "results"

# ── Validation primitives ────────────────────────────────────────────────


def check_manifest(result_dir: Path) -> list[str]:
    """Verify a manifest.json exists and contains required fields."""
    errors: list[str] = []
    manifest_path = result_dir / "manifest.json"

    if not manifest_path.exists():
        errors.append(f"Missing manifest: {manifest_path}")
        return errors

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        errors.append(f"Corrupt manifest JSON in {manifest_path}: {exc}")
        return errors

    required_fields = [
        "run_id",
        "gpu",
        "model",
        "workload",
        "mode",
        "quantization",
        "status",
    ]
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Manifest {manifest_path} missing or empty field: '{field}'")

    return errors


def check_lockfile(result_dir: Path) -> list[str]:
    """Verify a lockfile JSON exists and is valid JSON."""
    errors: list[str] = []

    # Try common lockfile names
    lockfile_names = ["lockfile.json", "lock.json", "reproducibility.json"]
    lockfile_path: Path | None = None
    for name in lockfile_names:
        candidate = result_dir / name
        if candidate.exists():
            lockfile_path = candidate
            break

    # Also check for any *lock*.json
    if lockfile_path is None:
        for p in result_dir.glob("*lock*.json"):
            lockfile_path = p
            break

    if lockfile_path is None:
        errors.append(f"Missing lockfile in {result_dir}")
        return errors

    try:
        with open(lockfile_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        errors.append(f"Corrupt lockfile JSON in {lockfile_path}: {exc}")
        return errors

    if not isinstance(data, dict):
        errors.append(f"Lockfile {lockfile_path} is not a JSON object")

    return errors


def check_raw_data(result_dir: Path) -> list[str]:
    """Verify raw JSONL data files exist and are not empty."""
    errors: list[str] = []

    jsonl_files = list(result_dir.glob("*.jsonl"))
    # Also check a raw/ subdirectory
    raw_subdir = result_dir / "raw"
    if raw_subdir.is_dir():
        jsonl_files.extend(raw_subdir.glob("*.jsonl"))

    if not jsonl_files:
        errors.append(f"No raw data JSONL files found in {result_dir}")
        return errors

    for jsonl_path in jsonl_files:
        if jsonl_path.stat().st_size == 0:
            errors.append(f"Empty JSONL file: {jsonl_path}")
            continue

        # Validate every line is parseable JSON
        line_errors = validate_jsonl(jsonl_path)
        errors.extend(line_errors)

    return errors


def validate_jsonl(jsonl_path: Path) -> list[str]:
    """Check that every line in a JSONL file is valid JSON."""
    errors: list[str] = []
    try:
        with open(jsonl_path, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue  # Blank lines are OK
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"Corrupt JSONL at {jsonl_path}:{line_num} — {exc}"
                    )
                    # Stop after 10 errors per file to avoid flooding
                    if len(errors) >= 10:
                        errors.append(
                            f"... (truncated, >10 errors in {jsonl_path})"
                        )
                        break
    except OSError as exc:
        errors.append(f"Cannot read {jsonl_path}: {exc}")

    return errors


def validate_result_dir(result_dir: Path) -> list[str]:
    """Run all validation checks on a single result directory."""
    all_errors: list[str] = []
    all_errors.extend(check_manifest(result_dir))
    all_errors.extend(check_lockfile(result_dir))
    all_errors.extend(check_raw_data(result_dir))
    return all_errors


def discover_result_dirs(results_root: Path) -> list[Path]:
    """Find all result directories under a root.

    A result directory is one that contains a manifest.json or any .jsonl
    file.
    """
    result_dirs: set[Path] = set()

    # Direct children that look like result dirs
    for p in results_root.iterdir():
        if p.is_dir():
            if (p / "manifest.json").exists():
                result_dirs.add(p)
            elif list(p.glob("*.jsonl")):
                result_dirs.add(p)

    # Recurse into raw/ if it exists
    raw_dir = results_root / "raw"
    if raw_dir.is_dir():
        for p in raw_dir.iterdir():
            if p.is_dir():
                result_dirs.add(p)

    # If the root itself has manifests / jsonl, include it
    if (results_root / "manifest.json").exists() or list(results_root.glob("*.jsonl")):
        result_dirs.add(results_root)

    return sorted(result_dirs)


# ── CLI ──────────────────────────────────────────────────────────────────


@click.command("validate-results")
@click.argument(
    "results_path",
    type=click.Path(exists=True),
    default=str(_DEFAULT_RESULTS_ROOT),
    show_default=False,
)
@click.option(
    "--strict",
    is_flag=True,
    help="Exit with code 1 on any warning (not just errors).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Print per-directory status even when no errors found.",
)
def main(results_path: str, strict: bool, verbose: bool) -> None:
    """Validate ISB-1 benchmark result integrity.

    RESULTS_PATH is the root results directory. Defaults to the product-local results/ tree.
    """
    root = Path(results_path).resolve()
    click.echo(f"Validating results in: {root}")
    click.echo("")

    result_dirs = discover_result_dirs(root)

    if not result_dirs:
        click.echo("No result directories found. Nothing to validate.")
        return

    click.echo(f"Found {len(result_dirs)} result directory(ies)")
    click.echo("")

    total_errors = 0
    dirs_with_errors = 0

    for rd in result_dirs:
        rel = rd.relative_to(root) if rd != root else Path(".")
        errors = validate_result_dir(rd)

        if errors:
            dirs_with_errors += 1
            total_errors += len(errors)
            click.echo(f"FAIL  {rel}/")
            for err in errors:
                click.echo(f"  - {err}")
        elif verbose:
            click.echo(f"OK    {rel}/")

    click.echo("")
    click.echo("────────────────────────────────────────────────")
    click.echo(f"Directories checked : {len(result_dirs)}")
    click.echo(f"Directories with errors : {dirs_with_errors}")
    click.echo(f"Total errors : {total_errors}")
    click.echo("────────────────────────────────────────────────")

    if total_errors > 0:
        click.echo("")
        click.echo("Result validation FAILED.")
        raise SystemExit(1)
    else:
        click.echo("")
        click.echo("All results passed validation.")


if __name__ == "__main__":
    main()
