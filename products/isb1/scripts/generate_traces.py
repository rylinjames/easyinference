#!/usr/bin/env python3
"""generate_traces.py — materialize ISB-1 workload traces as JSONL files."""

from __future__ import annotations

from pathlib import Path

import click

from harness.paths import (
    default_config_root,
    default_traces_root,
    resolve_existing_path,
    resolve_path,
)
from workloads.materialize import (
    default_request_count,
    load_workload_config,
    materialize_requests,
    save_requests,
)


@click.command("generate-traces")
@click.option(
    "--config-root",
    type=click.Path(path_type=Path),
    default=default_config_root(),
    show_default=False,
    callback=lambda _ctx, _param, value: resolve_existing_path(value),
    help="Root directory for ISB-1 config files. Defaults to the product-local configs/ tree.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=default_traces_root(),
    show_default=False,
    callback=lambda _ctx, _param, value: resolve_path(value),
    help="Output directory for generated traces.",
)
@click.option(
    "--num-requests",
    type=int,
    default=None,
    help="Override the configured request-pool size for every workload.",
)
@click.option(
    "--workload",
    "workload_filter",
    type=str,
    default=None,
    help="Generate a trace for one workload only.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be generated without writing.")
def main(
    config_root: Path,
    output_dir: Path,
    num_requests: int | None,
    workload_filter: str | None,
    dry_run: bool,
) -> None:
    """Pre-generate ISB-1 request traces from canonical workload configs."""
    workload_dir = config_root / "workloads"
    if not workload_dir.is_dir():
        click.echo(f"Workload config directory not found: {workload_dir}", err=True)
        raise SystemExit(1)

    workload_paths = sorted(workload_dir.glob("*.yaml"))
    if not workload_paths:
        click.echo("No workload configs found.", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(workload_paths)} workload config(s)")
    click.echo(f"Output directory: {output_dir}")
    if num_requests is not None:
        click.echo(f"Request-pool override: {num_requests}")
    click.echo("")

    generated = 0
    errors = 0

    for workload_path in workload_paths:
        workload_cfg, _ = load_workload_config(workload_path.stem, config_root=config_root)
        workload_name = workload_cfg.get("workload_name", workload_path.stem)
        if workload_filter and workload_name != workload_filter:
            continue

        resolved_request_count = num_requests if num_requests is not None else default_request_count(workload_cfg)
        trace_path = output_dir / f"{workload_name}.jsonl"

        click.echo(f"── {workload_name} ({workload_path.name}) ──────────────────")
        click.echo(f"  Requests  : {resolved_request_count}")
        click.echo(f"  Output    : {trace_path}")

        if dry_run:
            generated += 1
            continue

        try:
            requests = materialize_requests(
                workload_name,
                config_root=config_root,
                num_requests=resolved_request_count,
            )
            save_requests(requests, trace_path)
            click.echo(f"  Saved     : {trace_path} ({len(requests)} requests)")
            generated += 1
        except Exception as exc:  # noqa: BLE001
            click.echo(f"  ERROR: {exc}", err=True)
            errors += 1

    click.echo("")
    click.echo("────────────────────────────────────────────────")
    click.echo("Trace generation complete.")
    click.echo(f"  Generated : {generated}")
    click.echo(f"  Errors    : {errors}")
    click.echo(f"  Output    : {output_dir}")
    click.echo("────────────────────────────────────────────────")

    if errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
