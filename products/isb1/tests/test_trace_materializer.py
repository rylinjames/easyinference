"""Tests for shared workload materialization."""

from pathlib import Path

from workloads.base import Request
from workloads.materialize import (
    default_request_count,
    load_workload_config,
    materialize_requests,
    save_requests,
)


def test_default_request_count_reads_workload_config() -> None:
    workload_cfg, _ = load_workload_config("chat")
    assert default_request_count(workload_cfg) == 1000


def test_materialize_requests_respects_override() -> None:
    requests = materialize_requests("agent", num_requests=5)
    assert len(requests) == 5
    assert all(isinstance(request, Request) for request in requests)


def test_save_requests_writes_jsonl(tmp_path: Path) -> None:
    requests = materialize_requests("coding", num_requests=3)
    output_path = tmp_path / "coding.jsonl"
    saved_path = save_requests(requests, output_path)
    lines = saved_path.read_text(encoding="utf-8").splitlines()
    assert saved_path == output_path
    assert len(lines) == 3
