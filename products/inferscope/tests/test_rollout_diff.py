"""Tests for the rollout-diff module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from inferscope.benchmarks.rollout_diff import (
    RolloutDiffArtifact,
    RolloutEntry,
    TokenDivergence,
    TokenLogProb,
    diff_request,
    diff_rollouts,
    load_rollout_log,
)


def _make_entry(
    request_id: str,
    tokens: list[tuple[int, int, str, float]],
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    engine: str = "verl",
) -> RolloutEntry:
    return RolloutEntry(
        request_id=request_id,
        prompt="What is 2+2?",
        completion="4",
        model=model,
        engine=engine,
        tokens=[
            TokenLogProb(position=pos, token_id=tid, token_text=txt, log_prob=lp)
            for pos, tid, txt, lp in tokens
        ],
    )


def _write_jsonl(entries: list[RolloutEntry], path: Path) -> None:
    with path.open("w") as f:
        for entry in entries:
            f.write(entry.model_dump_json() + "\n")


class TestDiffRequest:
    def test_clean_match(self):
        training = _make_entry("r1", [(0, 1, "4", -0.5), (1, 2, ".", -1.0)])
        serving = _make_entry("r1", [(0, 1, "4", -0.5), (1, 2, ".", -1.0)])
        result = diff_request(training, serving, threshold=0.01)
        assert result.severity == "clean"
        assert result.divergent_tokens == 0
        assert result.total_tokens == 2

    def test_drift_detected(self):
        training = _make_entry("r1", [(0, 1, "4", -0.5), (1, 2, ".", -1.0)])
        serving = _make_entry("r1", [(0, 1, "4", -0.52), (1, 2, ".", -1.0)])
        result = diff_request(training, serving, threshold=0.01)
        assert result.severity == "drift"
        assert result.divergent_tokens == 1
        assert result.top_divergences[0].position == 0
        assert abs(result.top_divergences[0].abs_delta - 0.02) < 1e-6

    def test_critical_large_delta(self):
        training = _make_entry("r1", [
            (0, 1, "!", -0.1),
            (1, 2, "!", -0.1),
            (2, 3, "!", -0.1),
        ])
        serving = _make_entry("r1", [
            (0, 1, "!", -5.0),
            (1, 2, "!", -5.0),
            (2, 3, "!", -5.0),
        ])
        result = diff_request(training, serving, threshold=0.01)
        assert result.severity == "critical"
        assert result.divergent_tokens == 3

    def test_partial_overlap(self):
        training = _make_entry("r1", [(0, 1, "a", -1.0), (1, 2, "b", -2.0), (2, 3, "c", -3.0)])
        serving = _make_entry("r1", [(0, 1, "a", -1.0), (1, 2, "b", -2.0)])
        result = diff_request(training, serving, threshold=0.01)
        assert result.total_tokens == 2

    def test_top_k_limit(self):
        tokens = [(i, i, f"t{i}", -1.0) for i in range(20)]
        training = _make_entry("r1", tokens)
        serving_tokens = [(i, i, f"t{i}", -1.0 - (0.1 * i)) for i in range(20)]
        serving = _make_entry("r1", serving_tokens)
        result = diff_request(training, serving, threshold=0.01, top_k=5)
        assert len(result.top_divergences) == 5


class TestDiffRollouts:
    def test_end_to_end(self, tmp_path: Path):
        training_entries = [
            _make_entry("r1", [(0, 1, "4", -0.5), (1, 2, ".", -1.0)]),
            _make_entry("r2", [(0, 3, "yes", -0.3), (1, 4, "!", -0.8)]),
        ]
        serving_entries = [
            _make_entry("r1", [(0, 1, "4", -0.52), (1, 2, ".", -1.0)], engine="vllm"),
            _make_entry("r2", [(0, 3, "yes", -0.3), (1, 4, "!", -5.0)], engine="vllm"),
        ]
        training_path = tmp_path / "training.jsonl"
        serving_path = tmp_path / "serving.jsonl"
        _write_jsonl(training_entries, training_path)
        _write_jsonl(serving_entries, serving_path)

        artifact = diff_rollouts(training_path, serving_path, threshold=0.01)

        assert isinstance(artifact, RolloutDiffArtifact)
        assert artifact.summary.total_requests == 2
        assert artifact.summary.divergent_requests == 2
        assert artifact.model == "Qwen/Qwen2.5-7B-Instruct"

    def test_missing_serving_request(self, tmp_path: Path):
        training_entries = [
            _make_entry("r1", [(0, 1, "a", -1.0)]),
            _make_entry("r2", [(0, 2, "b", -2.0)]),
        ]
        serving_entries = [
            _make_entry("r1", [(0, 1, "a", -1.0)], engine="vllm"),
        ]
        training_path = tmp_path / "training.jsonl"
        serving_path = tmp_path / "serving.jsonl"
        _write_jsonl(training_entries, training_path)
        _write_jsonl(serving_entries, serving_path)

        artifact = diff_rollouts(training_path, serving_path)
        assert artifact.summary.total_requests == 1

    def test_save_and_load(self, tmp_path: Path):
        training_entries = [_make_entry("r1", [(0, 1, "x", -1.0)])]
        serving_entries = [_make_entry("r1", [(0, 1, "x", -1.1)], engine="vllm")]
        _write_jsonl(training_entries, tmp_path / "t.jsonl")
        _write_jsonl(serving_entries, tmp_path / "s.jsonl")

        artifact = diff_rollouts(tmp_path / "t.jsonl", tmp_path / "s.jsonl")
        saved = artifact.save_json(tmp_path / "diff.json")
        loaded = json.loads(saved.read_text())
        assert loaded["artifact_type"] == "rollout_diff"
        assert loaded["summary"]["total_requests"] == 1


class TestLoadRolloutLog:
    def test_load_valid_jsonl(self, tmp_path: Path):
        entries = [_make_entry("r1", [(0, 1, "a", -1.0)])]
        path = tmp_path / "log.jsonl"
        _write_jsonl(entries, path)
        loaded = load_rollout_log(path)
        assert len(loaded) == 1
        assert loaded[0].request_id == "r1"

    def test_empty_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "log.jsonl"
        entry = _make_entry("r1", [(0, 1, "a", -1.0)])
        path.write_text(entry.model_dump_json() + "\n\n\n")
        loaded = load_rollout_log(path)
        assert len(loaded) == 1


class TestTokenDivergence:
    def test_relative_delta(self):
        div = TokenDivergence(
            position=0,
            token_id=1,
            token_text="x",
            training_log_prob=-2.0,
            serving_log_prob=-2.5,
            abs_delta=0.5,
            relative_delta=0.25,
        )
        assert div.relative_delta == 0.25

    def test_zero_training_logprob(self):
        div = TokenDivergence(
            position=0,
            token_id=1,
            token_text="x",
            training_log_prob=0.0,
            serving_log_prob=-0.5,
            abs_delta=0.5,
            relative_delta=None,
        )
        assert div.relative_delta is None
