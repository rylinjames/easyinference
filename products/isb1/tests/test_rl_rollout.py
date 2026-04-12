"""Tests for the RL rollout workload generator."""

from __future__ import annotations

from pathlib import Path

from workloads.rl_rollout import RLRolloutGenerator


class TestRLRolloutGenerator:
    def test_generates_correct_count(self):
        gen = RLRolloutGenerator(seed=42, batch_size=8)
        requests = gen.generate(24)
        assert len(requests) == 24

    def test_batch_grouping(self):
        gen = RLRolloutGenerator(seed=42, batch_size=8)
        requests = gen.generate(24)
        batches = set(r.session_id for r in requests)
        assert len(batches) == 3

    def test_bimodal_pmax_distribution(self):
        gen = RLRolloutGenerator(seed=42, batch_size=32, pmax_distribution="bimodal")
        requests = gen.generate(100)
        pmaxes = [r.metadata["pmax"] for r in requests]
        assert min(pmaxes) >= 128
        assert max(pmaxes) <= 2048
        easy = sum(1 for p in pmaxes if p <= 256)
        assert easy > 30  # ~60% should be easy

    def test_fixed_pmax_distribution(self):
        gen = RLRolloutGenerator(seed=42, batch_size=8, pmax_distribution="fixed", pmax_fixed=1024)
        requests = gen.generate(16)
        pmaxes = [r.metadata["pmax"] for r in requests]
        assert all(p == 1024 for p in pmaxes)

    def test_deterministic_with_seed(self):
        gen1 = RLRolloutGenerator(seed=123, batch_size=4)
        gen2 = RLRolloutGenerator(seed=123, batch_size=4)
        r1 = gen1.generate(8)
        r2 = gen2.generate(8)
        assert [r.messages[1]["content"] for r in r1] == [r.messages[1]["content"] for r in r2]

    def test_messages_have_system_and_user(self):
        gen = RLRolloutGenerator(seed=42, batch_size=4)
        requests = gen.generate(4)
        for r in requests:
            assert len(r.messages) == 2
            assert r.messages[0]["role"] == "system"
            assert r.messages[1]["role"] == "user"

    def test_metadata_fields(self):
        gen = RLRolloutGenerator(seed=42, batch_size=4)
        requests = gen.generate(4)
        for r in requests:
            assert "workload_type" in r.metadata
            assert r.metadata["workload_type"] == "rl_rollout"
            assert "batch_id" in r.metadata
            assert "pmax" in r.metadata
            assert "pmax_distribution" in r.metadata

    def test_save_and_load(self, tmp_path: Path):
        gen = RLRolloutGenerator(seed=42, batch_size=4)
        requests = gen.generate(8)
        path = tmp_path / "rl_rollout.jsonl"
        gen.save(requests, path)
        loaded = gen.load(path)
        assert len(loaded) == 8
        assert loaded[0].messages[1]["content"] == requests[0].messages[1]["content"]
