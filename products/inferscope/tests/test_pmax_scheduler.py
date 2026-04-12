"""Tests for the adaptive P_max scheduler."""

from __future__ import annotations

from inferscope.tools.pmax_scheduler import (
    AdaptivePmaxScheduler,
    RewardHistory,
    recommend_pmax_schedule,
)


def _make_history(
    mean_reward: float = 0.5,
    std_reward: float = 0.2,
    mean_tokens: float = 500,
    max_tokens: float = 2000,
    num_truncated: int = 0,
    total: int = 32,
    pmax: int = 2048,
) -> RewardHistory:
    return RewardHistory(
        batch_id="batch_0000",
        mean_reward=mean_reward,
        std_reward=std_reward,
        min_reward=0.0,
        max_reward=1.0,
        mean_tokens_used=mean_tokens,
        max_tokens_used=max_tokens,
        num_truncated=num_truncated,
        total_requests=total,
        pmax_used=pmax,
    )


class TestFixedSchedule:
    def test_returns_constant_pmax(self):
        s = AdaptivePmaxScheduler(base_pmax=2048, strategy="fixed")
        schedule = s.schedule(32)
        assert all(b.pmax == 2048 for b in schedule.budgets)
        assert schedule.savings_vs_fixed_pct == 0.0

    def test_respects_gpu_budget(self):
        s = AdaptivePmaxScheduler(base_pmax=4096, gpu_token_budget=8192, strategy="fixed")
        schedule = s.schedule(32)
        assert all(b.pmax == 256 for b in schedule.budgets)


class TestVarianceScaled:
    def test_high_variance_increases_pmax(self):
        s = AdaptivePmaxScheduler(base_pmax=1024, gpu_token_budget=131072, strategy="variance_scaled")
        history = [_make_history(std_reward=0.8, mean_reward=0.5, mean_tokens=800, pmax=1024)]
        schedule = s.schedule(32, history)
        assert schedule.mean_pmax > 1024

    def test_low_utilization_decreases_pmax(self):
        s = AdaptivePmaxScheduler(base_pmax=2048, strategy="variance_scaled")
        history = [_make_history(std_reward=0.1, mean_tokens=200)]
        schedule = s.schedule(32, history)
        assert schedule.mean_pmax < 2048

    def test_high_truncation_increases_pmax(self):
        s = AdaptivePmaxScheduler(base_pmax=2048, strategy="variance_scaled")
        history = [_make_history(num_truncated=10, total=32, mean_tokens=1800)]
        schedule = s.schedule(32, history)
        assert schedule.mean_pmax >= 2048

    def test_no_history_falls_back_to_fixed(self):
        s = AdaptivePmaxScheduler(base_pmax=2048, strategy="variance_scaled")
        schedule = s.schedule(32, [])
        assert all(b.pmax == 2048 for b in schedule.budgets)


class TestTruncationAware:
    def test_many_truncations_increases(self):
        s = AdaptivePmaxScheduler(base_pmax=1024, gpu_token_budget=131072, strategy="truncation_aware")
        history = [_make_history(num_truncated=8, total=32, mean_tokens=900, pmax=1024)]
        schedule = s.schedule(32, history)
        assert schedule.mean_pmax > 1024

    def test_low_util_decreases(self):
        s = AdaptivePmaxScheduler(base_pmax=2048, strategy="truncation_aware")
        history = [_make_history(mean_tokens=300)]
        schedule = s.schedule(32, history)
        assert schedule.mean_pmax < 2048


class TestBimodal:
    def test_easy_hard_split(self):
        s = AdaptivePmaxScheduler(base_pmax=2048, strategy="bimodal")
        history = [_make_history(mean_tokens=500)]
        schedule = s.schedule(10, history)
        easy = [b for b in schedule.budgets if b.difficulty_estimate == "easy"]
        hard = [b for b in schedule.budgets if b.difficulty_estimate == "hard"]
        assert len(easy) == 6
        assert len(hard) == 4
        assert easy[0].pmax < hard[0].pmax

    def test_respects_gpu_budget(self):
        s = AdaptivePmaxScheduler(base_pmax=2048, gpu_token_budget=4096, strategy="bimodal")
        history = [_make_history(mean_tokens=500)]
        schedule = s.schedule(32, history)
        assert schedule.total_token_budget <= 4096


class TestRecommendPmaxSchedule:
    def test_returns_dict(self):
        result = recommend_pmax_schedule(batch_size=16, strategy="fixed")
        assert "schedule" in result
        assert "summary" in result
        assert result["schedule"]["batch_size"] == 16

    def test_with_history(self):
        history = [_make_history().model_dump(mode="json")]
        result = recommend_pmax_schedule(
            batch_size=32,
            strategy="variance_scaled",
            reward_history=history,
        )
        assert result["schedule"]["strategy"] == "variance_scaled"

    def test_savings_positive_when_reducing(self):
        history = [_make_history(mean_tokens=200).model_dump(mode="json")]
        result = recommend_pmax_schedule(
            batch_size=32,
            strategy="variance_scaled",
            reward_history=history,
        )
        assert result["schedule"]["savings_vs_fixed_pct"] > 0
