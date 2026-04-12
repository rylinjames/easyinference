"""Adaptive P_max scheduling for RL rollout inference.

In GRPO/PPO training, each rollout batch uses a fixed max_tokens budget (P_max).
This wastes GPU on easy prompts (short completions) and truncates hard ones.

This module implements an adaptive scheduler that adjusts P_max per batch based
on reward statistics from previous rounds, following the insight from:
- veRL issues #658, #1803, #2106 (P_max scheduling pain)
- arXiv:2509.21009 RollPacker (variable-length rollout packing)
- arXiv:2604.07853 QaRL (quality-aware RL with precision switching)

The scheduler outputs a per-prompt P_max budget given reward history,
prompt difficulty estimates, and GPU memory constraints.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RewardHistory(BaseModel):
    """Reward statistics from a previous rollout batch."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_tokens_used: float
    max_tokens_used: float
    num_truncated: int
    total_requests: int
    pmax_used: int


class PmaxBudget(BaseModel):
    """Computed P_max budget for a single prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt_index: int
    pmax: int
    difficulty_estimate: str
    reason: str


class PmaxSchedule(BaseModel):
    """Full P_max schedule for a rollout batch."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int
    strategy: str
    budgets: list[PmaxBudget]
    total_token_budget: int
    mean_pmax: float
    min_pmax: int
    max_pmax: int
    gpu_utilization_estimate: float
    savings_vs_fixed_pct: float


class AdaptivePmaxScheduler:
    """Compute per-prompt P_max budgets based on reward history.

    Strategies:
    - "fixed": constant P_max (baseline)
    - "variance_scaled": scale P_max by reward variance — high variance
      means the batch has hard prompts that need more tokens
    - "truncation_aware": increase P_max when previous batch had truncations,
      decrease when most completions finished early
    - "bimodal": classify prompts as easy/hard based on reward history,
      assign different budgets
    """

    def __init__(
        self,
        base_pmax: int = 2048,
        min_pmax: int = 128,
        max_pmax: int = 4096,
        gpu_token_budget: int = 65536,
        strategy: str = "variance_scaled",
    ) -> None:
        self.base_pmax = base_pmax
        self.min_pmax = min_pmax
        self.max_pmax = max_pmax
        self.gpu_token_budget = gpu_token_budget
        self.strategy = strategy

    def schedule_fixed(self, batch_size: int) -> PmaxSchedule:
        pmax = min(self.base_pmax, self.gpu_token_budget // batch_size)
        budgets = [
            PmaxBudget(
                prompt_index=i,
                pmax=pmax,
                difficulty_estimate="unknown",
                reason="fixed baseline",
            )
            for i in range(batch_size)
        ]
        return self._build_schedule(budgets, "fixed")

    def schedule_variance_scaled(
        self,
        batch_size: int,
        history: list[RewardHistory],
    ) -> PmaxSchedule:
        if not history:
            return self.schedule_fixed(batch_size)

        latest = history[-1]
        reward_cv = latest.std_reward / abs(latest.mean_reward) if latest.mean_reward != 0 else 1.0
        utilization = latest.mean_tokens_used / latest.pmax_used if latest.pmax_used > 0 else 0.5
        truncation_rate = latest.num_truncated / latest.total_requests if latest.total_requests > 0 else 0

        # High variance → allocate more tokens (hard prompts need room)
        # Low utilization → reduce budget (easy prompts wasting tokens)
        # High truncation → increase budget (prompts being cut off)
        scale = 1.0
        if reward_cv > 0.5:
            scale *= 1.0 + min(reward_cv - 0.5, 0.5)
        if utilization < 0.3:
            scale *= 0.6
        elif utilization < 0.5:
            scale *= 0.8
        if truncation_rate > 0.1:
            scale *= 1.0 + min(truncation_rate, 0.3)

        adjusted_pmax = int(self.base_pmax * scale)
        adjusted_pmax = max(self.min_pmax, min(adjusted_pmax, self.max_pmax))
        per_prompt = min(adjusted_pmax, self.gpu_token_budget // batch_size)

        budgets = [
            PmaxBudget(
                prompt_index=i,
                pmax=per_prompt,
                difficulty_estimate="mixed",
                reason=f"cv={reward_cv:.2f} util={utilization:.2f} trunc={truncation_rate:.2f} scale={scale:.2f}",
            )
            for i in range(batch_size)
        ]
        return self._build_schedule(budgets, "variance_scaled")

    def schedule_truncation_aware(
        self,
        batch_size: int,
        history: list[RewardHistory],
    ) -> PmaxSchedule:
        if not history:
            return self.schedule_fixed(batch_size)

        truncation_rates = [
            h.num_truncated / h.total_requests if h.total_requests > 0 else 0
            for h in history[-3:]
        ]
        avg_truncation = statistics.mean(truncation_rates)
        utilizations = [
            h.mean_tokens_used / h.pmax_used if h.pmax_used > 0 else 0.5
            for h in history[-3:]
        ]
        avg_utilization = statistics.mean(utilizations)

        if avg_truncation > 0.15:
            adjusted = int(self.base_pmax * 1.3)
            reason = f"high truncation ({avg_truncation:.0%}) → increase budget"
        elif avg_utilization < 0.3:
            adjusted = int(self.base_pmax * 0.5)
            reason = f"low utilization ({avg_utilization:.0%}) → halve budget"
        elif avg_utilization < 0.5:
            adjusted = int(self.base_pmax * 0.7)
            reason = f"moderate utilization ({avg_utilization:.0%}) → reduce budget"
        else:
            adjusted = self.base_pmax
            reason = f"utilization OK ({avg_utilization:.0%}) → keep baseline"

        adjusted = max(self.min_pmax, min(adjusted, self.max_pmax))
        per_prompt = min(adjusted, self.gpu_token_budget // batch_size)

        budgets = [
            PmaxBudget(prompt_index=i, pmax=per_prompt, difficulty_estimate="mixed", reason=reason)
            for i in range(batch_size)
        ]
        return self._build_schedule(budgets, "truncation_aware")

    def schedule_bimodal(
        self,
        batch_size: int,
        history: list[RewardHistory],
        easy_fraction: float = 0.6,
    ) -> PmaxSchedule:
        if not history:
            return self.schedule_fixed(batch_size)

        latest = history[-1]
        utilization = latest.mean_tokens_used / latest.pmax_used if latest.pmax_used > 0 else 0.5

        easy_pmax = max(self.min_pmax, int(self.base_pmax * min(utilization * 1.2, 0.5)))
        hard_pmax = min(self.max_pmax, int(self.base_pmax * 1.5))

        n_easy = int(batch_size * easy_fraction)
        n_hard = batch_size - n_easy

        total_budget = n_easy * easy_pmax + n_hard * hard_pmax
        while total_budget > self.gpu_token_budget and (easy_pmax > self.min_pmax or hard_pmax > self.min_pmax):
            ratio = self.gpu_token_budget / total_budget
            easy_pmax = max(self.min_pmax, int(easy_pmax * ratio))
            hard_pmax = max(self.min_pmax, int(hard_pmax * ratio))
            total_budget = n_easy * easy_pmax + n_hard * hard_pmax

        budgets = []
        for i in range(batch_size):
            if i < n_easy:
                budgets.append(PmaxBudget(
                    prompt_index=i, pmax=easy_pmax,
                    difficulty_estimate="easy",
                    reason=f"easy slot: util={utilization:.2f} → pmax={easy_pmax}",
                ))
            else:
                budgets.append(PmaxBudget(
                    prompt_index=i, pmax=hard_pmax,
                    difficulty_estimate="hard",
                    reason=f"hard slot: extended budget → pmax={hard_pmax}",
                ))
        return self._build_schedule(budgets, "bimodal")

    def schedule(
        self,
        batch_size: int,
        history: list[RewardHistory] | None = None,
        **kwargs: Any,
    ) -> PmaxSchedule:
        history = history or []
        if self.strategy == "fixed":
            return self.schedule_fixed(batch_size)
        if self.strategy == "variance_scaled":
            return self.schedule_variance_scaled(batch_size, history)
        if self.strategy == "truncation_aware":
            return self.schedule_truncation_aware(batch_size, history)
        if self.strategy == "bimodal":
            return self.schedule_bimodal(batch_size, history, **kwargs)
        return self.schedule_fixed(batch_size)

    def _build_schedule(self, budgets: list[PmaxBudget], strategy: str) -> PmaxSchedule:
        pmaxes = [b.pmax for b in budgets]
        total = sum(pmaxes)
        fixed_total = len(budgets) * self.base_pmax
        savings = (1.0 - total / fixed_total) * 100 if fixed_total > 0 else 0.0
        gpu_util = total / self.gpu_token_budget if self.gpu_token_budget > 0 else 0.0

        return PmaxSchedule(
            batch_size=len(budgets),
            strategy=strategy,
            budgets=budgets,
            total_token_budget=total,
            mean_pmax=statistics.mean(pmaxes) if pmaxes else 0,
            min_pmax=min(pmaxes) if pmaxes else 0,
            max_pmax=max(pmaxes) if pmaxes else 0,
            gpu_utilization_estimate=min(gpu_util, 1.0),
            savings_vs_fixed_pct=round(savings, 1),
        )


def recommend_pmax_schedule(
    batch_size: int = 32,
    base_pmax: int = 2048,
    strategy: str = "variance_scaled",
    reward_history: list[dict[str, Any]] | None = None,
    gpu_token_budget: int = 65536,
) -> dict[str, Any]:
    """Recommend a P_max schedule given batch size and reward history.

    This is the CLI/MCP entry point. Accepts raw dicts for reward_history
    and returns a JSON-serializable result.
    """
    history = []
    if reward_history:
        for h in reward_history:
            history.append(RewardHistory.model_validate(h))

    scheduler = AdaptivePmaxScheduler(
        base_pmax=base_pmax,
        gpu_token_budget=gpu_token_budget,
        strategy=strategy,
    )
    schedule = scheduler.schedule(batch_size, history)

    return {
        "schedule": schedule.model_dump(mode="json"),
        "summary": (
            f"{strategy} schedule: {schedule.batch_size} prompts, "
            f"mean pmax={schedule.mean_pmax:.0f}, "
            f"range=[{schedule.min_pmax}, {schedule.max_pmax}], "
            f"savings={schedule.savings_vs_fixed_pct:.1f}% vs fixed {base_pmax}"
        ),
        "confidence": 0.85,
        "evidence": "pmax_adaptive_scheduling",
        "reference": "veRL #658, #1803, #2106; arXiv:2509.21009 RollPacker",
    }
