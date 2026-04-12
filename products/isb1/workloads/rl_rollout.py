"""RL rollout workload generator for ISB-1 benchmarks.

Produces GRPO/PPO-style rollout request traces that mimic the inference
calls made during RL post-training. Each request is a prompt drawn from
a training distribution, with variable-length completions following
typical GRPO reward-conditioned output distributions.

Key characteristics vs other workload families:
- Batch-oriented: requests arrive in synchronized batches (not Poisson)
- Variable max_tokens per request (P_max scheduling problem)
- Completions are often short (easy prompts) or very long (hard prompts)
- High prefix reuse within a batch (shared system prompt)
- Log-prob capture is critical (training signal)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from workloads.base import Request, WorkloadGenerator, _new_request_id

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question concisely and accurately. "
    "Show your reasoning step by step."
)

_MATH_TEMPLATES = [
    "Solve: {a} × {b} + {c} = ?",
    "What is the derivative of x^{n} + {a}x^2?",
    "If f(x) = {a}x² + {b}x + {c}, find f({d}).",
    "Simplify: ({a}/{b}) × ({c}/{d})",
    "A train travels {a} km in {b} hours. What is its speed in m/s?",
    "Find the GCD of {a} and {b}.",
    "What is {a}^{n} mod {b}?",
    "How many ways can you choose {n} items from {a} items?",
]

_CODE_TEMPLATES = [
    "Write a Python function to {task}.",
    "Debug this code: `{code}`. What's wrong?",
    "Optimize this algorithm: {task}. Current complexity is O(n²).",
    "Write a unit test for a function that {task}.",
]

_CODE_TASKS = [
    "reverse a linked list",
    "find the longest palindromic substring",
    "implement a LRU cache",
    "merge two sorted arrays in place",
    "detect a cycle in a directed graph",
    "serialize and deserialize a binary tree",
    "find the kth largest element in an unsorted array",
    "implement a thread-safe queue",
    "compute the edit distance between two strings",
    "validate a balanced parentheses string",
]

_REASONING_TEMPLATES = [
    "Explain why {topic} in exactly {n} sentences.",
    "Compare {topic_a} and {topic_b}. Which is better for {use_case}?",
    "What are the tradeoffs of {topic}?",
    "Step by step, solve: {problem}",
]

_TOPICS = [
    "gradient descent converges",
    "transformers use attention",
    "TCP guarantees ordering",
    "B-trees are used in databases",
    "hash tables have O(1) average lookup",
    "floating point is not associative",
    "cache eviction matters for performance",
    "RL training needs rollouts",
]


class RLRolloutGenerator(WorkloadGenerator):
    """Generate GRPO/PPO-style rollout batches.

    Parameters:
        seed: Random seed for reproducibility.
        batch_size: Number of prompts per rollout batch.
        pmax_distribution: Distribution of max_tokens per request.
            "fixed" = constant P_max, "bimodal" = realistic easy/hard split.
        pmax_fixed: Fixed max_tokens when distribution is "fixed".
        pmax_easy: Max tokens for easy prompts in bimodal mode.
        pmax_hard: Max tokens for hard prompts in bimodal mode.
        easy_fraction: Fraction of prompts that are "easy" in bimodal mode.
    """

    def __init__(
        self,
        seed: int = 42,
        batch_size: int = 32,
        pmax_distribution: str = "bimodal",
        pmax_fixed: int = 2048,
        pmax_easy: int = 256,
        pmax_hard: int = 2048,
        easy_fraction: float = 0.6,
    ) -> None:
        super().__init__(seed=seed)
        self.batch_size = batch_size
        self.pmax_distribution = pmax_distribution
        self.pmax_fixed = pmax_fixed
        self.pmax_easy = pmax_easy
        self.pmax_hard = pmax_hard
        self.easy_fraction = easy_fraction

    def _sample_pmax(self) -> int:
        if self.pmax_distribution == "fixed":
            return self.pmax_fixed
        # Bimodal: easy prompts get short budgets, hard get long
        if self.rng.random() < self.easy_fraction:
            return int(self.rng.integers(128, self.pmax_easy + 1))
        return int(self.rng.integers(self.pmax_easy, self.pmax_hard + 1))

    def _generate_math_prompt(self) -> str:
        template = self.rng.choice(_MATH_TEMPLATES)
        return template.format(
            a=int(self.rng.integers(2, 100)),
            b=int(self.rng.integers(2, 100)),
            c=int(self.rng.integers(1, 50)),
            d=int(self.rng.integers(1, 20)),
            n=int(self.rng.integers(2, 8)),
        )

    def _generate_code_prompt(self) -> str:
        template = self.rng.choice(_CODE_TEMPLATES)
        task = self.rng.choice(_CODE_TASKS)
        return template.format(task=task, code=f"def solve(): return {int(self.rng.integers(1, 100))}")

    def _generate_reasoning_prompt(self) -> str:
        template = self.rng.choice(_REASONING_TEMPLATES)
        topic = self.rng.choice(_TOPICS)
        return template.format(
            topic=topic,
            topic_a=self.rng.choice(_TOPICS),
            topic_b=self.rng.choice(_TOPICS),
            use_case="production inference",
            problem=self._generate_math_prompt(),
            n=int(self.rng.integers(3, 8)),
        )

    def _generate_prompt(self) -> str:
        kind = self.rng.choice(["math", "code", "reasoning"], p=[0.4, 0.3, 0.3])
        if kind == "math":
            return self._generate_math_prompt()
        if kind == "code":
            return self._generate_code_prompt()
        return self._generate_reasoning_prompt()

    def generate(self, num_requests: int) -> list[Request]:
        requests: list[Request] = []
        num_batches = max(1, num_requests // self.batch_size)
        remaining = num_requests

        for batch_idx in range(num_batches):
            batch_count = min(self.batch_size, remaining)
            batch_id = f"batch_{batch_idx:04d}"

            for i in range(batch_count):
                pmax = self._sample_pmax()
                prompt = self._generate_prompt()

                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]

                requests.append(Request(
                    request_id=_new_request_id(self.rng),
                    messages=messages,
                    expected_output_tokens=pmax,
                    session_id=batch_id,
                    metadata={
                        "workload_type": "rl_rollout",
                        "batch_id": batch_id,
                        "batch_index": i,
                        "pmax": pmax,
                        "pmax_distribution": self.pmax_distribution,
                        "prompt_type": "mixed",
                    },
                ))
                remaining -= 1

            if remaining <= 0:
                break

        return requests
