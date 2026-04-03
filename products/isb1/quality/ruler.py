"""RULER (needle-in-a-haystack) long-context evaluation.

Generates retrieval tasks at multiple context lengths to measure whether a
model can accurately locate and return information embedded in long contexts.

Usage::

    python -m quality.ruler --model-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_NUM_TASKS_PER_LENGTH = 10
RULER_SEED = 42

# Approximate chars-per-token ratio for English prose (conservative).
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeedleTask:
    """A single needle-in-a-haystack retrieval task."""

    task_id: str
    context_length: int
    needle: str
    needle_position: float  # 0.0 = beginning, 1.0 = end
    haystack: str
    prompt: str
    expected_answer: str


@dataclass(slots=True)
class RULERReport:
    """Aggregated RULER evaluation report."""

    overall_accuracy: float
    accuracy_by_length: dict[int, float]
    num_correct: int
    num_total: int
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_accuracy": self.overall_accuracy,
            "accuracy_by_length": {
                str(k): v for k, v in self.accuracy_by_length.items()
            },
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def _generate_filler_text(rng: random.Random, num_chars: int) -> str:
    """Generate pseudo-random filler text of approximately *num_chars* characters.

    Uses a fixed vocabulary of common English words so the haystack reads as
    plausible low-entropy prose rather than random character noise.
    """
    words = [
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
        "was", "on", "are", "be", "with", "as", "at", "this", "have", "from",
        "or", "an", "by", "not", "but", "what", "all", "were", "when", "we",
        "there", "can", "had", "has", "its", "more", "if", "will", "each",
        "about", "up", "out", "them", "then", "she", "many", "some", "so",
        "these", "would", "other", "into", "been", "now", "could", "time",
        "very", "may", "no", "just", "over", "such", "new", "also", "any",
        "after", "use", "how", "our", "work", "first", "well", "way",
        "even", "because", "good", "give", "most", "find", "here", "thing",
        "long", "make", "look", "come", "go", "day", "still", "between",
        "name", "should", "much", "through", "great", "back", "old", "where",
    ]
    parts: list[str] = []
    length = 0
    while length < num_chars:
        word = rng.choice(words)
        parts.append(word)
        length += len(word) + 1  # +1 for the space
    return " ".join(parts)[:num_chars]


def _generate_needle(rng: random.Random) -> tuple[str, str]:
    """Generate a unique needle fact and its expected answer.

    Returns (needle_sentence, expected_answer).
    """
    secret = uuid.UUID(int=rng.getrandbits(128)).hex[:8]
    city = rng.choice([
        "Paris", "Tokyo", "Sydney", "Cairo", "Lima", "Oslo", "Seoul",
        "Dublin", "Prague", "Zurich", "Lisbon", "Vienna", "Athens",
        "Helsinki", "Warsaw", "Nairobi", "Bangkok", "Jakarta",
    ])
    needle = f"The special code for {city} is {secret}."
    return needle, secret


def generate_tasks(
    context_lengths: list[int] | None = None,
    num_tasks_per_length: int = DEFAULT_NUM_TASKS_PER_LENGTH,
    seed: int = RULER_SEED,
) -> list[NeedleTask]:
    """Generate a suite of needle-in-a-haystack tasks.

    Parameters:
        context_lengths: List of target context lengths in tokens.
        num_tasks_per_length: Number of tasks to create per length.
        seed: Random seed for reproducibility.

    Returns:
        List of :class:`NeedleTask` instances.
    """
    if context_lengths is None:
        context_lengths = DEFAULT_CONTEXT_LENGTHS

    rng = random.Random(seed)
    tasks: list[NeedleTask] = []

    for ctx_len in context_lengths:
        for task_idx in range(num_tasks_per_length):
            needle_text, expected = _generate_needle(rng)

            # Place needle at a random depth in the context.
            position = rng.random()

            # Build the haystack around the needle.
            total_chars = ctx_len * CHARS_PER_TOKEN
            needle_chars = len(needle_text)
            filler_chars = max(total_chars - needle_chars, 0)

            before_chars = int(filler_chars * position)
            after_chars = filler_chars - before_chars

            before_text = _generate_filler_text(rng, before_chars)
            after_text = _generate_filler_text(rng, after_chars)

            haystack = f"{before_text} {needle_text} {after_text}"

            # Extract the city from the needle for the question.
            city = needle_text.split("for ")[1].split(" is ")[0]
            prompt = (
                f"Read the following text carefully and answer the question at the end.\n\n"
                f"---BEGIN TEXT---\n{haystack}\n---END TEXT---\n\n"
                f"Question: What is the special code for {city}?\n"
                f"Answer with ONLY the code, nothing else."
            )

            tasks.append(NeedleTask(
                task_id=f"ruler_{ctx_len}_{task_idx}",
                context_length=ctx_len,
                needle=needle_text,
                needle_position=round(position, 4),
                haystack=haystack,
                prompt=prompt,
                expected_answer=expected,
            ))

    return tasks


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RULEREvaluator:
    """Run RULER needle-in-a-haystack evaluation against a vLLM endpoint.

    Parameters:
        model_url: Base URL of the OpenAI-compatible server.
        model_name: Model identifier.  Auto-detected if *None*.
        context_lengths: List of context lengths to test (in tokens).
        num_tasks_per_length: Number of tasks per context length.
        seed: Random seed for task generation.
    """

    def __init__(
        self,
        model_url: str = "http://localhost:8000",
        model_name: str | None = None,
        context_lengths: list[int] | None = None,
        num_tasks_per_length: int = DEFAULT_NUM_TASKS_PER_LENGTH,
        seed: int = RULER_SEED,
    ) -> None:
        self.model_url = model_url.rstrip("/")
        self.model_name = model_name or self._detect_model()
        self.context_lengths = context_lengths or DEFAULT_CONTEXT_LENGTHS
        self.num_tasks_per_length = num_tasks_per_length
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_model(self) -> str:
        resp = requests.get(f"{self.model_url}/v1/models", timeout=15)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if not models:
            raise RuntimeError("No models available on the server.")
        return models[0]["id"]

    def _query_model(self, prompt: str) -> str:
        """Send a chat completion and return the assistant response."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{self.model_url}/v1/chat/completions",
            json=payload,
            timeout=300,  # long contexts may be slow
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _check_answer(response: str, expected: str) -> bool:
        """Check whether *expected* appears in the model *response*."""
        return expected.lower() in response.strip().lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        """Generate tasks, query the model, and return accuracy report.

        Returns:
            Dictionary with ``overall_accuracy``, ``accuracy_by_length``,
            ``num_correct``, ``num_total``, and ``details``.
        """
        tasks = generate_tasks(
            context_lengths=self.context_lengths,
            num_tasks_per_length=self.num_tasks_per_length,
            seed=self.seed,
        )

        correct_by_length: dict[int, int] = {}
        total_by_length: dict[int, int] = {}
        details: list[dict[str, Any]] = []
        total_correct = 0

        for task in tasks:
            ctx_len = task.context_length
            total_by_length[ctx_len] = total_by_length.get(ctx_len, 0) + 1

            try:
                response = self._query_model(task.prompt)
            except Exception as exc:
                details.append({
                    "task_id": task.task_id,
                    "context_length": ctx_len,
                    "passed": False,
                    "needle_position": task.needle_position,
                    "error": str(exc),
                })
                continue

            passed = self._check_answer(response, task.expected_answer)
            if passed:
                total_correct += 1
                correct_by_length[ctx_len] = correct_by_length.get(ctx_len, 0) + 1

            details.append({
                "task_id": task.task_id,
                "context_length": ctx_len,
                "passed": passed,
                "needle_position": task.needle_position,
                "model_response": response.strip()[:200],
                "expected": task.expected_answer,
            })

        num_total = len(tasks)
        overall_accuracy = total_correct / num_total if num_total > 0 else 0.0

        accuracy_by_length: dict[int, float] = {}
        for ctx_len in sorted(total_by_length):
            total = total_by_length[ctx_len]
            correct = correct_by_length.get(ctx_len, 0)
            accuracy_by_length[ctx_len] = round(correct / total, 4) if total else 0.0

        report = RULERReport(
            overall_accuracy=round(overall_accuracy, 4),
            accuracy_by_length=accuracy_by_length,
            num_correct=total_correct,
            num_total=num_total,
            details=details,
        )
        return report.to_dict()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISB-1 RULER needle-in-a-haystack long-context evaluation.",
    )
    parser.add_argument(
        "--model-url", type=str, default="http://localhost:8000",
        help="Base URL of the vLLM / OpenAI-compatible server.",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Model name.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=None,
        help="Context lengths to test (in tokens).  "
             "Defaults to 4k, 8k, 16k, 32k, 64k, 128k.",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=DEFAULT_NUM_TASKS_PER_LENGTH,
        help="Number of tasks per context length.",
    )
    parser.add_argument(
        "--seed", type=int, default=RULER_SEED,
        help="Random seed for task generation.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write the JSON report.",
    )
    args = parser.parse_args()

    evaluator = RULEREvaluator(
        model_url=args.model_url,
        model_name=args.model_name,
        context_lengths=args.context_lengths,
        num_tasks_per_length=args.num_tasks,
        seed=args.seed,
    )
    report = evaluator.evaluate()

    report_json = json.dumps(report, indent=2)
    print(report_json)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json, encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
