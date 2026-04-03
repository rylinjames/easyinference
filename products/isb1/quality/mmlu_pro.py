"""MMLU-Pro accuracy evaluation via an OpenAI-compatible API.

Loads a reproducible 500-question subset (seed 42) of the MMLU-Pro benchmark,
sends each question to the model, parses the answer letter, and computes
accuracy.

Usage::

    python -m quality.mmlu_pro --model-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MMLU_PRO_SUBSET_SIZE = 500
MMLU_PRO_SEED = 42

# MMLU-Pro answer options are labelled A-J (up to 10 choices).
ANSWER_PATTERN = re.compile(r"(?:^|\b)([A-J])(?:\b|[.\s,)])", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MMLUQuestion:
    """A single MMLU-Pro multiple-choice question."""

    question_id: str
    question: str
    options: list[str]
    answer: str  # single letter A-J
    category: str


@dataclass(slots=True)
class MMLUProReport:
    """Aggregated MMLU-Pro evaluation report."""

    accuracy: float
    num_correct: int
    num_total: int
    accuracy_by_category: dict[str, float]
    delta_vs_bf16: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "accuracy_by_category": self.accuracy_by_category,
            "delta_vs_bf16": self.delta_vs_bf16,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class MMLUProEvaluator:
    """Evaluate a model on a 500-question MMLU-Pro subset.

    Parameters:
        model_url: Base URL of the vLLM / OpenAI-compatible server.
        model_name: Model identifier.  Auto-detected if *None*.
        data_path: Path to a JSONL file containing the full MMLU-Pro dataset.
            If *None*, the evaluator attempts to load from HuggingFace datasets.
        bf16_reference: Optional path to a JSON file with a BF16 accuracy
            baseline (key ``accuracy``).
    """

    def __init__(
        self,
        model_url: str = "http://localhost:8000",
        model_name: str | None = None,
        data_path: Path | None = None,
        bf16_reference: Path | None = None,
    ) -> None:
        self.model_url = model_url.rstrip("/")
        self.model_name = model_name or self._detect_model()
        self.data_path = data_path
        self.bf16_accuracy: float | None = None

        if bf16_reference is not None:
            ref = json.loads(Path(bf16_reference).read_text(encoding="utf-8"))
            self.bf16_accuracy = float(ref["accuracy"])

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

    def _load_dataset(self) -> list[dict[str, Any]]:
        """Load the full MMLU-Pro dataset from a local JSONL or HuggingFace."""
        if self.data_path is not None:
            records: list[dict[str, Any]] = []
            with open(self.data_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records

        # Fallback: load via HuggingFace datasets library.
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Either provide --data-path or install the `datasets` library: "
                "pip install datasets"
            ) from None

        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        return [dict(row) for row in ds]

    def _sample_subset(self, dataset: list[dict[str, Any]]) -> list[MMLUQuestion]:
        """Deterministically sample MMLU_PRO_SUBSET_SIZE questions."""
        rng = random.Random(MMLU_PRO_SEED)
        sampled = rng.sample(dataset, min(MMLU_PRO_SUBSET_SIZE, len(dataset)))

        questions: list[MMLUQuestion] = []
        for idx, row in enumerate(sampled):
            # MMLU-Pro rows have: question, options (list), answer (letter),
            # category, question_id (may vary by source).
            options = row.get("options", [])
            answer = row.get("answer", "")
            # Some dataset variants store the answer as an index.
            if isinstance(answer, int):
                answer = chr(ord("A") + answer)
            questions.append(MMLUQuestion(
                question_id=row.get("question_id", str(idx)),
                question=row["question"],
                options=options,
                answer=str(answer).strip().upper(),
                category=row.get("category", "unknown"),
            ))
        return questions

    def _format_prompt(self, q: MMLUQuestion) -> str:
        """Build a zero-shot multiple-choice prompt."""
        option_lines = "\n".join(
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(q.options)
        )
        return (
            f"Answer the following multiple-choice question. "
            f"Reply with ONLY the letter of the correct answer.\n\n"
            f"Question: {q.question}\n\n"
            f"{option_lines}\n\n"
            f"Answer:"
        )

    def _query_model(self, prompt: str) -> str:
        """Send a chat completion request and return the assistant content."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{self.model_url}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_answer(response: str) -> str | None:
        """Extract the first valid answer letter from the model response."""
        response = response.strip()
        # Check if the response is just a single letter.
        if len(response) == 1 and response.upper() in "ABCDEFGHIJ":
            return response.upper()
        match = ANSWER_PATTERN.search(response)
        return match.group(1).upper() if match else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        """Run the MMLU-Pro evaluation and return a report dict.

        Returns:
            Dictionary with ``accuracy``, ``num_correct``, ``num_total``,
            ``accuracy_by_category``, and ``delta_vs_bf16``.
        """
        dataset = self._load_dataset()
        questions = self._sample_subset(dataset)

        correct = 0
        category_correct: dict[str, int] = {}
        category_total: dict[str, int] = {}

        for q in questions:
            prompt = self._format_prompt(q)
            try:
                response = self._query_model(prompt)
            except Exception:
                # Count API errors as incorrect.
                category_total[q.category] = category_total.get(q.category, 0) + 1
                continue

            predicted = self._parse_answer(response)
            is_correct = predicted == q.answer

            category_total[q.category] = category_total.get(q.category, 0) + 1
            if is_correct:
                correct += 1
                category_correct[q.category] = category_correct.get(q.category, 0) + 1

        num_total = len(questions)
        accuracy = correct / num_total if num_total > 0 else 0.0

        accuracy_by_cat: dict[str, float] = {}
        for cat in sorted(category_total):
            cat_total = category_total[cat]
            cat_correct = category_correct.get(cat, 0)
            accuracy_by_cat[cat] = round(cat_correct / cat_total, 4) if cat_total else 0.0

        delta: float | None = None
        if self.bf16_accuracy is not None:
            delta = round((accuracy - self.bf16_accuracy) * 100.0, 4)

        report = MMLUProReport(
            accuracy=round(accuracy, 4),
            num_correct=correct,
            num_total=num_total,
            accuracy_by_category=accuracy_by_cat,
            delta_vs_bf16=delta,
        )
        return report.to_dict()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISB-1 MMLU-Pro accuracy evaluation (500-question subset, seed 42).",
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
        "--data-path", type=Path, default=None,
        help="Path to a local MMLU-Pro JSONL dataset file.",
    )
    parser.add_argument(
        "--bf16-reference", type=Path, default=None,
        help="Path to BF16 reference JSON with 'accuracy' key.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write the JSON report.",
    )
    args = parser.parse_args()

    evaluator = MMLUProEvaluator(
        model_url=args.model_url,
        model_name=args.model_name,
        data_path=args.data_path,
        bf16_reference=args.bf16_reference,
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
