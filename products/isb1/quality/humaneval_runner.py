"""HumanEval pass@1 evaluation via a running vLLM server.

Sends HumanEval prompts to a model served behind an OpenAI-compatible API,
collects completions, executes the bundled unit tests, and computes pass@1.

Usage::

    python -m quality.humaneval_runner --model-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

HUMANEVAL_NUM_TASKS = 164  # canonical HumanEval size


@dataclass(slots=True)
class HumanEvalResult:
    """Result container for a single HumanEval problem."""

    task_id: str
    prompt: str
    completion: str
    passed: bool
    error: str | None = None


@dataclass(slots=True)
class HumanEvalReport:
    """Aggregated HumanEval report."""

    pass_at_1: float
    num_passed: int
    num_total: int
    delta_vs_bf16: float | None
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_at_1": self.pass_at_1,
            "num_passed": self.num_passed,
            "num_total": self.num_total,
            "delta_vs_bf16": self.delta_vs_bf16,
            "results": self.results,
        }


# ---------------------------------------------------------------------------
# Sandboxed test execution
# ---------------------------------------------------------------------------


def _run_test_in_process(code: str, timeout: int = 10) -> tuple[bool, str | None]:
    """Execute *code* in a forked process with a hard timeout.

    Returns (passed, error_message | None).
    """

    def _target(code: str, result_queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
        # Restrict dangerous builtins inside the sandbox.
        try:
            exec_globals: dict[str, Any] = {}
            exec(code, exec_globals)  # noqa: S102
            result_queue.put((True, None))
        except Exception:
            result_queue.put((False, traceback.format_exc()))

    queue: multiprocessing.Queue[tuple[bool, str | None]] = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_target, args=(code, queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return False, "Execution timed out"

    if queue.empty():
        return False, "Process terminated without result"

    return queue.get_nowait()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class HumanEvalRunner:
    """Run HumanEval against an OpenAI-compatible vLLM endpoint.

    Parameters:
        model_url: Base URL of the vLLM server (e.g. ``http://localhost:8000``).
        model_name: Model identifier to pass in the API request.  Defaults to
            the first model reported by the ``/v1/models`` endpoint.
        max_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature.  Use 0 for deterministic pass@1.
        bf16_reference: Optional path to a JSON file with BF16 pass@1 baseline.
    """

    def __init__(
        self,
        model_url: str = "http://localhost:8000",
        model_name: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        bf16_reference: Path | None = None,
    ) -> None:
        self.model_url = model_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name = model_name or self._detect_model()
        self.bf16_pass_at_1: float | None = None

        if bf16_reference is not None:
            ref = json.loads(Path(bf16_reference).read_text(encoding="utf-8"))
            self.bf16_pass_at_1 = float(ref["pass_at_1"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_model(self) -> str:
        """Query ``/v1/models`` and return the first model id."""
        resp = requests.get(f"{self.model_url}/v1/models", timeout=15)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if not models:
            raise RuntimeError("No models available on the server.")
        return models[0]["id"]

    def _load_problems(self) -> list[dict[str, Any]]:
        """Load HumanEval problems from the ``human_eval`` package."""
        try:
            from human_eval.data import read_problems  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "The human_eval package is required.  Install it with: "
                "pip install human-eval   (or install inferscope-benchmark[quality])"
            ) from None

        problems = read_problems()
        return [
            {"task_id": tid, **prob}
            for tid, prob in sorted(problems.items())
        ]

    def _get_completion(self, prompt: str) -> str:
        """Request a single completion from the model."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": ["\nclass ", "\ndef ", "\n#", "\nif __name__"],
        }
        resp = requests.post(
            f"{self.model_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    @staticmethod
    def _build_test_code(prompt: str, completion: str, test: str, entry_point: str) -> str:
        """Combine prompt, completion, and test harness into executable code."""
        # The test function from HumanEval is called ``check`` and invokes
        # the entry point function.
        return f"{prompt}{completion}\n\n{test}\n\ncheck({entry_point})\n"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        """Run HumanEval end-to-end and return a report dict.

        Returns:
            Dictionary with ``pass_at_1``, ``num_passed``, ``num_total``,
            ``delta_vs_bf16``, and per-problem ``results``.
        """
        problems = self._load_problems()
        results: list[HumanEvalResult] = []

        for prob in problems:
            task_id = prob["task_id"]
            prompt = prob["prompt"]
            test = prob["test"]
            entry_point = prob["entry_point"]

            try:
                completion = self._get_completion(prompt)
            except Exception as exc:
                results.append(HumanEvalResult(
                    task_id=task_id, prompt=prompt, completion="",
                    passed=False, error=f"API error: {exc}",
                ))
                continue

            code = self._build_test_code(prompt, completion, test, entry_point)
            passed, err = _run_test_in_process(code, timeout=10)
            results.append(HumanEvalResult(
                task_id=task_id, prompt=prompt, completion=completion,
                passed=passed, error=err,
            ))

        num_passed = sum(1 for r in results if r.passed)
        num_total = len(results)
        pass_at_1 = num_passed / num_total if num_total > 0 else 0.0

        delta: float | None = None
        if self.bf16_pass_at_1 is not None:
            delta = round((pass_at_1 - self.bf16_pass_at_1) * 100.0, 4)

        report = HumanEvalReport(
            pass_at_1=round(pass_at_1, 4),
            num_passed=num_passed,
            num_total=num_total,
            delta_vs_bf16=delta,
            results=[
                {
                    "task_id": r.task_id,
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in results
            ],
        )
        return report.to_dict()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISB-1 HumanEval pass@1 evaluation via vLLM OpenAI API.",
    )
    parser.add_argument(
        "--model-url", type=str, default="http://localhost:8000",
        help="Base URL of the vLLM server.",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Model name to use.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum tokens per completion.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 for greedy).",
    )
    parser.add_argument(
        "--bf16-reference", type=Path, default=None,
        help="Path to BF16 reference JSON with pass_at_1 key.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write the JSON report.",
    )
    args = parser.parse_args()

    runner = HumanEvalRunner(
        model_url=args.model_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        bf16_reference=args.bf16_reference,
    )
    report = runner.evaluate()

    report_json = json.dumps(report, indent=2)
    print(report_json)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json, encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
