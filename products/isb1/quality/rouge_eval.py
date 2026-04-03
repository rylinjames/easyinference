"""ROUGE-based quality evaluation against BF16 reference outputs.

Compares model outputs to BF16 reference completions and determines whether
quantised or optimised models maintain acceptable generation quality.

Usage::

    python -m quality.rouge_eval --reference path/to/bf16_refs --test path/to/test_outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ROUGEScores:
    """Aggregated ROUGE scores for a single metric variant."""

    precision: float
    recall: float
    fmeasure: float


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Full quality-gate report returned by :meth:`ROUGEEvaluator.evaluate`."""

    rouge_1: ROUGEScores
    rouge_2: ROUGEScores
    rouge_l: ROUGEScores
    delta_vs_bf16: float
    quality_gate: str  # PASS | MARGINAL | FAIL
    num_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "rouge_1": {"precision": self.rouge_1.precision, "recall": self.rouge_1.recall,
                        "fmeasure": self.rouge_1.fmeasure},
            "rouge_2": {"precision": self.rouge_2.precision, "recall": self.rouge_2.recall,
                        "fmeasure": self.rouge_2.fmeasure},
            "rouge_l": {"precision": self.rouge_l.precision, "recall": self.rouge_l.recall,
                        "fmeasure": self.rouge_l.fmeasure},
            "delta_vs_bf16": self.delta_vs_bf16,
            "quality_gate": self.quality_gate,
            "num_samples": self.num_samples,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ROUGEEvaluator:
    """Compare test model outputs against BF16 reference outputs using ROUGE.

    Parameters:
        reference_dir: Directory containing JSONL files with BF16 reference
            outputs.  Each line must be a JSON object with at least
            ``request_id`` and ``output`` keys.
    """

    def __init__(self, reference_dir: Path) -> None:
        self.reference_dir = Path(reference_dir)
        self.references: dict[str, str] = {}
        self._load_references()
        self._scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_references(self) -> None:
        """Load all JSONL files from *reference_dir* into ``self.references``."""
        if not self.reference_dir.exists():
            raise FileNotFoundError(
                f"Reference directory does not exist: {self.reference_dir}"
            )

        jsonl_files = sorted(self.reference_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No JSONL files found in {self.reference_dir}"
            )

        for path in jsonl_files:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    rid = record["request_id"]
                    self.references[rid] = record["output"]

    @staticmethod
    def _load_test_outputs(test_dir: Path) -> dict[str, str]:
        """Load test outputs from a directory of JSONL files."""
        test_dir = Path(test_dir)
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

        outputs: dict[str, str] = {}
        for path in sorted(test_dir.glob("*.jsonl")):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    outputs[record["request_id"]] = record["output"]
        return outputs

    @staticmethod
    def _aggregate(scores_list: list[rouge_scorer.scoring.Score]) -> ROUGEScores:
        """Average a list of ``Score`` named-tuples into a single :class:`ROUGEScores`."""
        n = len(scores_list)
        if n == 0:
            return ROUGEScores(0.0, 0.0, 0.0)
        return ROUGEScores(
            precision=sum(s.precision for s in scores_list) / n,
            recall=sum(s.recall for s in scores_list) / n,
            fmeasure=sum(s.fmeasure for s in scores_list) / n,
        )

    @staticmethod
    def _quality_gate(delta_pct: float) -> str:
        """Determine the quality gate from the ROUGE-L F-measure delta.

        Rules:
            * delta >= -1.0 %  ->  PASS
            * -2.0 % <= delta < -1.0 %  ->  MARGINAL
            * delta < -2.0 %  ->  FAIL
        """
        if delta_pct >= -1.0:
            return "PASS"
        if delta_pct >= -2.0:
            return "MARGINAL"
        return "FAIL"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, test_outputs_dir: Path) -> dict[str, Any]:
        """Run ROUGE evaluation and return a quality report.

        Parameters:
            test_outputs_dir: Directory of JSONL files produced by the model
                under test.  Same schema as the reference files.

        Returns:
            A dictionary containing rouge_1, rouge_2, rouge_l scores,
            delta_vs_bf16 (percentage), and quality_gate verdict.
        """
        test_outputs = self._load_test_outputs(test_outputs_dir)

        # Only score request IDs present in both sets.
        common_ids = sorted(set(self.references) & set(test_outputs))
        if not common_ids:
            raise ValueError(
                "No overlapping request_ids between reference and test outputs."
            )

        r1_scores: list[rouge_scorer.scoring.Score] = []
        r2_scores: list[rouge_scorer.scoring.Score] = []
        rl_scores: list[rouge_scorer.scoring.Score] = []

        # BF16 self-scores (reference vs reference) yield perfect 1.0 baselines,
        # so the BF16 ROUGE-L F-measure baseline is 1.0 (100 %).
        bf16_rouge_l_fmeasure = 1.0

        for rid in common_ids:
            ref_text = self.references[rid]
            test_text = test_outputs[rid]
            scores = self._scorer.score(ref_text, test_text)
            r1_scores.append(scores["rouge1"])
            r2_scores.append(scores["rouge2"])
            rl_scores.append(scores["rougeL"])

        rouge_1 = self._aggregate(r1_scores)
        rouge_2 = self._aggregate(r2_scores)
        rouge_l = self._aggregate(rl_scores)

        # Delta is the percentage-point difference in ROUGE-L F-measure
        # relative to the BF16 baseline.
        delta_pct = (rouge_l.fmeasure - bf16_rouge_l_fmeasure) * 100.0
        gate = self._quality_gate(delta_pct)

        report = QualityReport(
            rouge_1=rouge_1,
            rouge_2=rouge_2,
            rouge_l=rouge_l,
            delta_vs_bf16=round(delta_pct, 4),
            quality_gate=gate,
            num_samples=len(common_ids),
        )
        return report.to_dict()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISB-1 ROUGE quality evaluation against BF16 reference outputs.",
    )
    parser.add_argument(
        "--reference", type=Path, required=True,
        help="Directory containing BF16 reference JSONL files.",
    )
    parser.add_argument(
        "--test", type=Path, required=True,
        help="Directory containing test-model JSONL output files.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write the JSON report.",
    )
    args = parser.parse_args()

    evaluator = ROUGEEvaluator(reference_dir=args.reference)
    report = evaluator.evaluate(test_outputs_dir=args.test)

    report_json = json.dumps(report, indent=2)
    print(report_json)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json, encoding="utf-8")
        print(f"\nReport written to {args.output}")

    sys.exit(0 if report["quality_gate"] != "FAIL" else 1)


if __name__ == "__main__":
    main()
