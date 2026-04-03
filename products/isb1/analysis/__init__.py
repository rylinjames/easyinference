"""ISB-1 Benchmark Analysis — metrics, statistics, and reporting pipeline."""

from analysis.metrics import CellMetrics, MetricComputer
from analysis.statistical import paired_ttest, bootstrap_ci, coefficient_of_variation, needs_more_trials
from analysis.aggregate import ResultAggregator
from analysis.claim_evaluator import ClaimVerdict, ClaimEvaluator
from analysis.comparisons import ComparisonGenerator
from analysis.leaderboard import LeaderboardGenerator

__all__ = [
    "CellMetrics",
    "MetricComputer",
    "paired_ttest",
    "bootstrap_ci",
    "coefficient_of_variation",
    "needs_more_trials",
    "ResultAggregator",
    "ClaimVerdict",
    "ClaimEvaluator",
    "ComparisonGenerator",
    "LeaderboardGenerator",
]
