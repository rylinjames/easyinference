"""Tests for analysis.statistical module."""

import numpy as np
import pytest

from analysis.statistical import (
    paired_ttest,
    bootstrap_ci,
    coefficient_of_variation,
    needs_more_trials,
    PairedTTestResult,
    BootstrapCIResult,
)


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------


class TestPairedTtestSignificant:
    """test_paired_ttest_significant: verify significant result detected."""

    def test_large_difference(self):
        """Two samples with a clear offset should produce a significant result."""
        rng = np.random.default_rng(42)
        n = 30
        a = rng.normal(loc=10.0, scale=1.0, size=n).tolist()
        b = rng.normal(loc=5.0, scale=1.0, size=n).tolist()
        result = paired_ttest(a, b)
        assert isinstance(result, PairedTTestResult)
        assert result.significant is True
        assert result.p_value < 0.05
        assert result.mean_diff > 0  # a > b

    def test_ci_does_not_contain_zero(self):
        """When significant, the 95% CI on the mean difference should exclude zero."""
        rng = np.random.default_rng(7)
        a = rng.normal(100, 2, size=50).tolist()
        b = rng.normal(90, 2, size=50).tolist()
        result = paired_ttest(a, b)
        assert result.significant is True
        assert result.ci_95_lower > 0, "CI should not contain zero for significant diff"


class TestPairedTtestNotSignificant:
    """test_paired_ttest_not_significant: verify non-significant result."""

    def test_identical_samples(self):
        """Identical samples should produce p~1.0 and non-significant result."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_ttest(data, data)
        assert result.significant is False
        assert result.mean_diff == pytest.approx(0.0)

    def test_overlapping_distributions(self):
        """Nearly identical distributions should not be significant."""
        rng = np.random.default_rng(42)
        a = rng.normal(10.0, 5.0, size=10).tolist()
        b = rng.normal(10.0, 5.0, size=10).tolist()
        result = paired_ttest(a, b)
        # With same mean and high variance, p should be large
        assert result.p_value > 0.01  # likely non-significant

    def test_minimum_samples(self):
        """Paired t-test requires at least 2 observations."""
        with pytest.raises(ValueError, match="at least 2"):
            paired_ttest([1.0], [2.0])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            paired_ttest([1.0, 2.0], [3.0])


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCIContainsMean:
    """test_bootstrap_ci_contains_mean: verify CI contains sample mean."""

    def test_mean_within_ci(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50.0, 5.0, size=100).tolist()
        result = bootstrap_ci(data, confidence=0.95, n_bootstrap=5000, rng_seed=42)
        assert isinstance(result, BootstrapCIResult)
        # The point estimate should be within the CI
        assert result.lower <= result.point_estimate <= result.upper
        # The sample mean should be close to the point estimate
        assert result.point_estimate == pytest.approx(np.mean(data), abs=1e-10)

    def test_wider_at_lower_confidence(self):
        """A 99% CI should be wider than a 90% CI."""
        data = list(np.random.default_rng(0).normal(0, 1, size=50))
        ci_90 = bootstrap_ci(data, confidence=0.90, n_bootstrap=5000, rng_seed=0)
        ci_99 = bootstrap_ci(data, confidence=0.99, n_bootstrap=5000, rng_seed=0)
        width_90 = ci_90.upper - ci_90.lower
        width_99 = ci_99.upper - ci_99.lower
        assert width_99 > width_90

    def test_single_value(self):
        result = bootstrap_ci([42.0], confidence=0.95)
        assert result.lower == pytest.approx(42.0)
        assert result.upper == pytest.approx(42.0)
        assert result.point_estimate == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Coefficient of variation
# ---------------------------------------------------------------------------


class TestCVComputation:
    """test_cv_computation: verify coefficient of variation."""

    def test_known_cv(self):
        """CV = std/mean for a known dataset."""
        data = [10.0, 10.0, 10.0, 10.0]
        assert coefficient_of_variation(data) == pytest.approx(0.0)

    def test_positive_cv(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = 3.0
        std = float(np.std(data, ddof=1))
        expected = std / mean
        assert coefficient_of_variation(data) == pytest.approx(expected, rel=1e-6)

    def test_single_value_returns_zero(self):
        assert coefficient_of_variation([5.0]) == 0.0

    def test_empty_returns_zero(self):
        assert coefficient_of_variation([]) == 0.0

    def test_zero_mean_returns_zero(self):
        # Values that sum to zero
        assert coefficient_of_variation([1.0, -1.0]) == 0.0


# ---------------------------------------------------------------------------
# Needs more trials
# ---------------------------------------------------------------------------


class TestNeedsMoreTrials:
    """test_needs_more_trials: verify threshold logic."""

    def test_high_cv_needs_more(self):
        # High variability
        data = [1.0, 10.0, 1.0, 10.0, 1.0]
        assert needs_more_trials(data, threshold=0.10) is True

    def test_low_cv_sufficient(self):
        data = [10.0, 10.1, 9.9, 10.0, 10.05]
        assert needs_more_trials(data, threshold=0.10) is False

    def test_constant_data(self):
        data = [5.0] * 10
        assert needs_more_trials(data, threshold=0.01) is False

    def test_custom_threshold(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        cv = coefficient_of_variation(data)
        # With threshold above CV, no more trials needed
        assert needs_more_trials(data, threshold=cv + 0.01) is False
        # With threshold below CV, more trials needed
        assert needs_more_trials(data, threshold=cv - 0.01) is True
