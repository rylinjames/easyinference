"""Statistical utilities for ISB-1 benchmark analysis.

Provides paired t-tests, BCa bootstrap confidence intervals,
coefficient of variation, and trial-sufficiency checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------

@dataclass
class PairedTTestResult:
    """Result of a paired t-test."""

    t_statistic: float
    p_value: float
    mean_diff: float
    ci_95_lower: float
    ci_95_upper: float
    significant: bool  # True when p_value < 0.05


def paired_ttest(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
) -> PairedTTestResult:
    """Perform a two-sided paired t-test on matched samples *a* and *b*.

    Parameters
    ----------
    a, b : array-like of float
        Matched measurement pairs (same length).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    PairedTTestResult
        Named fields: t_statistic, p_value, mean_diff,
        ci_95_lower, ci_95_upper, significant.

    Raises
    ------
    ValueError
        If *a* and *b* have different lengths or fewer than 2 elements.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"a and b must have the same length, got {len(a_arr)} and {len(b_arr)}"
        )
    if len(a_arr) < 2:
        raise ValueError("Need at least 2 paired observations for a t-test")

    diffs = a_arr - b_arr
    mean_diff = float(np.mean(diffs))
    n = len(diffs)
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))

    t_stat, p_val = stats.ttest_rel(a_arr, b_arr)
    t_stat = float(t_stat)
    p_val = float(p_val)

    # 95% CI for the mean difference
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return PairedTTestResult(
        t_statistic=t_stat,
        p_value=p_val,
        mean_diff=mean_diff,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        significant=p_val < alpha,
    )


# ---------------------------------------------------------------------------
# BCa Bootstrap confidence interval
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCIResult:
    """Result of a BCa bootstrap confidence interval."""

    lower: float
    upper: float
    point_estimate: float
    n_bootstrap: int


def bootstrap_ci(
    data: Sequence[float],
    statistic_fn=None,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    rng_seed: Optional[int] = None,
) -> BootstrapCIResult:
    """Compute a BCa (bias-corrected and accelerated) bootstrap confidence interval.

    Parameters
    ----------
    data : array-like of float
        The observed sample.
    statistic_fn : callable, optional
        A function that takes an array and returns a scalar.
        Defaults to ``np.mean``.
    confidence : float
        Confidence level (default 0.95).
    n_bootstrap : int
        Number of bootstrap resamples (default 10 000).
    rng_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    BootstrapCIResult
        lower, upper, point_estimate, n_bootstrap.
    """
    if statistic_fn is None:
        statistic_fn = np.mean

    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n < 2:
        val = float(statistic_fn(arr)) if n == 1 else 0.0
        return BootstrapCIResult(lower=val, upper=val, point_estimate=val, n_bootstrap=0)

    rng = np.random.default_rng(rng_seed)
    theta_hat = float(statistic_fn(arr))

    # Generate bootstrap distribution
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_stats = np.array([float(statistic_fn(arr[idx])) for idx in boot_indices])

    # --- Bias correction (z0) ---
    prop_less = np.mean(boot_stats < theta_hat)
    # Clamp to avoid infinities in ppf
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = float(stats.norm.ppf(prop_less))

    # --- Acceleration (a) via jackknife ---
    jack_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.concatenate([arr[:i], arr[i + 1 :]])
        jack_stats[i] = float(statistic_fn(jack_sample))
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = np.sum((jack_mean - jack_stats) ** 2)
    a_hat = float(num / (6.0 * (den ** 1.5))) if den > 0 else 0.0

    # --- Adjusted percentiles ---
    alpha = 1 - confidence
    z_alpha_lower = float(stats.norm.ppf(alpha / 2))
    z_alpha_upper = float(stats.norm.ppf(1 - alpha / 2))

    def _bca_percentile(z_alpha: float) -> float:
        numerator = z0 + z_alpha
        adjusted = z0 + numerator / (1 - a_hat * numerator)
        return float(stats.norm.cdf(adjusted)) * 100

    p_lower = _bca_percentile(z_alpha_lower)
    p_upper = _bca_percentile(z_alpha_upper)

    # Clamp to valid percentile range
    p_lower = np.clip(p_lower, 0.0, 100.0)
    p_upper = np.clip(p_upper, 0.0, 100.0)

    ci_lower = float(np.percentile(boot_stats, p_lower))
    ci_upper = float(np.percentile(boot_stats, p_upper))

    return BootstrapCIResult(
        lower=ci_lower,
        upper=ci_upper,
        point_estimate=theta_hat,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Coefficient of variation & trial sufficiency
# ---------------------------------------------------------------------------


def coefficient_of_variation(data: Sequence[float]) -> float:
    """Compute the coefficient of variation (CV = std / mean).

    Returns 0.0 if the mean is zero or the sample is empty.
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    mean = float(np.mean(arr))
    if mean == 0.0:
        return 0.0
    return float(np.std(arr, ddof=1) / abs(mean))


def needs_more_trials(data: Sequence[float], threshold: float = 0.10) -> bool:
    """Return True if the coefficient of variation exceeds *threshold*.

    A CV above the threshold (default 10%) indicates that additional
    trials are needed to achieve stable measurements.
    """
    return coefficient_of_variation(data) > threshold
