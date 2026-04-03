"""Tests for analysis.claim_evaluator.ClaimEvaluator."""

import pytest

from analysis.claim_evaluator import ClaimEvaluator, ClaimVerdict, _CLAIMS


def _good_metrics() -> dict:
    """Metrics that should pass all default-threshold claims."""
    return {
        "ttft_p99": 0.5,
        "tpot_p99": 0.05,
        "goodput": 100.0,
        "slo_attainment": 0.99,
        "error_rate": 0.001,
        "generation_throughput": 5000.0,
        "request_throughput": 50.0,
        "prefix_cache_hit_rate": 0.90,
        "kv_cache_utilization_p95": 0.70,
        "preemptions_per_minute": 1.0,
        "watts_per_token": 0.5,
        "itl_p99": 0.05,
        "e2e_p99": 2.0,
        "successful_requests": 1000,
        "total_requests": 1000,
    }


def _bad_metrics() -> dict:
    """Metrics that should fail all claims."""
    return {
        "ttft_p99": 10.0,
        "tpot_p99": 1.0,
        "goodput": 0.5,
        "slo_attainment": 0.50,
        "error_rate": 0.20,
        "generation_throughput": 10.0,
        "request_throughput": 0.5,
        "prefix_cache_hit_rate": 0.10,
        "kv_cache_utilization_p95": 0.99,
        "preemptions_per_minute": 50.0,
        "watts_per_token": 100.0,
        "itl_p99": 5.0,
        "e2e_p99": 60.0,
        "successful_requests": 1000,
        "total_requests": 1000,
    }


# Full set of thresholds for all 13 claims
_ALL_THRESHOLDS = {
    "C1": 2.0,    # ttft_p99 < 2.0
    "C2": 0.1,    # tpot_p99 < 0.1
    "C3": 10.0,   # goodput >= 10.0
    "C4": 0.95,   # slo_attainment >= 0.95
    "C5": 0.01,   # error_rate < 0.01
    "C6": 1000.0, # generation_throughput >= 1000
    "C7": 5.0,    # request_throughput >= 5.0
    "C8": 0.50,   # prefix_cache_hit_rate >= 0.50
    "C9": 0.95,   # kv_cache_utilization_p95 < 0.95
    "C10": 5.0,   # preemptions_per_minute < 5.0
    "C11": 10.0,  # watts_per_token < 10.0
    "C12": 0.2,   # itl_p99 < 0.2
    "C13": 10.0,  # e2e_p99 < 10.0
}


class TestClaimSupported:
    """test_claim_supported: mock results that should produce SUPPORTED verdict."""

    def test_c4_slo_attainment_supported(self):
        evaluator = ClaimEvaluator(thresholds=_ALL_THRESHOLDS)
        metrics = _good_metrics()
        result = evaluator.evaluate("C4", metrics)
        assert result.verdict == ClaimVerdict.SUPPORTED
        assert result.claim_id == "C4"

    def test_c5_error_rate_supported(self):
        evaluator = ClaimEvaluator(thresholds=_ALL_THRESHOLDS)
        metrics = _good_metrics()
        result = evaluator.evaluate("C5", metrics)
        assert result.verdict == ClaimVerdict.SUPPORTED

    def test_all_good_metrics_supported(self):
        evaluator = ClaimEvaluator(thresholds=_ALL_THRESHOLDS)
        metrics = _good_metrics()
        results = evaluator.evaluate_all(metrics)
        for r in results:
            assert r.verdict == ClaimVerdict.SUPPORTED, (
                f"Claim {r.claim_id} expected SUPPORTED, got {r.verdict}: {r.evidence}"
            )


class TestClaimRefuted:
    """test_claim_refuted: mock results that should produce REFUTED verdict."""

    def test_c1_ttft_too_high(self):
        evaluator = ClaimEvaluator(thresholds=_ALL_THRESHOLDS)
        metrics = _bad_metrics()
        result = evaluator.evaluate("C1", metrics)
        assert result.verdict == ClaimVerdict.REFUTED

    def test_c5_error_rate_too_high(self):
        evaluator = ClaimEvaluator(thresholds=_ALL_THRESHOLDS)
        metrics = _bad_metrics()
        result = evaluator.evaluate("C5", metrics)
        assert result.verdict == ClaimVerdict.REFUTED

    def test_all_bad_metrics_refuted(self):
        evaluator = ClaimEvaluator(thresholds=_ALL_THRESHOLDS)
        metrics = _bad_metrics()
        results = evaluator.evaluate_all(metrics)
        for r in results:
            assert r.verdict == ClaimVerdict.REFUTED, (
                f"Claim {r.claim_id} expected REFUTED, got {r.verdict}: {r.evidence}"
            )


class TestClaimInconclusive:
    """test_claim_inconclusive: mock results with insufficient data."""

    def test_too_few_requests(self):
        evaluator = ClaimEvaluator(thresholds=_ALL_THRESHOLDS, min_requests=100)
        metrics = _good_metrics()
        metrics["successful_requests"] = 10  # below min_requests
        metrics["total_requests"] = 10
        result = evaluator.evaluate("C1", metrics)
        assert result.verdict == ClaimVerdict.INCONCLUSIVE

    def test_missing_metric(self):
        evaluator = ClaimEvaluator(thresholds=_ALL_THRESHOLDS)
        metrics = _good_metrics()
        del metrics["ttft_p99"]
        result = evaluator.evaluate("C1", metrics)
        assert result.verdict == ClaimVerdict.INCONCLUSIVE

    def test_no_threshold_defined(self):
        """Claims without a threshold and no default should be INCONCLUSIVE."""
        evaluator = ClaimEvaluator(thresholds={})  # no thresholds at all
        metrics = _good_metrics()
        # C1 has no default_threshold, so should be INCONCLUSIVE
        result = evaluator.evaluate("C1", metrics)
        assert result.verdict == ClaimVerdict.INCONCLUSIVE


class TestAllClaimsDefined:
    """test_all_claims_defined: verify all 13 claims are defined."""

    def test_exactly_13_claims(self):
        ids = ClaimEvaluator.all_claim_ids()
        assert len(ids) == 13

    def test_claim_ids_c1_through_c13(self):
        ids = ClaimEvaluator.all_claim_ids()
        expected = [f"C{i}" for i in range(1, 14)]
        assert ids == expected

    def test_internal_claims_dict(self):
        assert len(_CLAIMS) == 13
        for cid, claim in _CLAIMS.items():
            assert "text" in claim
            assert "metric" in claim
            assert "direction" in claim
            assert claim["direction"] in ("less", "greater")

    def test_unknown_claim_raises(self):
        evaluator = ClaimEvaluator()
        with pytest.raises(ValueError, match="Unknown claim"):
            evaluator.evaluate("C99", _good_metrics())
