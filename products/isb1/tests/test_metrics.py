"""Tests for analysis.metrics.MetricComputer."""

import numpy as np
import pytest

from analysis.metrics import MetricComputer, CellMetrics, _compute_tpot, _compute_itl_gaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    ttft: float,
    e2e_latency: float,
    output_tokens: int,
    timestamp: float = 0.0,
    token_timestamps: list[float] | None = None,
    error: bool = False,
) -> dict:
    """Build a synthetic latency record."""
    rec = {
        "ttft": ttft,
        "e2e_latency": e2e_latency,
        "output_tokens": output_tokens,
        "timestamp": timestamp,
        "error": error,
    }
    if token_timestamps is not None:
        rec["token_timestamps"] = token_timestamps
    return rec


def _make_batch(n: int = 100, seed: int = 0) -> list[dict]:
    """Create *n* successful requests with controlled latency values."""
    rng = np.random.default_rng(seed)
    records = []
    t = 0.0
    for _ in range(n):
        ttft = 0.5 + rng.random() * 0.5          # 0.5-1.0s
        output_tokens = int(rng.integers(10, 200))
        # Decoding takes (output_tokens-1)*0.05s
        decode_time = (output_tokens - 1) * 0.05
        e2e = ttft + decode_time
        # Token timestamps: first token at ttft, then every 0.05s
        ts = [ttft + i * 0.05 for i in range(output_tokens)]
        records.append(
            _make_request(
                ttft=ttft,
                e2e_latency=e2e,
                output_tokens=output_tokens,
                timestamp=t,
                token_timestamps=ts,
            )
        )
        t += e2e * 0.1  # overlap requests
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTTFTComputation:
    """test_ttft_computation: verify TTFT percentiles computed correctly."""

    def test_known_ttfts(self):
        """TTFT percentiles should match numpy percentile on the input TTFTs."""
        ttfts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        data = [
            _make_request(ttft=t, e2e_latency=t + 1.0, output_tokens=20, timestamp=i)
            for i, t in enumerate(ttfts)
        ]
        mc = MetricComputer()
        result = mc.compute(data)

        assert result.ttft_p50 == pytest.approx(np.percentile(ttfts, 50), abs=1e-6)
        assert result.ttft_p95 == pytest.approx(np.percentile(ttfts, 95), abs=1e-6)
        assert result.ttft_p99 == pytest.approx(np.percentile(ttfts, 99), abs=1e-6)

    def test_single_request(self):
        """With one request, all percentiles should equal that TTFT."""
        data = [_make_request(ttft=0.42, e2e_latency=1.0, output_tokens=10, timestamp=0)]
        result = MetricComputer().compute(data)
        assert result.ttft_p50 == pytest.approx(0.42)
        assert result.ttft_p99 == pytest.approx(0.42)


class TestTPOTExcludesTTFT:
    """test_tpot_excludes_ttft: CRITICAL - verify TPOT = (e2e - ttft) / (output_tokens - 1)."""

    def test_tpot_formula(self):
        """TPOT must equal (e2e - ttft) / (output_tokens - 1), excluding TTFT."""
        ttft = 0.5
        output_tokens = 11  # 10 decode steps
        decode_time = 1.0   # 10 * 0.1
        e2e = ttft + decode_time
        expected_tpot = decode_time / (output_tokens - 1)  # 0.1

        data = [_make_request(ttft=ttft, e2e_latency=e2e, output_tokens=output_tokens, timestamp=0)]
        result = MetricComputer().compute(data)

        assert result.tpot_p50 == pytest.approx(expected_tpot, abs=1e-9)
        assert result.tpot_p99 == pytest.approx(expected_tpot, abs=1e-9)

    def test_tpot_not_e2e_divided_by_tokens(self):
        """TPOT must NOT be e2e / output_tokens (a common mistake)."""
        ttft = 2.0
        output_tokens = 11
        decode_time = 1.0
        e2e = ttft + decode_time

        wrong_tpot = e2e / output_tokens  # 3.0 / 11 = ~0.2727
        correct_tpot = decode_time / (output_tokens - 1)  # 1.0 / 10 = 0.1

        result = _compute_tpot(e2e, ttft, output_tokens)
        assert result == pytest.approx(correct_tpot, abs=1e-9)
        assert result != pytest.approx(wrong_tpot, abs=0.01)

    def test_tpot_single_token_returns_none(self):
        """With only 1 output token there are 0 decode steps; TPOT is undefined."""
        assert _compute_tpot(1.0, 0.5, 1) is None
        assert _compute_tpot(1.0, 0.5, 0) is None


class TestITLComputation:
    """test_itl_computation: verify ITL starts from token 2."""

    def test_itl_gaps_exclude_ttft(self):
        """ITL gaps should start from token index 2, excluding the TTFT gap."""
        # Token 0 at t=0.5, token 1 at t=0.6, token 2 at t=0.7, token 3 at t=0.85
        timestamps = [0.5, 0.6, 0.7, 0.85]
        gaps = _compute_itl_gaps(timestamps)
        # gap between token 1 and 2 = 0.1, gap between token 2 and 3 = 0.15
        assert len(gaps) == 2
        assert gaps[0] == pytest.approx(0.1)
        assert gaps[1] == pytest.approx(0.15)

    def test_itl_with_two_tokens(self):
        """Two tokens means zero ITL gaps (the only gap is the TTFT)."""
        timestamps = [0.5, 0.6]
        gaps = _compute_itl_gaps(timestamps)
        assert gaps == []

    def test_itl_single_token(self):
        """Single token produces no ITL gaps."""
        assert _compute_itl_gaps([0.5]) == []
        assert _compute_itl_gaps([]) == []

    def test_itl_percentiles_in_result(self):
        """ITL percentiles in CellMetrics should reflect decode-only gaps."""
        # 5 tokens: gaps are between consecutive tokens from index 2 onward
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
        data = [
            _make_request(
                ttft=0.1, e2e_latency=0.4, output_tokens=5,
                timestamp=0, token_timestamps=timestamps,
            )
        ]
        result = MetricComputer().compute(data)
        # Gaps: 0.2-0.1=0.1, 0.3-0.2=0.1, 0.4-0.3=0.1
        assert result.itl_p50 == pytest.approx(0.1, abs=1e-6)


class TestGoodputComputation:
    """test_goodput_computation: verify goodput = requests meeting both SLOs / wall_clock."""

    def test_all_within_slo(self):
        """When all requests meet SLOs, goodput should equal request_throughput."""
        # ttft_slo=2.0, tpot_slo=0.1 (defaults)
        data = [
            _make_request(ttft=0.5, e2e_latency=1.5, output_tokens=11, timestamp=i)
            for i in range(10)
        ]
        mc = MetricComputer(ttft_slo=2.0, tpot_slo=0.2)
        result = mc.compute(data)
        assert result.goodput == pytest.approx(result.request_throughput, rel=1e-6)

    def test_some_violate_slo(self):
        """Requests violating SLOs should be excluded from goodput."""
        good = [
            _make_request(ttft=0.5, e2e_latency=1.5, output_tokens=11, timestamp=i)
            for i in range(5)
        ]
        # Bad TTFT (3.0 > 2.0)
        bad = [
            _make_request(ttft=3.0, e2e_latency=4.0, output_tokens=11, timestamp=i + 5)
            for i in range(5)
        ]
        mc = MetricComputer(ttft_slo=2.0, tpot_slo=0.2)
        result = mc.compute(good + bad)
        # wall_clock = max end - min start
        wall_clock = (5 + 5 - 1 + 4.0) - 0  # last request starts at 9, ends at 13
        expected_goodput = 5 / wall_clock
        assert result.goodput == pytest.approx(expected_goodput, rel=1e-3)


class TestSLOAttainment:
    """test_slo_attainment: verify SLO attainment percentage."""

    def test_slo_attainment_fraction(self):
        """SLO attainment should be the fraction of successful requests meeting both SLOs."""
        good = [
            _make_request(ttft=0.5, e2e_latency=1.5, output_tokens=11, timestamp=i)
            for i in range(8)
        ]
        bad_ttft = [
            _make_request(ttft=5.0, e2e_latency=6.0, output_tokens=11, timestamp=i + 8)
            for i in range(2)
        ]
        mc = MetricComputer(ttft_slo=2.0, tpot_slo=0.2)
        result = mc.compute(good + bad_ttft)
        assert result.slo_attainment == pytest.approx(8 / 10, abs=1e-6)

    def test_perfect_attainment(self):
        data = [
            _make_request(ttft=0.1, e2e_latency=0.5, output_tokens=5, timestamp=i)
            for i in range(20)
        ]
        result = MetricComputer(ttft_slo=1.0, tpot_slo=1.0).compute(data)
        assert result.slo_attainment == pytest.approx(1.0)


class TestErrorRate:
    """test_error_rate: verify error rate computation."""

    def test_error_rate(self):
        good = [_make_request(ttft=0.5, e2e_latency=1.0, output_tokens=10, timestamp=i) for i in range(7)]
        bad = [_make_request(ttft=0.5, e2e_latency=1.0, output_tokens=10, timestamp=i + 7, error=True) for i in range(3)]
        result = MetricComputer().compute(good + bad)
        assert result.error_rate == pytest.approx(0.3, abs=1e-6)
        assert result.total_requests == 10
        assert result.successful_requests == 7

    def test_no_errors(self):
        data = [_make_request(ttft=0.5, e2e_latency=1.0, output_tokens=10, timestamp=i) for i in range(5)]
        result = MetricComputer().compute(data)
        assert result.error_rate == pytest.approx(0.0)

    def test_all_errors(self):
        data = [_make_request(ttft=0.5, e2e_latency=1.0, output_tokens=10, timestamp=i, error=True) for i in range(5)]
        result = MetricComputer().compute(data)
        assert result.error_rate == pytest.approx(1.0)


class TestEmptyData:
    """test_empty_data: edge case with no data."""

    def test_empty_latency_data(self):
        result = MetricComputer().compute([])
        assert result.ttft_p50 == 0.0
        assert result.tpot_p50 == 0.0
        assert result.itl_p50 == 0.0
        assert result.goodput == 0.0
        assert result.error_rate == 0.0
        assert result.total_requests == 0
        assert result.successful_requests == 0

    def test_cellmetrics_defaults(self):
        m = CellMetrics()
        assert m.ttft_p50 == 0.0
        assert m.total_requests == 0
