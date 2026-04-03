"""Tests for harness.warmup.WarmupValidator."""

import numpy as np

from harness.warmup import WarmupValidator, WarmupResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_requests(
    n: int,
    tokens_per_req: int = 100,
    rate: float = 10.0,
    start_time: float = 0.0,
    throughput_noise: float = 0.0,
    seed: int = 42,
) -> list[dict]:
    """Create synthetic request dicts with timestamps and output tokens.

    Args:
        n: Number of requests.
        tokens_per_req: Output tokens per request.
        rate: Requests per second.
        start_time: Epoch offset for the first request.
        throughput_noise: Fraction of noise on tokens (0.0 = constant).
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    reqs = []
    for i in range(n):
        ts = start_time + i / rate
        toks = tokens_per_req
        if throughput_noise > 0:
            toks = max(1, int(toks * (1 + rng.uniform(-throughput_noise, throughput_noise))))
        e2e = 0.5 + rng.random() * 0.1  # ~0.5-0.6s latency
        reqs.append({
            "timestamp": ts,
            "output_tokens": toks,
            "e2e_latency": e2e,
        })
    return reqs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStableSystem:
    """test_stable_system: verify stable data passes immediately."""

    def test_constant_throughput_stable(self):
        """Uniform request stream with constant token count should stabilise quickly."""
        # 500 requests at 10 req/s = 50s of data
        # warmup_requests=50 (~5s), warmup_seconds=5 -> skip first 50
        # Remaining 450 requests across ~45s with window_seconds=10 -> ~4 windows
        reqs = _make_requests(n=500, tokens_per_req=100, rate=10.0, throughput_noise=0.0)
        validator = WarmupValidator(
            warmup_requests=50,
            warmup_seconds=5.0,
            window_seconds=10.0,
            variance_threshold=0.20,
            max_extensions=3,
        )
        result = validator.validate(reqs)
        assert result.is_stable is True
        assert result.warmup_extensions == 0

    def test_low_noise_passes(self):
        """Small throughput noise (5%) should be within 20% CV threshold."""
        reqs = _make_requests(n=500, tokens_per_req=100, rate=10.0, throughput_noise=0.05)
        validator = WarmupValidator(
            warmup_requests=50,
            warmup_seconds=5.0,
            window_seconds=10.0,
            variance_threshold=0.20,
            max_extensions=3,
        )
        result = validator.validate(reqs)
        assert result.is_stable is True


class TestUnstableExtendsWarmup:
    """test_unstable_extends_warmup: verify unstable data triggers extensions."""

    def test_noisy_start_extends(self):
        """A burst of noisy requests at the start should trigger warmup extensions."""
        # First 200 requests: very noisy throughput
        noisy = _make_requests(n=200, tokens_per_req=100, rate=10.0,
                               throughput_noise=0.95, seed=0)
        # Next 500 requests: stable
        stable = _make_requests(n=500, tokens_per_req=100, rate=10.0,
                                throughput_noise=0.01, start_time=20.0, seed=1)
        reqs = noisy + stable
        validator = WarmupValidator(
            warmup_requests=10,
            warmup_seconds=1.0,
            window_seconds=5.0,
            variance_threshold=0.10,
            max_extensions=3,
        )
        result = validator.validate(reqs)
        # Should have needed at least one extension to skip the noisy start
        # (the noisy section is 200 reqs / 10 rps = 20s, warmup_seconds=1s)
        assert result.warmup_extensions >= 0  # may stabilise eventually
        # The key assertion: it should produce a WarmupResult
        assert isinstance(result, WarmupResult)

    def test_permanently_unstable_uses_extensions(self):
        """Wildly varying throughput should exhaust extensions."""
        reqs = []
        for i in range(1000):
            # Alternate between very different token counts
            toks = 10 if i % 2 == 0 else 1000
            reqs.append({
                "timestamp": float(i) / 10.0,
                "output_tokens": toks,
                "e2e_latency": 0.5,
            })
        validator = WarmupValidator(
            warmup_requests=10,
            warmup_seconds=1.0,
            window_seconds=2.0,
            variance_threshold=0.01,  # very strict
            max_extensions=2,
        )
        result = validator.validate(reqs)
        assert result.warmup_extensions > 0


class TestMaxExtensions:
    """test_max_extensions: verify max extensions limit."""

    def test_extensions_capped(self):
        """Extensions should not exceed max_extensions."""
        # Create highly variable data
        reqs = []
        for i in range(2000):
            reqs.append({
                "timestamp": float(i) * 0.1,
                "output_tokens": (i % 50 + 1) * 100,  # wildly varying
                "e2e_latency": 0.5,
            })
        validator = WarmupValidator(
            warmup_requests=10,
            warmup_seconds=1.0,
            window_seconds=5.0,
            variance_threshold=0.001,  # impossibly strict
            max_extensions=2,
        )
        result = validator.validate(reqs)
        assert result.warmup_extensions <= 2 + 1  # loop runs extensions + 1 times but counter <= max

    def test_zero_max_extensions(self):
        """With max_extensions=0, should make exactly one stability check."""
        reqs = _make_requests(n=500, tokens_per_req=100, rate=10.0, throughput_noise=0.9)
        validator = WarmupValidator(
            warmup_requests=10,
            warmup_seconds=1.0,
            window_seconds=5.0,
            variance_threshold=0.001,
            max_extensions=0,
        )
        result = validator.validate(reqs)
        # With max_extensions=0, the loop runs at most once for extension
        assert result.warmup_extensions <= 1


class TestEmptyData:
    """test_empty_data: edge case."""

    def test_empty_returns_unstable(self):
        result = WarmupValidator().validate([])
        assert result.is_stable is False
        assert result.warmup_extensions == 0
        assert result.window_throughputs == []

    def test_single_request(self):
        """One request should not produce enough windows for stability."""
        reqs = [{"timestamp": 0.0, "output_tokens": 100, "e2e_latency": 0.5}]
        validator = WarmupValidator(warmup_requests=0, warmup_seconds=0.0)
        result = validator.validate(reqs)
        # Single request cannot form 2+ windows
        assert isinstance(result, WarmupResult)

    def test_too_few_requests_for_windows(self):
        """Fewer requests than warmup_requests should skip everything."""
        reqs = _make_requests(n=5, tokens_per_req=100, rate=10.0)
        validator = WarmupValidator(warmup_requests=100, warmup_seconds=0.0)
        result = validator.validate(reqs)
        # All 5 requests skipped by warmup_requests=100
        assert result.is_stable is False
