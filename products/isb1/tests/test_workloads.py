"""Tests for workload generators and arrival time models."""

import numpy as np
import pytest

from workloads.base import Request
from workloads.chat import ChatWorkloadGenerator
from workloads.agent import AgentTraceGenerator
from workloads.rag import RAGTraceGenerator
from workloads.coding import CodingTraceGenerator
from workloads.arrivals import PoissonArrival, GammaArrival


# ---------------------------------------------------------------------------
# Chat workload
# ---------------------------------------------------------------------------


class TestChatGeneratorProducesRequests:
    """test_chat_generator_produces_requests: verify ChatWorkloadGenerator produces Request objects."""

    def test_produces_request_instances(self):
        gen = ChatWorkloadGenerator(seed=42)
        requests = gen.generate(10)
        assert len(requests) == 10
        for req in requests:
            assert isinstance(req, Request)

    def test_request_fields_populated(self):
        gen = ChatWorkloadGenerator(seed=42)
        requests = gen.generate(5)
        for req in requests:
            assert req.request_id
            assert len(req.messages) >= 2  # system + at least one user
            assert req.expected_output_tokens > 0
            assert req.metadata.get("workload") == "chat"

    def test_messages_in_openai_format(self):
        gen = ChatWorkloadGenerator(seed=42)
        requests = gen.generate(3)
        for req in requests:
            for msg in req.messages:
                assert "role" in msg
                assert "content" in msg
                assert msg["role"] in ("system", "user", "assistant")


class TestChatSessionIds:
    """test_chat_session_ids: verify multi-turn conversations share session_id."""

    def test_shared_session_id(self):
        gen = ChatWorkloadGenerator(seed=42, min_turns=3, max_turns=3)
        requests = gen.generate(20)
        # Group by session_id
        sessions: dict[str, list[Request]] = {}
        for req in requests:
            sid = req.session_id
            assert sid is not None, "Every chat request must have a session_id"
            sessions.setdefault(sid, []).append(req)

        # At least one session should have multiple requests (3 turns)
        multi_turn = [s for s in sessions.values() if len(s) > 1]
        assert len(multi_turn) > 0, "Expected at least one multi-turn session"

    def test_context_grows_across_turns(self):
        gen = ChatWorkloadGenerator(seed=42, min_turns=3, max_turns=3)
        requests = gen.generate(20)
        sessions: dict[str, list[Request]] = {}
        for req in requests:
            sessions.setdefault(req.session_id, []).append(req)

        for reqs in sessions.values():
            if len(reqs) >= 2:
                # Later turns should have more messages
                assert len(reqs[-1].messages) > len(reqs[0].messages)


# ---------------------------------------------------------------------------
# Agent workload
# ---------------------------------------------------------------------------


class TestAgentGeneratorToolSchemas:
    """test_agent_generator_tool_schemas: verify AgentTraceGenerator includes tool schemas."""

    def test_system_prompt_contains_tools(self):
        gen = AgentTraceGenerator(seed=42)
        requests = gen.generate(5)
        # The system prompt should reference tool names
        system_msg = requests[0].messages[0]
        assert system_msg["role"] == "system"
        # Default tools include 'search', 'code_execute', 'file_read'
        content = system_msg["content"]
        assert "search" in content or "code_execute" in content or "file_read" in content

    def test_metadata_includes_tool_count(self):
        gen = AgentTraceGenerator(seed=42)
        requests = gen.generate(5)
        for req in requests:
            assert req.metadata.get("workload") == "agent"
            assert req.metadata.get("num_tools", 0) > 0


# ---------------------------------------------------------------------------
# RAG workload
# ---------------------------------------------------------------------------


class TestRAGGeneratorContextLengths:
    """test_rag_generator_context_lengths: verify bimodal distribution."""

    def test_produces_requests(self):
        gen = RAGTraceGenerator(seed=42)
        requests = gen.generate(20)
        assert len(requests) == 20
        for req in requests:
            assert isinstance(req, Request)
            assert req.metadata.get("workload") == "rag"

    def test_bimodal_context_distribution(self):
        """RAG generator should produce both short and long contexts."""
        gen = RAGTraceGenerator(seed=42)
        requests = gen.generate(100)
        context_tokens = [req.metadata.get("approx_context_tokens", 0) for req in requests]
        # Bimodal: ~60% around 32K, ~40% around 96K
        # Check there are values in both modes
        short_count = sum(1 for t in context_tokens if t < 50000)
        long_count = sum(1 for t in context_tokens if t >= 50000)
        assert short_count > 0, "Expected some short-context requests"
        assert long_count > 0, "Expected some long-context requests"


# ---------------------------------------------------------------------------
# Coding workload
# ---------------------------------------------------------------------------


class TestCodingGeneratorPrefixReuse:
    """test_coding_generator_prefix_reuse: verify CodingTraceGenerator shares repo context."""

    def test_shared_session_id(self):
        gen = CodingTraceGenerator(seed=42)
        requests = gen.generate(20)
        # Group by session_id
        sessions: dict[str, list[Request]] = {}
        for req in requests:
            sid = req.session_id
            assert sid is not None
            sessions.setdefault(sid, []).append(req)

        # At least one session should have multiple requests (multi-turn)
        multi = [r for r in sessions.values() if len(r) > 1]
        assert len(multi) > 0, "Expected requests sharing the same session_id"

    def test_shared_context_prefix(self):
        gen = CodingTraceGenerator(seed=42)
        requests = gen.generate(20)
        sessions: dict[str, list[Request]] = {}
        for req in requests:
            sessions.setdefault(req.session_id, []).append(req)

        for reqs in sessions.values():
            if len(reqs) >= 2:
                # All requests in same session should share the system prompt (first message)
                first_system = reqs[0].messages[0]["content"]
                for r in reqs[1:]:
                    assert r.messages[0]["content"] == first_system


# ---------------------------------------------------------------------------
# Arrival models
# ---------------------------------------------------------------------------


class TestPoissonArrivalRate:
    """test_poisson_arrival_rate: verify PoissonArrival produces correct average rate."""

    def test_average_rate(self):
        rate = 10.0  # 10 req/s
        arrival = PoissonArrival(rate=rate, seed=42)
        timestamps = arrival.generate(10000)
        # Observed rate = num_requests / total_time
        total_time = timestamps[-1]
        observed_rate = len(timestamps) / total_time
        assert observed_rate == pytest.approx(rate, rel=0.05)

    def test_monotonically_increasing(self):
        arrival = PoissonArrival(rate=5.0, seed=0)
        timestamps = arrival.generate(100)
        assert np.all(np.diff(timestamps) > 0)

    def test_empty_request(self):
        arrival = PoissonArrival(rate=1.0, seed=0)
        timestamps = arrival.generate(0)
        assert len(timestamps) == 0

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            PoissonArrival(rate=0.0)
        with pytest.raises(ValueError):
            PoissonArrival(rate=-1.0)


class TestGammaArrivalBurstiness:
    """test_gamma_arrival_burstiness: verify GammaArrival produces bursty patterns."""

    def test_bursty_has_higher_variance(self):
        """Gamma with shape<1 should produce higher variance in inter-arrivals than Poisson."""
        n = 5000
        rate = 10.0
        poisson = PoissonArrival(rate=rate, seed=42)
        gamma = GammaArrival(rate=rate, shape=0.3, seed=42)

        p_times = poisson.generate(n)
        g_times = gamma.generate(n)

        p_gaps = np.diff(p_times)
        g_gaps = np.diff(g_times)

        # Gamma(shape=0.3) should have higher CV than Poisson(shape=1)
        p_cv = np.std(p_gaps) / np.mean(p_gaps)
        g_cv = np.std(g_gaps) / np.mean(g_gaps)
        assert g_cv > p_cv, "Gamma(0.3) should be burstier than Poisson"

    def test_average_rate_matches(self):
        rate = 5.0
        gamma = GammaArrival(rate=rate, shape=0.5, seed=42)
        timestamps = gamma.generate(5000)
        observed = len(timestamps) / timestamps[-1]
        assert observed == pytest.approx(rate, rel=0.1)

    def test_shape_one_similar_to_poisson(self):
        """Gamma with shape=1 should have similar CV to Poisson."""
        n = 5000
        rate = 10.0
        poisson = PoissonArrival(rate=rate, seed=99)
        gamma = GammaArrival(rate=rate, shape=1.0, seed=99)

        p_gaps = np.diff(poisson.generate(n))
        g_gaps = np.diff(gamma.generate(n))

        p_cv = np.std(p_gaps) / np.mean(p_gaps)
        g_cv = np.std(g_gaps) / np.mean(g_gaps)
        assert g_cv == pytest.approx(p_cv, rel=0.15)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """test_reproducibility: verify same seed produces same traces."""

    def test_chat_reproducibility(self):
        a = ChatWorkloadGenerator(seed=123).generate(10)
        b = ChatWorkloadGenerator(seed=123).generate(10)
        for ra, rb in zip(a, b):
            assert ra.messages == rb.messages
            assert ra.expected_output_tokens == rb.expected_output_tokens

    def test_agent_reproducibility(self):
        a = AgentTraceGenerator(seed=456).generate(5)
        b = AgentTraceGenerator(seed=456).generate(5)
        for ra, rb in zip(a, b):
            # Agent messages contain uuid-based tool_call IDs that are not
            # seeded by numpy, so compare the user-content messages only.
            user_msgs_a = [m["content"] for m in ra.messages if m.get("role") == "user"]
            user_msgs_b = [m["content"] for m in rb.messages if m.get("role") == "user"]
            assert user_msgs_a == user_msgs_b
            assert ra.expected_output_tokens == rb.expected_output_tokens

    def test_poisson_reproducibility(self):
        a = PoissonArrival(rate=10, seed=99).generate(100)
        b = PoissonArrival(rate=10, seed=99).generate(100)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = ChatWorkloadGenerator(seed=1).generate(5)
        b = ChatWorkloadGenerator(seed=2).generate(5)
        # At least one request should differ
        any_diff = any(ra.messages != rb.messages for ra, rb in zip(a, b))
        assert any_diff, "Different seeds should produce different traces"
