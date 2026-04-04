"""Regression tests for the deep research agent workload."""

from __future__ import annotations

from workloads.deep_research_agent import DeepResearchAgentGenerator


def test_deep_research_rolling_window_trims_full_turn_groups() -> None:
    generator = DeepResearchAgentGenerator(seed=1, rolling_window_size=2)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "summary"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "content": "t2"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a3"},
        {"role": "tool", "content": "t3"},
        {"role": "user", "content": "u3"},
    ]

    pruned = generator._prune_messages_for_window(messages)

    assert pruned[0]["role"] == "system"
    assert pruned[1]["role"] == "user"
    assert [msg["content"] for msg in pruned if msg["role"] == "assistant"] == ["a2", "a3"]
    assert [msg["content"] for msg in pruned if msg["role"] == "tool"] == ["t2", "t3"]
    assert [msg["content"] for msg in pruned if msg["role"] == "user"][-2:] == ["u2", "u3"]


def test_deep_research_restart_metadata_and_follow_up_are_set() -> None:
    generator = DeepResearchAgentGenerator(seed=7, min_turns=14, max_turns=14, retry_with_summary_prob=1.0)

    requests = generator.generate(14)

    restarted = [
        request for request in requests
        if request.metadata.get("session_restart") and request.metadata["turn"] < len(requests) - 1
    ]
    assert restarted, "expected at least one restarted request"

    restart_turn = restarted[0].metadata["turn"]
    later_user_messages = [
        message["content"]
        for request in requests[restart_turn + 1:]
        for message in request.messages
        if message["role"] == "user"
    ]
    assert any("Continue from the summary" in content for content in later_user_messages)
