"""Tests for MCP-side runtime profiling helpers."""

from __future__ import annotations

import pytest

from inferscope.server_profiling import _profile_runtime_for_mcp


@pytest.mark.asyncio
async def test_profile_runtime_for_mcp_resolves_auth_and_forces_private_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_resolve_auth_payload(payload, *, provider: str = "", default_scheme: str = "bearer"):
        captured["payload"] = payload
        captured["provider"] = provider
        captured["default_scheme"] = default_scheme
        return "AUTH"

    async def fake_profile_runtime(endpoint: str, **kwargs):
        captured["endpoint"] = endpoint
        captured["kwargs"] = kwargs
        return {"summary": "ok", "confidence": 1.0}

    monkeypatch.setattr("inferscope.server_profiling.resolve_auth_payload", fake_resolve_auth_payload)
    monkeypatch.setattr("inferscope.server_profiling.profile_runtime", fake_profile_runtime)

    result = await _profile_runtime_for_mcp(
        "http://localhost:8000",
        provider="fireworks",
        metrics_auth={"api_key": "secret"},
        include_identity=False,
    )

    assert result["summary"] == "ok"
    assert captured["provider"] == "fireworks"
    assert captured["payload"] == {"api_key": "secret"}
    assert captured["kwargs"]["allow_private"] is False
    assert captured["kwargs"]["metrics_auth"] == "AUTH"
    assert captured["kwargs"]["include_identity"] is False
