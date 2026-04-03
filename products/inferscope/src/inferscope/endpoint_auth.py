"""HTTP endpoint authentication helpers for inference and metrics traffic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

AuthScheme = Literal["none", "bearer", "api-key", "x-api-key", "raw"]


@dataclass(frozen=True)
class EndpointAuthConfig:
    """Authentication and extra headers for one HTTP endpoint."""

    api_key: str = ""
    auth_scheme: AuthScheme = "none"
    auth_header_name: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def build_headers(self) -> dict[str, str]:
        """Render authentication and extra headers for an outbound request."""
        resolved = dict(self.headers)
        if not self.api_key or self.auth_scheme == "none":
            return resolved

        if self.auth_scheme == "bearer":
            header_name = self.auth_header_name or "Authorization"
            header_value = f"Bearer {self.api_key}"
        elif self.auth_scheme == "api-key":
            header_name = self.auth_header_name or "Authorization"
            header_value = f"Api-Key {self.api_key}"
        elif self.auth_scheme == "x-api-key":
            header_name = self.auth_header_name or "X-API-Key"
            header_value = self.api_key
        else:
            header_name = self.auth_header_name or "Authorization"
            header_value = self.api_key

        resolved[header_name] = header_value
        return resolved


def default_auth_scheme(provider: str, *, fallback: AuthScheme = "bearer") -> AuthScheme:
    """Return a sane auth default for a known managed provider."""
    normalized = provider.strip().lower()
    if normalized in {"", "auto", "generic", "self-hosted", "self_hosted", "local"}:
        return fallback
    if normalized in {"fireworks", "huggingface", "hf", "vllm-hf", "vllm_hf"}:
        return "bearer"
    if normalized == "baseten":
        return "api-key"
    return fallback


def resolve_auth_config(
    api_key: str | None = None,
    *,
    provider: str = "",
    auth_scheme: str = "",
    auth_header_name: str = "",
    headers: Mapping[str, str] | None = None,
    default_scheme: AuthScheme = "bearer",
) -> EndpointAuthConfig | None:
    """Build a normalized auth config from simple runtime options."""
    clean_api_key = (api_key or "").strip()
    clean_headers = {str(key): str(value) for key, value in (headers or {}).items()}
    requested_scheme = auth_scheme.strip().lower()

    if not clean_api_key and not clean_headers:
        return None

    if not clean_api_key:
        resolved_scheme: AuthScheme = "none"
    else:
        fallback = default_auth_scheme(provider, fallback=default_scheme)
        resolved_scheme = requested_scheme if requested_scheme else fallback  # type: ignore[assignment]
        if resolved_scheme not in {"none", "bearer", "api-key", "x-api-key", "raw"}:
            allowed = ", ".join(["none", "bearer", "api-key", "x-api-key", "raw"])
            raise ValueError(f"Unsupported auth scheme '{auth_scheme}'. Use one of: {allowed}")

    return EndpointAuthConfig(
        api_key=clean_api_key,
        auth_scheme=resolved_scheme,
        auth_header_name=auth_header_name.strip(),
        headers=clean_headers,
    )


def resolve_auth_payload(
    payload: Mapping[str, Any] | None,
    *,
    provider: str = "",
    default_scheme: AuthScheme = "bearer",
) -> EndpointAuthConfig | None:
    """Build an auth config from an MCP-style payload dict."""
    if not payload:
        return None
    raw_headers = payload.get("headers", {})
    headers = raw_headers if isinstance(raw_headers, Mapping) else {}
    return resolve_auth_config(
        str(payload.get("api_key", "")).strip() or None,
        provider=provider,
        auth_scheme=str(payload.get("auth_scheme", "")),
        auth_header_name=str(payload.get("auth_header_name", "")),
        headers={str(key): str(value) for key, value in headers.items()},
        default_scheme=default_scheme,
    )


def build_auth_headers(
    auth: EndpointAuthConfig | None,
    *,
    include: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge default headers with optional auth headers."""
    resolved = {str(key): str(value) for key, value in (include or {}).items()}
    if auth is None:
        return resolved
    resolved.update(auth.build_headers())
    return resolved


def parse_header_values(values: list[str] | None, *, option_name: str = "header") -> dict[str, str]:
    """Parse repeated CLI values in the form Header=Value."""
    headers: dict[str, str] = {}
    for value in values or []:
        name, sep, header_value = value.partition("=")
        if not sep or not name.strip() or not header_value.strip():
            raise ValueError(f"Invalid {option_name} '{value}'. Use Header=Value")
        headers[name.strip()] = header_value.strip()
    return headers


_DEFAULT_PORTS = {"http": 80, "https": 443}


def same_origin(url_a: str, url_b: str) -> bool:
    """Return True when two URLs share scheme, host, and effective port."""
    parsed_a = urlsplit(url_a)
    parsed_b = urlsplit(url_b)
    port_a = parsed_a.port or _DEFAULT_PORTS.get(parsed_a.scheme.lower(), 0)
    port_b = parsed_b.port or _DEFAULT_PORTS.get(parsed_b.scheme.lower(), 0)
    return (
        parsed_a.scheme.lower(),
        (parsed_a.hostname or "").lower(),
        port_a,
    ) == (
        parsed_b.scheme.lower(),
        (parsed_b.hostname or "").lower(),
        port_b,
    )
