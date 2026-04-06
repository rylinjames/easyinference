"""SSRF regression coverage for `inferscope.security.validate_endpoint`.

Each test below corresponds to a documented bypass from
`improvements/easyinference/bugs/security_ssrf_gaps.md`. Currently
`security.py` has zero direct test coverage; this file is also the
first entry in `bugs/tests_zero_coverage_pile.md` Tier 1.
"""

from __future__ import annotations

import pytest

from inferscope.security import (
    InputValidationError,
    _BLOCKED_HOSTNAMES,
    _PRIVATE_NETWORKS,
    validate_endpoint,
)


# ----------------------------------------------------------------------------
# Sub-bug 1 — `0.0.0.0` and the IPv6 unspecified address
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://0.0.0.0:8080",
        "http://0.0.0.1:8080",       # rest of 0.0.0.0/8
        "http://0.255.255.255:8080",  # rest of 0.0.0.0/8
        "http://[::]:8080",          # IPv6 unspecified
    ],
)
def test_validate_endpoint_blocks_zero_network(bad_url: str) -> None:
    """`0.0.0.0/8` and `::/128` resolve to localhost from the local machine."""
    with pytest.raises(InputValidationError, match="private IP range"):
        validate_endpoint(bad_url, allow_private=False)


# ----------------------------------------------------------------------------
# Sub-bug 2 — IPv4-mapped IPv6 addresses
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://[::ffff:127.0.0.1]:8000",
        "http://[::ffff:127.1.2.3]:8000",
        "http://[::ffff:192.168.1.1]:8000",
        "http://[::ffff:10.0.0.5]:8000",
        "http://[::ffff:172.16.0.1]:8000",
        "http://[::ffff:169.254.169.254]:8000",  # AWS metadata via IPv6 wrap
        "http://[::ffff:0.0.0.0]:8000",          # zero network via IPv6 wrap
    ],
)
def test_validate_endpoint_unwraps_ipv4_mapped_ipv6(bad_url: str) -> None:
    """IPv4-mapped IPv6 (::ffff:x.y.z.w) must be unwrapped before the IP-range check.

    httpx and requests resolve these to the underlying IPv4 address, so they
    must be blocked even though the literal value parses as an IPv6Address.
    """
    with pytest.raises(InputValidationError, match="private IP range"):
        validate_endpoint(bad_url, allow_private=False)


def test_validate_endpoint_allows_public_ipv4_mapped_ipv6() -> None:
    """IPv4-mapped IPv6 wrapping a PUBLIC IPv4 address should still pass."""
    # 8.8.8.8 wrapped: ::ffff:8.8.8.8
    assert validate_endpoint("http://[::ffff:8.8.8.8]:8000", allow_private=False)


# ----------------------------------------------------------------------------
# Sub-bug 4 — Hostname block list (loopback aliases)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://localhost:8080",
        "http://LOCALHOST:8080",  # case-insensitive
        "http://localhost.localdomain:8080",
        "http://ip6-localhost:8080",
        "http://ip6-loopback:8080",
        "http://loopback:8080",
        "http://local.host:8080",
        "http://broadcasthost:8080",
    ],
)
def test_validate_endpoint_blocks_loopback_hostnames(bad_url: str) -> None:
    """All known loopback / private hostname aliases must be blocked."""
    with pytest.raises(InputValidationError, match="blocked hostname"):
        validate_endpoint(bad_url, allow_private=False)


# ----------------------------------------------------------------------------
# Defense-in-depth — existing private IP checks still work
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:8000",
        "http://10.0.0.1:8000",
        "http://172.16.0.1:8000",
        "http://192.168.1.1:8000",
        "http://169.254.169.254:80",  # AWS / GCP / Azure metadata
        "http://[::1]:8000",          # IPv6 loopback literal
        "http://[fe80::1]:8000",      # IPv6 link-local
        "http://[fc00::1]:8000",      # IPv6 ULA
    ],
)
def test_validate_endpoint_blocks_classic_private_ips(bad_url: str) -> None:
    """The original `_PRIVATE_NETWORKS` blocks must continue to fire."""
    with pytest.raises(InputValidationError, match="private IP range"):
        validate_endpoint(bad_url, allow_private=False)


# ----------------------------------------------------------------------------
# Allowlist — public addresses should pass cleanly
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good_url",
    [
        "http://api.openai.com",
        "https://api.anthropic.com:443",
        "http://8.8.8.8",
        "http://203.0.113.10:8000",  # TEST-NET-3 reserved but public-routable form
        "https://example.com/v1/chat/completions",
    ],
)
def test_validate_endpoint_allows_public_addresses(good_url: str) -> None:
    """Public-looking addresses must pass without raising."""
    assert validate_endpoint(good_url, allow_private=False).startswith(("http://", "https://"))


# ----------------------------------------------------------------------------
# allow_private=True bypasses everything (the CLI trust boundary)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://[::ffff:127.0.0.1]:8000",
        "http://ip6-localhost:8000",
    ],
)
def test_validate_endpoint_allow_private_skips_all_checks(url: str) -> None:
    """`allow_private=True` is the CLI trust boundary — everything passes."""
    assert validate_endpoint(url, allow_private=True)


# ----------------------------------------------------------------------------
# Module-level invariants — make sure the constants stay in sync
# ----------------------------------------------------------------------------


def test_private_networks_includes_zero_network() -> None:
    """Sub-bug 1 must stay fixed: 0.0.0.0/8 must remain in _PRIVATE_NETWORKS."""
    nets = {str(n) for n in _PRIVATE_NETWORKS}
    assert "0.0.0.0/8" in nets


def test_private_networks_includes_ipv4_mapped_ipv6_prefix() -> None:
    """Sub-bug 2 must stay fixed: the ::ffff:0:0/96 prefix must remain covered
    in _PRIVATE_NETWORKS as a defense-in-depth supplement to the unwrap logic.

    Test by membership rather than string equality because Python's
    `ipaddress` normalizes the network's string form."""
    import ipaddress

    sample = ipaddress.ip_address("::ffff:127.0.0.1")
    assert any(sample in net for net in _PRIVATE_NETWORKS), (
        "::ffff:0:0/96 prefix not present in _PRIVATE_NETWORKS"
    )


def test_blocked_hostnames_covers_all_known_loopback_aliases() -> None:
    """Sub-bug 4 must stay fixed: every known loopback alias must be in the set."""
    for alias in ("localhost", "ip6-localhost", "ip6-loopback", "loopback", "broadcasthost"):
        assert alias in _BLOCKED_HOSTNAMES, f"{alias!r} missing from _BLOCKED_HOSTNAMES"
