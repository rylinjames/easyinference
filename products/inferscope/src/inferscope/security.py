"""Security utilities — input validation, URL sanitization, SSRF protection.

All user-facing inputs should pass through these validators before use.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

# Maximum lengths for user inputs
MAX_MODEL_NAME_LENGTH = 256
MAX_GPU_NAME_LENGTH = 64
MAX_ENDPOINT_LENGTH = 2048

# Allowed URL schemes for engine endpoints
ALLOWED_SCHEMES = {"http", "https"}

# Private/reserved IP ranges that should be blocked for SSRF protection
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),  # RFC 791 "this network" — resolves to 127.0.0.1 from local
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local (incl. cloud metadata 169.254.169.254)
    ipaddress.ip_network("::/128"),  # IPv6 unspecified — analog of 0.0.0.0
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped IPv6 prefix (defense-in-depth; we also unwrap)
    ipaddress.ip_network("fc00::/7"),  # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]

# Hostnames that map to loopback / private targets even without an IP literal.
# Defense-in-depth supplement to the IP-range check; runs even when DNS resolution is unavailable.
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "loopback",
        "local.host",
        "broadcasthost",
    }
)


class InputValidationError(ValueError):
    """Raised when user input fails validation."""


def validate_model_name(name: str) -> str:
    """Validate and sanitize a model name input.

    Allows: alphanumeric, hyphens, dots, underscores, slashes (for HF-style names).
    """
    if not name or len(name) > MAX_MODEL_NAME_LENGTH:
        raise InputValidationError(f"Model name must be 1-{MAX_MODEL_NAME_LENGTH} characters, got {len(name)}")
    # Allow HuggingFace-style names: org/model-name_v1.2
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$", name):
        raise InputValidationError(
            f"Invalid model name: '{name}'. Allowed: alphanumeric, hyphens, dots, underscores, slashes."
        )
    return name.strip()


def validate_gpu_name(name: str) -> str:
    """Validate and sanitize a GPU name input."""
    if not name or len(name) > MAX_GPU_NAME_LENGTH:
        raise InputValidationError(f"GPU name must be 1-{MAX_GPU_NAME_LENGTH} characters, got {len(name)}")
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_ -]*$", name):
        raise InputValidationError(f"Invalid GPU name: '{name}'. Allowed: alphanumeric, hyphens, underscores, spaces.")
    return name.strip()


def validate_endpoint(endpoint: str, allow_private: bool = False) -> str:
    """Validate and sanitize an HTTP endpoint URL.

    Prevents SSRF by blocking private IP ranges (unless explicitly allowed).

    Args:
        endpoint: URL to validate (e.g., "http://localhost:8000")
        allow_private: If True, allow private/localhost IPs (for local development)

    Returns:
        Validated URL string

    Raises:
        InputValidationError: If URL is invalid or targets a blocked address
    """
    if not endpoint or len(endpoint) > MAX_ENDPOINT_LENGTH:
        raise InputValidationError(f"Endpoint must be 1-{MAX_ENDPOINT_LENGTH} characters")

    # Parse URL
    try:
        parsed = urlparse(endpoint)
    except Exception as e:
        raise InputValidationError(f"Invalid URL: {e}") from e

    # Check scheme
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise InputValidationError(f"URL scheme '{parsed.scheme}' not allowed. Use: {ALLOWED_SCHEMES}")

    # Check host exists
    if not parsed.hostname:
        raise InputValidationError("URL must have a hostname")

    # SSRF protection: block private IPs unless explicitly allowed
    if not allow_private:
        hostname_is_ip = True
        try:
            ip = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            hostname_is_ip = False

        if hostname_is_ip:
            # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1) before
            # the IP-range check. httpx/requests resolve these to the underlying
            # IPv4 address, so we must check the unwrapped form against the
            # private IPv4 networks.
            check_ip: ipaddress.IPv4Address | ipaddress.IPv6Address
            if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
                check_ip = ip.ipv4_mapped
            else:
                check_ip = ip
            for network in _PRIVATE_NETWORKS:
                if check_ip in network:
                    raise InputValidationError(
                        f"Endpoint targets private IP range ({network}). Use allow_private=True for local development."
                    )
        else:
            # hostname is not an IP — DNS resolution happens at request time.
            # Block known loopback / private hostname aliases as defense-in-depth.
            # Note: this does not catch DNS rebinding (e.g. *.nip.io, *.sslip.io)
            # because we do not resolve the hostname at validation time. For
            # network-exposed deployments, add an SSRF gate at the reverse proxy.
            hostname_lower = parsed.hostname.lower()
            if hostname_lower in _BLOCKED_HOSTNAMES:
                raise InputValidationError(
                    f"Endpoint targets blocked hostname '{hostname_lower}'. Use allow_private=True for local development."
                )

    return endpoint.rstrip("/")


def validate_positive_int(value: int, name: str, max_value: int = 1_000_000) -> int:
    """Validate that an integer is positive and within bounds."""
    if value < 0:
        raise InputValidationError(f"{name} must be non-negative, got {value}")
    if value > max_value:
        raise InputValidationError(f"{name} must be <= {max_value}, got {value}")
    return value


def validate_float_range(value: float, name: str, min_val: float = 0.0, max_val: float = 1e9) -> float:
    """Validate that a float is within bounds."""
    if value < min_val or value > max_val:
        raise InputValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value
