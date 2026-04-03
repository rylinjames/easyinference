"""Production-grade structured logging for InferScope.

Uses structlog for JSON-structured output in production and
human-readable colored output in development.

Usage:
    from inferscope.logging import get_logger
    log = get_logger(component="recommender")
    log.info("recommendation_generated", model="DeepSeek-R1", gpu="mi355x", engine="atom")
"""

from __future__ import annotations

import os
import re
import sys
from typing import cast
from urllib.parse import urlsplit, urlunsplit

import structlog

REDACTED = "[REDACTED]"
_URL_PATTERN = re.compile(r"https?://[^\s'\"<>]+")
_BEARER_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9._-]+")
_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "headers",
    "payload",
    "messages",
    "body",
    "prompt",
    "content",
    "response_body",
}
_URL_KEYS = {"endpoint", "metrics_endpoint", "url", "otlp_endpoint"}


def sanitize_log_url(value: str) -> str:
    """Strip credentials and query parameters from URLs before logging."""
    try:
        parsed = urlsplit(value)
    except Exception:
        return value
    if not parsed.scheme or not parsed.netloc:
        return value
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def sanitize_log_text(value: str) -> str:
    """Sanitize free-form text before it is emitted to structured logs."""
    sanitized = _URL_PATTERN.sub(lambda match: sanitize_log_url(match.group(0)), value)
    sanitized = _BEARER_PATTERN.sub(f"Bearer {REDACTED}", sanitized)
    return sanitized


def _redact_value(key: str, value: object) -> object:
    normalized_key = key.lower()
    if normalized_key in _SENSITIVE_KEYS:
        return REDACTED
    if normalized_key in _URL_KEYS and isinstance(value, str):
        return sanitize_log_url(value)
    if normalized_key in {"error", "error_summary", "message"} and isinstance(value, str):
        return sanitize_log_text(value)
    if isinstance(value, dict):
        return {inner_key: _redact_value(str(inner_key), inner_value) for inner_key, inner_value in value.items()}
    if isinstance(value, list):
        return [_redact_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(key, item) for item in value)
    return value


def redact_sensitive_fields(
    _: structlog.types.WrappedLogger,
    __: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Redact sensitive fields before rendering structured logs."""
    return {key: _redact_value(key, value) for key, value in event_dict.items()}


def configure_logging(json_output: bool | None = None, level: str = "INFO") -> None:
    """Configure structlog for the application."""
    if json_output is None:
        env_format = os.getenv("INFERSCOPE_LOG_FORMAT", "").lower()
        if env_format == "json":
            json_output = True
        elif env_format == "console":
            json_output = False
        else:
            json_output = not sys.stderr.isatty()

    log_level = os.getenv("INFERSCOPE_LOG_LEVEL", level).upper()
    level_map = getattr(structlog.processors, "NAME_TO_LEVEL", None) or getattr(
        structlog.processors, "_NAME_TO_LEVEL", {}
    )
    numeric_level = level_map.get(log_level.lower(), 20)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_sensitive_fields,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            context_class=dict,
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=False,
        )
    else:
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            context_class=dict,
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=False,
        )


def get_logger(**initial_context: object) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with optional initial context."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(**initial_context))


configure_logging()
