"""Telemetry collection — Prometheus, DCGM, AMD DME adapters."""

from inferscope.telemetry.failure_taxonomy import ClassifiedFailure, FailureMode, classify_failure_modes, dominant_failure_mode

__all__ = [
    "ClassifiedFailure",
    "FailureMode",
    "classify_failure_modes",
    "dominant_failure_mode",
]
