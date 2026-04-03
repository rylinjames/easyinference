"""KV disaggregated transfer profiling — NIXL transfer overhead measurement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransferPoint:
    """Transfer metrics at a specific sequence length."""
    isl: int
    transfer_latency_ms: float | None = None
    transfer_bandwidth_gbps: float | None = None
    decode_idle_fraction: float | None = None
    nixl_failures: float = 0.0
    kvbm_offload_events: float = 0.0
    kvbm_onboard_events: float = 0.0
    confidence_kind: str = "direct"


@dataclass
class KVDisaggTransferResult:
    """Result of disaggregated KV transfer profiling."""
    model_name: str
    topology: str
    transfer_curve: list[TransferPoint] = field(default_factory=list)
    avg_decode_idle_fraction: float | None = None
    transport_type: str = ""  # "nixl_rdma" | "nixl_nvlink" | "p2p_nccl" | "lmcache"
    support_tier: str = ""
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "topology": self.topology,
            "transfer_curve": [
                {
                    "isl": t.isl,
                    "transfer_latency_ms": t.transfer_latency_ms,
                    "transfer_bandwidth_gbps": round(t.transfer_bandwidth_gbps, 2) if t.transfer_bandwidth_gbps else None,
                    "decode_idle_fraction": round(t.decode_idle_fraction, 3) if t.decode_idle_fraction else None,
                    "nixl_failures": t.nixl_failures,
                    "confidence_kind": t.confidence_kind,
                }
                for t in self.transfer_curve
            ],
            "avg_decode_idle_fraction": round(self.avg_decode_idle_fraction, 3) if self.avg_decode_idle_fraction else None,
            "transport_type": self.transport_type,
            "support_tier": self.support_tier,
            "warnings": self.warnings,
            "summary": self.summary,
        }
