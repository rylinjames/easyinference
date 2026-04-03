"""Abstract base classes for engine adapters and config compilers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.optimization.serving_profile import ServingProfile


@dataclass
class EngineConfig:
    """Compiled engine-specific configuration."""

    engine: str
    cli_flags: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    command: str = ""
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    support_tier: str = "supported"
    support_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "cli_flags": self.cli_flags,
            "metadata": self.metadata,
            "env_vars": self.env_vars,
            "command": self.command,
            "warnings": self.warnings,
            "notes": self.notes,
            "support_tier": self.support_tier,
            "support_reason": self.support_reason,
        }


@dataclass
class DeploymentInventory:
    """What we know about the deployment environment."""

    gpu_type: str = ""
    gpu_arch: str = ""  # sm_80, sm_90a, sm_100, gfx942, gfx950
    gpu_count: int = 1
    gpu_memory_gb: float = 0.0
    gpu_memory_bandwidth_tb_s: float = 0.0
    interconnect: str = ""  # nvlink4, nvlink5, infinity_fabric_3, pcie_gen4, etc.
    interconnect_bandwidth_gb_s: float = 0.0
    fp8_support: bool = False
    fp4_support: bool = False
    fp8_format: str = ""  # OCP | FNUZ
    node_count: int = 1
    has_rdma: bool = False
    rdma_type: str = ""  # UCX | libfabric | EFA | ""
    has_encoder_gpu: bool = False  # Dedicated multimodal encoder GPUs available
    platform_family: str = ""
    has_grace: bool = False
    grace_memory_gb: float = 0.0
    grace_memory_bandwidth_gb_s: float = 0.0
    c2c_bandwidth_gb_s: float = 0.0
    has_decompression_engine: bool = False
    has_helix_parallelism: bool = False
    has_accelerated_softmax: bool = False
    platform_features: dict[str, Any] = field(default_factory=dict)


class ConfigCompiler(ABC):
    """Translates a normalized ServingProfile into engine-specific config.

    This is THE abstraction that makes InferScope engine-agnostic.
    """

    @abstractmethod
    def compile(self, profile: ServingProfile, inventory: DeploymentInventory) -> EngineConfig:
        """Convert normalized profile to engine-specific config."""

    @abstractmethod
    def engine_name(self) -> str:
        """Return the engine identifier."""


class EngineAdapter(ABC):
    """Connects to a running engine to scrape metrics and detect config.

    All endpoint URLs are validated before HTTP requests to prevent SSRF.
    """

    def _validate_endpoint(self, endpoint: str, *, allow_private: bool = True) -> str:
        """Validate endpoint URL before making HTTP requests."""
        from inferscope.security import validate_endpoint

        return validate_endpoint(endpoint, allow_private=allow_private)

    @abstractmethod
    async def detect_engine(self, endpoint: str) -> bool:
        """Return True if this adapter can handle the given endpoint."""

    @abstractmethod
    async def get_metrics(self, endpoint: str) -> dict[str, Any]:
        """Scrape Prometheus metrics from the engine."""

    @abstractmethod
    async def get_config(
        self,
        endpoint: str,
        *,
        allow_private: bool = True,
        auth: EndpointAuthConfig | None = None,
    ) -> dict[str, Any]:
        """Retrieve current running configuration."""

    @abstractmethod
    def engine_name(self) -> str:
        """Return the engine identifier."""
