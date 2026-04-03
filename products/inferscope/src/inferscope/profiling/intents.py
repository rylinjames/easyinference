"""Advisory profiling intents exposed by the recommender DAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProfilingTool = Literal["nsys", "rocprofv3"]
ProfilingMode = Literal["advisory"]


@dataclass(frozen=True)
class ProfilingIntent:
    """A lightweight description of which profiling path fits the current GPU vendor."""

    tool: ProfilingTool
    mode: ProfilingMode = "advisory"
    summary: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "tool": self.tool,
            "mode": self.mode,
            "summary": self.summary,
        }


def resolve_profiling_intent(gpu_vendor: str) -> ProfilingIntent:
    """Resolve the advisory profiling tool for a GPU vendor."""
    vendor = gpu_vendor.strip().lower()
    if vendor == "amd":
        return ProfilingIntent(
            tool="rocprofv3",
            summary=(
                "ProfilingNode: Routed future kernel/profiling work to the rocprofv3 "
                "advisory seam (external execution remains out-of-band)."
            ),
        )
    return ProfilingIntent(
        tool="nsys",
        summary=(
            "ProfilingNode: Routed future kernel/profiling work to the nsys "
            "advisory seam (external execution remains out-of-band)."
        ),
    )
