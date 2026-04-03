"""Abstract base class and data structures for ISB-1 workload generators."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Request:
    """A single inference request in OpenAI chat-completion format.

    Attributes:
        request_id: Unique identifier for this request.
        messages: Conversation history in OpenAI chat format
            (list of dicts with ``role`` and ``content`` keys).
        expected_output_tokens: Estimated number of tokens the model should
            produce for this request, used for capacity planning.
        session_id: Optional identifier grouping requests that belong to the
            same logical conversation or session.
        metadata: Arbitrary key-value metadata attached to the request
            (e.g. workload type, turn index, context length).
    """

    request_id: str
    messages: list[dict[str, Any]]
    expected_output_tokens: int
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Request:
        """Construct a :class:`Request` from a plain dictionary."""
        return cls(
            request_id=data["request_id"],
            messages=data["messages"],
            expected_output_tokens=data["expected_output_tokens"],
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )


def _new_request_id(rng=None) -> str:
    """Generate a short request identifier.

    When *rng* (a ``numpy.random.Generator``) is provided, the ID is
    deterministic for a given RNG state.  Otherwise falls back to
    ``uuid.uuid4`` for backwards compatibility.
    """
    if rng is not None:
        return rng.bytes(8).hex()
    return uuid.uuid4().hex[:16]


class WorkloadGenerator(ABC):
    """Base class for all ISB-1 workload trace generators.

    Subclasses must implement :meth:`generate` which produces a list of
    :class:`Request` objects.  The base class provides JSONL persistence
    via :meth:`save` and :meth:`load`.

    Parameters:
        seed: Random seed passed to ``numpy.random.default_rng`` for full
            reproducibility of generated traces.
    """

    def __init__(self, seed: int = 42) -> None:
        # Import here so the module is importable even without numpy installed,
        # but generation will fail fast with a clear error.
        import numpy as np

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, num_requests: int) -> list[Request]:
        """Generate *num_requests* inference requests.

        Args:
            num_requests: Number of :class:`Request` objects to produce.

        Returns:
            A list of :class:`Request` instances ready for the benchmark
            harness.
        """

    # ------------------------------------------------------------------
    # JSONL persistence
    # ------------------------------------------------------------------

    def save(self, requests: list[Request], path: str | Path) -> None:
        """Write a list of requests to a JSONL file.

        Each line is a self-contained JSON object representing one
        :class:`Request`.

        Args:
            requests: The request trace to persist.
            path: Destination file path.  Parent directories are created
                automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for req in requests:
                fh.write(json.dumps(req.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: str | Path) -> list[Request]:
        """Load a request trace from a JSONL file.

        Args:
            path: Path to a JSONL file previously written by :meth:`save`.

        Returns:
            A list of :class:`Request` reconstructed from the file.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        requests: list[Request] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    requests.append(Request.from_dict(json.loads(line)))
        return requests
