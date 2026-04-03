"""Chat workload generator for ISB-1 benchmarks.

Produces multi-turn conversational request traces.  When a ShareGPT
dataset is available the generator samples real conversations; otherwise
it falls back to fully synthetic generation using a realistic English
vocabulary bank.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from workloads.base import Request, WorkloadGenerator, _new_request_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ShareGPT auto-download
# ---------------------------------------------------------------------------

_SHAREGPT_HF_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
_SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


def _resolve_sharegpt_path(path: Path) -> Path:
    """Return the path to a ShareGPT JSON file, downloading if necessary.

    Download locations tried in order:
    1. *path* as given (already exists)
    2. ``~/.cache/isb1/{filename}`` (cached download)

    If neither exists, downloads from HuggingFace to the cache dir.
    """
    if path.is_file():
        return path

    # Check cache
    cache_dir = Path.home() / ".cache" / "isb1"
    cached = cache_dir / _SHAREGPT_FILENAME
    if cached.is_file():
        logger.info("Using cached ShareGPT dataset at %s", cached)
        return cached

    # Download
    logger.info("Downloading ShareGPT V3 dataset (~600 MB)...")
    try:
        import urllib.request

        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".tmp")
        urllib.request.urlretrieve(_SHAREGPT_HF_URL, tmp)  # noqa: S310
        tmp.rename(cached)
        logger.info("ShareGPT dataset saved to %s", cached)
        return cached
    except Exception:
        logger.warning("Failed to download ShareGPT dataset, falling back to synthetic", exc_info=True)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return path  # Return original path — _try_load_sharegpt will see it doesn't exist

# ---------------------------------------------------------------------------
# Realistic vocabulary bank (NOT lorem ipsum)
# ---------------------------------------------------------------------------

_TOPICS = [
    "machine learning model training",
    "distributed systems architecture",
    "personal financial planning",
    "healthy meal preparation",
    "European travel itineraries",
    "home renovation projects",
    "effective team management strategies",
    "climate change mitigation policies",
    "mobile application development",
    "scientific research methodology",
    "creative writing techniques",
    "electric vehicle maintenance",
    "supply chain optimisation",
    "college admission strategies",
    "mental health and mindfulness practices",
    "open-source software licensing",
    "urban gardening in small spaces",
    "cybersecurity best practices",
    "data pipeline engineering",
    "remote work productivity tips",
]

_USER_OPENERS = [
    "Can you explain how {topic} works in practice?",
    "I need help understanding the fundamentals of {topic}.",
    "What are the most important considerations when dealing with {topic}?",
    "Could you walk me through a typical approach to {topic}?",
    "I've been reading about {topic} and have some questions.",
    "What would you recommend as a starting point for {topic}?",
    "I'm working on a project related to {topic}. Any advice?",
    "How do experts typically handle challenges in {topic}?",
    "What are the common pitfalls to avoid with {topic}?",
    "I'd like a comprehensive overview of {topic}.",
]

_USER_FOLLOWUPS = [
    "That makes sense. Can you elaborate on the part about {aspect}?",
    "Interesting. How does that compare to alternative approaches?",
    "What are the potential risks or downsides of this approach?",
    "Could you provide a concrete example to illustrate that?",
    "Thanks. What tools or resources would you recommend for this?",
    "I see. How would this apply in a real-world scenario?",
    "That's helpful. What should I prioritise first?",
    "Are there any recent developments that change this advice?",
    "How long does it typically take to see results with this method?",
    "What metrics should I track to measure progress?",
    "Can you break that down into smaller, actionable steps?",
    "How does the cost compare across these different options?",
    "What's the difference between the beginner and advanced approach?",
    "Is there a way to automate some of these steps?",
    "What would change if the budget were significantly smaller?",
]

_ASPECTS = [
    "the implementation details",
    "the cost-benefit analysis",
    "the timeline and milestones",
    "the performance characteristics",
    "the scalability requirements",
    "the initial setup process",
    "the monitoring strategy",
    "the integration with existing systems",
    "the security implications",
    "the maintenance overhead",
]

_ASSISTANT_SENTENCES = [
    "The key factor here is understanding the relationship between the variables involved.",
    "In practice, most teams start with a straightforward approach and iterate from there.",
    "There are several well-established frameworks that address this exact problem.",
    "Research consistently shows that incremental improvements yield the best long-term results.",
    "The most effective strategy depends on your specific constraints and objectives.",
    "A common mistake is underestimating the time required for proper testing and validation.",
    "Industry best practices suggest starting with a minimal viable approach.",
    "The trade-off between complexity and maintainability is central to this decision.",
    "Performance benchmarks indicate that the recommended approach handles most workloads well.",
    "Documentation and knowledge sharing are often overlooked but critically important.",
    "Experienced practitioners typically recommend a phased rollout strategy.",
    "Automated tooling can significantly reduce the manual effort involved.",
    "The underlying principles remain consistent even as specific technologies evolve.",
    "Cross-functional collaboration is essential for achieving sustainable results.",
    "Regular review cycles help identify issues before they become significant problems.",
    "Data-driven decision making is fundamental to optimising outcomes.",
    "Modularity in design allows components to be updated independently.",
    "Version control and reproducibility should be considered from the start.",
    "Clear communication of requirements prevents costly misunderstandings later.",
    "The total cost of ownership includes ongoing maintenance and support.",
]

_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. Provide clear, accurate, "
    "and well-structured responses. When appropriate, use examples to "
    "illustrate your points. Be concise but thorough."
)


def _build_assistant_response(rng: np.random.Generator, n_sentences: int) -> str:
    """Compose a synthetic assistant reply from the sentence bank."""
    indices = rng.choice(len(_ASSISTANT_SENTENCES), size=n_sentences, replace=False)
    return " ".join(_ASSISTANT_SENTENCES[i] for i in indices)


class ChatWorkloadGenerator(WorkloadGenerator):
    """Generate multi-turn chat workload traces.

    If *sharegpt_path* points to an existing ShareGPT-format JSON file the
    generator samples real conversations (filtered to 1-5 turns).  Otherwise
    it synthesises conversations from a curated English vocabulary bank.

    Parameters:
        seed: Random seed for reproducibility.
        sharegpt_path: Optional path to a ShareGPT JSON dataset.
        min_turns: Minimum number of user turns per conversation (default 1).
        max_turns: Maximum number of user turns per conversation (default 5).
    """

    def __init__(
        self,
        seed: int = 42,
        sharegpt_path: str | Path | None = None,
        min_turns: int = 1,
        max_turns: int = 5,
    ) -> None:
        super().__init__(seed=seed)
        self.min_turns = min_turns
        self.max_turns = max_turns

        self._sharegpt_data: list[list[dict[str, str]]] | None = None
        if sharegpt_path is not None:
            resolved = _resolve_sharegpt_path(Path(sharegpt_path))
            self._try_load_sharegpt(resolved)

    # ------------------------------------------------------------------
    # ShareGPT loading
    # ------------------------------------------------------------------

    def _try_load_sharegpt(self, path: Path) -> None:
        """Attempt to load and filter a ShareGPT JSON dataset."""
        if not path.is_file():
            return
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return

        filtered: list[list[dict[str, str]]] = []
        for entry in raw:
            convos = entry.get("conversations", [])
            # Count user turns
            user_turns = sum(1 for m in convos if m.get("from") == "human")
            if self.min_turns <= user_turns <= self.max_turns:
                filtered.append(convos)

        if filtered:
            self._sharegpt_data = filtered

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, num_requests: int) -> list[Request]:
        """Generate *num_requests* chat requests across multi-turn sessions.

        Each conversation (session) produces one :class:`Request` per user
        turn, with messages accumulating to simulate growing context.

        Args:
            num_requests: Target number of requests to produce.  The actual
                count may be slightly higher to avoid splitting a
                conversation mid-way.

        Returns:
            A list of :class:`Request` in session-sequential order.
        """
        requests: list[Request] = []

        while len(requests) < num_requests:
            if self._sharegpt_data is not None:
                session_requests = self._generate_from_sharegpt()
            else:
                session_requests = self._generate_synthetic()
            requests.extend(session_requests)

        return requests[:num_requests]

    def _generate_from_sharegpt(self) -> list[Request]:
        """Create a session from a randomly sampled ShareGPT conversation."""
        idx = int(self.rng.integers(0, len(self._sharegpt_data)))  # type: ignore[arg-type]
        convo = self._sharegpt_data[idx]  # type: ignore[index]
        session_id = self.rng.bytes(6).hex()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        requests: list[Request] = []
        turn = 0

        for entry in convo:
            role = "user" if entry.get("from") == "human" else "assistant"
            content = entry.get("value", "")
            messages.append({"role": role, "content": content})

            if role == "user":
                # Estimate output tokens from the next assistant message
                expected = 256
                for future in convo[convo.index(entry) + 1 :]:
                    if future.get("from") != "human":
                        expected = max(64, len(future.get("value", "")) // 4)
                        break

                requests.append(
                    Request(
                        request_id=_new_request_id(self.rng),
                        messages=list(messages),
                        expected_output_tokens=expected,
                        session_id=session_id,
                        metadata={
                            "workload": "chat",
                            "source": "sharegpt",
                            "turn": turn,
                        },
                    )
                )
                turn += 1

        return requests

    def _generate_synthetic(self) -> list[Request]:
        """Create a fully synthetic multi-turn conversation."""
        num_turns = int(self.rng.integers(self.min_turns, self.max_turns + 1))
        session_id = self.rng.bytes(6).hex()
        topic = _TOPICS[int(self.rng.integers(0, len(_TOPICS)))]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        requests: list[Request] = []

        for turn in range(num_turns):
            # ---- user message ----
            if turn == 0:
                template = _USER_OPENERS[
                    int(self.rng.integers(0, len(_USER_OPENERS)))
                ]
                user_text = template.format(topic=topic)
            else:
                template = _USER_FOLLOWUPS[
                    int(self.rng.integers(0, len(_USER_FOLLOWUPS)))
                ]
                aspect = _ASPECTS[int(self.rng.integers(0, len(_ASPECTS)))]
                user_text = template.format(aspect=aspect)

            messages.append({"role": "user", "content": user_text})

            # Estimate output length (longer for first turn, shorter follow-ups)
            n_sentences = int(self.rng.integers(3, 8)) if turn == 0 else int(
                self.rng.integers(2, 5)
            )
            expected_output_tokens = n_sentences * 30  # rough estimate

            requests.append(
                Request(
                    request_id=_new_request_id(self.rng),
                    messages=list(messages),
                    expected_output_tokens=expected_output_tokens,
                    session_id=session_id,
                    metadata={
                        "workload": "chat",
                        "source": "synthetic",
                        "turn": turn,
                        "topic": topic,
                    },
                )
            )

            # ---- synthetic assistant reply (added to context for next turn) ----
            assistant_text = _build_assistant_response(self.rng, n_sentences)
            messages.append({"role": "assistant", "content": assistant_text})

        return requests
