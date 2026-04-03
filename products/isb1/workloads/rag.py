"""RAG (Retrieval-Augmented Generation) trace generator for ISB-1 benchmarks.

Produces synthetic RAG queries with long contexts assembled from multiple
retrieved document chunks.  Context lengths follow a bimodal distribution:
60 % of requests target ~32K tokens and 40 % target ~96K tokens.
"""

from __future__ import annotations

from typing import Any


from workloads.base import Request, WorkloadGenerator, _new_request_id

# ---------------------------------------------------------------------------
# Realistic vocabulary and sentence banks (NOT lorem ipsum)
# ---------------------------------------------------------------------------

_DOMAIN_TOPICS = [
    "distributed computing", "natural language processing",
    "database management systems", "network security protocols",
    "cloud infrastructure design", "software testing methodologies",
    "machine learning pipelines", "data warehousing strategies",
    "microservices architecture", "real-time analytics platforms",
    "compiler optimisation techniques", "operating system internals",
    "cryptographic protocols", "recommendation systems",
    "computer vision applications", "reinforcement learning",
    "graph database modelling", "event-driven architectures",
    "continuous integration pipelines", "performance engineering",
]

_DOC_TITLES = [
    "Technical Architecture Overview: {topic}",
    "Best Practices for Production {topic}",
    "{topic}: A Comprehensive Survey",
    "Scaling {topic} in Enterprise Environments",
    "Lessons Learned: Deploying {topic} at Scale",
    "Performance Analysis of {topic} Systems",
    "{topic}: Design Patterns and Anti-Patterns",
    "Security Considerations for {topic}",
    "Monitoring and Observability in {topic}",
    "Cost Optimisation Strategies for {topic}",
    "Migration Guide: Transitioning to Modern {topic}",
    "Troubleshooting Common Issues in {topic}",
]

# Sentence fragments that can be combined to form realistic technical prose.
# Each list is a pool; the generator composes paragraphs by sampling and
# concatenating sentences.

_INTRO_SENTENCES = [
    "This document provides a detailed overview of the key concepts and design decisions.",
    "The following sections describe the architecture, implementation details, and operational considerations.",
    "Understanding the fundamental principles is essential before proceeding with implementation.",
    "The system was designed to handle high throughput while maintaining low latency.",
    "Several iterations of the design were evaluated before settling on the current approach.",
    "The requirements gathered from stakeholders emphasised reliability and scalability.",
]

_BODY_SENTENCES = [
    "The primary component processes incoming requests through a multi-stage pipeline that validates, transforms, and routes data to the appropriate downstream services.",
    "Load balancing is achieved through a combination of consistent hashing and weighted round-robin algorithms that distribute traffic based on real-time capacity metrics.",
    "The caching layer sits between the application tier and the persistence layer, reducing database load by approximately 80 percent for read-heavy workloads.",
    "Fault tolerance is implemented using a circuit breaker pattern that monitors error rates and automatically diverts traffic when a downstream service becomes degraded.",
    "The data model uses a normalised schema for transactional operations and a denormalised representation for analytical queries, bridged by a change data capture pipeline.",
    "Authentication follows the OAuth 2.0 authorisation code flow with PKCE, ensuring that tokens are never exposed to client-side code.",
    "Monitoring relies on a combination of structured logging, distributed tracing, and custom metrics exported to a time-series database for real-time dashboards.",
    "The deployment pipeline runs automated tests at three levels: unit tests covering individual functions, integration tests verifying service interactions, and end-to-end tests simulating user workflows.",
    "Configuration management uses a hierarchical approach where environment-specific overrides are merged with sensible defaults at startup.",
    "The message queue decouples producers from consumers, allowing each to scale independently and providing a natural buffer during traffic spikes.",
    "Index maintenance runs as a background process during off-peak hours, rebuilding fragmented indices to maintain consistent query performance.",
    "Schema migrations are applied through a versioned migration framework that supports both forward and rollback operations.",
    "Rate limiting is enforced at the API gateway level using a sliding window counter algorithm with configurable thresholds per client.",
    "The search subsystem uses an inverted index with BM25 scoring, augmented by semantic similarity from dense vector embeddings.",
    "Data retention policies are enforced automatically, with cold data migrated to object storage after 90 days and purged after the regulatory retention period.",
    "The notification service supports multiple delivery channels including email, SMS, push notifications, and webhook callbacks.",
    "Secrets management follows a zero-trust model where credentials are injected at runtime from a vault service and never stored on disk.",
    "The batch processing framework partitions workloads across available nodes using a work-stealing scheduler that adapts to heterogeneous processing speeds.",
    "Network policies restrict pod-to-pod communication to explicitly allowed paths, reducing the blast radius of a potential compromise.",
    "The feature flag system enables gradual rollouts by targeting specific user segments based on configurable predicates.",
    "Connection pooling is managed by a dedicated sidecar proxy that maintains persistent connections to upstream services.",
    "The event sourcing pattern captures every state change as an immutable event, enabling full audit trails and temporal queries.",
    "Horizontal scaling is triggered by autoscaler rules that monitor CPU utilisation, request queue depth, and custom application metrics.",
    "The API versioning strategy uses URL path segments for major versions and content negotiation headers for minor revisions.",
    "Disaster recovery procedures include automated failover to a secondary region with a recovery time objective of under five minutes.",
    "The ETL pipeline processes approximately two million records per hour, applying data quality checks and enrichment transformations at each stage.",
    "Memory management in the hot path uses arena allocation to minimise garbage collection pauses during latency-sensitive operations.",
    "The consensus protocol ensures that all replicas agree on the order of operations, providing linearisable read and write semantics.",
    "Capacity planning uses historical growth curves extrapolated with seasonal adjustments to forecast resource requirements six months ahead.",
    "The observability stack correlates logs, metrics, and traces through a shared request identifier propagated via HTTP headers.",
]

_CONCLUSION_SENTENCES = [
    "In summary, the architecture balances performance, reliability, and maintainability through well-established design patterns.",
    "Future work will focus on reducing operational complexity and improving automated recovery mechanisms.",
    "The metrics collected over the past quarter confirm that the system meets all published service-level objectives.",
    "Ongoing improvements to the monitoring stack will provide deeper visibility into long-tail latency issues.",
    "Teams adopting this approach should plan for an initial ramp-up period as operational procedures are established.",
    "Feedback from production operations has been incorporated into the next iteration of the design.",
]

# ---------------------------------------------------------------------------
# User query templates for RAG scenarios
# ---------------------------------------------------------------------------

_USER_QUERIES = [
    "Based on the retrieved documents, what is the recommended approach for {topic}?",
    "Summarise the key architectural decisions described in the context above.",
    "What are the main trade-offs discussed in the provided documents regarding {topic}?",
    "Using the context provided, explain how {topic} handles fault tolerance.",
    "According to the documents, what monitoring strategy is recommended for {topic}?",
    "What security considerations are highlighted in the retrieved context?",
    "Compare the different approaches to {topic} mentioned in the documents.",
    "Extract the most important performance metrics from the provided context.",
    "Based on the context, what are the prerequisites for implementing {topic}?",
    "Identify any potential risks or limitations described in the documents.",
    "What deployment strategy do the documents recommend for {topic}?",
    "Summarise the data management practices described in the retrieved context.",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_RAG_SYSTEM_PROMPT = (
    "You are a knowledgeable technical assistant. The user's question will be "
    "accompanied by relevant document chunks retrieved from an internal knowledge "
    "base. Use ONLY the information provided in the retrieved context to answer "
    "the question. If the context does not contain sufficient information, say so "
    "explicitly. Cite specific sections when possible. Provide clear, well-organised "
    "responses."
)

_RETRIEVAL_INSTRUCTION = (
    "The following document chunks were retrieved from the knowledge base and are "
    "provided as context for answering the user's question. Each chunk includes a "
    "source identifier and relevance score.\n"
)


class RAGTraceGenerator(WorkloadGenerator):
    """Generate synthetic RAG workload traces with long retrieval contexts.

    Each request contains a system prompt, a retrieval instruction block,
    5-20 document chunks (each 1 000-5 000 tokens), and a user query.
    Context lengths are drawn from a bimodal distribution: 60 % target
    approximately 32K tokens and 40 % target approximately 96K tokens.

    Parameters:
        seed: Random seed for reproducibility.
        min_chunks: Minimum number of document chunks per request (default 5).
        max_chunks: Maximum number of document chunks per request (default 20).
    """

    def __init__(
        self,
        seed: int = 42,
        min_chunks: int = 5,
        max_chunks: int = 20,
    ) -> None:
        super().__init__(seed=seed)
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks

    # ------------------------------------------------------------------
    # Chunk generation
    # ------------------------------------------------------------------

    def _generate_paragraph(self, target_sentences: int) -> str:
        """Build a paragraph from the sentence bank."""
        indices = self.rng.choice(
            len(_BODY_SENTENCES),
            size=min(target_sentences, len(_BODY_SENTENCES)),
            replace=False,
        )
        return " ".join(_BODY_SENTENCES[i] for i in indices)

    def _generate_chunk(self, chunk_idx: int, topic: str, target_tokens: int) -> str:
        """Generate a single document chunk with a target token count.

        One token is roughly 4 characters of English text, so we target
        ``target_tokens * 4`` characters.
        """
        target_chars = target_tokens * 4

        title_template = _DOC_TITLES[int(self.rng.integers(0, len(_DOC_TITLES)))]
        title = title_template.format(topic=topic.title())
        relevance = round(float(self.rng.uniform(0.70, 0.99)), 2)

        header = (
            f"[Source: doc_{chunk_idx:03d} | Title: \"{title}\" | "
            f"Relevance: {relevance}]\n\n"
        )

        # Opening sentence
        intro = _INTRO_SENTENCES[int(self.rng.integers(0, len(_INTRO_SENTENCES)))]
        body_parts: list[str] = [intro]
        current_chars = len(header) + len(intro)

        # Fill with body paragraphs until we approach the target
        while current_chars < target_chars * 0.90:
            n_sentences = int(self.rng.integers(3, 8))
            paragraph = self._generate_paragraph(n_sentences)
            body_parts.append(paragraph)
            current_chars += len(paragraph) + 2  # account for newlines

        # Conclusion
        conclusion = _CONCLUSION_SENTENCES[
            int(self.rng.integers(0, len(_CONCLUSION_SENTENCES)))
        ]
        body_parts.append(conclusion)

        return header + "\n\n".join(body_parts)

    # ------------------------------------------------------------------
    # Bimodal context length selection
    # ------------------------------------------------------------------

    def _select_context_params(self) -> tuple[int, int]:
        """Return (num_chunks, tokens_per_chunk) following the bimodal distribution.

        60 % of requests target ~32K tokens, 40 % target ~96K tokens.
        """
        if self.rng.random() < 0.60:
            # Short context: ~32K tokens
            target_total = int(self.rng.normal(loc=32000, scale=3000))
            target_total = max(16000, min(48000, target_total))
        else:
            # Long context: ~96K tokens
            target_total = int(self.rng.normal(loc=96000, scale=8000))
            target_total = max(64000, min(128000, target_total))

        # Choose number of chunks within bounds
        num_chunks = int(self.rng.integers(self.min_chunks, self.max_chunks + 1))
        tokens_per_chunk = max(1000, min(5000, target_total // num_chunks))
        return num_chunks, tokens_per_chunk

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, num_requests: int) -> list[Request]:
        """Generate *num_requests* RAG requests with long retrieval contexts.

        Args:
            num_requests: Number of requests to produce.

        Returns:
            A list of :class:`Request` instances, each containing a system
            prompt, retrieval instruction, document chunks, and user query.
        """
        requests: list[Request] = []

        for _ in range(num_requests):
            topic = _DOMAIN_TOPICS[int(self.rng.integers(0, len(_DOMAIN_TOPICS)))]
            num_chunks, tokens_per_chunk = self._select_context_params()

            # Build chunks
            chunks: list[str] = []
            total_chunk_tokens = 0
            for ci in range(num_chunks):
                # Vary individual chunk sizes around the mean
                chunk_tokens = int(
                    self.rng.integers(
                        max(1000, tokens_per_chunk - 1000),
                        min(5000, tokens_per_chunk + 1000) + 1,
                    )
                )
                chunk = self._generate_chunk(ci, topic, chunk_tokens)
                chunks.append(chunk)
                total_chunk_tokens += len(chunk) // 4  # approximate token count

            # Assemble context
            context_block = _RETRIEVAL_INSTRUCTION + "\n---\n".join(chunks)

            # User query
            query_template = _USER_QUERIES[
                int(self.rng.integers(0, len(_USER_QUERIES)))
            ]
            user_query = query_template.format(topic=topic)

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"{context_block}\n\n{user_query}"},
            ]

            expected_output = int(self.rng.integers(200, 800))

            requests.append(
                Request(
                    request_id=_new_request_id(self.rng),
                    messages=messages,
                    expected_output_tokens=expected_output,
                    session_id=None,
                    metadata={
                        "workload": "rag",
                        "topic": topic,
                        "num_chunks": num_chunks,
                        "approx_context_tokens": total_chunk_tokens,
                        "target_tokens_per_chunk": tokens_per_chunk,
                    },
                )
            )

        return requests
