"""Rollout-diff: detect log-prob divergence between training and serving engines.

Compares per-token log-probabilities from an RL training rollout (veRL/OpenRLHF)
against a serving engine replay (vLLM/SGLang) of the same prompt+completion pairs.
Surfaces numerical parity bugs that cause silent RL degradation (e.g., the "!!!!"
collapse documented in veRL issues #891, #747, #751).
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TokenLogProb(BaseModel):
    """A single token with its log-probability from one engine."""

    model_config = ConfigDict(extra="forbid")

    position: int
    token_id: int
    token_text: str
    log_prob: float


class RolloutEntry(BaseModel):
    """One prompt+completion pair with per-token log-probs."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    prompt: str
    completion: str
    tokens: list[TokenLogProb]
    model: str = ""
    engine: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenDivergence(BaseModel):
    """Per-token divergence between training and serving log-probs."""

    model_config = ConfigDict(extra="forbid")

    position: int
    token_id: int
    token_text: str
    training_log_prob: float
    serving_log_prob: float
    abs_delta: float
    relative_delta: float | None = None


class RequestDiff(BaseModel):
    """Divergence summary for one request."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    total_tokens: int
    divergent_tokens: int
    max_abs_delta: float
    mean_abs_delta: float
    divergent_positions: list[int]
    top_divergences: list[TokenDivergence]
    severity: Literal["clean", "drift", "critical"]


class RolloutDiffSummary(BaseModel):
    """Aggregate summary across all diffed requests."""

    model_config = ConfigDict(extra="forbid")

    total_requests: int
    total_tokens: int
    divergent_requests: int
    divergent_tokens: int
    mean_abs_delta: float
    max_abs_delta: float
    p50_abs_delta: float
    p95_abs_delta: float
    p99_abs_delta: float
    severity_counts: dict[str, int]
    position_histogram: dict[str, int]


class RolloutDiffArtifact(BaseModel):
    """Serializable artifact for a rollout-diff analysis."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: str = "1"
    artifact_type: str = "rollout_diff"
    diff_id: str = Field(default_factory=lambda: uuid4().hex)
    training_source: str
    serving_source: str
    threshold: float
    model: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    summary: RolloutDiffSummary
    request_diffs: list[RequestDiff]

    def save_json(self, path: str | Path) -> Path:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(
            self.model_dump(mode="json"), indent=2, allow_nan=False,
        ))
        return file_path

    @property
    def default_filename(self) -> str:
        return f"rollout-diff-{self.diff_id[:12]}.json"


def load_rollout_log(path: str | Path) -> list[RolloutEntry]:
    """Load a JSONL rollout log file."""
    entries: list[RolloutEntry] = []
    file_path = Path(path)
    for line in file_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        entries.append(RolloutEntry.model_validate(data))
    return entries


def _classify_severity(max_delta: float, mean_delta: float, threshold: float) -> Literal["clean", "drift", "critical"]:
    if max_delta < threshold:
        return "clean"
    if mean_delta > threshold * 5 or max_delta > threshold * 20:
        return "critical"
    return "drift"


def _bucket_position(position: int) -> str:
    if position < 10:
        return "0-9"
    if position < 50:
        return "10-49"
    if position < 200:
        return "50-199"
    if position < 500:
        return "200-499"
    return "500+"


def diff_request(
    training: RolloutEntry,
    serving: RolloutEntry,
    *,
    threshold: float = 0.01,
    top_k: int = 10,
) -> RequestDiff:
    """Diff per-token log-probs between a training and serving rollout of the same request."""
    training_by_pos = {t.position: t for t in training.tokens}
    serving_by_pos = {t.position: t for t in serving.tokens}
    common_positions = sorted(set(training_by_pos) & set(serving_by_pos))

    divergences: list[TokenDivergence] = []
    all_deltas: list[float] = []

    for pos in common_positions:
        t_tok = training_by_pos[pos]
        s_tok = serving_by_pos[pos]
        abs_delta = abs(t_tok.log_prob - s_tok.log_prob)
        all_deltas.append(abs_delta)

        if abs_delta >= threshold:
            rel_delta = None
            if t_tok.log_prob != 0 and math.isfinite(t_tok.log_prob):
                rel_delta = abs_delta / abs(t_tok.log_prob)
            divergences.append(TokenDivergence(
                position=pos,
                token_id=t_tok.token_id,
                token_text=t_tok.token_text,
                training_log_prob=t_tok.log_prob,
                serving_log_prob=s_tok.log_prob,
                abs_delta=abs_delta,
                relative_delta=rel_delta,
            ))

    sorted_divs = sorted(divergences, key=lambda d: d.abs_delta, reverse=True)
    max_delta = max(all_deltas) if all_deltas else 0.0
    mean_delta = statistics.mean(all_deltas) if all_deltas else 0.0

    return RequestDiff(
        request_id=training.request_id,
        total_tokens=len(common_positions),
        divergent_tokens=len(divergences),
        max_abs_delta=max_delta,
        mean_abs_delta=mean_delta,
        divergent_positions=[d.position for d in sorted_divs[:top_k]],
        top_divergences=sorted_divs[:top_k],
        severity=_classify_severity(max_delta, mean_delta, threshold),
    )


def diff_rollouts(
    training_log: str | Path,
    serving_log: str | Path,
    *,
    threshold: float = 0.01,
    top_k: int = 10,
) -> RolloutDiffArtifact:
    """Compare two rollout logs and produce a diff artifact."""
    training_entries = load_rollout_log(training_log)
    serving_entries = load_rollout_log(serving_log)

    serving_by_id = {e.request_id: e for e in serving_entries}

    request_diffs: list[RequestDiff] = []
    all_deltas: list[float] = []
    position_buckets: dict[str, int] = {}

    for t_entry in training_entries:
        s_entry = serving_by_id.get(t_entry.request_id)
        if s_entry is None:
            continue
        rd = diff_request(t_entry, s_entry, threshold=threshold, top_k=top_k)
        request_diffs.append(rd)

        for div in rd.top_divergences:
            all_deltas.append(div.abs_delta)
            bucket = _bucket_position(div.position)
            position_buckets[bucket] = position_buckets.get(bucket, 0) + 1

    all_deltas_sorted = sorted(all_deltas) if all_deltas else [0.0]
    total_tokens = sum(rd.total_tokens for rd in request_diffs)
    divergent_tokens = sum(rd.divergent_tokens for rd in request_diffs)
    divergent_requests = sum(1 for rd in request_diffs if rd.severity != "clean")

    severity_counts: dict[str, int] = {"clean": 0, "drift": 0, "critical": 0}
    for rd in request_diffs:
        severity_counts[rd.severity] += 1

    def _percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        idx = int(len(data) * pct / 100)
        return data[min(idx, len(data) - 1)]

    model = ""
    if training_entries:
        model = training_entries[0].model

    summary = RolloutDiffSummary(
        total_requests=len(request_diffs),
        total_tokens=total_tokens,
        divergent_requests=divergent_requests,
        divergent_tokens=divergent_tokens,
        mean_abs_delta=statistics.mean(all_deltas) if all_deltas else 0.0,
        max_abs_delta=max(all_deltas) if all_deltas else 0.0,
        p50_abs_delta=_percentile(all_deltas_sorted, 50),
        p95_abs_delta=_percentile(all_deltas_sorted, 95),
        p99_abs_delta=_percentile(all_deltas_sorted, 99),
        severity_counts=severity_counts,
        position_histogram=position_buckets,
    )

    return RolloutDiffArtifact(
        training_source=str(training_log),
        serving_source=str(serving_log),
        threshold=threshold,
        model=model,
        summary=summary,
        request_diffs=request_diffs,
    )
