"""Rank- and score-based fusion for sparse and dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


Ranking = Sequence[tuple[int, float]]


@dataclass(slots=True)
class FusedCandidate:
    row_index: int
    score: float = 0.0
    component_scores: dict[str, float] = field(default_factory=dict)
    component_ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Ranking],
    *,
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[FusedCandidate]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    candidates: dict[int, FusedCandidate] = {}
    for name, ranking in rankings.items():
        weight = float((weights or {}).get(name, 1.0))
        for rank, (row_index, component_score) in enumerate(ranking, start=1):
            candidate = candidates.setdefault(row_index, FusedCandidate(row_index=row_index))
            candidate.score += weight / (rrf_k + rank)
            candidate.component_scores[name] = float(component_score)
            candidate.component_ranks[name] = rank
    return sorted(candidates.values(), key=lambda item: (-item.score, item.row_index))


def _min_max_scores(ranking: Ranking) -> dict[int, float]:
    if not ranking:
        return {}
    values = [float(score) for _, score in ranking]
    low = min(values)
    high = max(values)
    if high == low:
        return {row_index: 1.0 for row_index, _ in ranking}
    return {row_index: (float(score) - low) / (high - low) for row_index, score in ranking}


def normalized_weighted_fusion(
    rankings: Mapping[str, Ranking],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[FusedCandidate]:
    candidates: dict[int, FusedCandidate] = {}
    for name, ranking in rankings.items():
        weight = float((weights or {}).get(name, 1.0))
        normalized = _min_max_scores(ranking)
        for rank, (row_index, component_score) in enumerate(ranking, start=1):
            candidate = candidates.setdefault(row_index, FusedCandidate(row_index=row_index))
            candidate.score += weight * normalized[row_index]
            candidate.component_scores[name] = float(component_score)
            candidate.component_ranks[name] = rank
    return sorted(candidates.values(), key=lambda item: (-item.score, item.row_index))
