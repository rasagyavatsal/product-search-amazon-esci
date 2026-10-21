"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""

from collections import defaultdict


def rrf(rankings: list[list[str]], k: int = 60, top_n: int | None = None,
        weights: list[float] | None = None) -> list[str]:
    """Fuse ranked lists: score(d) = sum_i w_i / (k + rank_i(d)), ranks from 1."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += w / (k + rank)
    fused = sorted(scores, key=scores.__getitem__, reverse=True)
    return fused[:top_n] if top_n else fused
