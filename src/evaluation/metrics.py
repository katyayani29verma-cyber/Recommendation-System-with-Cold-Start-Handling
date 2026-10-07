"""Ranking metrics (README: "Metrics").

All functions take a ranked list of movie IDs (best first) and a relevance
mapping restricted to the user's hidden test set, so these never need to
know the full catalog.
"""
from __future__ import annotations

import numpy as np


def dcg_at_k(ranked_ids: list[int], relevance: dict[int, float], k: int) -> float:
    gains = [relevance.get(mid, 0.0) for mid in ranked_ids[:k]]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(np.asarray(gains) / discounts)) if gains else 0.0


def ndcg_at_k(ranked_ids: list[int], relevance: dict[int, float], k: int) -> float:
    """Normalized DCG@k. `relevance` may be graded (e.g. star ratings)."""
    dcg = dcg_at_k(ranked_ids, relevance, k)
    ideal_order = sorted(relevance.values(), reverse=True)
    idcg = dcg_at_k(list(range(len(ideal_order))), dict(enumerate(ideal_order)), k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Average precision at k using binary relevance."""
    if not relevant_ids:
        return 0.0
    hits = 0
    precisions = []
    for i, mid in enumerate(ranked_ids[:k], start=1):
        if mid in relevant_ids:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return float(np.sum(precisions) / min(len(relevant_ids), k))


def mean_average_precision_at_k(
    ranked_ids_per_user: list[list[int]],
    relevant_ids_per_user: list[set[int]],
    k: int,
) -> float:
    scores = [
        average_precision_at_k(ranked, relevant, k)
        for ranked, relevant in zip(ranked_ids_per_user, relevant_ids_per_user)
    ]
    return float(np.mean(scores)) if scores else 0.0


def mean_ndcg_at_k(
    ranked_ids_per_user: list[list[int]],
    relevance_per_user: list[dict[int, float]],
    k: int,
) -> float:
    scores = [
        ndcg_at_k(ranked, relevance, k)
        for ranked, relevance in zip(ranked_ids_per_user, relevance_per_user)
    ]
    return float(np.mean(scores)) if scores else 0.0
