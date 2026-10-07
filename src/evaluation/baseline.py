"""Fixed onboarding baseline (README: "Fixed onboarding").

A predetermined sequence of movie-pair questions, built once from
core-training data and item vectors -- never from any evaluation user's
answers, so the comparison against adaptive selection is fair.

Candidates are restricted to popular movies (enough core-training ratings
that nearly every user could plausibly have an opinion), and pairs are
chosen by farthest-point sampling over item vectors so the fixed sequence
spans diverse taste directions instead of asking near-duplicate questions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.interfaces import RecommenderCore


def select_popular_movies(core_train: pd.DataFrame, min_ratings: int, top_n: int | None = None) -> list[int]:
    counts = core_train.groupby("movieId").size().sort_values(ascending=False)
    popular = counts[counts >= min_ratings]
    ids = popular.index.tolist()
    return ids[:top_n] if top_n else ids


def build_fixed_sequence(
    core: RecommenderCore,
    candidate_ids: list[int],
    budget: int,
    seed: int = 0,
) -> list[tuple[int, int]]:
    """Return `budget` (movie_a, movie_b) pairs via farthest-point sampling
    over item vectors, so the fixed sequence is diverse and deterministic
    given the same candidates and vectors.
    """
    n_needed = 2 * budget
    if len(candidate_ids) < n_needed:
        raise ValueError(
            f"Need at least {n_needed} candidate movies for a budget of {budget}, "
            f"got {len(candidate_ids)}"
        )

    vectors = np.stack([core.get_item_vector(mid) for mid in candidate_ids])
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(candidate_ids)))

    chosen = [start]
    min_dist = np.linalg.norm(vectors - vectors[start], axis=1)
    min_dist[start] = -np.inf
    while len(chosen) < n_needed:
        nxt = int(np.argmax(min_dist))
        chosen.append(nxt)
        dist_to_new = np.linalg.norm(vectors - vectors[nxt], axis=1)
        min_dist = np.minimum(min_dist, dist_to_new)
        min_dist[chosen] = -np.inf

    chosen_ids = [candidate_ids[i] for i in chosen]
    return list(zip(chosen_ids[0::2], chosen_ids[1::2]))
