"""Evaluation harness comparing fixed vs. adaptive onboarding (README:
"Evaluation plan").

Simulates pairwise answers from each evaluation user's onboarding pool
(never the hidden test set), replays them through a `UserModel`
(active_learning/interfaces.py), and scores the resulting ranking against
the hidden test set after every question.

This module is deliberately agnostic to *how* vectors or posteriors are
produced -- it only calls the Track A / Track B contracts -- so it keeps
working unchanged once the mock implementations are swapped for the real
ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.active_learning.interfaces import ItemPair, PairwiseAnswer, PosteriorState, UserModel
from src.core.interfaces import RecommenderCore


def simulate_answer(
    onboarding_ratings: dict[int, float], pair: ItemPair, session_id: str = "sim"
) -> PairwiseAnswer:
    """Answer a pair from a simulated user's onboarding-pool ratings.

    Ties and ratings unavailable in the pool both produce "skip": the
    dataset cannot tell us a genuine preference in either case, so we must
    not guess one.
    """
    r_a = onboarding_ratings.get(pair.movie_a)
    r_b = onboarding_ratings.get(pair.movie_b)
    if r_a is None or r_b is None or r_a == r_b:
        choice = "skip"
    elif r_a > r_b:
        choice = "A"
    else:
        choice = "B"
    return PairwiseAnswer(session_id=session_id, movie_a=pair.movie_a, movie_b=pair.movie_b, choice=choice)


def reference_vector(onboarding_ratings: dict[int, float], core: RecommenderCore, n_dims: int) -> np.ndarray:
    """A simple rating-weighted item-vector average, used only to define the
    convergence target (README: "a chosen fraction of a reference score
    obtained from additional onboarding-history ratings").
    """
    if not onboarding_ratings:
        return np.zeros(n_dims)
    mean_rating = np.mean(list(onboarding_ratings.values()))
    vec = np.zeros(n_dims)
    for mid, rating in onboarding_ratings.items():
        try:
            vec += (rating - mean_rating) * core.get_item_vector(mid)
        except KeyError:
            continue
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass
class SessionTrajectory:
    states: list[PosteriorState]
    answers: list[PairwiseAnswer] = field(default_factory=list)


def run_fixed_session(
    model: UserModel,
    fixed_pairs: list[tuple[int, int]],
    onboarding_ratings: dict[int, float],
) -> SessionTrajectory:
    state = model.initialize_user()
    states, answers = [state], []
    for movie_a, movie_b in fixed_pairs:
        answer = simulate_answer(onboarding_ratings, ItemPair(movie_a, movie_b))
        state = model.update_user(state, answer)
        states.append(state)
        answers.append(answer)
    return SessionTrajectory(states=states, answers=answers)


def run_adaptive_session(
    model: UserModel,
    candidate_ids: list[int],
    onboarding_ratings: dict[int, float],
    budget: int,
) -> SessionTrajectory:
    state = model.initialize_user()
    states, answers = [state], []
    remaining = list(candidate_ids)
    for _ in range(budget):
        if len(remaining) < 2:
            break
        pair = model.select_next_pair(state, remaining)
        # Never reuse a movie within a session, matching the API's behaviour.
        remaining = [m for m in remaining if m not in (pair.movie_a, pair.movie_b)]
        answer = simulate_answer(onboarding_ratings, pair)
        state = model.update_user(state, answer)
        states.append(state)
        answers.append(answer)
    return SessionTrajectory(states=states, answers=answers)


def ratings_dict(ratings: pd.DataFrame, user_id: int) -> dict[int, float]:
    user_rows = ratings[ratings["userId"] == user_id]
    return dict(zip(user_rows["movieId"], user_rows["rating"]))


def relevance_for_user(
    hidden_test: pd.DataFrame, user_id: int, binary_threshold: float
) -> tuple[dict[int, float], set[int]]:
    rows = hidden_test[hidden_test["userId"] == user_id]
    graded = dict(zip(rows["movieId"], rows["rating"]))
    relevant = {mid for mid, r in graded.items() if r >= binary_threshold}
    return graded, relevant
