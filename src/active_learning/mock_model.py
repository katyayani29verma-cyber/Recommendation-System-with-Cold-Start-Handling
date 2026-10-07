"""Placeholder user model implementing the Track B contract.

Stand-in only: Track B will replace this with real Bayesian posterior
inference and expected-information-gain / BALD question selection. This
version keeps a single point-estimate vector (no uncertainty) updated with a
simple perceptron-style nudge toward the chosen movie's vector and away from
the rejected one, following the pairwise likelihood sketched in the README:

    P(choose A | p) = sigmoid(dot(p, q_A - q_B) / temperature)

It lets Track C build and test the evaluation harness end-to-end before the
real inference engine lands. `select_next_pair` here just pulls from a
pre-built queue (used for the fixed baseline) or picks randomly among
remaining candidates (a stand-in for "adaptive" until Track B's EIG/BALD
selector is ready).
"""
from __future__ import annotations

import numpy as np

from src.active_learning.interfaces import ItemPair, PairwiseAnswer, PosteriorState
from src.core.interfaces import RecommenderCore


class MockPointEstimateUserModel:
    def __init__(self, core: RecommenderCore, n_dims: int, learning_rate: float = 0.5, seed: int = 0):
        self.core = core
        self.n_dims = n_dims
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)

    def initialize_user(self) -> PosteriorState:
        return PosteriorState(mean_vector=np.zeros(self.n_dims))

    def select_next_pair(self, state: PosteriorState, candidate_ids: list[int]) -> ItemPair:
        # Placeholder selection: Track B replaces this with EIG/BALD scoring.
        a, b = self._rng.choice(candidate_ids, size=2, replace=False)
        return ItemPair(movie_a=int(a), movie_b=int(b))

    def update_user(self, state: PosteriorState, answer: PairwiseAnswer) -> PosteriorState:
        if answer.choice == "skip":
            return state
        q_a = self.core.get_item_vector(answer.movie_a)
        q_b = self.core.get_item_vector(answer.movie_b)
        winner, loser = (q_a, q_b) if answer.choice == "A" else (q_b, q_a)
        direction = winner - loser
        new_vector = state.mean_vector + self.learning_rate * direction
        return PosteriorState(mean_vector=new_vector)
