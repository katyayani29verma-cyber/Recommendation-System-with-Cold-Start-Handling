"""Track B's proposed interface contract (README: "Shared interface contract").

Track C depends on this contract, not on Track B's inference method, so the
evaluation harness can run the fixed-vs-adaptive comparison against
`MockPointEstimateUserModel` (see mock_model.py) today and swap in Track B's
real Bayesian posterior later without changing the harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class ItemPair:
    movie_a: int
    movie_b: int


@dataclass(frozen=True)
class PairwiseAnswer:
    session_id: str
    movie_a: int
    movie_b: int
    choice: Literal["A", "B", "skip"]


@dataclass
class PosteriorState:
    """Opaque to the harness; Track B defines what actually lives here
    (e.g. posterior mean/covariance or particle samples). The harness only
    needs `mean_vector` to rank movies.
    """

    mean_vector: np.ndarray
    extra: Any = None


class UserModel(Protocol):
    def initialize_user(self) -> PosteriorState: ...

    def select_next_pair(self, state: PosteriorState, candidate_ids: list[int]) -> ItemPair: ...

    def update_user(self, state: PosteriorState, answer: PairwiseAnswer) -> PosteriorState: ...
