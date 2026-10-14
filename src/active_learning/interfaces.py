"""Shared interface for user-preference models."""
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
