"""Shared interface for trained and mock recommendation cores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    score: float


class RecommenderCore(Protocol):
    n_dims: int

    def get_item_vector(self, movie_id: int) -> np.ndarray: ...

    def recommend(
        self,
        user_vector: np.ndarray,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[Recommendation]: ...

    def embed_new_item(self, metadata: dict) -> np.ndarray: ...
