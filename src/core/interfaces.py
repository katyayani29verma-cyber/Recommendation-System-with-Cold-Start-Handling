"""Track A's proposed interface contract (README: "Shared interface contract").

Track C depends on this contract, not on Track A's implementation, so the
evaluation harness, API, and demo can be built and tested against
`MockRecommenderCore` (see mock_vectors.py) today and swapped for Track A's
real trained model later without touching calling code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    score: float


class RecommenderCore(Protocol):
    def get_item_vector(self, movie_id: int) -> np.ndarray: ...

    def recommend(
        self,
        user_vector: np.ndarray,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[Recommendation]: ...

    def embed_new_item(self, metadata: dict) -> np.ndarray: ...
