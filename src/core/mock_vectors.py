"""Placeholder movie vectors implementing the Track A contract.

Stand-in only: real vectors come from Track A's trained collaborative model.
This lets Track C build and test the evaluation harness, API, and demo
end-to-end before that handoff happens. Swapping it out later means
constructing a different `RecommenderCore` with the same methods -- nothing
else should need to change.
"""
from __future__ import annotations

import numpy as np

from src.core.interfaces import Recommendation


class MockRecommenderCore:
    def __init__(self, movie_ids: list[int], n_dims: int = 32, seed: int = 7):
        self.n_dims = n_dims
        rng = np.random.default_rng(seed)
        vectors = rng.normal(size=(len(movie_ids), n_dims))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = {mid: vec for mid, vec in zip(movie_ids, vectors)}
        self._rng = rng

    def get_item_vector(self, movie_id: int) -> np.ndarray:
        if movie_id not in self._vectors:
            raise KeyError(f"Unknown movie_id: {movie_id}")
        return self._vectors[movie_id]

    def recommend(
        self,
        user_vector: np.ndarray,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[Recommendation]:
        exclude_ids = exclude_ids or set()
        scored = [
            Recommendation(movie_id=mid, score=float(np.dot(user_vector, vec)))
            for mid, vec in self._vectors.items()
            if mid not in exclude_ids
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def embed_new_item(self, metadata: dict) -> np.ndarray:
        # Deterministic pseudo-embedding keyed on title, standing in for the
        # content encoder + content-to-latent mapping Track A will train.
        seed = abs(hash(metadata.get("title", ""))) % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.n_dims)
        return vec / np.linalg.norm(vec)
