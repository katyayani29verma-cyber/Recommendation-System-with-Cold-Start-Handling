"""Load final ALS artifacts and provide recommendation operations."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path

import numpy as np

from src.core.data_prep import file_sha256
from src.core.interfaces import Recommendation


class ALSRecommenderCore:
    def __init__(
        self,
        movie_ids: np.ndarray,
        item_vectors: np.ndarray,
        available_mask: np.ndarray | None = None,
        *,
        artifact_version: str = "memory",
        prior_mean: np.ndarray | None = None,
        prior_covariance: np.ndarray | None = None,
        content_encoder: Callable[[dict], np.ndarray] | None = None,
    ):
        ids = np.asarray(movie_ids)
        vectors = np.asarray(item_vectors)

        if ids.ndim != 1:
            raise ValueError("Movie IDs must be one-dimensional.")

        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Movie IDs must be integers.")

        if len(ids) == 0 or np.any(ids < 1):
            raise ValueError("Movie IDs must be positive and nonempty.")

        if len(np.unique(ids)) != len(ids):
            raise ValueError("Movie IDs must be unique.")

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(ids)
            or vectors.shape[1] < 1
        ):
            raise ValueError("Movie vectors do not match the movie IDs.")

        if not np.issubdtype(vectors.dtype, np.floating):
            raise ValueError("Movie vectors must contain floating-point values.")

        if not np.isfinite(vectors).all():
            raise ValueError("Movie vectors contain invalid numbers.")

        if available_mask is None:
            available = np.ones(len(ids), dtype=bool)
        else:
            available = np.asarray(available_mask)

            if available.shape != (len(ids),):
                raise ValueError("Availability mask has the wrong shape.")

            if available.dtype != np.dtype(bool):
                raise ValueError("Availability mask must contain booleans.")

        self.n_dims = vectors.shape[1]
        self.artifact_version = artifact_version

        self._movie_ids = ids.astype(np.int64, copy=True)
        self._item_vectors = vectors.copy()
        self._available = available.copy()

        self._movie_to_row = {
            int(movie_id): row
            for row, movie_id in enumerate(self._movie_ids)
        }

        # Protect shared model arrays against accidental modification.
        self._movie_ids.setflags(write=False)
        self._item_vectors.setflags(write=False)
        self._available.setflags(write=False)

        self.prior_mean = None
        self.prior_covariance = None

        if prior_mean is not None:
            mean = self._validate_vector(prior_mean).copy()
            mean.setflags(write=False)
            self.prior_mean = mean

        if prior_covariance is not None:
            covariance = np.asarray(
                prior_covariance,
                dtype=np.float64,
            )

            if covariance.shape != (self.n_dims, self.n_dims):
                raise ValueError("Prior covariance has the wrong shape.")

            if not np.isfinite(covariance).all():
                raise ValueError("Prior covariance contains invalid numbers.")

            if not np.allclose(
                covariance,
                covariance.T,
                rtol=1e-7,
                atol=1e-10,
            ):
                raise ValueError("Prior covariance must be symmetric.")

            covariance = covariance.copy()
            covariance.setflags(write=False)
            self.prior_covariance = covariance

        self._content_encoder = content_encoder

    @property
    def movie_ids(self) -> np.ndarray:
        return self._movie_ids.copy()

    @property
    def available_movie_ids(self) -> list[int]:
        return [
            int(movie_id)
            for movie_id in self._movie_ids[self._available]
        ]

    def _validate_vector(self, vector) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float32)

        if values.shape != (self.n_dims,):
            raise ValueError(
                f"Expected a vector with shape ({self.n_dims},)."
            )

        if not np.isfinite(values).all():
            raise ValueError("The vector contains invalid numbers.")

        return values

    @staticmethod
    def _validate_movie_id(movie_id) -> int:
        if isinstance(movie_id, (bool, np.bool_)):
            raise KeyError("Movie IDs must be integers.")

        if not isinstance(movie_id, (int, np.integer)):
            raise KeyError("Movie IDs must be integers.")

        return int(movie_id)

    def has_item(self, movie_id: int) -> bool:
        movie_id = self._validate_movie_id(movie_id)
        row = self._movie_to_row.get(movie_id)

        return (
            row is not None
            and bool(self._available[row])
        )

    def get_item_vector(self, movie_id: int) -> np.ndarray:
        movie_id = self._validate_movie_id(movie_id)
        row = self._movie_to_row.get(movie_id)

        if row is None or not self._available[row]:
            raise KeyError(
                f"No available vector for movie ID {movie_id}."
            )

        return self._item_vectors[row].copy()

    def recommend(
        self,
        user_vector: np.ndarray,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[Recommendation]:
        if isinstance(k, (bool, np.bool_)):
            raise ValueError("k must be a positive integer.")

        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer.")

        user_vector = self._validate_vector(user_vector)

        eligible = self._available.copy()

        for movie_id in exclude_ids or set():
            movie_id = self._validate_movie_id(movie_id)
            row = self._movie_to_row.get(movie_id)

            if row is not None:
                eligible[row] = False

        rows = np.flatnonzero(eligible)

        if len(rows) == 0:
            return []

        # One matrix multiplication calculates every movie score.
        scores = self._item_vectors @ user_vector

        if not np.isfinite(scores).all():
            raise ValueError("Calculated scores contain invalid numbers.")

        count = min(int(k), len(rows))
        eligible_scores = scores[rows]

        if count == len(rows):
            candidates = rows
        else:
            selected = np.argpartition(
                eligible_scores,
                len(rows) - count,
            )[-count:]

            cutoff = eligible_scores[selected].min()

            better = rows[eligible_scores > cutoff]
            tied = rows[eligible_scores == cutoff]

            # Stable movie-ID ordering at the top-k boundary.
            tied_order = np.argsort(
                self._movie_ids[tied],
                kind="stable",
            )

            needed = count - len(better)

            candidates = np.concatenate([
                better,
                tied[tied_order[:needed]],
            ])

        ordering = np.lexsort((
            self._movie_ids[candidates],
            -scores[candidates],
        ))

        selected_rows = candidates[ordering]

        return [
            Recommendation(
                movie_id=int(self._movie_ids[row]),
                score=float(scores[row]),
            )
            for row in selected_rows
        ]

    def embed_new_item(self, metadata: dict) -> np.ndarray:
        if self._content_encoder is None:
            raise RuntimeError(
                "A trained content encoder must be loaded "
                "before embedding new movies."
            )

        vector = self._content_encoder(metadata)

        return self._validate_vector(vector).copy()


def load_core(
    artifact_dir: str | Path,
    *,
    expected_version: str | None = None,
    content_encoder: Callable[[dict], np.ndarray] | None = None,
) -> ALSRecommenderCore:
    from src.data.config import REPO_ROOT

    directory = Path(artifact_dir)

    if not directory.is_absolute():
        directory = REPO_ROOT / directory

    version = directory.name

    if expected_version is not None and version != expected_version:
        raise ValueError(
            f"Expected version {expected_version}, received {version}."
        )

    with (directory / "manifest.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact schema version.")

    if manifest.get("stage") != "full_core_training":
        raise ValueError("Load the final refitted model artifacts.")

    if manifest.get("vector_normalization") != "raw_learned_factors":
        raise ValueError("Expected raw learned ALS factors.")

    required_files = [
        "als_model.npz",
        "user_vectors.npy",
        "item_vectors.npy",
        "user_ids.npy",
        "movie_ids.npy",
        "user_has_positive.npy",
        "item_has_positive.npy",
        "prior_mean.npy",
        "prior_covariance.npy",
    ]

    for filename in required_files:
        actual = file_sha256(directory / filename)
        expected = manifest["file_sha256"][filename]

        if actual != expected:
            raise ValueError(
                f"Artifact checksum mismatch: {filename}"
            )

    def read_array(filename):
        return np.load(
            directory / filename,
            allow_pickle=False,
        )

    item_vectors = read_array("item_vectors.npy")
    movie_ids = read_array("movie_ids.npy")
    available = read_array("item_has_positive.npy")
    prior_mean = read_array("prior_mean.npy")
    prior_covariance = read_array("prior_covariance.npy")

    factors = manifest["parameters"]["factors"]

    if item_vectors.shape != (len(movie_ids), factors):
        raise ValueError("Movie-vector dimensions do not match.")

    if list(item_vectors.shape) != manifest["item_vector_shape"]:
        raise ValueError("Movie-vector dimensions differ from the manifest.")

    if int(available.sum()) != manifest["movies_with_positive_support"]:
        raise ValueError("Movie support count differs from the manifest.")

    return ALSRecommenderCore(
        movie_ids=movie_ids,
        item_vectors=item_vectors,
        available_mask=available,
        artifact_version=version,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        content_encoder=content_encoder,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact-dir",
        required=True,
    )

    parser.add_argument(
        "--expected-version",
        default=None,
    )

    args = parser.parse_args()

    core = load_core(
        args.artifact_dir,
        expected_version=args.expected_version,
    )

    print(f"Loaded version: {core.artifact_version}")
    print(f"Latent dimensions: {core.n_dims}")
    print(f"Catalog vector rows: {len(core.movie_ids):,}")
    print(f"Scorable movies: {len(core.available_movie_ids):,}")

    print("\nRecommendations for the average training preference:")

    for recommendation in core.recommend(core.prior_mean, k=10):
        print(
            f"Movie {recommendation.movie_id}: "
            f"{recommendation.score:.6f}"
        )


if __name__ == "__main__":
    main()