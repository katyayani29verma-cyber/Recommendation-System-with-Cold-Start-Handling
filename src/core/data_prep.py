"""Prepare validated, reproducible interaction data for ALS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


@dataclass
class PreparedInteractions:
    matrix: csr_matrix
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_to_row: dict[int, int]
    movie_to_column: dict[int, int]
    positive_threshold: float


def validate_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Check records without modifying the original table."""

    required = ["userId", "movieId", "rating"]

    missing_columns = set(required) - set(ratings.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if ratings.empty:
        raise ValueError("The training ratings table is empty.")

    clean = ratings[required].copy()

    if clean.isna().any().any():
        raise ValueError("User IDs, movie IDs, and ratings cannot be missing.")

    for column in required:
        contains_boolean = clean[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).any()

        if contains_boolean:
            raise ValueError(f"{column} cannot contain boolean values.")

        try:
            clean[column] = pd.to_numeric(
                clean[column],
                errors="raise",
            )
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"{column} contains a nonnumeric value."
            ) from error

        values = clean[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains an infinite or invalid value.")

    for column in ["userId", "movieId"]:
        values = clean[column]

        invalid_ids = (
            (values < 1)
            | (values >= 2**63)
            | (values % 1 != 0)
        )

        if invalid_ids.any():
            raise ValueError(
                f"{column} must contain positive integer IDs below 2**63."
            )

        clean[column] = values.astype(np.int64)

    if not clean["rating"].between(0.5, 5.0).all():
        raise ValueError("MovieLens ratings must be between 0.5 and 5.0.")

    if clean.duplicated(["userId", "movieId"]).any():
        raise ValueError(
            "Duplicate user/movie ratings found. Resolve them explicitly."
        )

    return clean


def prepare_interactions(
    ratings: pd.DataFrame,
    positive_threshold: float,
    *,
    evaluation_user_ids: set[int],
    catalog_movie_ids: set[int],
) -> PreparedInteractions:
    """Create a binary sparse matrix with user rows and movie columns."""

    if isinstance(positive_threshold, (bool, np.bool_)):
        raise ValueError("The positive threshold must be numeric.")

    try:
        threshold = float(positive_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("The positive threshold must be numeric.") from error

    if not np.isfinite(threshold) or not 0.5 <= threshold <= 5.0:
        raise ValueError("The positive threshold must be between 0.5 and 5.0.")

    clean = validate_ratings(ratings)

    training_users = set(clean["userId"].tolist())
    overlapping_users = training_users & evaluation_user_ids

    if overlapping_users:
        raise ValueError(
            "Evaluation users found in training data: "
            f"{sorted(overlapping_users)[:10]}"
        )

    unknown_movies = (
        set(clean["movieId"].tolist()) - catalog_movie_ids
    )

    if unknown_movies:
        raise ValueError(
            "Training movies missing from the catalog: "
            f"{sorted(unknown_movies)[:10]}"
        )

    # Select positive interactions using the shared configuration.
    positive = clean.loc[clean["rating"] >= threshold]

    if positive.empty:
        raise ValueError(
            "No positive interactions remain at the configured threshold."
        )

    # Preserve all core users, including those without positive interactions.
    user_ids = np.sort(clean["userId"].unique()).astype(np.int64)

    # These are movies with positive core-training interactions.
    movie_ids = np.sort(positive["movieId"].unique()).astype(np.int64)

    # Sorted IDs produce stable matrix positions.
    user_to_row = {
        int(user_id): row
        for row, user_id in enumerate(user_ids)
    }

    movie_to_column = {
        int(movie_id): column
        for column, movie_id in enumerate(movie_ids)
    }

    rows = (
        positive["userId"]
        .map(user_to_row)
        .to_numpy(dtype=np.int64)
    )

    columns = (
        positive["movieId"]
        .map(movie_to_column)
        .to_numpy(dtype=np.int64)
    )

    # Store a positive indicator for each retained interaction.
    # ALS confidence weighting will be handled by the training component.
    values = np.ones(len(positive), dtype=np.float32)

    matrix = csr_matrix(
        (values, (rows, columns)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    )

    matrix.sort_indices()

    if matrix.nnz != len(positive):
        raise RuntimeError("Matrix entries do not match the positive records.")

    if not np.all(matrix.data == 1.0):
        raise RuntimeError("The interaction matrix must contain binary positives.")

    return PreparedInteractions(
        matrix=matrix,
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_to_row=user_to_row,
        movie_to_column=movie_to_column,
        positive_threshold=threshold,
    )


def file_sha256(path: Path) -> str:
    """Calculate a fingerprint identifying a file's contents."""

    digest = sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_prepared(
    prepared: PreparedInteractions,
    output_dir: Path,
    *,
    provenance: dict,
) -> None:
    """Save the matrix, ordered ID arrays, and preparation metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(
        output_dir / "interactions.npz",
        prepared.matrix,
    )

    np.save(
        output_dir / "user_ids.npy",
        prepared.user_ids,
        allow_pickle=False,
    )

    np.save(
        output_dir / "movie_ids.npy",
        prepared.movie_ids,
        allow_pickle=False,
    )

    filenames = [
        "interactions.npz",
        "user_ids.npy",
        "movie_ids.npy",
    ]

    users_without_positives = int(
        np.count_nonzero(np.diff(prepared.matrix.indptr) == 0)
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_orientation": "user_by_movie",
        "interaction_encoding": "binary_positive_indicator",
        "dtype": str(prepared.matrix.dtype),
        "shape": list(prepared.matrix.shape),
        "positive_interactions": int(prepared.matrix.nnz),
        "users_without_positive_interactions": users_without_positives,
        "positive_rating_threshold": prepared.positive_threshold,
        "provenance": provenance,
        "file_sha256": {
            filename: file_sha256(output_dir / filename)
            for filename in filenames
        },
    }

    # Save the manifest after the files it describes.
    with (output_dir / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            allow_nan=False,
        )


def read_user_ids(path: Path) -> set[int]:
    """Read and validate a saved split-membership file."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing split-membership file: {path}")

    ids = [
        int(line.strip())
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    if not ids:
        raise ValueError(f"The membership file is empty: {path}")

    if any(user_id < 1 for user_id in ids):
        raise ValueError(f"Membership IDs must be positive: {path}")

    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate membership IDs found: {path}")

    return set(ids)


def main() -> None:
    from src.data.config import REPO_ROOT, load_config

    config = load_config("data")

    processed_dir = (
        REPO_ROOT / config["dataset"]["processed_dir"]
    )

    ratings_path = processed_dir / "core_train.csv"

    movies_path = (
        REPO_ROOT
        / config["dataset"]["raw_dir"]
        / config["dataset"]["name"]
        / "movies.csv"
    )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    core_ids_path = processed_dir / "core_user_ids.txt"
    eval_ids_path = processed_dir / "eval_user_ids.txt"

    core_user_ids = read_user_ids(core_ids_path)
    evaluation_user_ids = read_user_ids(eval_ids_path)

    if core_user_ids & evaluation_user_ids:
        raise ValueError("Core and evaluation membership files overlap.")

    clean = validate_ratings(ratings)

    if set(clean["userId"]) != core_user_ids:
        raise ValueError(
            "core_train.csv does not match core_user_ids.txt."
        )

    prepared = prepare_interactions(
        clean,
        positive_threshold=config["positive_rating_threshold"],
        evaluation_user_ids=evaluation_user_ids,
        catalog_movie_ids=set(movies["movieId"].tolist()),
    )

    output_dir = processed_dir / "als_input"

    provenance = {
        "dataset": config["dataset"]["name"],
        "split_config": config["split"],
        "source_ratings": str(ratings_path.relative_to(REPO_ROOT)),
        "source_ratings_sha256": file_sha256(ratings_path),
        "source_movies_sha256": file_sha256(movies_path),
        "core_user_ids_sha256": file_sha256(core_ids_path),
        "eval_user_ids_sha256": file_sha256(eval_ids_path),
    }

    save_prepared(
        prepared,
        output_dir,
        provenance=provenance,
    )

    print(f"Training ratings: {len(clean):,}")
    print(f"Positive threshold: {prepared.positive_threshold}")
    print(f"Positive interactions: {prepared.matrix.nnz:,}")
    print(f"Users: {len(prepared.user_ids):,}")
    print(
        "Movies with positive training interactions: "
        f"{len(prepared.movie_ids):,}"
    )
    print(f"Matrix shape: {prepared.matrix.shape}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()