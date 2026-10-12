"""Prepare positive-only and signed inputs with shared IDs and holdouts."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from src.core.data_prep import (
    file_sha256,
    read_user_ids,
    validate_ratings,
)
from src.core.validation_split import split_interactions


def prepare_signed_interactions(
    ratings: pd.DataFrame,
    positive_threshold: float,
    negative_threshold: float,
    *,
    evaluation_user_ids: set[int],
    catalog_movie_ids: set[int],
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    """Build two matrices with the same user and movie positions."""

    for value in (positive_threshold, negative_threshold):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Rating thresholds must be numeric.")

    positive_threshold = float(positive_threshold)
    negative_threshold = float(negative_threshold)

    if not (
        np.isfinite(positive_threshold)
        and np.isfinite(negative_threshold)
        and 0.5 <= negative_threshold < positive_threshold <= 5.0
    ):
        raise ValueError(
            "Thresholds must satisfy: "
            "0.5 <= negative < positive <= 5.0."
        )

    clean = validate_ratings(ratings)

    overlap = set(clean["userId"]) & evaluation_user_ids
    if overlap:
        raise ValueError(
            f"Evaluation users found in training: {sorted(overlap)[:10]}"
        )

    unknown_movies = set(clean["movieId"]) - catalog_movie_ids
    if unknown_movies:
        raise ValueError(
            "Movies missing from the catalog: "
            f"{sorted(unknown_movies)[:10]}"
        )

    positives = clean.loc[
        clean["rating"] >= positive_threshold
    ]

    negatives = clean.loc[
        clean["rating"] <= negative_threshold
    ]

    if positives.empty:
        raise ValueError("No positive interactions remain.")

    labelled = pd.concat(
        [positives, negatives],
        ignore_index=True,
    )

    # Include every core user.
    user_ids = np.sort(
        clean["userId"].unique()
    ).astype(np.int64)

    # Shared catalog: movies with positive or negative training evidence.
    movie_ids = np.sort(
        labelled["movieId"].unique()
    ).astype(np.int64)

    user_to_row = {
        int(user_id): row
        for row, user_id in enumerate(user_ids)
    }

    movie_to_column = {
        int(movie_id): column
        for column, movie_id in enumerate(movie_ids)
    }

    shape = (len(user_ids), len(movie_ids))

    positive_rows = positives["userId"].map(
        user_to_row
    ).to_numpy(dtype=np.int64)

    positive_columns = positives["movieId"].map(
        movie_to_column
    ).to_numpy(dtype=np.int64)

    positive_matrix = csr_matrix(
        (
            np.ones(len(positives), dtype=np.float32),
            (positive_rows, positive_columns),
        ),
        shape=shape,
        dtype=np.float32,
    )

    labelled_rows = labelled["userId"].map(
        user_to_row
    ).to_numpy(dtype=np.int64)

    labelled_columns = labelled["movieId"].map(
        movie_to_column
    ).to_numpy(dtype=np.int64)

    labels = np.where(
        labelled["rating"].to_numpy() >= positive_threshold,
        1.0,
        -1.0,
    ).astype(np.float32)

    signed_matrix = csr_matrix(
        (
            labels,
            (labelled_rows, labelled_columns),
        ),
        shape=shape,
        dtype=np.float32,
    )

    positive_matrix.sort_indices()
    signed_matrix.sort_indices()

    if positive_matrix.nnz != len(positives):
        raise RuntimeError("Positive interaction count does not match.")

    if signed_matrix.nnz != len(positives) + len(negatives):
        raise RuntimeError("Signed interaction count does not match.")

    return positive_matrix, signed_matrix, user_ids, movie_ids


def split_signed_interactions(
    positive_matrix: csr_matrix,
    signed_matrix: csr_matrix,
    fraction: float,
    seed: int,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Hold out positives only; retain every explicit negative in training."""

    if positive_matrix.shape != signed_matrix.shape:
        raise ValueError("Positive and signed matrix shapes must match.")

    if not np.all(np.isin(signed_matrix.data, [-1.0, 1.0])):
        raise ValueError("Signed entries must be +1 or -1.")

    # Confirm that both matrices contain the same positive evidence.
    extracted_positives = signed_matrix.copy()
    extracted_positives.data = np.maximum(
        extracted_positives.data,
        0,
    )
    extracted_positives.eliminate_zeros()

    if (extracted_positives != positive_matrix).nnz:
        raise ValueError("Positive evidence differs between the matrices.")

    positive_training, validation = split_interactions(
        positive_matrix,
        fraction=fraction,
        seed=seed,
    )

    # Subtract only the held-out positive entries.
    signed_training = (
        signed_matrix - validation
    ).tocsr()

    signed_training.eliminate_zeros()
    signed_training.sort_indices()

    if signed_training.multiply(validation).nnz:
        raise RuntimeError("Signed training overlaps validation.")

    if (signed_training + validation != signed_matrix).nnz:
        raise RuntimeError("Signed split does not reconstruct its source.")

    original_negatives = np.count_nonzero(
        signed_matrix.data < 0
    )

    training_negatives = np.count_nonzero(
        signed_training.data < 0
    )

    if original_negatives != training_negatives:
        raise RuntimeError("Explicit negatives were lost during splitting.")

    return positive_training, signed_training, validation


def main() -> None:
    from src.data.config import REPO_ROOT, load_config

    data_config = load_config("data")
    core_config = load_config("core")

    processed_dir = (
        REPO_ROOT / data_config["dataset"]["processed_dir"]
    )

    ratings_path = processed_dir / "core_train.csv"
    core_ids_path = processed_dir / "core_user_ids.txt"
    eval_ids_path = processed_dir / "eval_user_ids.txt"

    movies_path = (
        REPO_ROOT
        / data_config["dataset"]["raw_dir"]
        / data_config["dataset"]["name"]
        / "movies.csv"
    )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    core_ids = read_user_ids(core_ids_path)
    eval_ids = read_user_ids(eval_ids_path)

    if core_ids & eval_ids:
        raise ValueError("Core and evaluation memberships overlap.")

    clean = validate_ratings(ratings)

    if set(clean["userId"]) != core_ids:
        raise ValueError(
            "core_train.csv does not match core_user_ids.txt."
        )

    positive, signed, user_ids, movie_ids = (
        prepare_signed_interactions(
            clean,
            positive_threshold=data_config[
                "positive_rating_threshold"
            ],
            negative_threshold=data_config[
                "negative_rating_threshold"
            ],
            evaluation_user_ids=eval_ids,
            catalog_movie_ids=set(movies["movieId"]),
        )
    )

    positive_training, signed_training, validation = (
        split_signed_interactions(
            positive,
            signed,
            fraction=core_config["validation"]["fraction"],
            seed=core_config["validation"]["seed"],
        )
    )

    if validation.nnz == 0:
        raise ValueError("No positive validation interactions were selected.")

    # Separate directory: retain the previous preparation artifacts.
    output_dir = processed_dir / "als_input_signed"
    output_dir.mkdir(parents=True, exist_ok=True)

    matrices = {
        "positive_interactions.npz": positive,
        "signed_interactions.npz": signed,
        "positive_train.npz": positive_training,
        "signed_train.npz": signed_training,
        "validation_interactions.npz": validation,
    }

    for filename, matrix in matrices.items():
        save_npz(output_dir / filename, matrix)

    np.save(
        output_dir / "user_ids.npy",
        user_ids,
        allow_pickle=False,
    )

    np.save(
        output_dir / "movie_ids.npy",
        movie_ids,
        allow_pickle=False,
    )

    negative_count = int(
        np.count_nonzero(signed.data < 0)
    )

    neutral_count = len(clean) - positive.nnz - negative_count

    validation_users = int(
        np.count_nonzero(np.diff(validation.indptr))
    )

    filenames = list(matrices) + [
        "user_ids.npy",
        "movie_ids.npy",
    ]

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_orientation": "user_by_movie",
        "interaction_encoding": "signed_positive_negative_indicators",
        "shape": list(signed.shape),
        "positive_rating_threshold": float(
            data_config["positive_rating_threshold"]
        ),
        "negative_rating_threshold": float(
            data_config["negative_rating_threshold"]
        ),
        "positive_interactions": int(positive.nnz),
        "negative_interactions": negative_count,
        "neutral_ratings": int(neutral_count),
        "positive_training_interactions": int(
            positive_training.nnz
        ),
        "signed_training_interactions": int(
            signed_training.nnz
        ),
        "validation_interactions": int(validation.nnz),
        "users_with_validation": validation_users,
        "validation_config": core_config["validation"],
        "dataset": data_config["dataset"]["name"],
        "source_sha256": {
            "core_train.csv": file_sha256(ratings_path),
            "movies.csv": file_sha256(movies_path),
            "core_user_ids.txt": file_sha256(core_ids_path),
            "eval_user_ids.txt": file_sha256(eval_ids_path),
        },
        "file_sha256": {
            filename: file_sha256(output_dir / filename)
            for filename in filenames
        },
    }

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

    print(f"Core-training ratings: {len(clean):,}")
    print(f"Positive interactions: {positive.nnz:,}")
    print(f"Negative interactions: {negative_count:,}")
    print(f"Neutral ratings: {neutral_count:,}")
    print(
        "Positive training interactions: "
        f"{positive_training.nnz:,}"
    )
    print(
        "Signed training interactions: "
        f"{signed_training.nnz:,}"
    )
    print(f"Validation positives: {validation.nnz:,}")
    print(f"Users with validation: {validation_users:,}")
    print(f"Matrix shape: {signed.shape}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()