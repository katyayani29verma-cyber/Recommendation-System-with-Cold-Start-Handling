"""Create a reproducible validation split from prepared core interactions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from src.core.data_prep import file_sha256


def split_interactions(
    matrix: csr_matrix,
    fraction: float,
    seed: int,
) -> tuple[csr_matrix, csr_matrix]:
    """Separate positive interactions while preserving training support."""

    fraction = float(fraction)

    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("Validation fraction must be between 0 and 1.")

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Validation seed must be an integer.")

    if seed < 0:
        raise ValueError("Validation seed must be nonnegative.")

    # Work on a copy so the original matrix remains unchanged.
    source = csr_matrix(matrix, copy=True)
    source.sum_duplicates()
    source.eliminate_zeros()
    source.sort_indices()

    if source.nnz == 0:
        raise ValueError("The interaction matrix has no positive entries.")

    if not np.all(source.data == 1.0):
        raise ValueError("Expected a binary positive-interaction matrix.")

    rng = np.random.default_rng(seed)

    # Each stored matrix value corresponds to one positive interaction.
    holdout_mask = np.zeros(source.nnz, dtype=bool)

    # Count how many users currently provide training support per movie.
    item_support = np.bincount(
        source.indices,
        minlength=source.shape[1],
    )

    for user_row in range(source.shape[0]):
        start = source.indptr[user_row]
        end = source.indptr[user_row + 1]

        number_of_positives = end - start

        # A user needs at least two positives to contribute a holdout.
        if number_of_positives < 2:
            continue

        requested = max(
            1,
            int(np.floor(number_of_positives * fraction)),
        )

        # Always leave at least one positive interaction in training.
        requested = min(requested, number_of_positives - 1)

        positions = rng.permutation(np.arange(start, end))
        selected = 0

        for position in positions:
            movie_column = source.indices[position]

            # Keep at least one training interaction for every movie.
            if item_support[movie_column] <= 1:
                continue

            holdout_mask[position] = True
            item_support[movie_column] -= 1
            selected += 1

            if selected == requested:
                break

    # Recover the user-row position of each stored interaction.
    row_numbers = np.repeat(
        np.arange(source.shape[0]),
        np.diff(source.indptr),
    )

    validation = csr_matrix(
        (
            source.data[holdout_mask],
            (
                row_numbers[holdout_mask],
                source.indices[holdout_mask],
            ),
        ),
        shape=source.shape,
        dtype=source.dtype,
    )

    training = source.copy()
    training.data[holdout_mask] = 0
    training.eliminate_zeros()

    # Verify the split's essential properties.
    if training.multiply(validation).nnz != 0:
        raise RuntimeError("Training and validation overlap.")

    if (training + validation != source).nnz != 0:
        raise RuntimeError("The split does not reconstruct the source matrix.")

    original_user_support = np.diff(source.indptr)
    training_user_support = np.diff(training.indptr)

    if np.any(
        (original_user_support > 0)
        & (training_user_support == 0)
    ):
        raise RuntimeError("A user lost all training interactions.")

    original_item_support = np.bincount(
        source.indices,
        minlength=source.shape[1],
    )

    remaining_item_support = np.bincount(
        training.indices,
        minlength=training.shape[1],
    )

    if np.any(
        (original_item_support > 0)
        & (remaining_item_support == 0)
    ):
        raise RuntimeError("A movie lost all training interactions.")

    return training, validation


def load_prepared_matrix(
    input_dir: Path,
    ratings_path: Path,
) -> tuple[csr_matrix, dict]:
    """Check the preprocessing artifacts before using them."""

    manifest_path = input_dir / "manifest.json"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported preprocessing manifest version.")

    if manifest.get("matrix_orientation") != "user_by_movie":
        raise ValueError("Expected user rows and movie columns.")

    if manifest.get("interaction_encoding") != "binary_positive_indicator":
        raise ValueError("Expected binary positive interactions.")

    filenames = [
        "interactions.npz",
        "user_ids.npy",
        "movie_ids.npy",
    ]

    for filename in filenames:
        expected_checksum = manifest["file_sha256"][filename]
        actual_checksum = file_sha256(input_dir / filename)

        if actual_checksum != expected_checksum:
            raise ValueError(
                f"Preprocessing checksum mismatch: {filename}"
            )

    recorded_source_checksum = manifest["provenance"][
        "source_ratings_sha256"
    ]

    if file_sha256(ratings_path) != recorded_source_checksum:
        raise ValueError(
            "core_train.csv changed. Run preprocessing again "
            "before creating this split."
        )

    matrix = load_npz(input_dir / "interactions.npz").tocsr()

    user_ids = np.load(
        input_dir / "user_ids.npy",
        allow_pickle=False,
    )

    movie_ids = np.load(
        input_dir / "movie_ids.npy",
        allow_pickle=False,
    )

    if user_ids.ndim != 1 or movie_ids.ndim != 1:
        raise ValueError("ID mappings must be one-dimensional arrays.")

    expected_shape = (len(user_ids), len(movie_ids))

    if matrix.shape != expected_shape:
        raise ValueError("Matrix dimensions do not match the ID mappings.")

    if list(matrix.shape) != manifest["shape"]:
        raise ValueError("Matrix dimensions do not match the manifest.")

    if matrix.nnz != manifest["positive_interactions"]:
        raise ValueError("Interaction count does not match the manifest.")

    return matrix, manifest


def main() -> None:
    from src.data.config import REPO_ROOT, load_config

    data_config = load_config("data")
    core_config = load_config("core")

    processed_dir = (
        REPO_ROOT / data_config["dataset"]["processed_dir"]
    )

    input_dir = processed_dir / "als_input"
    ratings_path = processed_dir / "core_train.csv"

    matrix, source_manifest = load_prepared_matrix(
        input_dir,
        ratings_path,
    )

    fraction = core_config["validation"]["fraction"]
    seed = core_config["validation"]["seed"]

    training, validation = split_interactions(
        matrix,
        fraction=fraction,
        seed=seed,
    )

    if validation.nnz == 0:
        raise ValueError(
            "No validation interactions could be selected while "
            "preserving user and movie training support."
        )

    output_dir = input_dir / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(
        output_dir / "train_interactions.npz",
        training,
    )

    save_npz(
        output_dir / "validation_interactions.npz",
        validation,
    )

    validation_users = int(
        np.count_nonzero(np.diff(validation.indptr))
    )

    eligible_users = int(
        np.count_nonzero(np.diff(matrix.indptr) >= 2)
    )

    output_files = [
        "train_interactions.npz",
        "validation_interactions.npz",
    ]

    manifest = {
        "schema_version": 1,
        "validation_fraction_requested": float(fraction),
        "validation_seed": int(seed),
        "matrix_orientation": "user_by_movie",
        "shape": list(matrix.shape),
        "source_positive_interactions": int(matrix.nnz),
        "training_interactions": int(training.nnz),
        "validation_interactions": int(validation.nnz),
        "eligible_users": eligible_users,
        "users_with_validation_interactions": validation_users,
        "eligible_users_without_holdouts": (
            eligible_users - validation_users
        ),
        "id_mapping_location": "parent als_input directory",
        "source_dataset": source_manifest["provenance"]["dataset"],
        "source_manifest_sha256": file_sha256(
            input_dir / "manifest.json"
        ),
        "file_sha256": {
            filename: file_sha256(output_dir / filename)
            for filename in output_files
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

    print(f"Source interactions: {matrix.nnz:,}")
    print(f"Training interactions: {training.nnz:,}")
    print(f"Validation interactions: {validation.nnz:,}")
    print(f"Users eligible for a holdout: {eligible_users:,}")
    print(f"Users with validation interactions: {validation_users:,}")
    print(f"Matrix shape: {training.shape}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()