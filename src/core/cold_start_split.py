"""Create disjoint movie groups for an item cold-start experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from src.core.data_prep import file_sha256


def split_by_movie(
    interactions,
    validation_fraction=0.1,
    test_fraction=0.1,
    seed=42,
):
    """Move all interactions for each movie into one group."""

    fractions = []

    for value in (validation_fraction, test_fraction):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Split fractions must be numeric.")

        value = float(value)

        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("Each fraction must be between zero and one.")

        fractions.append(value)

    validation_fraction, test_fraction = fractions

    if validation_fraction + test_fraction >= 1:
        raise ValueError("The fractions must leave movies for training.")

    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("Seed must be a nonnegative integer.")

    matrix = csr_matrix(interactions, copy=True)

    if np.iscomplexobj(matrix.data):
        raise ValueError("Interactions must be real-valued.")

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    if matrix.nnz == 0 or not np.all(matrix.data == 1):
        raise ValueError("Expected a nonempty binary positive matrix.")

    matrix = matrix.astype(np.float32)

    supported_columns = np.flatnonzero(
        np.asarray(matrix.getnnz(axis=0)).ravel() > 0
    )

    movie_count = len(supported_columns)

    if movie_count < 3:
        raise ValueError("At least three supported movies are required.")

    validation_count = max(
        1,
        int(movie_count * validation_fraction),
    )
    test_count = max(
        1,
        int(movie_count * test_fraction),
    )

    if validation_count + test_count >= movie_count:
        raise ValueError("The requested split leaves no warm movies.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(supported_columns)

    validation_columns = np.sort(shuffled[:validation_count])

    test_columns = np.sort(
        shuffled[
            validation_count:
            validation_count + test_count
        ]
    )

    warm_columns = np.sort(
        shuffled[validation_count + test_count:]
    )

    groups = {
        "warm": warm_columns,
        "validation": validation_columns,
        "test": test_columns,
    }

    result = {}

    for name, columns in groups.items():
        mask = np.zeros(matrix.shape[1], dtype=np.float32)
        mask[columns] = 1.0

        group_matrix = matrix.multiply(mask).tocsr()
        group_matrix.eliminate_zeros()
        group_matrix.sort_indices()

        result[f"{name}_columns"] = columns
        result[f"{name}_interactions"] = group_matrix

    warm = result["warm_interactions"]
    validation = result["validation_interactions"]
    test = result["test_interactions"]

    # Verify that whole cold-movie columns are absent from training.
    if (
        warm[:, validation_columns].nnz
        or warm[:, test_columns].nnz
        or validation[:, test_columns].nnz
    ):
        raise RuntimeError("Movie groups overlap.")

    if (warm + validation + test != matrix).nnz:
        raise RuntimeError("The split does not reconstruct the source.")

    warm_support = (
        np.asarray(warm.getnnz(axis=1)).ravel() > 0
    )

    result["validation_users"] = (
        warm_support
        & (np.asarray(validation.getnnz(axis=1)).ravel() > 0)
    )

    result["test_users"] = (
        warm_support
        & (np.asarray(test.getnnz(axis=1)).ravel() > 0)
    )

    return result


def main():
    from src.core.refit import restore_full_training
    from src.core.train import load_training_inputs
    from src.data.config import REPO_ROOT, load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
    )
    args = parser.parse_args()

    data_config = load_config("data")
    core_config = deepcopy(load_config("core"))

    # This experiment uses the selected positive-only approach.
    core_config["als"]["interaction_mode"] = "positive"

    (
        previous_training,
        previous_validation,
        user_ids,
        movie_ids,
        input_directory,
    ) = load_training_inputs(data_config, core_config)

    # Restore the core positives, then make a new movie-level split.
    full_positive = restore_full_training(
        previous_training,
        previous_validation,
    )

    processed_directory = (
        REPO_ROOT / data_config["dataset"]["processed_dir"]
    )

    evaluation_user_ids = np.loadtxt(
        processed_directory / "eval_user_ids.txt",
        dtype=np.int64,
        ndmin=1,
    )

    if np.intersect1d(user_ids, evaluation_user_ids).size:
        raise ValueError("Track C evaluation users entered the experiment.")

    split = split_by_movie(
        full_positive,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    if not split["validation_users"].any():
        raise ValueError("No users qualify for cold validation.")

    if not split["test_users"].any():
        raise ValueError("No users qualify for cold testing.")

    identity = {
        "source_manifest_sha256": file_sha256(
            input_directory / "manifest.json"
        ),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "split_method": "random_supported_movie_partition_v1",
    }

    run_hash = sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    output_directory = (
        processed_directory
        / "cold_start"
        / f"item-split-{run_hash}"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    filenames = []

    for group in ("warm", "validation", "test"):
        filename = f"{group}_interactions.npz"

        save_npz(
            output_directory / filename,
            split[f"{group}_interactions"],
        )

        filenames.append(filename)

    arrays = {
        "user_ids.npy": user_ids,
        "movie_ids.npy": movie_ids,
        "warm_movie_ids.npy": movie_ids[split["warm_columns"]],
        "validation_movie_ids.npy": movie_ids[
            split["validation_columns"]
        ],
        "test_movie_ids.npy": movie_ids[split["test_columns"]],
        "validation_user_mask.npy": split["validation_users"],
        "test_user_mask.npy": split["test_users"],
    }

    for filename, values in arrays.items():
        np.save(
            output_directory / filename,
            values,
            allow_pickle=False,
        )

        filenames.append(filename)

    manifest = {
        "schema_version": 1,
        "stage": "item_cold_start_split",
        **identity,
        "matrix_orientation": "user_by_movie",
        "interaction_encoding": "binary_positive",
        "shape": list(full_positive.shape),
        "positive_rating_threshold": data_config[
            "positive_rating_threshold"
        ],
        "source_positive_interactions": int(full_positive.nnz),
        "movie_counts": {
            group: len(split[f"{group}_columns"])
            for group in ("warm", "validation", "test")
        },
        "interaction_counts": {
            group: int(split[f"{group}_interactions"].nnz)
            for group in ("warm", "validation", "test")
        },
        "eligible_validation_users": int(
            split["validation_users"].sum()
        ),
        "eligible_test_users": int(split["test_users"].sum()),
        "eligibility_rule": (
            "At least one warm positive and at least one positive "
            "in the relevant cold group."
        ),
        "file_sha256": {
            filename: file_sha256(output_directory / filename)
            for filename in filenames
        },
    }

    with (output_directory / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            allow_nan=False,
        )

    print(f"Source positives: {full_positive.nnz:,}")
    print(f"Matrix shape: {full_positive.shape}")

    for group in ("warm", "validation", "test"):
        print(
            f"{group.capitalize()} movies: "
            f"{manifest['movie_counts'][group]:,}"
        )
        print(
            f"{group.capitalize()} interactions: "
            f"{manifest['interaction_counts'][group]:,}"
        )

    print(
        "Eligible validation users: "
        f"{manifest['eligible_validation_users']:,}"
    )
    print(
        "Eligible test users: "
        f"{manifest['eligible_test_users']:,}"
    )
    print("Cold-movie interactions in warm training: 0")
    print(f"Saved to: {output_directory}")


if __name__ == "__main__":
    main()