"""Train experimental ALS using only the warm movie group."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz

from src.core.als_core import load_core
from src.core.data_prep import file_sha256
from src.core.hybrid_core import validate_ids
from src.core.train import train_als, validate_settings
from src.data.config import REPO_ROOT


def compact_warm_matrix(interactions, movie_ids, warm_movie_ids):
    """Keep warm columns in the exact order of warm_movie_ids."""
    movie_ids = validate_ids(movie_ids)
    warm_movie_ids = validate_ids(warm_movie_ids)

    matrix = csr_matrix(interactions, copy=True)

    if np.iscomplexobj(matrix.data):
        raise ValueError("Interactions must be real-valued.")

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    if matrix.shape[1] != len(movie_ids):
        raise ValueError("Movie IDs do not match matrix columns.")

    if matrix.nnz == 0 or not np.all(matrix.data == 1):
        raise ValueError("Expected nonempty binary positive interactions.")

    movie_to_column = {
        int(movie_id): column
        for column, movie_id in enumerate(movie_ids)
    }

    if any(int(movie_id) not in movie_to_column for movie_id in warm_movie_ids):
        raise ValueError("A warm movie is missing from the catalog.")

    columns = np.array([
        movie_to_column[int(movie_id)]
        for movie_id in warm_movie_ids
    ])

    compact = matrix[:, columns].astype(np.float32).tocsr()

    # Any removed nonzero entry would belong to a non-warm movie.
    if compact.nnz != matrix.nnz:
        raise ValueError("Training contains interactions outside warm movies.")

    if np.any(np.asarray(compact.getnnz(axis=0)).ravel() == 0):
        raise ValueError("Every warm movie must have training support.")

    return compact


def resolve_directory(value):
    directory = Path(value)

    if not directory.is_absolute():
        directory = REPO_ROOT / directory

    return directory.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--selected-als-artifact", required=True)
    args = parser.parse_args()

    split_directory = resolve_directory(args.split_dir)
    selected_directory = resolve_directory(args.selected_als_artifact)

    split_manifest_path = split_directory / "manifest.json"
    selected_manifest_path = selected_directory / "manifest.json"

    with split_manifest_path.open("r", encoding="utf-8") as handle:
        split_manifest = json.load(handle)

    with selected_manifest_path.open("r", encoding="utf-8") as handle:
        selected_manifest = json.load(handle)

    if (
        split_manifest.get("schema_version") != 1
        or split_manifest.get("stage") != "item_cold_start_split"
    ):
        raise ValueError("Expected an item cold-start split.")

    if selected_manifest.get("interaction_mode") != "positive":
        raise ValueError("This experiment expects positive-only ALS.")

    # Verify the selected artifact before using its settings.
    selected_core = load_core(selected_directory)

    if (
        split_manifest["source_manifest_sha256"]
        != selected_manifest["source_manifest_sha256"]
    ):
        raise ValueError("The split and selected model use different source data.")

    checksums = split_manifest["file_sha256"]

    required = {
        "warm_interactions.npz",
        "user_ids.npy",
        "movie_ids.npy",
        "warm_movie_ids.npy",
        "validation_movie_ids.npy",
        "test_movie_ids.npy",
    }

    if not required.issubset(checksums):
        raise ValueError("The split manifest is missing required files.")

    for filename, expected_hash in checksums.items():
        path = (split_directory / filename).resolve()

        if not path.is_relative_to(split_directory):
            raise ValueError("A split file path escapes its directory.")

        if not path.is_file():
            raise FileNotFoundError(path)

        if file_sha256(path) != expected_hash:
            raise ValueError(f"Split checksum mismatch: {filename}")

    def read_array(filename):
        return np.load(
            split_directory / filename,
            allow_pickle=False,
        )

    user_ids = validate_ids(read_array("user_ids.npy"))
    movie_ids = validate_ids(read_array("movie_ids.npy"))
    warm_ids = validate_ids(read_array("warm_movie_ids.npy"))
    validation_ids = validate_ids(read_array("validation_movie_ids.npy"))
    test_ids = validate_ids(read_array("test_movie_ids.npy"))

    groups = [set(warm_ids), set(validation_ids), set(test_ids)]

    if (
        groups[0] & groups[1]
        or groups[0] & groups[2]
        or groups[1] & groups[2]
    ):
        raise ValueError("Warm, validation, and test movies overlap.")

    if set.union(*groups) != set(selected_core.available_movie_ids):
        raise ValueError("Movie groups differ from the supported source catalog.")

    if not np.array_equal(movie_ids, selected_core.movie_ids):
        raise ValueError("Catalog ordering differs from the selected model.")

    selected_user_ids = np.load(
        selected_directory / "user_ids.npy",
        allow_pickle=False,
    )

    if not np.array_equal(user_ids, selected_user_ids):
        raise ValueError("User ordering differs from the selected model.")

    full_width_warm = load_npz(
        split_directory / "warm_interactions.npz"
    ).tocsr()

    if (
        full_width_warm.shape != (len(user_ids), len(movie_ids))
        or list(full_width_warm.shape) != split_manifest["shape"]
    ):
        raise ValueError("Warm matrix dimensions do not match.")

    training = compact_warm_matrix(
        full_width_warm,
        movie_ids,
        warm_ids,
    )

    if (
        training.nnz != split_manifest["interaction_counts"]["warm"]
        or len(warm_ids) != split_manifest["movie_counts"]["warm"]
    ):
        raise ValueError("Warm counts differ from the split manifest.")

    parameters = validate_settings(selected_manifest["parameters"])

    print(f"Experimental users: {len(user_ids):,}")
    print(f"Warm training movies: {len(warm_ids):,}")
    print(f"Warm training interactions: {training.nnz:,}")
    print(f"Training matrix shape: {training.shape}")
    print(f"Latent dimensions: {parameters['factors']}")
    print("Cold movies included in ALS training: 0")

    # A new model is fitted here. Existing learned factors are not reused.
    model, history = train_als(training, parameters)

    identity = {
        "stage": "item_cold_start_training",
        "split_manifest_sha256": file_sha256(split_manifest_path),
        "selected_model_manifest_sha256": file_sha256(selected_manifest_path),
        "parameters": parameters,
    }

    run_hash = sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    output_directory = (
        REPO_ROOT
        / "artifacts"
        / f"als-cold-experiment-{parameters['factors']}d-{run_hash}"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    model.save(str(output_directory / "als_model.npz"))

    arrays = {
        "user_ids.npy": user_ids,
        "movie_ids.npy": warm_ids,
        "user_vectors.npy": model.user_factors,
        "item_vectors.npy": model.item_factors,
        "user_has_warm_positive.npy": (
            np.asarray(training.getnnz(axis=1)).ravel() > 0
        ),
    }

    for filename, values in arrays.items():
        np.save(
            output_directory / filename,
            values,
            allow_pickle=False,
        )

    filenames = ["als_model.npz"] + list(arrays)

    manifest = {
        "schema_version": 1,
        **identity,
        "artifact_version": output_directory.name,
        "interaction_mode": "positive",
        "training_shape": list(training.shape),
        "training_interactions": int(training.nnz),
        "user_vector_shape": list(model.user_factors.shape),
        "item_vector_shape": list(model.item_factors.shape),
        "vector_normalization": "raw_learned_factors",
        "training_history": history,
        "file_sha256": {
            filename: file_sha256(output_directory / filename)
            for filename in filenames
        },
    }

    with (output_directory / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    print("\nExperimental ALS training completed.")
    print(f"User-vector shape: {model.user_factors.shape}")
    print(f"Warm-movie-vector shape: {model.item_factors.shape}")
    print(f"Final training loss: {history[-1]['training_loss']:.6f}")
    print(f"Saved to: {output_directory}")


if __name__ == "__main__":
    main()