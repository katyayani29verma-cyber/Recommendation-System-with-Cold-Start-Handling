"""Refit a selected ALS configuration using all approved core interactions."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import implicit
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix

from src.core.data_prep import file_sha256
from src.core.train import (
    load_training_inputs,
    train_als,
    validate_settings,
)


def restore_full_training(
    training: csr_matrix,
    validation: csr_matrix,
) -> csr_matrix:
    """Restore held-out positives without overlapping or invalid entries."""

    training = csr_matrix(training, copy=True)
    validation = csr_matrix(validation, copy=True)

    training.sum_duplicates()
    validation.sum_duplicates()
    training.eliminate_zeros()
    validation.eliminate_zeros()

    if training.shape != validation.shape:
        raise ValueError("Training and validation shapes must match.")

    if not np.all(np.isin(training.data, [-1.0, 1.0])):
        raise ValueError("Training labels must be +1 or -1.")

    if not np.all(validation.data == 1.0):
        raise ValueError("Validation labels must be positive.")

    if training.multiply(validation).nnz:
        raise ValueError("Training and validation overlap.")

    full_training = (training + validation).tocsr()
    full_training.sort_indices()

    if full_training.nnz != training.nnz + validation.nnz:
        raise RuntimeError("Restored interaction count does not match.")

    return full_training


def main() -> None:
    from src.data.config import REPO_ROOT, load_config

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--selected-artifact",
        required=True,
        help="The evaluated validation-model folder selected for refitting.",
    )

    args = parser.parse_args()

    selected_dir = Path(args.selected_artifact)

    if not selected_dir.is_absolute():
        selected_dir = REPO_ROOT / selected_dir

    selected_manifest_path = selected_dir / "manifest.json"

    with selected_manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        selected_manifest = json.load(handle)

    if selected_manifest.get("schema_version") != 1:
        raise ValueError("Unsupported selected-model manifest version.")

    if selected_manifest.get("stage") != "validation_training":
        raise ValueError("Select an evaluated validation-trained model.")

    selected_mode = selected_manifest["interaction_mode"]

    if selected_mode not in ("positive", "signed"):
        raise ValueError("Unsupported selected interaction mode.")

    for filename in (
        "als_model.npz",
        "user_vectors.npy",
        "item_vectors.npy",
        "user_ids.npy",
        "movie_ids.npy",
    ):
        actual_checksum = file_sha256(selected_dir / filename)
        expected_checksum = selected_manifest["file_sha256"][filename]

        if actual_checksum != expected_checksum:
            raise ValueError(
                f"Selected-model checksum mismatch: {filename}"
            )

    # Confirm that the evaluation report belongs to this exact model.
    report_dir = (
        REPO_ROOT
        / "reports"
        / f"core-validation-{selected_dir.name}"
    )

    with (report_dir / "metadata.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        report_metadata = json.load(handle)

    if (
        report_metadata["model_manifest_sha256"]
        != file_sha256(selected_manifest_path)
    ):
        raise ValueError(
            "The evaluation report is stale. Evaluate the selected model again."
        )

    selection_scores = pd.read_csv(
        report_dir / "summary.csv"
    ).to_dict(orient="records")

    data_config = load_config("data")
    core_config = load_config("core")

    # Use the selected model's settings, not whichever ALS mode is in YAML.
    selected_config = deepcopy(core_config)
    selected_config["als"]["interaction_mode"] = selected_mode

    training, validation, user_ids, movie_ids, input_dir = (
        load_training_inputs(
            data_config,
            selected_config,
        )
    )

    if (
        file_sha256(input_dir / "manifest.json")
        != selected_manifest["source_manifest_sha256"]
    ):
        raise ValueError(
            "Prepared data changed since model selection. "
            "Repeat the validation comparison before refitting."
        )

    selected_user_ids = np.load(
        selected_dir / "user_ids.npy",
        allow_pickle=False,
    )

    selected_movie_ids = np.load(
        selected_dir / "movie_ids.npy",
        allow_pickle=False,
    )

    if not np.array_equal(user_ids, selected_user_ids):
        raise ValueError("Selected user ordering does not match.")

    if not np.array_equal(movie_ids, selected_movie_ids):
        raise ValueError("Selected movie ordering does not match.")

    full_training = restore_full_training(
        training,
        validation,
    )

    parameters = validate_settings(
        selected_manifest["parameters"]
    )

    positives = int(
        np.count_nonzero(full_training.data > 0)
    )

    negatives = int(
        np.count_nonzero(full_training.data < 0)
    )

    print(f"Selected mode: {selected_mode}")
    print(f"Latent dimensions: {parameters['factors']}")
    print(f"Full training positives: {positives:,}")
    print(f"Full training negatives: {negatives:,}")
    print(f"Matrix shape: {full_training.shape}")

    started = perf_counter()

    model, history = train_als(
        full_training,
        parameters,
    )

    elapsed = perf_counter() - started

    # Identify users and movies with positive collaborative support.
    positive_entries = full_training > 0

    user_has_positive = (
        np.asarray(positive_entries.getnnz(axis=1)).ravel() > 0
    )

    item_has_positive = (
        np.asarray(positive_entries.getnnz(axis=0)).ravel() > 0
    )

    # Empirical statistics for Track B to use when designing its prior.
    supported_user_vectors = model.user_factors[
        user_has_positive
    ].astype(np.float64)

    if len(supported_user_vectors) < 2:
        raise ValueError("At least two supported users are needed for statistics.")

    prior_mean = supported_user_vectors.mean(axis=0)

    prior_covariance = np.atleast_2d(
        np.cov(
            supported_user_vectors,
            rowvar=False,
            ddof=1,
        )
    )

    run_identity = {
        "stage": "full_core_training",
        "interaction_mode": selected_mode,
        "parameters": parameters,
        "source_manifest_sha256": file_sha256(
            input_dir / "manifest.json"
        ),
        "selected_model_manifest_sha256": file_sha256(
            selected_manifest_path
        ),
    }

    run_hash = sha256(
        json.dumps(
            run_identity,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]

    artifact_root = (
        REPO_ROOT
        / core_config["als"].get("artifact_root", "artifacts")
    )

    output_dir = artifact_root / (
        f"als-final-{selected_mode}-{parameters['factors']}d-{run_hash}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(output_dir / "als_model.npz"))

    arrays = {
        "user_vectors.npy": model.user_factors,
        "item_vectors.npy": model.item_factors,
        "user_ids.npy": user_ids,
        "movie_ids.npy": movie_ids,
        "user_has_positive.npy": user_has_positive,
        "item_has_positive.npy": item_has_positive,
        "prior_mean.npy": prior_mean,
        "prior_covariance.npy": prior_covariance,
    }

    for filename, values in arrays.items():
        np.save(
            output_dir / filename,
            values,
            allow_pickle=False,
        )

    filenames = ["als_model.npz"] + list(arrays)

    manifest = {
        "schema_version": 1,
        "stage": "full_core_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interaction_mode": selected_mode,
        "parameters": parameters,
        "positive_training_interactions": positives,
        "negative_training_interactions": negatives,
        "training_shape": list(full_training.shape),
        "user_vector_shape": list(model.user_factors.shape),
        "item_vector_shape": list(model.item_factors.shape),
        "users_with_positive_support": int(user_has_positive.sum()),
        "movies_with_positive_support": int(item_has_positive.sum()),
        "vector_normalization": "raw_learned_factors",
        "score": "dot(user_vector, item_vector)",
        "model_alpha": 1.0,
        "elapsed_seconds": elapsed,
        "training_history": history,
        "source_manifest_sha256": run_identity[
            "source_manifest_sha256"
        ],
        "selected_model_manifest_sha256": run_identity[
            "selected_model_manifest_sha256"
        ],
        "selection_report_sha256": file_sha256(
            report_dir / "summary.csv"
        ),
        "selection_validation_scores": selection_scores,
        "prior_statistics": (
            "Empirical mean and sample covariance of user factors "
            "for users with positive training support."
        ),
        "versions": {
            "python": platform.python_version(),
            "implicit": implicit.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
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

    print("\nFull-core training completed.")
    print(f"User-vector shape: {model.user_factors.shape}")
    print(f"Movie-vector shape: {model.item_factors.shape}")
    print(
        "Movies with positive training support: "
        f"{int(item_has_positive.sum()):,}"
    )
    print(f"Final training loss: {history[-1]['training_loss']:.6f}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()