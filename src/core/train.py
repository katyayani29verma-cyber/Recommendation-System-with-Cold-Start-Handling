"""Train CPU ALS from verified positive-only or signed inputs."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import implicit
import numpy as np
import scipy
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, load_npz
from threadpoolctl import threadpool_limits

from src.core.data_prep import file_sha256


def validate_settings(settings: dict) -> dict:
    checked = {}

    integer_settings = {
        "factors": 1,
        "iterations": 1,
        "num_threads": 0,
        "seed": 0,
    }

    for name, minimum in integer_settings.items():
        value = settings[name]

        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer.")

        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")

        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")

        checked[name] = int(value)

    # This default keeps existing positive-only callers compatible.
    negative_confidence = settings.get(
        "negative_confidence",
        settings["positive_confidence"],
    )

    numeric_settings = {
        "regularization": settings["regularization"],
        "positive_confidence": settings["positive_confidence"],
        "negative_confidence": negative_confidence,
    }

    for name, value in numeric_settings.items():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be numeric.")

        try:
            value = float(value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name} must be numeric.") from error

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        checked[name] = value

    if checked["regularization"] <= 0:
        raise ValueError("regularization must be greater than zero.")

    for name in ("positive_confidence", "negative_confidence"):
        if not 1.0 <= checked[name] <= np.finfo(np.float32).max:
            raise ValueError(
                f"{name} must be at least 1 and representable as float32."
            )

    return checked


def confidence_matrix(
    interactions: csr_matrix,
    settings: dict,
) -> csr_matrix:
    """Convert recorded labels into ALS confidence entries."""

    parameters = validate_settings(settings)

    matrix = csr_matrix(
        interactions,
        dtype=np.float32,
        copy=True,
    )

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    if matrix.nnz == 0:
        raise ValueError("The training matrix has no recorded interactions.")

    if not np.all(np.isin(matrix.data, [-1.0, 1.0])):
        raise ValueError("Recorded labels must be +1 or -1.")

    # Only stored nonzero entries are changed.
    # Unknown entries remain unstored.
    matrix.data = np.where(
        matrix.data > 0,
        parameters["positive_confidence"],
        -parameters["negative_confidence"],
    ).astype(np.float32)

    return matrix


def train_als(
    training: csr_matrix,
    settings: dict,
    *,
    show_progress: bool = True,
) -> tuple[AlternatingLeastSquares, list[dict]]:
    parameters = validate_settings(settings)
    weighted = confidence_matrix(training, parameters)

    history = []

    def record_iteration(iteration, elapsed, loss):
        if loss is None or not np.isfinite(loss):
            raise RuntimeError("ALS produced an invalid training loss.")

        history.append({
            "iteration": int(iteration) + 1,
            "elapsed_seconds": float(elapsed),
            "training_loss": float(loss),
        })

    with threadpool_limits(limits=1, user_api="blas"):
        model = AlternatingLeastSquares(
            factors=parameters["factors"],
            regularization=parameters["regularization"],
            alpha=1.0,
            iterations=parameters["iterations"],
            num_threads=parameters["num_threads"],
            random_state=parameters["seed"],
            dtype=np.float32,
            calculate_training_loss=True,
        )

        model.fit(
            weighted,
            show_progress=show_progress,
            callback=record_iteration,
        )

    expected_user_shape = (
        weighted.shape[0],
        parameters["factors"],
    )

    expected_movie_shape = (
        weighted.shape[1],
        parameters["factors"],
    )

    if model.user_factors.shape != expected_user_shape:
        raise RuntimeError("Unexpected user-vector dimensions.")

    if model.item_factors.shape != expected_movie_shape:
        raise RuntimeError("Unexpected movie-vector dimensions.")

    if not np.isfinite(model.user_factors).all():
        raise RuntimeError("User vectors contain invalid numbers.")

    if not np.isfinite(model.item_factors).all():
        raise RuntimeError("Movie vectors contain invalid numbers.")

    return model, history


def load_training_inputs(data_config: dict, core_config: dict):
    from src.data.config import REPO_ROOT

    mode = core_config["als"]["interaction_mode"]

    if mode not in ("positive", "signed"):
        raise ValueError("interaction_mode must be positive or signed.")

    processed_dir = (
        REPO_ROOT / data_config["dataset"]["processed_dir"]
    )

    input_dir = processed_dir / "als_input_signed"

    with (input_dir / "manifest.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported signed-data manifest version.")

    if manifest.get("matrix_orientation") != "user_by_movie":
        raise ValueError("Expected user rows and movie columns.")

    if (
        manifest.get("interaction_encoding")
        != "signed_positive_negative_indicators"
    ):
        raise ValueError("Unexpected interaction encoding.")

    if (
        manifest["positive_rating_threshold"]
        != data_config["positive_rating_threshold"]
        or manifest["negative_rating_threshold"]
        != data_config["negative_rating_threshold"]
        or manifest["validation_config"]
        != core_config["validation"]
    ):
        raise ValueError(
            "Preparation settings changed. Run signed_data_prep again."
        )

    movies_path = (
        REPO_ROOT
        / data_config["dataset"]["raw_dir"]
        / data_config["dataset"]["name"]
        / "movies.csv"
    )

    source_paths = {
        "core_train.csv": processed_dir / "core_train.csv",
        "movies.csv": movies_path,
        "core_user_ids.txt": processed_dir / "core_user_ids.txt",
        "eval_user_ids.txt": processed_dir / "eval_user_ids.txt",
    }

    for name, path in source_paths.items():
        if file_sha256(path) != manifest["source_sha256"][name]:
            raise ValueError(
                f"Source file changed: {name}. Run signed_data_prep again."
            )

    required_files = [
        "positive_interactions.npz",
        "signed_interactions.npz",
        "positive_train.npz",
        "signed_train.npz",
        "validation_interactions.npz",
        "user_ids.npy",
        "movie_ids.npy",
    ]

    for name in required_files:
        if file_sha256(input_dir / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Prepared-file checksum mismatch: {name}")

    positive_source = load_npz(
        input_dir / "positive_interactions.npz"
    ).tocsr()

    signed_source = load_npz(
        input_dir / "signed_interactions.npz"
    ).tocsr()

    positive_training = load_npz(
        input_dir / "positive_train.npz"
    ).tocsr()

    signed_training = load_npz(
        input_dir / "signed_train.npz"
    ).tocsr()

    validation = load_npz(
        input_dir / "validation_interactions.npz"
    ).tocsr()

    user_ids = np.load(
        input_dir / "user_ids.npy",
        allow_pickle=False,
    )

    movie_ids = np.load(
        input_dir / "movie_ids.npy",
        allow_pickle=False,
    )

    if user_ids.ndim != 1 or movie_ids.ndim != 1:
        raise ValueError("ID arrays must be one-dimensional.")

    expected_shape = (len(user_ids), len(movie_ids))

    matrices = (
        positive_source,
        signed_source,
        positive_training,
        signed_training,
        validation,
    )

    if any(matrix.shape != expected_shape for matrix in matrices):
        raise ValueError("Matrix dimensions do not match the ID arrays.")

    if list(expected_shape) != manifest["shape"]:
        raise ValueError("Matrix dimensions do not match the manifest.")

    if not np.all(validation.data == 1.0):
        raise ValueError("Validation must contain positive labels only.")

    for training, source in (
        (positive_training, positive_source),
        (signed_training, signed_source),
    ):
        if training.multiply(validation).nnz:
            raise ValueError("Training and validation overlap.")

        if (training + validation != source).nnz:
            raise ValueError("A split does not reconstruct its source.")

    training = (
        signed_training
        if mode == "signed"
        else positive_training
    )

    return training, validation, user_ids, movie_ids, input_dir


def main() -> None:
    from src.data.config import REPO_ROOT, load_config

    data_config = load_config("data")
    core_config = load_config("core")
    settings = core_config["als"]
    parameters = validate_settings(settings)
    mode = settings["interaction_mode"]

    training, validation, user_ids, movie_ids, input_dir = (
        load_training_inputs(data_config, core_config)
    )

    positives = int(np.count_nonzero(training.data > 0))
    negatives = int(np.count_nonzero(training.data < 0))

    print(f"Training mode: {mode}")
    print(f"Positive training interactions: {positives:,}")
    print(f"Negative training interactions: {negatives:,}")
    print(f"Validation positives held out: {validation.nnz:,}")
    print(f"Input matrix shape: {training.shape}")
    print(f"Latent dimensions: {parameters['factors']}")

    started = perf_counter()

    model, history = train_als(
        training,
        parameters,
    )

    elapsed = perf_counter() - started

    # Distinguish runs by mode, settings, and prepared-data identity.
    run_identity = {
        "mode": mode,
        "parameters": parameters,
        "source_manifest_sha256": file_sha256(
            input_dir / "manifest.json"
        ),
    }

    run_hash = sha256(
        json.dumps(
            run_identity,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]

    folder_name = (
        f"als-{mode}-{parameters['factors']}d-{run_hash}"
    )

    output_dir = (
        REPO_ROOT / settings["artifact_root"] / folder_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(output_dir / "als_model.npz"))

    arrays = {
        "user_vectors.npy": model.user_factors,
        "item_vectors.npy": model.item_factors,
        "user_ids.npy": user_ids,
        "movie_ids.npy": movie_ids,
    }

    for name, values in arrays.items():
        np.save(
            output_dir / name,
            values,
            allow_pickle=False,
        )

    filenames = ["als_model.npz"] + list(arrays)

    manifest = {
        "schema_version": 1,
        "stage": "validation_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interaction_mode": mode,
        "parameters": parameters,
        "positive_training_interactions": positives,
        "negative_training_interactions": negatives,
        "validation_positives": int(validation.nnz),
        "training_shape": list(training.shape),
        "user_vector_shape": list(model.user_factors.shape),
        "item_vector_shape": list(model.item_factors.shape),
        "vector_normalization": "raw_learned_factors",
        "score": "dot(user_vector, item_vector)",
        "model_alpha": 1.0,
        "confidence_convention": {
            "positive_target": 1.0,
            "positive_confidence": parameters["positive_confidence"],
            "explicit_negative_target": 0.0,
            "explicit_negative_confidence": parameters["negative_confidence"],
            "unknown_target": 0.0,
            "unknown_confidence": 1.0,
        },
        "elapsed_seconds": elapsed,
        "training_history": history,
        "source_manifest_sha256": run_identity[
            "source_manifest_sha256"
        ],
        "versions": {
            "python": platform.python_version(),
            "implicit": implicit.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "file_sha256": {
            name: file_sha256(output_dir / name)
            for name in filenames
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

    print("\nTraining completed.")
    print(f"User-vector shape: {model.user_factors.shape}")
    print(f"Movie-vector shape: {model.item_factors.shape}")
    print(f"Final training loss: {history[-1]['training_loss']:.6f}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()