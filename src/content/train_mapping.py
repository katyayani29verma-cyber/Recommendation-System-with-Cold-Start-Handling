"""Create text embeddings and train a mapping into the final ALS space."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from src.content.metadata import FORMAT_VERSION, metadata_text
from src.core.als_core import load_core
from src.core.data_prep import file_sha256


def fit_mapping(
    embeddings: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> Ridge:
    """Fit a multi-output ridge mapping to raw ALS vectors."""

    if np.iscomplexobj(embeddings) or np.iscomplexobj(targets):
        raise ValueError("Mapping inputs must be real-valued.")

    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)

    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[0] != y.shape[0]
        or x.shape[0] < 2
        or x.shape[1] < 1
        or y.shape[1] < 1
    ):
        raise ValueError("Invalid embedding/target dimensions.")

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Mapping inputs contain invalid numbers.")

    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("Ridge alpha must be positive.")

    alpha = float(alpha)

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be positive and finite.")

    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver="cholesky",
    )

    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(x, y)

    return model


def verify_files(directory: Path, checksums: dict) -> None:
    root = directory.resolve()

    for filename, expected in checksums.items():
        path = (directory / filename).resolve()

        if not path.is_relative_to(root):
            raise ValueError("A manifest path escapes its artifact directory.")

        if not path.is_file():
            raise FileNotFoundError(path)

        if file_sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {filename}")


def embedding_cache(
    movies: pd.DataFrame,
    movies_path: Path,
    config: dict,
    cache_dir: Path,
    rebuild: bool,
):
    """Create or reuse embeddings, recording the encoder's exact revision."""

    signature = {
        "metadata_source_sha256": file_sha256(movies_path),
        "text_format_version": FORMAT_VERSION,
        "model_name": config["model_name"],
        "requested_revision": config.get("model_revision"),
        "sentence_transformers_version": version("sentence-transformers"),
        "normalize_text_embeddings": True,
        "batch_size": int(config["batch_size"]),
        "cpu_threads": int(config["cpu_threads"]),
    }

    manifest_path = cache_dir / "manifest.json"

    if manifest_path.is_file() and not rebuild:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest.get("signature") == signature:
            verify_files(cache_dir, manifest["file_sha256"])

            ids = np.load(
                cache_dir / "movie_ids.npy",
                allow_pickle=False,
            )

            embeddings = np.load(
                cache_dir / "embeddings.npy",
                allow_pickle=False,
            )

            expected_ids = movies["movieId"].to_numpy(dtype=np.int64)

            if not np.array_equal(ids, expected_ids):
                raise ValueError("Cached movie ordering does not match.")

            if embeddings.shape != (
                len(ids),
                manifest["embedding_dimension"],
            ):
                raise ValueError("Cached embedding dimensions do not match.")

            if not np.isfinite(embeddings).all():
                raise ValueError("Cached embeddings contain invalid numbers.")

            print("Reusing verified text embeddings.")

            return embeddings, ids, cache_dir / "encoder", manifest

    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    import torch

    if config["batch_size"] < 1 or config["cpu_threads"] < 1:
        raise ValueError("Batch size and CPU threads must be positive.")

    cache_dir.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(
        config["model_name"],
        revision=config.get("model_revision"),
        token=False,
    )

    resolved_revision = info.sha

    if not resolved_revision:
        raise RuntimeError("Could not resolve the text model's revision.")

    torch.set_num_threads(int(config["cpu_threads"]))

    encoder = SentenceTransformer(
        config["model_name"],
        revision=resolved_revision,
        device="cpu",
        token=False,
        trust_remote_code=False,
        cache_folder=str(cache_dir / "hf_cache"),
    )

    texts = [
        metadata_text({
            "title": row.title,
            "genres": row.genres,
        })
        for row in movies.itertuples(index=False)
    ]

    print(f"Encoding {len(texts):,} movies on CPU.")

    embeddings = encoder.encode(
        texts,
        batch_size=int(config["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(movies):
        raise RuntimeError("Unexpected text-embedding dimensions.")

    if not np.isfinite(embeddings).all():
        raise RuntimeError("The encoder produced invalid numbers.")

    ids = movies["movieId"].to_numpy(dtype=np.int64)

    np.save(
        cache_dir / "embeddings.npy",
        embeddings,
        allow_pickle=False,
    )

    np.save(
        cache_dir / "movie_ids.npy",
        ids,
        allow_pickle=False,
    )

    encoder_dir = cache_dir / "encoder"

    encoder.save_pretrained(
        str(encoder_dir),
        safe_serialization=True,
    )

    files = [
        cache_dir / "embeddings.npy",
        cache_dir / "movie_ids.npy",
    ]

    files += [
        path
        for path in encoder_dir.rglob("*")
        if path.is_file()
    ]

    manifest = {
        "schema_version": 1,
        "signature": signature,
        "resolved_revision": resolved_revision,
        "embedding_dimension": embeddings.shape[1],
        "movie_count": len(ids),
        "torch_version": torch.__version__,
        "file_sha256": {
            path.relative_to(cache_dir).as_posix(): file_sha256(path)
            for path in files
        },
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            allow_nan=False,
        )

    return embeddings, ids, encoder_dir, manifest


def main() -> None:
    from src.data.config import REPO_ROOT, load_config

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--als-artifact",
        required=True,
    )

    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config("content")
    data_config = load_config("data")

    als_dir = Path(args.als_artifact)

    if not als_dir.is_absolute():
        als_dir = REPO_ROOT / als_dir

    core = load_core(als_dir)

    movies_path = (
        REPO_ROOT
        / data_config["dataset"]["raw_dir"]
        / data_config["dataset"]["name"]
        / "movies.csv"
    )

    movies = pd.read_csv(movies_path)

    required_columns = {"movieId", "title", "genres"}

    if not required_columns.issubset(movies.columns):
        raise ValueError("Movie metadata columns are missing.")

    if movies["movieId"].isna().any():
        raise ValueError("Movie IDs cannot be missing.")

    if not pd.api.types.is_integer_dtype(movies["movieId"]):
        raise ValueError("Movie IDs must be integers.")

    if movies["movieId"].duplicated().any():
        raise ValueError("Movie IDs must be unique.")

    if (movies["movieId"] < 1).any():
        raise ValueError("Movie IDs must be positive.")

    movies = movies.sort_values("movieId").reset_index(drop=True)

    # Validate formatting before beginning downloads or encoding.
    for row in movies.itertuples(index=False):
        metadata_text({
            "title": row.title,
            "genres": row.genres,
        })

    cache_dir = REPO_ROOT / config["cache_dir"]

    embeddings, catalog_ids, encoder_dir, cache_manifest = (
        embedding_cache(
            movies,
            movies_path,
            config,
            cache_dir,
            args.rebuild_embeddings,
        )
    )

    catalog_to_row = {
        int(movie_id): row
        for row, movie_id in enumerate(catalog_ids)
    }

    warm_ids = np.array(
        sorted(core.available_movie_ids),
        dtype=np.int64,
    )

    if len(warm_ids) < 3:
        raise ValueError("At least three supported movies are required.")

    if any(int(movie_id) not in catalog_to_row for movie_id in warm_ids):
        raise ValueError("Some ALS movies are missing from the metadata.")

    warm_rows = np.array([
        catalog_to_row[int(movie_id)]
        for movie_id in warm_ids
    ])

    x = embeddings[warm_rows].astype(np.float64)

    y = np.stack([
        core.get_item_vector(int(movie_id))
        for movie_id in warm_ids
    ]).astype(np.float64)

    fraction = float(config["validation_fraction"])

    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Mapping validation fraction must be between 0 and 1.")

    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(len(warm_ids))

    validation_count = min(
        max(1, int(len(warm_ids) * fraction)),
        len(warm_ids) - 2,
    )

    validation_rows = order[:validation_count]
    fitting_rows = order[validation_count:]

    mean_target = y[fitting_rows].mean(axis=0)

    baseline_mse = float(
        np.mean((y[validation_rows] - mean_target) ** 2)
    )

    trials = []

    for alpha in config["ridge_alphas"]:
        mapping = fit_mapping(
            x[fitting_rows],
            y[fitting_rows],
            alpha,
        )

        predicted = mapping.predict(x[validation_rows])

        mse = float(
            np.mean((predicted - y[validation_rows]) ** 2)
        )

        if not np.isfinite(mse):
            raise RuntimeError("Mapping validation produced an invalid error.")

        trials.append({
            "alpha": float(alpha),
            "validation_mse": mse,
        })

    if not trials:
        raise ValueError("At least one ridge alpha is required.")

    best = min(
        trials,
        key=lambda trial: (trial["validation_mse"], trial["alpha"]),
    )

    # Refit the selected mapping using every supported training movie.
    mapping = fit_mapping(x, y, best["alpha"])

    weights = mapping.coef_.T.astype(np.float32)
    bias = np.asarray(mapping.intercept_, dtype=np.float32)

    predictions = (
        embeddings.astype(np.float64)
        @ weights.astype(np.float64)
        + bias.astype(np.float64)
    ).astype(np.float32)

    if predictions.shape != (len(catalog_ids), core.n_dims):
        raise RuntimeError("Unexpected projected-vector dimensions.")

    if not np.isfinite(predictions).all():
        raise RuntimeError("Projected movie vectors contain invalid numbers.")

    identity = {
        "als_manifest_sha256": file_sha256(als_dir / "manifest.json"),
        "embedding_manifest_sha256": file_sha256(
            cache_dir / "manifest.json"
        ),
        "selected_alpha": best["alpha"],
        "validation_fraction": fraction,
        "validation_seed": int(config["seed"]),
    }

    run_hash = sha256(
        json.dumps(
            identity,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]

    output_dir = (
        REPO_ROOT
        / config["artifact_root"]
        / f"content-{core.n_dims}d-{run_hash}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "mapping_weights.npy": weights,
        "mapping_bias.npy": bias,
        "catalog_movie_ids.npy": catalog_ids,
        "catalog_predictions.npy": predictions,
        "warm_movie_ids.npy": warm_ids,
        "mapping_validation_movie_ids.npy": warm_ids[validation_rows],
    }

    for filename, values in arrays.items():
        np.save(
            output_dir / filename,
            values,
            allow_pickle=False,
        )

    # Include the pretrained encoder for inference without new downloads.
    shutil.copytree(
        encoder_dir,
        output_dir / "encoder",
        dirs_exist_ok=True,
    )

    files = [
        output_dir / filename
        for filename in arrays
    ]

    files += [
        path
        for path in (output_dir / "encoder").rglob("*")
        if path.is_file()
    ]

    manifest = {
        "schema_version": 1,
        "stage": "content_mapping",
        "artifact_version": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_als_version": core.artifact_version,
        "source_als_manifest_sha256": identity["als_manifest_sha256"],
        "embedding_cache_manifest_sha256": identity[
            "embedding_manifest_sha256"
        ],
        "metadata_source_sha256": file_sha256(movies_path),
        "text_format_version": FORMAT_VERSION,
        "text_model_name": config["model_name"],
        "text_model_revision": cache_manifest["resolved_revision"],
        "embedding_dimension": embeddings.shape[1],
        "latent_dimension": core.n_dims,
        "normalize_text_embeddings": True,
        "normalize_latent_vectors": False,
        "ridge_alpha": best["alpha"],
        "mapping_training_movies": len(warm_ids),
        "mapping_selection_fit_movies": len(fitting_rows),
        "mapping_selection_validation_movies": len(validation_rows),
        "mapping_validation_mse": best["validation_mse"],
        "mean_baseline_validation_mse": baseline_mse,
        "ridge_trials": trials,
        "validation_type": (
            "Held-out-item latent-vector reconstruction. "
            "This is not a true item-cold-start ranking experiment."
        ),
        "catalog_movie_count": len(catalog_ids),
        "versions": {
            "sentence_transformers": version("sentence-transformers"),
            "torch": cache_manifest["torch_version"],
            "scikit_learn": version("scikit-learn"),
            "numpy": np.__version__,
        },
        "file_sha256": {
            path.relative_to(output_dir).as_posix(): file_sha256(path)
            for path in files
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

    print("\nContent mapping trained.")
    print(f"Catalog text embeddings: {embeddings.shape}")
    print(f"Supported ALS target movies: {len(warm_ids):,}")
    print(f"Selected ridge alpha: {best['alpha']}")
    print(f"Mapping validation MSE: {best['validation_mse']:.8f}")
    print(f"Mean-baseline validation MSE: {baseline_mse:.8f}")
    print(f"Projected catalog vectors: {predictions.shape}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()