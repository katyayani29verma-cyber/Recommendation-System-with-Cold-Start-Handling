"""Refit the validation-selected content mapping for the final ALS model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import shutil

import numpy as np
import pandas as pd

from src.content.cold_start_validate import (
    read_checked,
    read_manifest,
    resolve_path,
    rows_for_ids,
)
from src.content.metadata import FORMAT_VERSION
from src.content.train_mapping import fit_mapping, verify_files
from src.core.als_core import load_core
from src.core.data_prep import file_sha256
from src.core.hybrid_core import load_hybrid_core, validate_ids
from src.data.config import REPO_ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--als-artifact", required=True)
    parser.add_argument("--selection-artifact", required=True)
    parser.add_argument("--split-dir", required=True)
    args = parser.parse_args()

    als_directory = resolve_path(args.als_artifact)
    selection_directory = resolve_path(args.selection_artifact)
    split_directory = resolve_path(args.split_dir)

    data_config = load_config("data")
    content_config = load_config("content")
    cache_directory = resolve_path(content_config["cache_dir"])

    core = load_core(als_directory)

    als_manifest = read_manifest(
        als_directory, "full_core_training"
    )
    selection = read_manifest(
        selection_directory, "cold_mapping_selection"
    )
    split_manifest = read_manifest(
        split_directory, "item_cold_start_split"
    )
    cache_manifest = read_manifest(cache_directory)

    verify_files(selection_directory, selection["file_sha256"])
    verify_files(cache_directory, cache_manifest["file_sha256"])

    if (
        selection.get("selection_metric") != "ndcg_at_10"
        or selection.get("evaluation_scope") != "cold_movies_only"
        or selection.get("protocol_version") != 1
    ):
        raise ValueError("Unexpected content-selection protocol.")

    if (
        selection["split_manifest_sha256"]
        != file_sha256(split_directory / "manifest.json")
        or split_manifest["source_manifest_sha256"]
        != als_manifest["source_manifest_sha256"]
    ):
        raise ValueError("Selection and final ALS use different source data.")

    if (
        selection["embedding_manifest_sha256"]
        != file_sha256(cache_directory / "manifest.json")
    ):
        raise ValueError("Text embeddings changed since mapping selection.")

    movies_path = (
        REPO_ROOT
        / data_config["dataset"]["raw_dir"]
        / data_config["dataset"]["name"]
        / "movies.csv"
    )

    metadata_hash = file_sha256(movies_path)
    signature = cache_manifest["signature"]

    if (
        metadata_hash != selection["metadata_sha256"]
        or metadata_hash != signature["metadata_source_sha256"]
        or signature["text_format_version"] != FORMAT_VERSION
        or signature["normalize_text_embeddings"] is not True
    ):
        raise ValueError("Metadata or text formatting changed.")

    # This value was chosen on cold validation, not on cold test results.
    alpha = float(selection["selected_alphas"]["transformer"])

    transformer_trials = [
        trial
        for trial in selection["validation_trials"]
        if trial["method"] == "transformer"
    ]

    if not transformer_trials:
        raise ValueError("Transformer validation trials are missing.")

    best_trial = max(
        transformer_trials,
        key=lambda trial: (trial["ndcg_at_10"], -trial["alpha"]),
    )

    if alpha != float(best_trial["alpha"]):
        raise ValueError("Selected alpha disagrees with validation results.")

    catalog_ids = validate_ids(
        read_checked(cache_directory, "movie_ids.npy", cache_manifest)
    )
    embeddings = read_checked(
        cache_directory, "embeddings.npy", cache_manifest
    )

    if (
        embeddings.shape != (
            len(catalog_ids),
            cache_manifest["embedding_dimension"],
        )
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Invalid cached text embeddings.")

    warm_ids = np.array(
        sorted(core.available_movie_ids),
        dtype=np.int64,
    )
    warm_rows = rows_for_ids(catalog_ids, warm_ids)

    targets = np.stack([
        core.get_item_vector(int(movie_id))
        for movie_id in warm_ids
    ])

    print(f"Selected Transformer alpha: {alpha}")
    print(f"Final ALS version: {core.artifact_version}")
    print(f"Supported mapping-training movies: {len(warm_ids):,}")

    mapping = fit_mapping(
        embeddings[warm_rows],
        targets,
        alpha,
    )

    weights = mapping.coef_.T.astype(np.float64)
    bias = np.asarray(mapping.intercept_, dtype=np.float64)

    predictions = (
        embeddings.astype(np.float64) @ weights + bias
    ).astype(np.float32)

    if (
        predictions.shape != (len(catalog_ids), core.n_dims)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("Invalid final catalog predictions.")

    identity = {
        "source_als_manifest_sha256": file_sha256(
            als_directory / "manifest.json"
        ),
        "selection_manifest_sha256": file_sha256(
            selection_directory / "manifest.json"
        ),
        "embedding_cache_manifest_sha256": file_sha256(
            cache_directory / "manifest.json"
        ),
        "ridge_alpha": alpha,
        "refit_protocol_version": 1,
    }

    run_hash = sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    output_directory = (
        REPO_ROOT
        / content_config["artifact_root"]
        / f"content-final-{core.n_dims}d-{run_hash}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    arrays = {
        "mapping_weights.npy": weights,
        "mapping_bias.npy": bias,
        "catalog_movie_ids.npy": catalog_ids,
        "catalog_predictions.npy": predictions,
        "warm_movie_ids.npy": warm_ids,
        # No additional holdout is made during this final refit.
        # Validation selection is recorded in selection_manifest.json.
        "mapping_validation_movie_ids.npy": np.array([], dtype=np.int64),
    }

    for filename, values in arrays.items():
        np.save(
            output_directory / filename,
            values,
            allow_pickle=False,
        )

    shutil.copytree(
        cache_directory / "encoder",
        output_directory / "encoder",
        dirs_exist_ok=True,
    )
    shutil.copyfile(
        selection_directory / "manifest.json",
        output_directory / "selection_manifest.json",
    )

    filenames = list(arrays) + ["selection_manifest.json"]
    filenames += [
        path.relative_to(output_directory).as_posix()
        for path in (output_directory / "encoder").rglob("*")
        if path.is_file()
    ]

    manifest = {
        "schema_version": 1,
        "stage": "content_mapping",
        "artifact_version": output_directory.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **identity,
        "source_als_version": core.artifact_version,
        "metadata_source_sha256": metadata_hash,
        "text_format_version": FORMAT_VERSION,
        "text_model_name": signature["model_name"],
        "text_model_revision": cache_manifest["resolved_revision"],
        "embedding_dimension": embeddings.shape[1],
        "latent_dimension": core.n_dims,
        "normalize_text_embeddings": True,
        "normalize_latent_vectors": False,
        "mapping_training_movies": len(warm_ids),
        "catalog_movie_count": len(catalog_ids),
        "selection_method": "cold_validation_ndcg_at_10",
        "refit_scope": "all_supported_core_training_movies",
        "mapping_validation_movie_ids_meaning": (
            "Empty: no additional holdout in the final refit. "
            "Selection was performed in the separate cold-start experiment."
        ),
        "versions": {
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
        },
        "file_sha256": {
            filename: file_sha256(output_directory / filename)
            for filename in filenames
        },
    }

    with (output_directory / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    # Load the files we just saved through the actual hybrid loader.
    hybrid = load_hybrid_core(als_directory, output_directory)

    # Verify that encoding real movie metadata agrees with its cached vector.
    cold_ids = [
        int(movie_id)
        for movie_id in catalog_ids
        if int(movie_id) not in hybrid.als_movie_ids
    ]

    if cold_ids:
        movie_id = cold_ids[0]
        movies = pd.read_csv(movies_path).set_index("movieId")
        movie = movies.loc[movie_id]

        encoded = hybrid.embed_new_item({
            "title": movie["title"],
            "genres": (
                movie["genres"]
                if pd.notna(movie["genres"])
                else ""
            ),
        })

        np.testing.assert_allclose(
            encoded,
            hybrid.get_item_vector(movie_id),
            rtol=1e-4,
            atol=1e-6,
        )

        print("Real-movie encoding agrees with the saved prediction.")

    recommendations = hybrid.recommend(hybrid.prior_mean, k=10)

    if len(recommendations) != min(10, len(catalog_ids)):
        raise RuntimeError("Unexpected recommendation count.")

    print("\nFinal content refit completed.")
    print(f"Ridge alpha: {alpha}")
    print(f"Catalog predictions: {predictions.shape}")
    print(f"Hybrid scorable movies: {len(hybrid.available_movie_ids):,}")
    print("Saved hybrid loaded and recommendation check passed.")
    print(f"Saved to: {output_directory}")


if __name__ == "__main__":
    main()