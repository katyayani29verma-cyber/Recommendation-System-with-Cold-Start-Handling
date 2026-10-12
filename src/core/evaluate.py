"""Evaluate a validation-trained ALS model against popularity."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

from src.core.data_prep import file_sha256
from src.core.train import load_training_inputs
from src.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
)


def rank_scores(
    scores: np.ndarray,
    movie_ids: np.ndarray,
    excluded_columns: set[int],
    k: int,
) -> list[int]:
    """Return eligible movie IDs in descending score order."""

    scores = np.asarray(scores)

    if scores.ndim != 1 or movie_ids.ndim != 1:
        raise ValueError("Scores and movie IDs must be one-dimensional.")

    if len(scores) != len(movie_ids):
        raise ValueError("Scores and movie IDs must have equal lengths.")

    if not np.isfinite(scores).all():
        raise ValueError("Recommendation scores must be finite.")

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a positive integer.")

    if k < 1:
        raise ValueError("k must be a positive integer.")

    if any(
        column < 0 or column >= len(movie_ids)
        for column in excluded_columns
    ):
        raise ValueError("An excluded column is outside the catalog.")

    eligible = np.ones(len(movie_ids), dtype=bool)

    if excluded_columns:
        eligible[list(excluded_columns)] = False

    columns = np.flatnonzero(eligible)

    # Highest score first; break equal-score ties by movie ID.
    ordering = np.lexsort(
        (
            movie_ids[columns],
            -scores[columns],
        )
    )

    selected_columns = columns[ordering[:k]]

    return [
        int(movie_ids[column])
        for column in selected_columns
    ]


def exclusion_columns(
    observed_movie_ids,
    validation_columns,
    movie_to_column: dict[int, int],
) -> set[int]:
    """Exclude observed history while leaving held-out positives eligible."""

    observed_columns = {
        movie_to_column[int(movie_id)]
        for movie_id in observed_movie_ids
        if int(movie_id) in movie_to_column
    }

    return observed_columns - {
        int(column)
        for column in validation_columns
    }


def main() -> None:
    from src.data.config import REPO_ROOT, load_config

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Folder containing the validation-trained model artifacts.",
    )

    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[5, 10, 20],
    )

    args = parser.parse_args()

    ks = sorted(set(args.k))

    if not ks or min(ks) < 1:
        raise ValueError("Every k must be positive.")

    artifact_dir = Path(args.artifact_dir)

    if not artifact_dir.is_absolute():
        artifact_dir = REPO_ROOT / artifact_dir

    with (artifact_dir / "manifest.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        model_manifest = json.load(handle)

    if model_manifest.get("schema_version") != 1:
        raise ValueError("Unsupported model manifest version.")

    if model_manifest.get("stage") != "validation_training":
        raise ValueError(
            "Evaluate a validation-trained model, not a model "
            "refitted on the full core data."
        )

    for filename in (
        "als_model.npz",
        "user_vectors.npy",
        "item_vectors.npy",
        "user_ids.npy",
        "movie_ids.npy",
    ):
        actual = file_sha256(artifact_dir / filename)
        expected = model_manifest["file_sha256"][filename]

        if actual != expected:
            raise ValueError(f"Model checksum mismatch: {filename}")

    data_config = load_config("data")
    core_config = load_config("core")

    # Load shared signed data for checking both model variants.
    shared_config = deepcopy(core_config)
    shared_config["als"]["interaction_mode"] = "signed"

    _, validation, user_ids, movie_ids, input_dir = (
        load_training_inputs(
            data_config,
            shared_config,
        )
    )

    if (
        file_sha256(input_dir / "manifest.json")
        != model_manifest["source_manifest_sha256"]
    ):
        raise ValueError(
            "The model belongs to a different prepared-data version."
        )

    user_vectors = np.load(
        artifact_dir / "user_vectors.npy",
        allow_pickle=False,
    )

    item_vectors = np.load(
        artifact_dir / "item_vectors.npy",
        allow_pickle=False,
    )

    model_user_ids = np.load(
        artifact_dir / "user_ids.npy",
        allow_pickle=False,
    )

    model_movie_ids = np.load(
        artifact_dir / "movie_ids.npy",
        allow_pickle=False,
    )

    if not np.array_equal(user_ids, model_user_ids):
        raise ValueError("The model's user ordering does not match.")

    if not np.array_equal(movie_ids, model_movie_ids):
        raise ValueError("The model's movie ordering does not match.")

    factors = model_manifest["parameters"]["factors"]

    if user_vectors.shape != (len(user_ids), factors):
        raise ValueError("Unexpected user-vector shape.")

    if item_vectors.shape != (len(movie_ids), factors):
        raise ValueError("Unexpected movie-vector shape.")

    if not (
        np.isfinite(user_vectors).all()
        and np.isfinite(item_vectors).all()
    ):
        raise ValueError("The model vectors contain invalid numbers.")

    # Popularity uses positive TRAINING counts, never validation counts.
    positive_training = load_npz(
        input_dir / "positive_train.npz"
    ).tocsr()

    popularity_scores = np.asarray(
        positive_training.sum(axis=0)
    ).ravel()

    processed_dir = (
        REPO_ROOT / data_config["dataset"]["processed_dir"]
    )

    ratings = pd.read_csv(
        processed_dir / "core_train.csv"
    )

    observed_by_user = {
        int(user_id): set(group["movieId"].tolist())
        for user_id, group in ratings.groupby("userId")
    }

    movie_to_column = {
        int(movie_id): column
        for column, movie_id in enumerate(movie_ids)
    }

    rows = []

    with threadpool_limits(limits=1, user_api="blas"):
        for user_row, user_id in enumerate(user_ids):
            user_id = int(user_id)

            start = validation.indptr[user_row]
            end = validation.indptr[user_row + 1]
            held_out_columns = validation.indices[start:end]

            if len(held_out_columns) == 0:
                continue

            relevant_ids = {
                int(movie_ids[column])
                for column in held_out_columns
            }

            relevance = {
                movie_id: 1.0
                for movie_id in relevant_ids
            }

            # Same exclusions for ALS and popularity.
            # This includes known positives, negatives, and neutral ratings.
            excluded = exclusion_columns(
                observed_by_user.get(user_id, set()),
                held_out_columns,
                movie_to_column,
            )

            als_scores = (
                user_vectors[user_row]
                @ item_vectors.T
            )

            als_ranked = rank_scores(
                als_scores,
                movie_ids,
                excluded,
                max(ks),
            )

            popularity_ranked = rank_scores(
                popularity_scores,
                movie_ids,
                excluded,
                max(ks),
            )

            row = {
                "user_id": user_id,
                "validation_positives": len(relevant_ids),
            }

            for k in ks:
                row[f"als_ndcg@{k}"] = ndcg_at_k(
                    als_ranked,
                    relevance,
                    k,
                )

                row[f"popularity_ndcg@{k}"] = ndcg_at_k(
                    popularity_ranked,
                    relevance,
                    k,
                )

                row[f"als_map@{k}"] = average_precision_at_k(
                    als_ranked,
                    relevant_ids,
                    k,
                )

                row[f"popularity_map@{k}"] = average_precision_at_k(
                    popularity_ranked,
                    relevant_ids,
                    k,
                )

            rows.append(row)

    if not rows:
        raise ValueError("No users have eligible validation positives.")

    results = pd.DataFrame(rows)
    summary_rows = []

    for k in ks:
        als_ndcg = float(results[f"als_ndcg@{k}"].mean())
        popularity_ndcg = float(
            results[f"popularity_ndcg@{k}"].mean()
        )

        als_map = float(results[f"als_map@{k}"].mean())
        popularity_map = float(
            results[f"popularity_map@{k}"].mean()
        )

        summary_rows.append({
            "k": k,
            "als_ndcg": als_ndcg,
            "popularity_ndcg": popularity_ndcg,
            "ndcg_difference": als_ndcg - popularity_ndcg,
            "als_map": als_map,
            "popularity_map": popularity_map,
            "map_difference": als_map - popularity_map,
        })

    summary = pd.DataFrame(summary_rows)

    output_dir = (
        REPO_ROOT
        / "reports"
        / f"core-validation-{artifact_dir.name}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "per_user.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    metadata = {
        "model_artifact": str(artifact_dir),
        "model_manifest_sha256": file_sha256(
            artifact_dir / "manifest.json"
        ),
        "interaction_mode": model_manifest["interaction_mode"],
        "parameters": model_manifest["parameters"],
        "evaluated_users": len(results),
        "catalog_movies": len(movie_ids),
        "relevance": "binary held-out positive interactions",
        "popularity": "positive training-interaction counts",
        "exclusions": (
            "observed core-user movie history, excluding "
            "the held-out validation positives"
        ),
        "k_values": ks,
    }

    with (output_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            allow_nan=False,
        )

    print(f"Model mode: {model_manifest['interaction_mode']}")
    print(f"Evaluated users: {len(results):,}")
    print(f"Catalog movies: {len(movie_ids):,}")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
    