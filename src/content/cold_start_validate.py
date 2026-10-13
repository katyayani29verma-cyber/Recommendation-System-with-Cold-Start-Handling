"""Select content mappings using cold validation recommendation quality."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz

from src.content.metadata import FORMAT_VERSION
from src.content.train_mapping import fit_mapping
from src.core.data_prep import file_sha256
from src.core.hybrid_core import validate_ids
from src.data.config import REPO_ROOT, load_config
from src.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_manifest(directory, expected_stage=None):
    with (directory / "manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema.")

    if (
        expected_stage is not None
        and manifest.get("stage") != expected_stage
    ):
        raise ValueError(f"Expected artifact stage: {expected_stage}")

    return manifest


def read_checked(directory, filename, manifest):
    path = (directory / filename).resolve()

    if not path.is_relative_to(directory.resolve()):
        raise ValueError("File path escapes its directory.")

    if file_sha256(path) != manifest["file_sha256"][filename]:
        raise ValueError(f"Checksum mismatch: {filename}")

    if filename.endswith(".npz"):
        return load_npz(path).tocsr()

    return np.load(path, allow_pickle=False)


def rows_for_ids(catalog_ids, requested_ids):
    catalog_ids = validate_ids(catalog_ids)
    requested_ids = validate_ids(requested_ids)

    lookup = {
        int(movie_id): row
        for row, movie_id in enumerate(catalog_ids)
    }

    if any(int(movie_id) not in lookup for movie_id in requested_ids):
        raise ValueError("A requested movie is missing from the catalog.")

    return np.array(
        [lookup[int(movie_id)] for movie_id in requested_ids],
        dtype=np.int64,
    )


def genre_features(genre_strings, warm_rows):
    """Learn the genre vocabulary using warm movies only."""
    genre_sets = [
        {
            genre.strip()
            for genre in str(value).split("|")
            if genre.strip()
            and genre.strip() != "(no genres listed)"
        }
        for value in genre_strings
    ]

    vocabulary = sorted({
        genre
        for row in warm_rows
        for genre in genre_sets[int(row)]
    })

    if not vocabulary:
        raise ValueError("Warm movies have no usable genres.")

    features = np.array([
        [float(genre in movie_genres) for genre in vocabulary]
        for movie_genres in genre_sets
    ], dtype=np.float64)

    return features, vocabulary


def ranking_results(
    scores,
    candidate_ids,
    relevance,
    user_ids,
    k=10,
    tie_seed=42,
):
    """Return per-user binary ranking metrics for one candidate group."""
    candidate_ids = validate_ids(candidate_ids)
    user_ids = validate_ids(user_ids)

    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or k < 1
    ):
        raise ValueError("k must be a positive integer.")

    if np.iscomplexobj(scores):
        raise ValueError("Scores must be real-valued.")

    scores = np.asarray(scores, dtype=np.float64)

    if (
        scores.shape != (len(user_ids), len(candidate_ids))
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Invalid score matrix.")

    relevance = csr_matrix(relevance, copy=True)
    relevance.sum_duplicates()
    relevance.eliminate_zeros()
    relevance.sort_indices()

    if (
        relevance.shape != scores.shape
        or not np.all(relevance.data == 1)
        or np.any(relevance.getnnz(axis=1) == 0)
    ):
        raise ValueError("Every evaluated user needs binary positive relevance.")

    tie_rng = np.random.default_rng(tie_seed)
    records = []

    for row, user_id in enumerate(user_ids):
        # Seeded tie-breaking avoids systematically favoring low movie IDs.
        tie_order = tie_rng.permutation(len(candidate_ids))
        order = np.lexsort((tie_order, -scores[row]))
        ranked = candidate_ids[order[:k]].tolist()

        relevant_columns = relevance.indices[
            relevance.indptr[row]:relevance.indptr[row + 1]
        ]
        relevant_ids = set(candidate_ids[relevant_columns].tolist())

        records.append({
            "user_id": int(user_id),
            "ndcg": ndcg_at_k(
                ranked,
                {movie_id: 1.0 for movie_id in relevant_ids},
                k,
            ),
            "ap": average_precision_at_k(ranked, relevant_ids, k),
            "recall": len(set(ranked) & relevant_ids) / len(relevant_ids),
        })

    return pd.DataFrame(records)


def summarize(results):
    return {
        "ndcg_at_10": float(results["ndcg"].mean()),
        "map_at_10": float(results["ap"].mean()),
        "recall_at_10": float(results["recall"].mean()),
    }


def load_experiment(
    split_directory,
    als_directory,
    cache_directory,
    movies_path,
    group="validation",
):
    """Load aligned features, factors, and one cold evaluation group."""
    if group not in ("validation", "test"):
        raise ValueError("Expected validation or test.")

    split_manifest = read_manifest(
        split_directory, "item_cold_start_split"
    )
    als_manifest = read_manifest(
        als_directory, "item_cold_start_training"
    )
    cache_manifest = read_manifest(cache_directory)

    if (
        als_manifest["split_manifest_sha256"]
        != file_sha256(split_directory / "manifest.json")
    ):
        raise ValueError("The experimental ALS model belongs to another split.")

    if (
        als_manifest.get("interaction_mode") != "positive"
        or als_manifest.get("vector_normalization")
        != "raw_learned_factors"
    ):
        raise ValueError("Expected positive-only raw ALS factors.")

    signature = cache_manifest["signature"]

    if (
        signature["metadata_source_sha256"] != file_sha256(movies_path)
        or signature["text_format_version"] != FORMAT_VERSION
        or signature["normalize_text_embeddings"] is not True
    ):
        raise ValueError("Cached embeddings do not match the movie metadata.")

    def split_array(name):
        return read_checked(split_directory, name, split_manifest)

    def als_array(name):
        return read_checked(als_directory, name, als_manifest)

    user_ids = validate_ids(split_array("user_ids.npy"))
    catalog_ids = validate_ids(split_array("movie_ids.npy"))
    warm_ids = validate_ids(split_array("warm_movie_ids.npy"))
    cold_ids = validate_ids(split_array(f"{group}_movie_ids.npy"))

    if set(warm_ids) & set(cold_ids):
        raise ValueError("Warm and cold movies overlap.")

    if (
        not np.array_equal(user_ids, als_array("user_ids.npy"))
        or not np.array_equal(warm_ids, als_array("movie_ids.npy"))
    ):
        raise ValueError("Experimental ALS ID ordering differs from the split.")

    user_vectors = als_array("user_vectors.npy")
    warm_vectors = als_array("item_vectors.npy")
    dimensions = als_manifest["parameters"]["factors"]

    if (
        user_vectors.shape != (len(user_ids), dimensions)
        or warm_vectors.shape != (len(warm_ids), dimensions)
        or not np.isfinite(user_vectors).all()
        or not np.isfinite(warm_vectors).all()
    ):
        raise ValueError("Invalid experimental ALS factors.")

    warm_matrix = split_array("warm_interactions.npz")
    cold_matrix = split_array(f"{group}_interactions.npz")
    expected_shape = (len(user_ids), len(catalog_ids))

    for matrix in (warm_matrix, cold_matrix):
        matrix.sum_duplicates()
        matrix.eliminate_zeros()

        if matrix.shape != expected_shape or not np.all(matrix.data == 1):
            raise ValueError("Invalid interaction matrix.")

    warm_columns = rows_for_ids(catalog_ids, warm_ids)
    cold_columns = rows_for_ids(catalog_ids, cold_ids)

    if (
        warm_matrix[:, warm_columns].nnz != warm_matrix.nnz
        or cold_matrix[:, cold_columns].nnz != cold_matrix.nnz
        or warm_matrix[:, cold_columns].nnz != 0
    ):
        raise ValueError("Interactions escaped their movie group.")

    warm_support = np.asarray(warm_matrix.getnnz(axis=1)).ravel() > 0
    eligible = (
        warm_support
        & (np.asarray(cold_matrix.getnnz(axis=1)).ravel() > 0)
    )

    if (
        not eligible.any()
        or not np.array_equal(
            eligible, split_array(f"{group}_user_mask.npy")
        )
        or not np.array_equal(
            warm_support, als_array("user_has_warm_positive.npy")
        )
        or int(eligible.sum())
        != split_manifest[f"eligible_{group}_users"]
    ):
        raise ValueError("User eligibility does not match saved artifacts.")

    embedding_ids = validate_ids(
        read_checked(cache_directory, "movie_ids.npy", cache_manifest)
    )
    embeddings = read_checked(
        cache_directory, "embeddings.npy", cache_manifest
    )

    if (
        embeddings.shape != (
            len(embedding_ids),
            cache_manifest["embedding_dimension"],
        )
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Invalid cached embeddings.")

    movies = pd.read_csv(movies_path).sort_values("movieId")

    if not np.array_equal(
        movies["movieId"].to_numpy(), embedding_ids
    ):
        raise ValueError("Metadata ordering differs from cached embeddings.")

    warm_rows = rows_for_ids(embedding_ids, warm_ids)
    cold_rows = rows_for_ids(embedding_ids, cold_ids)

    genres, vocabulary = genre_features(
        movies["genres"].fillna("").tolist(),
        warm_rows,
    )

    return {
        "features": {
            "transformer": (
                embeddings[warm_rows],
                embeddings[cold_rows],
            ),
            "genre_only": (
                genres[warm_rows],
                genres[cold_rows],
            ),
        },
        "genre_vocabulary": vocabulary,
        "warm_vectors": warm_vectors,
        "user_vectors": user_vectors[eligible],
        "user_ids": user_ids[eligible],
        "candidate_ids": cold_ids,
        "relevance": cold_matrix[eligible][:, cold_columns].tocsr(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--als-artifact", required=True)
    args = parser.parse_args()

    split_directory = resolve_path(args.split_dir)
    als_directory = resolve_path(args.als_artifact)

    content_config = load_config("content")
    data_config = load_config("data")

    cache_directory = resolve_path(content_config["cache_dir"])
    movies_path = (
        REPO_ROOT
        / data_config["dataset"]["raw_dir"]
        / data_config["dataset"]["name"]
        / "movies.csv"
    )

    data = load_experiment(
        split_directory,
        als_directory,
        cache_directory,
        movies_path,
        group="validation",
    )

    alphas = sorted(set(float(x) for x in content_config["ridge_alphas"]))

    if not alphas or any(not np.isfinite(x) or x <= 0 for x in alphas):
        raise ValueError("Ridge alphas must be positive and finite.")

    print(f"Warm mapping examples: {len(data['warm_vectors']):,}")
    print(f"Cold validation movies: {len(data['candidate_ids']):,}")
    print(f"Eligible validation users: {len(data['user_ids']):,}")
    print("Selection metric: NDCG@10")

    trials = []
    summaries = []
    per_user = []
    selected_parameters = {}
    selected_alphas = {}

    for method, (warm_features, cold_features) in data["features"].items():
        best = None

        for alpha in alphas:
            mapping = fit_mapping(
                warm_features,
                data["warm_vectors"],
                alpha,
            )

            weights = mapping.coef_.T.astype(np.float64)
            bias = np.asarray(mapping.intercept_, dtype=np.float64)

            # Use the exact coefficients that will be saved.
            predicted_vectors = (
                cold_features.astype(np.float64) @ weights + bias
            )
            scores = (
                data["user_vectors"].astype(np.float64)
                @ predicted_vectors.T
            )

            results = ranking_results(
                scores,
                data["candidate_ids"],
                data["relevance"],
                data["user_ids"],
            )
            metrics = summarize(results)

            trials.append({
                "method": method,
                "alpha": alpha,
                **metrics,
            })

            print(
                f"{method:12s} alpha={alpha:6g} "
                f"NDCG@10={metrics['ndcg_at_10']:.6f}"
            )

            # Prefer smaller alpha when validation NDCG is exactly tied.
            key = (metrics["ndcg_at_10"], -alpha)

            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "alpha": alpha,
                    "metrics": metrics,
                    "results": results,
                    "weights": weights,
                    "bias": bias,
                }

        selected_alphas[method] = best["alpha"]
        selected_parameters[f"{method}_weights.npy"] = best["weights"]
        selected_parameters[f"{method}_bias.npy"] = best["bias"]

        summaries.append({
            "method": method,
            "alpha": best["alpha"],
            **best["metrics"],
        })
        per_user.append(best["results"].assign(method=method))

    # Average random rankings across repeated, reproducible runs.
    random_seeds = list(range(1000, 1020))
    random_results = []

    for seed in random_seeds:
        scores = np.random.default_rng(seed).random(
            (len(data["user_ids"]), len(data["candidate_ids"]))
        )

        random_results.append(ranking_results(
            scores,
            data["candidate_ids"],
            data["relevance"],
            data["user_ids"],
        ))

    averaged_random = (
        pd.concat(random_results, ignore_index=True)
        .groupby("user_id", as_index=False)[["ndcg", "ap", "recall"]]
        .mean()
    )

    summaries.append({
        "method": "random_20_runs",
        "alpha": None,
        **summarize(averaged_random),
    })
    per_user.append(averaged_random.assign(method="random_20_runs"))

    identity = {
        "split_manifest_sha256": file_sha256(
            split_directory / "manifest.json"
        ),
        "als_manifest_sha256": file_sha256(
            als_directory / "manifest.json"
        ),
        "embedding_manifest_sha256": file_sha256(
            cache_directory / "manifest.json"
        ),
        "metadata_sha256": file_sha256(movies_path),
        "ridge_alphas": alphas,
        "selection_metric": "ndcg_at_10",
        "tie_seed": 42,
        "random_seeds": random_seeds,
        "protocol_version": 1,
    }

    run_hash = sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    artifact_directory = (
        REPO_ROOT / "artifacts" / f"cold-mappings-{run_hash}"
    )
    report_directory = (
        REPO_ROOT / "reports" / f"cold-validation-{run_hash}"
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)

    for filename, values in selected_parameters.items():
        np.save(
            artifact_directory / filename,
            values,
            allow_pickle=False,
        )

    manifest = {
        "schema_version": 1,
        "stage": "cold_mapping_selection",
        **identity,
        "artifact_version": artifact_directory.name,
        "selected_alphas": selected_alphas,
        "genre_vocabulary": data["genre_vocabulary"],
        "warm_training_movies": len(data["warm_vectors"]),
        "validation_candidates": len(data["candidate_ids"]),
        "validation_users": len(data["user_ids"]),
        "evaluation_scope": "cold_movies_only",
        "relevance": "binary_positive",
        "validation_trials": trials,
        "validation_summary": summaries,
        "file_sha256": {
            filename: file_sha256(artifact_directory / filename)
            for filename in selected_parameters
        },
    }

    with (artifact_directory / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    summary = pd.DataFrame(summaries)

    summary.to_csv(report_directory / "summary.csv", index=False)
    pd.DataFrame(trials).to_csv(
        report_directory / "trials.csv", index=False
    )
    pd.concat(per_user, ignore_index=True).to_csv(
        report_directory / "per_user.csv", index=False
    )

    print("\nSelected mappings — cold validation results:")
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))
    print(f"\nSelected mappings saved to: {artifact_directory}")
    print(f"Validation reports saved to: {report_directory}")


if __name__ == "__main__":
    main()