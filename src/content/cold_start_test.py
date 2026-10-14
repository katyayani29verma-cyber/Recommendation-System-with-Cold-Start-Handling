"""Evaluate frozen content mappings on held-out cold test movies."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from src.content.cold_start_validate import (
    load_experiment,
    ranking_results,
    read_checked,
    read_manifest,
    resolve_path,
    summarize,
)
from src.core.data_prep import file_sha256
from src.data.config import REPO_ROOT, load_config


def paired_ndcg_interval(first, second, seed=42):
    """Bootstrap the mean NDCG difference over paired users."""
    if np.iscomplexobj(first) or np.iscomplexobj(second):
        raise ValueError("Metric values must be real-valued.")

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    if (
        first.ndim != 1
        or first.shape != second.shape
        or len(first) == 0
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("Expected matching finite per-user metric arrays.")

    differences = first - second

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(2000, len(differences)),
    )

    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

    return {
        "mean_ndcg_difference": float(differences.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--als-artifact", required=True)
    parser.add_argument("--mapping-artifact", required=True)
    args = parser.parse_args()

    split_directory = resolve_path(args.split_dir)
    als_directory = resolve_path(args.als_artifact)
    mapping_directory = resolve_path(args.mapping_artifact)

    content_config = load_config("content")
    data_config = load_config("data")

    cache_directory = resolve_path(content_config["cache_dir"])
    movies_path = (
        REPO_ROOT
        / data_config["dataset"]["raw_dir"]
        / data_config["dataset"]["name"]
        / "movies.csv"
    )

    selection = read_manifest(
        mapping_directory,
        "cold_mapping_selection",
    )

    # Ensure that the frozen mappings belong to these exact inputs.
    expected_sources = {
        "split_manifest_sha256": split_directory / "manifest.json",
        "als_manifest_sha256": als_directory / "manifest.json",
        "embedding_manifest_sha256": cache_directory / "manifest.json",
        "metadata_sha256": movies_path,
    }

    for key, path in expected_sources.items():
        if selection[key] != file_sha256(path):
            raise ValueError(f"Selected mapping source mismatch: {key}")

    if (
        selection.get("protocol_version") != 1
        or selection.get("selection_metric") != "ndcg_at_10"
        or selection.get("evaluation_scope") != "cold_movies_only"
        or selection.get("relevance") != "binary_positive"
    ):
        raise ValueError("Unexpected mapping-selection protocol.")

    data = load_experiment(
        split_directory,
        als_directory,
        cache_directory,
        movies_path,
        group="test",
    )

    if data["genre_vocabulary"] != selection["genre_vocabulary"]:
        raise ValueError("Genre feature ordering differs from selection.")

    methods = ("transformer", "genre_only")

    if set(selection["selected_alphas"]) != set(methods):
        raise ValueError("Expected both selected mapping methods.")

    print(f"Cold test movies: {len(data['candidate_ids']):,}")
    print(f"Eligible test users: {len(data['user_ids']):,}")
    print("Evaluation: cold test movies only")
    print("Using frozen mappings selected on validation data.")

    summaries = []
    results_by_method = {}

    for method in methods:
        _, test_features = data["features"][method]

        weights = read_checked(
            mapping_directory,
            f"{method}_weights.npy",
            selection,
        )
        bias = read_checked(
            mapping_directory,
            f"{method}_bias.npy",
            selection,
        )

        dimensions = data["user_vectors"].shape[1]

        if (
            weights.shape != (test_features.shape[1], dimensions)
            or bias.shape != (dimensions,)
            or not np.isfinite(weights).all()
            or not np.isfinite(bias).all()
        ):
            raise ValueError(f"Invalid saved mapping: {method}")

        predicted_vectors = (
            test_features.astype(np.float64) @ weights + bias
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
            k=10,
            tie_seed=selection["tie_seed"],
        )

        results_by_method[method] = results

        summaries.append({
            "method": method,
            "alpha": selection["selected_alphas"][method],
            **summarize(results),
        })

    # Use the same predefined random-baseline procedure as validation.
    random_runs = []

    for seed in selection["random_seeds"]:
        scores = np.random.default_rng(seed).random(
            (len(data["user_ids"]), len(data["candidate_ids"]))
        )

        random_runs.append(ranking_results(
            scores,
            data["candidate_ids"],
            data["relevance"],
            data["user_ids"],
            k=10,
            tie_seed=selection["tie_seed"],
        ))

    averaged_random = (
        pd.concat(random_runs, ignore_index=True)
        .groupby("user_id", as_index=False)[["ndcg", "ap", "recall"]]
        .mean()
    )

    results_by_method["random_20_runs"] = averaged_random

    summaries.append({
        "method": "random_20_runs",
        "alpha": None,
        **summarize(averaged_random),
    })

    # Align by user ID before calculating paired differences.
    transformer = results_by_method["transformer"].set_index("user_id")
    comparisons = []

    for baseline in ("genre_only", "random_20_runs"):
        baseline_results = results_by_method[baseline].set_index("user_id")

        if set(transformer.index) != set(baseline_results.index):
            raise ValueError("Methods were evaluated on different users.")

        baseline_results = baseline_results.reindex(transformer.index)

        comparisons.append({
            "comparison": f"transformer_minus_{baseline}",
            **paired_ndcg_interval(
                transformer["ndcg"].to_numpy(),
                baseline_results["ndcg"].to_numpy(),
            ),
        })

    identity = {
        "mapping_manifest_sha256": file_sha256(
            mapping_directory / "manifest.json"
        ),
        "split_manifest_sha256": selection["split_manifest_sha256"],
        "als_manifest_sha256": selection["als_manifest_sha256"],
        "evaluation_group": "test",
        "k": 10,
        "bootstrap_seed": 42,
        "bootstrap_repetitions": 2000,
        "protocol_version": 1,
    }

    report_hash = sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    report_directory = (
        REPO_ROOT / "reports" / f"cold-test-{report_hash}"
    )
    report_directory.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summaries)
    comparison_table = pd.DataFrame(comparisons)

    summary.to_csv(report_directory / "summary.csv", index=False)
    comparison_table.to_csv(
        report_directory / "ndcg_comparisons.csv",
        index=False,
    )

    per_user = pd.concat(
        [
            results.assign(method=method)
            for method, results in results_by_method.items()
        ],
        ignore_index=True,
    )
    per_user.to_csv(report_directory / "per_user.csv", index=False)

    metadata = {
        "schema_version": 1,
        "stage": "item_cold_start_test_report",
        **identity,
        "selected_alphas": selection["selected_alphas"],
        "candidate_movies": len(data["candidate_ids"]),
        "evaluated_users": len(data["user_ids"]),
        "evaluation_scope": "cold_movies_only",
        "relevance": "binary_positive",
        "confidence_interval_scope": (
            "Paired bootstrap over eligible users, conditional on "
            "this movie split and these fitted models."
        ),
        "random_baseline_runs": len(selection["random_seeds"]),
        "file_sha256": {
            filename: file_sha256(report_directory / filename)
            for filename in (
                "summary.csv",
                "ndcg_comparisons.csv",
                "per_user.csv",
            )
        },
    }

    with (report_directory / "metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)

    print("\nHeld-out cold-test results:")
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))

    print("\nPaired NDCG@10 differences with 95% bootstrap intervals:")
    print(comparison_table.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))

    print(f"\nReports saved to: {report_directory}")


if __name__ == "__main__":
    main()