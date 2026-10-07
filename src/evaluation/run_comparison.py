"""End-to-end fixed-vs-adaptive onboarding comparison (README: "Fair
baseline comparison").

Runs today against mock Track A vectors and a mock Track B point-estimate
user model (see src/core/mock_vectors.py and src/active_learning/mock_model.py).
Swap `RecommenderCore` and `UserModel` for the real implementations once
those tracks land artifacts -- nothing else in this script should need to
change, since both are called only through their shared interfaces.

The comparison is repeated over `seeds` (configs/eval.yaml). A seed controls
the fixed sequence's starting point and the user model's randomness; the
user split itself is shared and stays fixed. Results are reported for all
evaluation users and for the niche subset (README: "Niche-taste subset").

Usage:
    python -m src.evaluation.run_comparison
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.active_learning.mock_model import MockPointEstimateUserModel
from src.core.interfaces import RecommenderCore
from src.core.mock_vectors import MockRecommenderCore
from src.data.config import REPO_ROOT, load_config
from src.data.niche import compute_niche_user_ids, compute_popular_movie_consensus
from src.data.splits import ColdStartSplit, load_ratings, make_cold_start_split
from src.evaluation import baseline, harness
from src.evaluation.metrics import average_precision_at_k, ndcg_at_k


def run_mode(
    mode: str,
    seed: int,
    core: RecommenderCore,
    split: ColdStartSplit,
    candidate_ids: list[int],
    fixed_pairs: list[tuple[int, int]],
    niche_ids: set[int],
    eval_cfg: dict,
) -> list[dict]:
    n_dims = eval_cfg["mock_vectors"]["n_dims"]
    ks = eval_cfg["ranking_k"]
    top_k = max(ks)
    budget = eval_cfg["question_budget"]
    # Offset keeps the two modes' random streams independent for a given seed.
    model = MockPointEstimateUserModel(core, n_dims=n_dims, seed=2 * seed + (mode == "adaptive"))

    rows: list[dict] = []
    for user_id in split.eval_user_ids:
        user_id = int(user_id)
        onboarding_ratings = harness.ratings_dict(split.onboarding_pool, user_id)
        graded_relevance, relevant_ids = harness.relevance_for_user(
            split.hidden_test, user_id, eval_cfg["relevance_rating_threshold"]
        )
        if not graded_relevance or not onboarding_ratings:
            continue

        seen = set(onboarding_ratings)
        ref_vec = harness.reference_vector(onboarding_ratings, core, n_dims)
        ref_ranked = [r.movie_id for r in core.recommend(ref_vec, k=top_k, exclude_ids=seen)]
        ref_ndcg = ndcg_at_k(ref_ranked, graded_relevance, top_k)
        target = eval_cfg["target_fraction_of_reference"] * ref_ndcg

        if mode == "fixed":
            traj = harness.run_fixed_session(model, fixed_pairs, onboarding_ratings)
        else:
            traj = harness.run_adaptive_session(model, candidate_ids, onboarding_ratings, budget)

        user_rows = []
        reached_at = None
        n_skips = 0
        for step, state in enumerate(traj.states):
            if step > 0 and traj.answers[step - 1].choice == "skip":
                n_skips += 1
            ranked = [r.movie_id for r in core.recommend(state.mean_vector, k=top_k, exclude_ids=seen)]
            if reached_at is None and ref_ndcg > 0 and ndcg_at_k(ranked, graded_relevance, top_k) >= target:
                reached_at = step

            row = {
                "seed": seed,
                "mode": mode,
                "user_id": user_id,
                "niche": user_id in niche_ids,
                "step": step,
                "n_skips": n_skips,
            }
            for k in ks:
                row[f"ndcg@{k}"] = ndcg_at_k(ranked[:k], graded_relevance, k)
                row[f"map@{k}"] = average_precision_at_k(ranked[:k], relevant_ids, k)
            user_rows.append(row)

        for row in user_rows:
            row["reached_target"] = reached_at is not None
            row["questions_to_target"] = reached_at
        rows.extend(user_rows)
    return rows


def summarize(results: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    """Per-(group, mode, step): mean over users within each seed, then mean
    and std of that across seeds. Std across seeds is the seed-to-seed
    uncertainty; it does not include user sampling variance.
    """
    metrics = [f"ndcg@{k}" for k in ks] + [f"map@{k}" for k in ks] + ["n_skips"]
    groups = {"all": results, "niche": results[results["niche"]]}
    frames = []
    for name, df in groups.items():
        if df.empty:
            continue
        per_seed = df.groupby(["mode", "step", "seed"])[metrics].mean().reset_index()
        agg = per_seed.groupby(["mode", "step"])[metrics].agg(["mean", "std"])
        agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
        agg = agg.reset_index()
        agg.insert(0, "subset", name)
        agg.insert(3, "n_users", df.groupby(["mode", "step"])["user_id"].nunique().values)
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def target_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Target-reached rate and median questions-to-target (reached users only)."""
    final = results[results["step"] == results["step"].max()]
    out = []
    for name, df in {"all": final, "niche": final[final["niche"]]}.items():
        if df.empty:
            continue
        per_seed = df.groupby(["mode", "seed"]).agg(
            reached_rate=("reached_target", "mean"),
            median_questions=("questions_to_target", "median"),
        )
        agg = per_seed.groupby("mode").agg(["mean", "std"])
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        agg = agg.reset_index()
        agg.insert(0, "subset", name)
        out.append(agg)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    data_cfg = load_config("data")
    eval_cfg = load_config("eval")

    ratings = load_ratings()
    split = make_cold_start_split(ratings, data_cfg["split"])

    consensus = compute_popular_movie_consensus(split.core_train, data_cfg["niche_subset"])
    niche_ids = {int(u) for u in compute_niche_user_ids(split.onboarding_pool, consensus, data_cfg["niche_subset"])}
    print(f"Evaluation users: {len(split.eval_user_ids)}, niche subset: {len(niche_ids)}")

    all_movie_ids = sorted(ratings["movieId"].unique().tolist())
    core = MockRecommenderCore(
        all_movie_ids, n_dims=eval_cfg["mock_vectors"]["n_dims"], seed=eval_cfg["mock_vectors"]["seed"]
    )
    candidate_ids = baseline.select_popular_movies(
        split.core_train, data_cfg["niche_subset"]["min_ratings_for_popularity"]
    )

    rows: list[dict] = []
    for seed in eval_cfg["seeds"]:
        fixed_pairs = baseline.build_fixed_sequence(core, candidate_ids, eval_cfg["question_budget"], seed=seed)
        for mode in ("fixed", "adaptive"):
            rows.extend(run_mode(mode, seed, core, split, candidate_ids, fixed_pairs, niche_ids, eval_cfg))

    results = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "onboarding_comparison.csv", index=False)

    ks = eval_cfg["ranking_k"]
    summary = summarize(results, ks)
    summary.to_csv(out_dir / "onboarding_summary.csv", index=False)
    targets = target_summary(results)
    targets.to_csv(out_dir / "onboarding_targets.csv", index=False)

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        cols = ["subset", "mode", "step", "n_users", f"ndcg@{max(ks)}_mean", f"ndcg@{max(ks)}_std", "n_skips_mean"]
        print(summary[cols].to_string(index=False))
        print("\nTarget reached (mean/std across seeds; median questions among reached users):")
        print(targets.to_string(index=False))


if __name__ == "__main__":
    main()
