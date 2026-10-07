"""Shared user cold-start split protocol (README: "User cold-start protocol").

Produces three disjoint pieces from the ratings table:

- core_train:        ratings from "core" users, used to train the
                      collaborative model (Track A) and to compute
                      popularity/consensus statistics for the niche subset.
- onboarding_pool:    a portion of each held-out evaluation user's ratings,
                      used to *simulate* pairwise answers during onboarding
                      (Track B draws comparisons from here, never from the
                      hidden test set).
- hidden_test:        the remaining ratings for evaluation users; rankings
                      are scored against this set and it must never be used
                      to pick questions or fit the user's posterior.

Ties and unavailable answers: two movies with equal simulated rating produce
no usable preference signal and are treated as a skip, since the dataset
cannot tell us which the user actually prefers. A comparison is only
simulated when both movies have a rating available in the user's onboarding
pool; otherwise the pair is ineligible for that user and must be skipped
rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.config import REPO_ROOT, load_config


@dataclass
class ColdStartSplit:
    core_train: pd.DataFrame
    onboarding_pool: pd.DataFrame
    hidden_test: pd.DataFrame
    eval_user_ids: np.ndarray
    core_user_ids: np.ndarray


def load_ratings(processed_dir: str | None = None) -> pd.DataFrame:
    cfg = load_config("data")["dataset"]
    raw_dir = REPO_ROOT / cfg["raw_dir"] / cfg["name"]
    path = raw_dir / "ratings.csv"
    return pd.read_csv(path)


def load_movies(processed_dir: str | None = None) -> pd.DataFrame:
    cfg = load_config("data")["dataset"]
    raw_dir = REPO_ROOT / cfg["raw_dir"] / cfg["name"]
    path = raw_dir / "movies.csv"
    return pd.read_csv(path)


def make_cold_start_split(ratings: pd.DataFrame, cfg: dict | None = None) -> ColdStartSplit:
    """Split ratings into core-training users and held-out evaluation users,
    then split each evaluation user's ratings into an onboarding pool and a
    hidden test set.
    """
    cfg = cfg or load_config("data")["split"]
    rng = np.random.default_rng(cfg["seed"])

    all_user_ids = ratings["userId"].unique()
    rng.shuffle(all_user_ids)

    n_eval = int(len(all_user_ids) * cfg["eval_user_fraction"])
    candidate_eval_ids = set(all_user_ids[:n_eval])

    # Drop eval users without enough ratings to support a meaningful
    # onboarding pool + hidden test set; they fall back into core training.
    counts = ratings.groupby("userId").size()
    eval_user_ids = np.array(
        [
            uid
            for uid in candidate_eval_ids
            if counts.get(uid, 0) >= cfg["min_ratings_per_eval_user"]
        ]
    )
    eval_user_set = set(eval_user_ids)
    core_user_ids = np.array([uid for uid in all_user_ids if uid not in eval_user_set])

    core_train = ratings[ratings["userId"].isin(core_user_ids)].reset_index(drop=True)
    eval_ratings = ratings[ratings["userId"].isin(eval_user_set)]

    pool_frac = cfg["onboarding_pool_fraction"]
    pool_rows, test_rows = [], []
    for uid, group in eval_ratings.groupby("userId"):
        group = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        n_pool = max(1, int(len(group) * pool_frac))
        pool_rows.append(group.iloc[:n_pool])
        test_rows.append(group.iloc[n_pool:])

    onboarding_pool = pd.concat(pool_rows).reset_index(drop=True) if pool_rows else eval_ratings.iloc[0:0]
    hidden_test = pd.concat(test_rows).reset_index(drop=True) if test_rows else eval_ratings.iloc[0:0]

    return ColdStartSplit(
        core_train=core_train,
        onboarding_pool=onboarding_pool,
        hidden_test=hidden_test,
        eval_user_ids=eval_user_ids,
        core_user_ids=core_user_ids,
    )


def save_split(split: ColdStartSplit, processed_dir: str | None = None) -> None:
    cfg = load_config("data")["dataset"]
    out_dir = REPO_ROOT / (processed_dir or cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    split.core_train.to_csv(out_dir / "core_train.csv", index=False)
    split.onboarding_pool.to_csv(out_dir / "onboarding_pool.csv", index=False)
    split.hidden_test.to_csv(out_dir / "hidden_test.csv", index=False)
    np.savetxt(out_dir / "eval_user_ids.txt", split.eval_user_ids, fmt="%d")
    np.savetxt(out_dir / "core_user_ids.txt", split.core_user_ids, fmt="%d")


if __name__ == "__main__":
    ratings = load_ratings()
    split = make_cold_start_split(ratings)
    save_split(split)
    print(
        f"core_train={len(split.core_train)} rows, "
        f"onboarding_pool={len(split.onboarding_pool)} rows, "
        f"hidden_test={len(split.hidden_test)} rows, "
        f"eval_users={len(split.eval_user_ids)}, core_users={len(split.core_user_ids)}"
    )
