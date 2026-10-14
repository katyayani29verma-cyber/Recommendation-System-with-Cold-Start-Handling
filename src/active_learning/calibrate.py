
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def _log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood; labels are 1 if A was preferred else 0."""
    z = logits / temperature
    return float(-np.mean(labels * _log_sigmoid(z) + (1 - labels) * _log_sigmoid(-z)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    res = minimize_scalar(
        lambda log_t: nll(logits, labels, float(np.exp(log_t))),
        bounds=(np.log(1e-3), np.log(1e3)),
        method="bounded",
    )
    return float(np.exp(res.x))


def build_pairs(
    ratings: pd.DataFrame,
    user_vectors: dict[int, np.ndarray],
    item_vectors: dict[int, np.ndarray],
    popular_ids: set[int],
    *,
    max_pairs_per_user: int,
    positive_threshold: float,
    pair_filter: str,
    seed: int,
) -> pd.DataFrame:
    """Returns columns: user_id, logit, label, rating_a, rating_b."""
    rng = np.random.default_rng(seed)
    rows = []
    sub = ratings[ratings["movieId"].isin(popular_ids) & ratings["movieId"].isin(item_vectors.keys())]
    for uid, g in sub.groupby("userId"):
        if uid not in user_vectors or len(g) < 2:
            continue
        mids, rs = g["movieId"].to_numpy(), g["rating"].to_numpy(float)
        n = len(mids)
        i = rng.integers(0, n, size=4 * max_pairs_per_user)
        j = rng.integers(0, n, size=4 * max_pairs_per_user)
        ok = rs[i] != rs[j]
        if pair_filter == "same_side":
            ok &= (rs[i] >= positive_threshold) == (rs[j] >= positive_threshold)
        elif pair_filter != "all":
            raise ValueError("pair_filter must be 'same_side' or 'all'")
        i, j = i[ok][:max_pairs_per_user], j[ok][:max_pairs_per_user]
        p = user_vectors[uid]
        for a, b in zip(i, j):
            x = item_vectors[int(mids[a])] - item_vectors[int(mids[b])]
            rows.append((int(uid), float(p @ x), float(rs[a] > rs[b]), rs[a], rs[b]))
    return pd.DataFrame(rows, columns=["user_id", "logit", "label", "rating_a", "rating_b"])


def calibrate(pairs: pd.DataFrame, seed: int = 0) -> dict:
    users = pairs["user_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    fit_users = set(users[: len(users) // 2])
    is_fit = pairs["user_id"].isin(fit_users)
    fit, held = pairs[is_fit], pairs[~is_fit]
    t_fit = fit_temperature(fit["logit"].to_numpy(), fit["label"].to_numpy())
    t_all = fit_temperature(pairs["logit"].to_numpy(), pairs["label"].to_numpy())
    h_logit, h_label = held["logit"].to_numpy(), held["label"].to_numpy()
    return {
        "temperature": t_all,
        "temperature_fit_half": t_fit,
        "n_pairs": int(len(pairs)),
        "n_users": int(len(users)),
        "heldout_nll_at_fit_T": nll(h_logit, h_label, t_fit),
        "heldout_nll_at_T1": nll(h_logit, h_label, 1.0),
        "heldout_nll_chance": float(np.log(2)),
        "heldout_accuracy": float(np.mean((h_logit > 0) == (h_label > 0.5))),
    }


def main() -> None:
    from src.core.als_core import load_core
    from src.data.config import REPO_ROOT, load_config
    from src.data.splits import load_ratings, make_cold_start_split
    from src.evaluation import baseline

    parser = argparse.ArgumentParser()
    parser.add_argument("--als-artifact", default=load_config("eval")["real_core"]["als_artifact"])
    args = parser.parse_args()

    data_cfg, al_cfg = load_config("data"), load_config("active_learning")["calibration"]
    split = make_cold_start_split(load_ratings(), data_cfg["split"])  # core_train only below
    core = load_core(args.als_artifact)

    directory = Path(args.als_artifact)
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    user_ids = np.load(directory / "user_ids.npy", allow_pickle=False)
    user_vecs = np.load(directory / "user_vectors.npy", allow_pickle=False).astype(np.float64)
    users = {int(u): v for u, v in zip(user_ids, user_vecs)}
    items = {int(m): core.get_item_vector(int(m)).astype(np.float64) for m in core.available_movie_ids}

    popular = set(baseline.select_popular_movies(split.core_train, data_cfg["niche_subset"]["min_ratings_for_popularity"]))
    pairs = build_pairs(
        split.core_train, users, items, popular,
        max_pairs_per_user=al_cfg["max_pairs_per_user"],
        positive_threshold=data_cfg["positive_rating_threshold"],
        pair_filter=al_cfg["pair_filter"],
        seed=al_cfg["seed"],
    )
    result = calibrate(pairs, seed=al_cfg["seed"])
    result.update({"als_artifact": str(args.als_artifact), "pair_filter": al_cfg["pair_filter"]})
    out = REPO_ROOT / "reports" / "temperature_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
