
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.active_learning.interfaces import ItemPair, PairwiseAnswer


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def simulate(
    core,
    make_model,
    user_vectors: np.ndarray,
    candidate_ids: list[int],
    fixed_pairs: list[tuple[int, int]],
    *,
    temperature: float,
    budget: int = 10,
    seeds=(0,),
    top_k: int = 20,
) -> pd.DataFrame:
    """Returns one row per (seed, mode, user, step)."""
    all_ids = np.asarray(sorted(core.available_movie_ids)) if hasattr(core, "available_movie_ids") else None
    rows = []
    for seed in seeds:
        for mode in ("adaptive", "random", "fixed"):
            model = make_model(seed)
            rng = np.random.default_rng(10_000 + seed)  # same answer noise stream per mode
            for uid, p in enumerate(user_vectors):
                truth_scores = None
                state, remaining = model.initialize_user(), list(candidate_ids)
                asked: set[int] = set()
                for step in range(budget + 1):
                    if step > 0:
                        if mode == "adaptive":
                            pair = model.select_next_pair(state, remaining)
                        elif mode == "random":
                            a, b = rng.choice(remaining, size=2, replace=False)
                            pair = ItemPair(int(a), int(b))
                        else:
                            pair = ItemPair(*fixed_pairs[step - 1])
                        remaining = [m for m in remaining if m not in (pair.movie_a, pair.movie_b)]
                        asked |= {pair.movie_a, pair.movie_b}
                        x = core.get_item_vector(pair.movie_a) - core.get_item_vector(pair.movie_b)
                        choice = "A" if rng.random() < _sigmoid(float(p @ x) / temperature) else "B"
                        state = model.update_user(state, PairwiseAnswer("sim", pair.movie_a, pair.movie_b, choice))
                    mean = np.asarray(state.mean_vector, dtype=np.float64)
                    est = {r.movie_id for r in core.recommend(mean, k=top_k, exclude_ids=asked)}
                    tru = {r.movie_id for r in core.recommend(p, k=top_k, exclude_ids=asked)}
                    denom = np.linalg.norm(mean) * np.linalg.norm(p)
                    rows.append(
                        {
                            "seed": seed, "mode": mode, "user": uid, "step": step,
                            "cosine": float(mean @ p / denom) if denom > 0 else 0.0,
                            f"top{top_k}_overlap": len(est & tru) / top_k,
                        }
                    )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in ("seed", "mode", "user", "step")]
    per_seed = df.groupby(["mode", "step", "seed"])[metrics].mean().reset_index()
    out = per_seed.groupby(["mode", "step"])[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def main() -> None:
    from pathlib import Path

    from src.active_learning.factory import build_core, build_user_model, resolve_temperature
    from src.data.config import REPO_ROOT, load_config
    from src.data.splits import load_ratings, make_cold_start_split
    from src.evaluation import baseline

    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    eval_cfg, data_cfg, al_cfg = load_config("eval"), load_config("data"), load_config("active_learning")
    eval_cfg["mock_vectors"]["enabled"] = False
    ratings = load_ratings()
    core = build_core(eval_cfg, sorted(ratings["movieId"].unique().tolist()))

    directory = Path(eval_cfg["real_core"]["als_artifact"])
    directory = directory if directory.is_absolute() else REPO_ROOT / directory
    vecs = np.load(directory / "user_vectors.npy", allow_pickle=False).astype(np.float64)
    has_pos = np.load(directory / "user_has_positive.npy", allow_pickle=False)
    vecs = vecs[has_pos]
    vecs = vecs[np.random.default_rng(0).permutation(len(vecs))[: args.users]]

    split = make_cold_start_split(ratings, data_cfg["split"])
    cands = baseline.select_popular_movies(split.core_train, data_cfg["niche_subset"]["min_ratings_for_popularity"])
    fixed = baseline.build_fixed_sequence(core, cands, eval_cfg["question_budget"], seed=0)

    T = resolve_temperature(al_cfg)
    df = simulate(
        core, lambda s: build_user_model(core, eval_cfg, seed=s, al_cfg=al_cfg), vecs, cands, fixed,
        temperature=T, budget=eval_cfg["question_budget"], seeds=args.seeds,
    )
    out = REPO_ROOT / "reports" / "track_b_simulation.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    s = summarize(df)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(s[s["step"].isin([0, 1, 3, 5, 10])].to_string(index=False))


if __name__ == "__main__":
    main()
