import numpy as np
import pandas as pd

from src.data.splits import make_cold_start_split


def make_synthetic_ratings(n_users: int = 50, n_movies: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, n_users + 1):
        n_ratings = rng.integers(15, 25)
        movie_ids = rng.choice(np.arange(1, n_movies + 1), size=n_ratings, replace=False)
        for mid in movie_ids:
            rows.append(
                {
                    "userId": uid,
                    "movieId": int(mid),
                    "rating": float(rng.choice([1, 2, 3, 4, 5])),
                    "timestamp": 0,
                }
            )
    return pd.DataFrame(rows)


SPLIT_CFG = {
    "seed": 42,
    "eval_user_fraction": 0.2,
    "onboarding_pool_fraction": 0.5,
    "min_ratings_per_eval_user": 10,
}


def test_core_and_eval_users_are_disjoint():
    ratings = make_synthetic_ratings()
    split = make_cold_start_split(ratings, SPLIT_CFG)
    assert set(split.core_user_ids).isdisjoint(set(split.eval_user_ids))


def test_core_train_contains_no_eval_users():
    ratings = make_synthetic_ratings()
    split = make_cold_start_split(ratings, SPLIT_CFG)
    assert split.core_train["userId"].isin(split.eval_user_ids).sum() == 0


def test_onboarding_pool_and_hidden_test_disjoint_and_cover_eval_ratings():
    ratings = make_synthetic_ratings()
    split = make_cold_start_split(ratings, SPLIT_CFG)

    pool_keys = set(zip(split.onboarding_pool["userId"], split.onboarding_pool["movieId"]))
    test_keys = set(zip(split.hidden_test["userId"], split.hidden_test["movieId"]))
    assert pool_keys.isdisjoint(test_keys)

    eval_ratings = ratings[ratings["userId"].isin(split.eval_user_ids)]
    assert len(pool_keys) + len(test_keys) == len(eval_ratings)


def test_every_eval_user_has_nonempty_pool():
    ratings = make_synthetic_ratings()
    split = make_cold_start_split(ratings, SPLIT_CFG)
    pool_users = set(split.onboarding_pool["userId"].unique())
    assert set(split.eval_user_ids) <= pool_users


def test_deterministic_given_same_seed():
    ratings = make_synthetic_ratings()
    split_a = make_cold_start_split(ratings, SPLIT_CFG)
    split_b = make_cold_start_split(ratings, SPLIT_CFG)
    assert sorted(split_a.eval_user_ids.tolist()) == sorted(split_b.eval_user_ids.tolist())
