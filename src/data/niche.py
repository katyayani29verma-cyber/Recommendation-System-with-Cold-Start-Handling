"""Niche-taste subset membership (README: "Niche-taste subset").

A user is "niche" if their ratings of popular movies disagree with the
consensus (mean rating) on those movies, computed from core-training data
only. Membership is decided from a held-out portion of each evaluation
user's own history (we reuse the onboarding pool, never the hidden test
set) so subset assignment cannot leak into the recommendation metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.config import load_config


def compute_popular_movie_consensus(core_train: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """Mean rating per movie among movies popular enough in core-training data."""
    cfg = cfg or load_config("data")["niche_subset"]
    counts = core_train.groupby("movieId").size()
    popular_ids = counts[counts >= cfg["min_ratings_for_popularity"]].index
    means = core_train[core_train["movieId"].isin(popular_ids)].groupby("movieId")["rating"].mean()
    return means


def compute_niche_user_ids(
    onboarding_pool: pd.DataFrame,
    popular_consensus: pd.Series,
    cfg: dict | None = None,
) -> np.ndarray:
    """Evaluation users whose correlation with consensus on popular movies
    they rated falls below the configured threshold.
    """
    cfg = cfg or load_config("data")["niche_subset"]
    min_n = cfg["min_popular_ratings_for_membership"]
    threshold = cfg["consensus_correlation_threshold"]

    niche_ids = []
    pool = onboarding_pool[onboarding_pool["movieId"].isin(popular_consensus.index)]
    for uid, group in pool.groupby("userId"):
        if len(group) < min_n:
            continue
        user_ratings = group.set_index("movieId")["rating"]
        consensus = popular_consensus.loc[user_ratings.index]
        if user_ratings.nunique() < 2 or consensus.nunique() < 2:
            corr = 0.0
        else:
            corr = np.corrcoef(user_ratings.values, consensus.values)[0, 1]
        if corr < threshold:
            niche_ids.append(uid)
    return np.array(niche_ids)
