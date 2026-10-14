from __future__ import annotations

import json
from pathlib import Path

from src.active_learning.bayesian_model import BayesianPairwiseUserModel
from src.active_learning.mock_model import MockPointEstimateUserModel
from src.core.interfaces import RecommenderCore
from src.data.config import REPO_ROOT, load_config

CALIBRATION_FILE = REPO_ROOT / "reports" / "temperature_calibration.json"


def build_core(eval_cfg: dict, all_movie_ids: list[int]) -> RecommenderCore:
    if eval_cfg["mock_vectors"]["enabled"]:
        from src.core.mock_vectors import MockRecommenderCore

        return MockRecommenderCore(
            all_movie_ids, n_dims=eval_cfg["mock_vectors"]["n_dims"], seed=eval_cfg["mock_vectors"]["seed"]
        )
    from src.core.hybrid_core import load_hybrid_core

    real = eval_cfg["real_core"]
    return load_hybrid_core(real["als_artifact"], real["content_artifact"])


def resolve_temperature(al_cfg: dict, calibration_file: Path = CALIBRATION_FILE) -> float:
    t = al_cfg["user_model"].get("temperature")
    if t is not None:
        return float(t)
    if not calibration_file.exists():
        raise FileNotFoundError(
            "No temperature set. Run `python -m src.active_learning.calibrate` or set "
            "user_model.temperature in configs/active_learning.yaml."
        )
    return float(json.loads(calibration_file.read_text())["temperature"])


def build_user_model(core: RecommenderCore, eval_cfg: dict, seed: int = 0, al_cfg: dict | None = None):
    al_cfg = al_cfg or load_config("active_learning")
    cfg = al_cfg["user_model"]
    if eval_cfg["mock_vectors"]["enabled"] or cfg["kind"] == "mock":
        return MockPointEstimateUserModel(core, n_dims=core.n_dims, seed=seed)
    return BayesianPairwiseUserModel(
        core,
        temperature=resolve_temperature(al_cfg),
        prior=cfg["prior"],
        prior_std=cfg["prior_std"],
        prior_shrinkage=cfg["prior_shrinkage"],
        n_samples=cfg["n_samples"],
        n_candidate_pairs=cfg["n_candidate_pairs"],
        seed=seed,
    )
