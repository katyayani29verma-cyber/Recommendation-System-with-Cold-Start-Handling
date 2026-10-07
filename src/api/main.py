"""FastAPI service wiring together the Track A and Track B contracts behind
HTTP endpoints for the Streamlit demo (README: "Online onboarding").

Runs today with mock vectors/model; swap `build_core()` / `build_model()`
for the real Track A/B artifacts once available.

Usage:
    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import uuid
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.active_learning.interfaces import ItemPair, PairwiseAnswer, PosteriorState
from src.active_learning.mock_model import MockPointEstimateUserModel
from src.core.interfaces import RecommenderCore
from src.core.mock_vectors import MockRecommenderCore
from src.data.config import load_config
from src.data.splits import load_movies, load_ratings, make_cold_start_split
from src.evaluation import baseline

app = FastAPI(title="Adaptive Taste API")


class SessionState(BaseModel):
    mode: Literal["fixed", "adaptive"]
    posterior_mean: list[float]
    asked: list[tuple[int, int]] = []
    fixed_index: int = 0


_sessions: dict[str, SessionState] = {}
_core: RecommenderCore | None = None
_model: MockPointEstimateUserModel | None = None
_fixed_pairs: list[tuple[int, int]] = []
_candidate_ids: list[int] = []
_question_budget: int = 0
_movies_by_id: dict[int, dict] = {}


def _ensure_loaded() -> None:
    global _core, _model, _fixed_pairs, _candidate_ids, _movies_by_id, _question_budget
    if _core is not None:
        return

    data_cfg = load_config("data")
    eval_cfg = load_config("eval")

    ratings = load_ratings()
    movies = load_movies()
    _movies_by_id = movies.set_index("movieId").to_dict(orient="index")

    all_movie_ids = sorted(ratings["movieId"].unique().tolist())
    n_dims = eval_cfg["mock_vectors"]["n_dims"]
    _core = MockRecommenderCore(all_movie_ids, n_dims=n_dims, seed=eval_cfg["mock_vectors"]["seed"])
    _model = MockPointEstimateUserModel(_core, n_dims=n_dims)

    # Popularity comes from core-training data only, same as the evaluation.
    split = make_cold_start_split(ratings, data_cfg["split"])
    _candidate_ids = baseline.select_popular_movies(
        split.core_train, data_cfg["niche_subset"]["min_ratings_for_popularity"]
    )
    _question_budget = eval_cfg["question_budget"]
    _fixed_pairs = baseline.build_fixed_sequence(_core, _candidate_ids, _question_budget, seed=0)


class StartSessionRequest(BaseModel):
    mode: Literal["fixed", "adaptive"] = "adaptive"


class StartSessionResponse(BaseModel):
    session_id: str
    next_pair: "MoviePairResponse"


class MoviePairResponse(BaseModel):
    movie_a: dict
    movie_b: dict


class AnswerRequest(BaseModel):
    session_id: str
    movie_a: int
    movie_b: int
    choice: Literal["A", "B", "skip"]


class AnswerResponse(BaseModel):
    next_pair: MoviePairResponse | None
    done: bool


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    score: float


def _movie_payload(movie_id: int) -> dict:
    info = _movies_by_id.get(movie_id, {})
    return {"movie_id": movie_id, "title": info.get("title", f"Movie {movie_id}")}


def _next_pair_for_session(session_id: str, state: SessionState) -> MoviePairResponse | None:
    if len(state.asked) >= _question_budget:
        return None
    if state.mode == "fixed":
        if state.fixed_index >= len(_fixed_pairs):
            return None
        movie_a, movie_b = _fixed_pairs[state.fixed_index]
    else:
        remaining = [m for m in _candidate_ids if m not in {mid for pair in state.asked for mid in pair}]
        if len(remaining) < 2:
            return None
        posterior = PosteriorState(mean_vector=np.array(state.posterior_mean))
        pair = _model.select_next_pair(posterior, remaining)
        movie_a, movie_b = pair.movie_a, pair.movie_b
    return MoviePairResponse(movie_a=_movie_payload(movie_a), movie_b=_movie_payload(movie_b))


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    _ensure_loaded()
    session_id = str(uuid.uuid4())
    posterior = _model.initialize_user()
    state = SessionState(mode=req.mode, posterior_mean=posterior.mean_vector.tolist())
    _sessions[session_id] = state

    next_pair = _next_pair_for_session(session_id, state)
    if next_pair is None:
        raise HTTPException(500, "No eligible questions available")
    return StartSessionResponse(session_id=session_id, next_pair=next_pair)


@app.post("/session/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    _ensure_loaded()
    state = _sessions.get(req.session_id)
    if state is None:
        raise HTTPException(404, "Unknown session_id")

    posterior = PosteriorState(mean_vector=np.array(state.posterior_mean))
    pairwise_answer = PairwiseAnswer(
        session_id=req.session_id, movie_a=req.movie_a, movie_b=req.movie_b, choice=req.choice
    )
    posterior = _model.update_user(posterior, pairwise_answer)
    state.posterior_mean = posterior.mean_vector.tolist()
    state.asked.append((req.movie_a, req.movie_b))
    if state.mode == "fixed":
        state.fixed_index += 1

    next_pair = _next_pair_for_session(req.session_id, state)
    return AnswerResponse(next_pair=next_pair, done=next_pair is None)


@app.get("/session/{session_id}/recommend", response_model=list[RecommendationItem])
def recommend(session_id: str, k: int = 10) -> list[RecommendationItem]:
    _ensure_loaded()
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(404, "Unknown session_id")

    exclude = {mid for pair in state.asked for mid in pair}
    recs = _core.recommend(np.array(state.posterior_mean), k=k, exclude_ids=exclude)
    return [
        RecommendationItem(movie_id=r.movie_id, title=_movie_payload(r.movie_id)["title"], score=r.score)
        for r in recs
    ]


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
