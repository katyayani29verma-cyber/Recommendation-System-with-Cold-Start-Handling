"""FastAPI service wiring the configured recommendation and user models
behind HTTP endpoints for the Streamlit demo.

Usage:
    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.active_learning.factory import build_core, build_user_model
from src.active_learning.interfaces import PairwiseAnswer, PosteriorState, UserModel
from src.core.interfaces import RecommenderCore
from src.data.config import load_config
from src.data.splits import load_movies, load_ratings, make_cold_start_split
from src.evaluation import baseline

app = FastAPI(title="Adaptive Taste API")


@dataclass
class SessionState:
    mode: Literal["fixed", "adaptive"]
    posterior: PosteriorState
    asked: list[tuple[int, int]] = field(default_factory=list)
    fixed_index: int = 0


_sessions: dict[str, SessionState] = {}
_core: RecommenderCore | None = None
_model: UserModel | None = None
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
    _core = build_core(eval_cfg, all_movie_ids)
    _model = build_user_model(_core, eval_cfg)

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


def _next_pair_for_session(state: SessionState) -> MoviePairResponse | None:
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
        pair = _model.select_next_pair(state.posterior, remaining)
        movie_a, movie_b = pair.movie_a, pair.movie_b
    return MoviePairResponse(movie_a=_movie_payload(movie_a), movie_b=_movie_payload(movie_b))


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    _ensure_loaded()
    session_id = str(uuid.uuid4())
    posterior = _model.initialize_user()
    state = SessionState(mode=req.mode, posterior=posterior)
    _sessions[session_id] = state

    next_pair = _next_pair_for_session(state)
    if next_pair is None:
        raise HTTPException(500, "No eligible questions available")
    return StartSessionResponse(session_id=session_id, next_pair=next_pair)


@app.post("/session/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    _ensure_loaded()
    state = _sessions.get(req.session_id)
    if state is None:
        raise HTTPException(404, "Unknown session_id")

    pairwise_answer = PairwiseAnswer(
        session_id=req.session_id, movie_a=req.movie_a, movie_b=req.movie_b, choice=req.choice
    )
    state.posterior = _model.update_user(state.posterior, pairwise_answer)
    state.asked.append((req.movie_a, req.movie_b))
    if state.mode == "fixed":
        state.fixed_index += 1

    next_pair = _next_pair_for_session(state)
    return AnswerResponse(next_pair=next_pair, done=next_pair is None)


@app.get("/session/{session_id}/recommend", response_model=list[RecommendationItem])
def recommend(session_id: str, k: int = 10) -> list[RecommendationItem]:
    _ensure_loaded()
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(404, "Unknown session_id")

    exclude = {mid for pair in state.asked for mid in pair}
    recs = _core.recommend(state.posterior.mean_vector, k=k, exclude_ids=exclude)
    return [
        RecommendationItem(movie_id=r.movie_id, title=_movie_payload(r.movie_id)["title"], score=r.score)
        for r in recs
    ]


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
