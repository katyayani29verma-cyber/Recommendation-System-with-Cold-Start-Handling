import pytest
from fastapi.testclient import TestClient

from src.api.main import app

pytest.importorskip("httpx")

client = TestClient(app)


@pytest.fixture(autouse=True)
def configured_test_models(monkeypatch):
    from src.active_learning.bayesian_model import BayesianPairwiseUserModel
    from src.api import main as api_main
    from src.core.mock_vectors import MockRecommenderCore

    original_load_config = api_main.load_config

    def load_test_config(name):
        config = original_load_config(name)
        if name == "eval":
            config["mock_vectors"]["enabled"] = True
        return config

    monkeypatch.setattr(api_main, "load_config", load_test_config)
    monkeypatch.setattr(
        api_main,
        "build_core",
        lambda cfg, movie_ids: MockRecommenderCore(
            movie_ids,
            n_dims=cfg["mock_vectors"]["n_dims"],
            seed=cfg["mock_vectors"]["seed"],
        ),
    )
    monkeypatch.setattr(
        api_main,
        "build_user_model",
        lambda core, cfg: BayesianPairwiseUserModel(
            core,
            prior="isotropic",
            temperature=1.0,
        ),
    )
    api_main._core = None
    api_main._model = None
    api_main._sessions.clear()
    api_main._fixed_pairs = []
    api_main._candidate_ids = []
    api_main._movies_by_id = {}
    api_main._question_budget = 0
    yield
    api_main._core = None
    api_main._model = None
    api_main._sessions.clear()


def test_start_session_returns_a_pair():
    r = client.post("/session/start", json={"mode": "adaptive"})
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert "movie_a" in body["next_pair"] and "movie_b" in body["next_pair"]


def test_full_session_flow_reaches_recommendations():
    from src.active_learning.bayesian_model import GaussianPosterior
    from src.api import main as api_main

    start = client.post("/session/start", json={"mode": "adaptive"}).json()
    session_id = start["session_id"]
    pair = start["next_pair"]

    r = client.post(
        "/session/answer",
        json={
            "session_id": session_id,
            "movie_a": pair["movie_a"]["movie_id"],
            "movie_b": pair["movie_b"]["movie_id"],
            "choice": "A",
        },
    )
    assert r.status_code == 200
    assert r.json()["done"] is False
    posterior = api_main._sessions[session_id].posterior
    assert isinstance(posterior.extra, GaussianPosterior)
    assert posterior.extra.n_answers == 1

    recs = client.get(f"/session/{session_id}/recommend")
    assert recs.status_code == 200
    assert len(recs.json()) == 10


def test_unknown_session_returns_404():
    r = client.get("/session/does-not-exist/recommend")
    assert r.status_code == 404


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_adaptive_session_stops_at_question_budget():
    from src.api import main as api_main

    start = client.post("/session/start", json={"mode": "adaptive"}).json()
    session_id, pair = start["session_id"], start["next_pair"]
    answered = 0
    while pair is not None:
        r = client.post(
            "/session/answer",
            json={
                "session_id": session_id,
                "movie_a": pair["movie_a"]["movie_id"],
                "movie_b": pair["movie_b"]["movie_id"],
                "choice": "A",
            },
        ).json()
        answered += 1
        pair = r["next_pair"]
        assert answered <= api_main._question_budget
    assert answered == api_main._question_budget
    assert r["done"] is True
