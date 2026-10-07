import pytest
from fastapi.testclient import TestClient

from src.api.main import app

pytest.importorskip("httpx")

client = TestClient(app)


def test_start_session_returns_a_pair():
    r = client.post("/session/start", json={"mode": "adaptive"})
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert "movie_a" in body["next_pair"] and "movie_b" in body["next_pair"]


def test_full_session_flow_reaches_recommendations():
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
