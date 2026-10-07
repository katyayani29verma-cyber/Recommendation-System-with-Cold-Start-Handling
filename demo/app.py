"""Streamlit demo (README: "What we are building").

Shows two users side by side, one on the fixed onboarding sequence and one
on adaptive selection, so divergence in question order (and eventually
recommendation quality) is visible as answers differ.

Talks to the FastAPI service; start it first:
    uvicorn src.api.main:app --reload

Then run:
    streamlit run demo/app.py
"""
from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Adaptive Taste", layout="wide")
st.title("Adaptive Taste — fixed vs. adaptive onboarding")


def start_session(mode: str) -> dict:
    resp = requests.post(f"{API_BASE}/session/start", json={"mode": mode}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def submit_answer(session_id: str, movie_a: int, movie_b: int, choice: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/session/answer",
        json={"session_id": session_id, "movie_a": movie_a, "movie_b": movie_b, "choice": choice},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_recommendations(session_id: str) -> list[dict]:
    resp = requests.get(f"{API_BASE}/session/{session_id}/recommend", timeout=10)
    resp.raise_for_status()
    return resp.json()


def render_column(label: str, mode: str) -> None:
    st.subheader(label)
    key = f"state_{mode}"

    if key not in st.session_state:
        try:
            data = start_session(mode)
        except requests.RequestException as e:
            st.error(f"Could not reach API at {API_BASE}: {e}")
            return
        st.session_state[key] = {"session_id": data["session_id"], "pair": data["next_pair"], "done": False}

    state = st.session_state[key]

    if state["done"] or state["pair"] is None:
        st.success("Onboarding complete.")
        if st.button("Show recommendations", key=f"rec_{mode}"):
            recs = get_recommendations(state["session_id"])
            for r in recs:
                st.write(f"{r['title']} (score: {r['score']:.3f})")
        return

    pair = state["pair"]
    movie_a, movie_b = pair["movie_a"], pair["movie_b"]
    st.write("Which would you rather watch?")
    col_a, col_b, col_skip = st.columns(3)

    choice = None
    if col_a.button(movie_a["title"], key=f"a_{mode}_{movie_a['movie_id']}"):
        choice = "A"
    if col_b.button(movie_b["title"], key=f"b_{mode}_{movie_b['movie_id']}"):
        choice = "B"
    if col_skip.button("Skip", key=f"skip_{mode}_{movie_a['movie_id']}_{movie_b['movie_id']}"):
        choice = "skip"

    if choice:
        result = submit_answer(state["session_id"], movie_a["movie_id"], movie_b["movie_id"], choice)
        state["pair"] = result["next_pair"]
        state["done"] = result["done"]
        st.rerun()


left, right = st.columns(2)
with left:
    render_column("Fixed onboarding", "fixed")
with right:
    render_column("Adaptive onboarding", "adaptive")
