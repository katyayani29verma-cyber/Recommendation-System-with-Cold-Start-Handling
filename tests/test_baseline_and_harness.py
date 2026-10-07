import pandas as pd

from src.active_learning.interfaces import ItemPair
from src.core.mock_vectors import MockRecommenderCore
from src.evaluation import baseline, harness


def make_core_train(n_movies: int = 20, ratings_per_movie: int = 10) -> pd.DataFrame:
    rows = []
    uid = 0
    for mid in range(1, n_movies + 1):
        for _ in range(ratings_per_movie):
            rows.append({"userId": uid, "movieId": mid, "rating": 4.0, "timestamp": 0})
            uid += 1
    return pd.DataFrame(rows)


def test_select_popular_movies_respects_threshold():
    core_train = make_core_train(n_movies=10, ratings_per_movie=5)
    popular = baseline.select_popular_movies(core_train, min_ratings=5)
    assert set(popular) == set(range(1, 11))

    none_popular = baseline.select_popular_movies(core_train, min_ratings=6)
    assert none_popular == []


def test_build_fixed_sequence_produces_distinct_movies_and_budget_pairs():
    movie_ids = list(range(1, 21))
    core = MockRecommenderCore(movie_ids, n_dims=8, seed=1)
    pairs = baseline.build_fixed_sequence(core, movie_ids, budget=5, seed=0)

    assert len(pairs) == 5
    used = [mid for pair in pairs for mid in pair]
    assert len(used) == len(set(used)), "fixed sequence should not reuse movies"


def test_build_fixed_sequence_raises_when_not_enough_candidates():
    movie_ids = list(range(1, 5))
    core = MockRecommenderCore(movie_ids, n_dims=4, seed=1)
    try:
        baseline.build_fixed_sequence(core, movie_ids, budget=5, seed=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_simulate_answer_picks_higher_rated_movie():
    pair = ItemPair(movie_a=1, movie_b=2)
    answer = harness.simulate_answer({1: 5.0, 2: 2.0}, pair)
    assert answer.choice == "A"


def test_simulate_answer_skips_on_tie():
    pair = ItemPair(movie_a=1, movie_b=2)
    answer = harness.simulate_answer({1: 3.0, 2: 3.0}, pair)
    assert answer.choice == "skip"


def test_simulate_answer_skips_when_rating_missing():
    pair = ItemPair(movie_a=1, movie_b=2)
    answer = harness.simulate_answer({1: 5.0}, pair)
    assert answer.choice == "skip"


def test_adaptive_session_never_repeats_a_movie():
    from src.active_learning.mock_model import MockPointEstimateUserModel

    movie_ids = list(range(1, 21))
    core = MockRecommenderCore(movie_ids, n_dims=8, seed=1)
    model = MockPointEstimateUserModel(core, n_dims=8, seed=3)
    ratings = {mid: float(mid % 5 + 1) for mid in movie_ids}

    traj = harness.run_adaptive_session(model, movie_ids, ratings, budget=10)

    used = [mid for a in traj.answers for mid in (a.movie_a, a.movie_b)]
    assert len(traj.answers) == 10
    assert len(used) == len(set(used))
