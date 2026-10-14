import numpy as np
import pytest

from src.active_learning.bayesian_model import BayesianPairwiseUserModel, GaussianPosterior, _sigmoid
from src.active_learning.interfaces import ItemPair, PairwiseAnswer


class SyntheticCore:
    def __init__(self, n_items=300, d=16, seed=0):
        r = np.random.default_rng(seed)
        A = r.normal(size=(d, d)) / np.sqrt(d)
        self.n_dims = d
        self.prior_covariance = 0.5 * A @ A.T + 0.02 * np.eye(d)
        self.prior_mean = 0.1 * r.normal(size=d)
        self.q = 0.6 * r.normal(size=(n_items, d)) * (0.5 + r.random((n_items, 1)))

    def get_item_vector(self, movie_id):
        if not 0 <= movie_id < len(self.q):
            raise KeyError(movie_id)
        return self.q[movie_id]


@pytest.fixture
def core():
    return SyntheticCore()


def ans(a, b, choice):
    return PairwiseAnswer("s", a, b, choice)


def test_initial_state_shapes_and_prior(core):
    m = BayesianPairwiseUserModel(core, temperature=1.0)
    s = m.initialize_user()
    assert s.mean_vector.ndim == 1 and s.mean_vector.shape == (core.n_dims,)
    assert s.extra.covariance.shape == (core.n_dims, core.n_dims)
    np.testing.assert_allclose(s.mean_vector, core.prior_mean)


def test_isotropic_prior_without_core_stats():
    bare = SyntheticCore()
    bare.prior_mean = bare.prior_covariance = None

    m = BayesianPairwiseUserModel(bare, temperature=1.0, prior="isotropic", prior_std=2.0)
    s = m.initialize_user()
    assert np.allclose(s.mean_vector, 0) and np.allclose(s.extra.covariance, 4 * np.eye(16))
    with pytest.raises(ValueError):
        BayesianPairwiseUserModel(bare, temperature=1.0, prior="empirical")


def test_variance_shrinks_after_answers(core):
    m = BayesianPairwiseUserModel(core, temperature=1.0)
    s = m.initialize_user()
    tr0 = np.trace(s.extra.covariance)
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        prev = np.trace(s.extra.covariance)
        s = m.update_user(s, ans(a, b, "A"))
        assert np.trace(s.extra.covariance) < prev
    assert np.trace(s.extra.covariance) < tr0
    assert s.extra.n_answers == 3
    assert np.linalg.eigvalsh(s.extra.covariance).min() > 0  # stays PD


def test_skip_leaves_state_unchanged(core):
    m = BayesianPairwiseUserModel(core, temperature=1.0)
    s = m.initialize_user()
    s2 = m.update_user(s, ans(0, 1, "skip"))
    assert s2 is s


def test_answer_moves_mean_toward_winner(core):
    m = BayesianPairwiseUserModel(core, temperature=1.0)
    s = m.initialize_user()
    x = core.q[0] - core.q[1]
    sa = m.update_user(s, ans(0, 1, "A"))
    sb = m.update_user(s, ans(0, 1, "B"))
    assert x @ sa.mean_vector > x @ s.mean_vector > x @ sb.mean_vector


def test_update_matches_dense_laplace(core):
    """The 1-D reduction must equal a brute-force d-dimensional Newton MAP
    and Laplace covariance."""
    T = 0.7
    m = BayesianPairwiseUserModel(core, temperature=T)
    s = m.initialize_user()
    got = m.update_user(s, ans(10, 20, "B"))

    x = -(core.q[10] - core.q[20])
    mean0, S0 = s.mean_vector, s.extra.covariance
    P0 = np.linalg.inv(S0)
    p = mean0.copy()
    for _ in range(50):
        sg = _sigmoid(x @ p / T)
        grad = P0 @ (p - mean0) - (1 - sg) * x / T
        H = P0 + sg * (1 - sg) * np.outer(x, x) / T**2
        p = p - np.linalg.solve(H, grad)
    sg = _sigmoid(x @ p / T)
    cov = np.linalg.inv(P0 + sg * (1 - sg) * np.outer(x, x) / T**2)
    np.testing.assert_allclose(got.mean_vector, p, atol=1e-7)
    np.testing.assert_allclose(got.extra.covariance, cov, atol=1e-8)


def test_bald_prefers_ambiguous_pair(core):
    m = BayesianPairwiseUserModel(core, temperature=1.0, n_samples=2000)
    s = m.initialize_user()
    # Narrow posterior centred on a known p so one pair is a near-certain
    # answer and another is a coin flip.
    p = np.ones(core.n_dims)
    s = type(s)(mean_vector=p, extra=GaussianPosterior(1e-4 * np.eye(core.n_dims), 5))
    certain = 20.0 * p / np.linalg.norm(p)  # p.x huge -> answer near-certain
    ambiguous = np.zeros(core.n_dims)
    ambiguous[0] = 1.0
    ambiguous -= (ambiguous @ p) * p / (p @ p)  # orthogonal to p -> p(A) ~ 0.5
    scores = m.bald_scores(s, np.stack([certain, ambiguous]))
    assert scores[1] > scores[0]


def test_select_next_pair_contract(core):
    m = BayesianPairwiseUserModel(core, temperature=1.0, n_candidate_pairs=50)
    s = m.initialize_user()
    cands = list(range(10, 60))
    for _ in range(20):
        pair = m.select_next_pair(s, cands)
        assert isinstance(pair, ItemPair)
        assert pair.movie_a != pair.movie_b
        assert pair.movie_a in cands and pair.movie_b in cands
    two = m.select_next_pair(s, [3, 4])
    assert {two.movie_a, two.movie_b} == {3, 4}
    with pytest.raises(ValueError):
        m.select_next_pair(s, [3])


def _simulate(core, mode, temperature, n_users=120, budget=10, seed=0):
    model = BayesianPairwiseUserModel(core, temperature=temperature, seed=seed)
    L = np.linalg.cholesky(core.prior_covariance)
    users = np.random.default_rng(100)  # identical users for both modes
    picker = np.random.default_rng(seed)
    cos = []
    for _ in range(n_users):
        p = core.prior_mean + L @ users.standard_normal(core.n_dims)
        state, remaining = model.initialize_user(), list(range(len(core.q)))
        for _ in range(budget):
            if mode == "adaptive":
                pair = model.select_next_pair(state, remaining)
            else:
                a, b = picker.choice(remaining, 2, replace=False)
                pair = ItemPair(int(a), int(b))
            remaining = [i for i in remaining if i not in (pair.movie_a, pair.movie_b)]
            z = p @ (core.q[pair.movie_a] - core.q[pair.movie_b]) / temperature
            choice = "A" if users.random() < _sigmoid(z) else "B"
            state = model.update_user(state, ans(pair.movie_a, pair.movie_b, choice))
        cos.append(state.mean_vector @ p / np.linalg.norm(state.mean_vector) / np.linalg.norm(p))
    return float(np.mean(cos))


def test_adaptive_beats_random_on_simulated_users(core):
    assert _simulate(core, "adaptive", 1.0) > _simulate(core, "random", 1.0)


def test_state_roundtrip(core):
    m = BayesianPairwiseUserModel(core, temperature=1.0)
    s = m.update_user(m.initialize_user(), ans(0, 1, "A"))
    s2 = BayesianPairwiseUserModel.state_from_dict(BayesianPairwiseUserModel.state_to_dict(s))
    np.testing.assert_allclose(s2.mean_vector, s.mean_vector)
    np.testing.assert_allclose(s2.extra.covariance, s.extra.covariance)
    assert s2.extra.n_answers == 1


def test_fixed_and_adaptive_share_the_same_update(core):
    """Same answers in the same order -> identical posterior regardless of
    which mode produced the questions."""
    m = BayesianPairwiseUserModel(core, temperature=1.0)
    seq = [ans(0, 1, "A"), ans(2, 3, "skip"), ans(4, 5, "B")]
    s1 = s2 = m.initialize_user()
    for a in seq:
        s1 = m.update_user(s1, a)
        s2 = m.update_user(s2, a)
    np.testing.assert_allclose(s1.mean_vector, s2.mean_vector)


def test_fit_temperature_recovers_true_value():
    from src.active_learning.calibrate import fit_temperature

    rng = np.random.default_rng(0)
    logits = rng.normal(scale=2.0, size=40000)
    for true_t in (0.4, 1.0, 2.5):
        labels = (rng.random(len(logits)) < _sigmoid(logits / true_t)).astype(float)
        assert abs(fit_temperature(logits, labels) - true_t) / true_t < 0.1


def test_build_pairs_same_side_filter(core):
    import pandas as pd

    from src.active_learning.calibrate import build_pairs

    r = np.random.default_rng(0)
    rows = [(u, m, float(r.choice([2, 3, 4, 4.5, 5]))) for u in range(5) for m in range(30)]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    users = {u: r.normal(size=core.n_dims) for u in range(5)}
    items = {m: core.q[m] for m in range(30)}
    pairs = build_pairs(ratings, users, items, set(range(30)), max_pairs_per_user=20,
                        positive_threshold=4.0, pair_filter="same_side", seed=0)
    assert len(pairs) > 0 and set(pairs["label"]) <= {0.0, 1.0}
    assert (pairs["rating_a"] != pairs["rating_b"]).all()
    assert ((pairs["rating_a"] >= 4) == (pairs["rating_b"] >= 4)).all()
    assert (pairs["label"] == (pairs["rating_a"] > pairs["rating_b"])).all()
