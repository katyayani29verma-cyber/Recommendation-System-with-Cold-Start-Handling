from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.active_learning.interfaces import ItemPair, PairwiseAnswer, PosteriorState
from src.core.interfaces import RecommenderCore

_LN2 = np.log(2.0)


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-safe


def _binary_entropy(p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


@dataclass
class GaussianPosterior:

    covariance: np.ndarray  # (d, d) symmetric PSD
    n_answers: int = 0  # informative (non-skip) answers absorbed

    def to_dict(self) -> dict:
        return {"covariance": self.covariance.tolist(), "n_answers": self.n_answers}

    @classmethod
    def from_dict(cls, data: dict) -> "GaussianPosterior":
        return cls(np.asarray(data["covariance"], dtype=np.float64), int(data["n_answers"]))


class BayesianPairwiseUserModel:
    def __init__(
        self,
        core: RecommenderCore,
        n_dims: int | None = None,
        *,
        temperature: float = 1.0,
        prior: str = "empirical",  # "empirical" (Track A stats) or "isotropic"
        prior_std: float = 1.0,  # sigma for the isotropic prior
        prior_shrinkage: float = 0.1,  # blend empirical cov toward its mean-variance identity
        n_samples: int = 64,  # posterior samples M for BALD
        n_candidate_pairs: int = 300,
        seed: int = 0,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if prior not in ("empirical", "isotropic"):
            raise ValueError("prior must be 'empirical' or 'isotropic'")
        self.core = core
        self.n_dims = int(n_dims if n_dims is not None else core.n_dims)
        self.temperature = float(temperature)
        self.n_samples = int(n_samples)
        self.n_candidate_pairs = int(n_candidate_pairs)
        self._rng = np.random.default_rng(seed)
        self._vec_cache: dict[int, np.ndarray] = {}

        self._prior_mean, self._prior_cov = self._build_prior(prior, prior_std, prior_shrinkage)

    # ------------------------------------------------------------------ prior
    def _build_prior(self, prior: str, prior_std: float, shrinkage: float):
        d = self.n_dims
        mean = getattr(self.core, "prior_mean", None)
        cov = getattr(self.core, "prior_covariance", None)
        if prior == "empirical" and mean is not None and cov is not None:
            mean = np.asarray(mean, dtype=np.float64).copy()
            cov = np.asarray(cov, dtype=np.float64)
            if mean.shape != (d,) or cov.shape != (d, d):
                raise ValueError("Core prior statistics do not match n_dims.")
            cov = 0.5 * (cov + cov.T)
            # Shrink toward a scaled identity so the prior is well conditioned.
            cov = (1.0 - shrinkage) * cov + shrinkage * (np.trace(cov) / d) * np.eye(d)
            return mean, cov
        if prior == "empirical":
            raise ValueError("prior='empirical' needs core.prior_mean and core.prior_covariance.")
        return np.zeros(d), (prior_std**2) * np.eye(d)

    # ------------------------------------------------------------- UserModel
    def initialize_user(self) -> PosteriorState:
        return PosteriorState(
            mean_vector=self._prior_mean.copy(),
            extra=GaussianPosterior(self._prior_cov.copy(), 0),
        )

    def _item_vec(self, movie_id: int) -> np.ndarray:
        v = self._vec_cache.get(movie_id)
        if v is None:
            v = np.asarray(self.core.get_item_vector(int(movie_id)), dtype=np.float64)
            self._vec_cache[movie_id] = v
        return v

    def update_user(self, state: PosteriorState, answer: PairwiseAnswer) -> PosteriorState:
        if answer.choice == "skip":
            return state  # no likelihood term; state is untouched
        if answer.choice not in ("A", "B"):
            raise ValueError(f"Unknown choice: {answer.choice!r}")

        post: GaussianPosterior = state.extra
        m, S, T = state.mean_vector, post.covariance, self.temperature
        sign = 1.0 if answer.choice == "A" else -1.0
        x = sign * (self._item_vec(answer.movie_a) - self._item_vec(answer.movie_b))

        u = S @ x
        v = float(x @ u)  # prior variance of x.p
        mu = float(x @ m)
        # 1-D Newton on g(a) = 0.5 a^2 v - log sigmoid((mu + a v)/T), a >= 0.
        a = 0.0
        for _ in range(30):
            s = _sigmoid((mu + a * v) / T)
            grad = a * v - (1.0 - s) * v / T
            hess = v + s * (1.0 - s) * v * v / (T * T)
            step = grad / hess if hess > 0 else 0.0
            a -= step
            if abs(step) < 1e-10:
                break
        new_mean = m + a * u

        z = (mu + a * v) / T
        w = _sigmoid(z) * (1.0 - _sigmoid(z))
        new_cov = S - (w / T**2) * np.outer(u, u) / (1.0 + w * v / T**2)
        new_cov = 0.5 * (new_cov + new_cov.T)

        return PosteriorState(new_mean, GaussianPosterior(new_cov, post.n_answers + 1))

    def select_next_pair(self, state: PosteriorState, candidate_ids: list[int]) -> ItemPair:
        ids = np.asarray(candidate_ids)
        if len(ids) < 2:
            raise ValueError("Need at least two candidate movies.")
        n = len(ids)
        n_pairs = min(self.n_candidate_pairs, n * (n - 1) // 2)
        a_idx = self._rng.integers(0, n, size=3 * n_pairs)
        b_idx = self._rng.integers(0, n, size=3 * n_pairs)
        keep = a_idx != b_idx
        a_idx, b_idx = a_idx[keep][:n_pairs], b_idx[keep][:n_pairs]
        if len(a_idx) == 0:  # tiny candidate list, fall back to a fixed pair
            return ItemPair(int(ids[0]), int(ids[1]))

        X = np.stack([self._item_vec(int(ids[i])) - self._item_vec(int(ids[j])) for i, j in zip(a_idx, b_idx)])
        scores = self.bald_scores(state, X)
        best = int(np.argmax(scores))
        return ItemPair(movie_a=int(ids[a_idx[best]]), movie_b=int(ids[b_idx[best]]))

    # ------------------------------------------------------------------ BALD
    def bald_scores(self, state: PosteriorState, X: np.ndarray) -> np.ndarray:
        """BALD for each row x of X (= q_A - q_B). Returns shape (len(X),)."""
        post: GaussianPosterior = state.extra
        L = np.linalg.cholesky(post.covariance + 1e-10 * np.eye(self.n_dims))
        samples = state.mean_vector + self._rng.standard_normal((self.n_samples, self.n_dims)) @ L.T
        p = _sigmoid(samples @ X.T / self.temperature)  # (M, N)
        return _binary_entropy(p.mean(axis=0)) - _binary_entropy(p).mean(axis=0)

    # ------------------------------------------------- (de)serialisation (API)
    @staticmethod
    def state_to_dict(state: PosteriorState) -> dict:
        return {"mean": np.asarray(state.mean_vector).tolist(), "posterior": state.extra.to_dict()}

    @staticmethod
    def state_from_dict(data: dict) -> PosteriorState:
        return PosteriorState(np.asarray(data["mean"], dtype=np.float64), GaussianPosterior.from_dict(data["posterior"]))
