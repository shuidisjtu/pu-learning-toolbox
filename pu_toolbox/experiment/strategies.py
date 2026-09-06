"""Built-in strategies for the experiment layer (P0).

Design notes: SCAR/SAR share the fixed-count policy (protocol §2.1,
n_L = round(c·n_+), uniform without replacement); SAR-LBE matches
PU-Bench commit 2d95a19 (implementation_plan.md §2.2). The posterior
helper model is fitted on REAL labels (source train) — never on PU views.
"""

# ruff: noqa: N803

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from pu_toolbox.core.random import check_random_state

from .protocols import Generator


def _n_labeled(y_true: np.ndarray, c: float) -> int:
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0
    return min(n_pos, max(1, int(np.round(n_pos * c))))


def _fit_posterior(X: np.ndarray, y_true: np.ndarray, seed: int | None):
    return LogisticRegression(solver="lbfgs", max_iter=100, random_state=seed).fit(X, y_true)


class SCARGenerator(Generator):
    """Fixed-count SCAR (protocol §2.1): uniform sampling without replacement."""

    def generate(self, X, y_true, c, seed=None):
        rng = check_random_state(seed)
        n_labeled = _n_labeled(y_true, c)
        pos = np.where(y_true == 1)[0]
        y_pu = np.zeros(len(y_true), dtype=int)
        if n_labeled > 0:
            y_pu[rng.choice(pos, size=n_labeled, replace=False)] = 1
        n_pos = int(len(pos))
        return y_pu, {
            "mechanism": "scar",
            "c_realized": n_labeled / n_pos if n_pos else 0.0,
            "n_labeled": n_labeled,
        }


def _lbe_sample(scores: np.ndarray, n_labeled: int, rng, weights: np.ndarray) -> np.ndarray:
    """Weighted sampling without replacement used by both LBE variants."""
    if n_labeled >= len(scores):
        return np.arange(len(scores))
    if np.all(weights == 0):
        return rng.choice(len(scores), size=n_labeled, replace=False)
    return rng.choice(len(scores), size=n_labeled, replace=False, p=weights / weights.sum())


class SARLBEAGenerator(Generator):
    """LBE-A: p ∝ scores**k with smoothing p = 0.9p + 0.1·uniform (PU-Bench 2d95a19)."""

    def __init__(self, k: float = 10, smoothing: tuple[float, float] = (0.9, 0.1)) -> None:
        self.k = k
        self.smoothing = smoothing

    def generate(self, X, y_true, c, seed=None):
        rng = check_random_state(seed)
        n_labeled = _n_labeled(y_true, c)
        model = _fit_posterior(X, y_true, seed)
        scores = model.predict_proba(X)[:, 1]
        weights = np.clip(scores, 1e-9, None) ** self.k
        w1, w0 = self.smoothing
        weights = w1 * weights / weights.sum() + w0 * np.ones_like(weights) / len(weights)
        y_pu = np.zeros(len(y_true), dtype=int)
        if n_labeled > 0:
            chosen = _lbe_sample(scores, n_labeled, rng, weights)
            y_pu[chosen] = 1
        n_pos = int(np.sum(y_true == 1))
        return y_pu, {
            "mechanism": "sar_lbe_a",
            "c_realized": n_labeled / n_pos if n_pos else 0.0,
            "k": self.k,
            "smoothing": self.smoothing,
            "posterior_fit_on": "real_labels",
            "posterior_version": "PU-Bench 2d95a19/lbfgs(max_iter=100)",
            "scores_file": None,
        }


class SARLBEBGenerator(Generator):
    """LBE-B: p ∝ (1.5 + shrink_coef - scores)**k, negative clipped, uniform fallback."""

    def __init__(self, shrink_coef: float = 1.0, k: float = 10) -> None:
        self.shrink_coef = shrink_coef
        self.k = k

    def generate(self, X, y_true, c, seed=None):
        rng = check_random_state(seed)
        n_labeled = _n_labeled(y_true, c)
        model = _fit_posterior(X, y_true, seed)
        scores = model.predict_proba(X)[:, 1]
        weights = np.clip(1.5 + self.shrink_coef - scores, 0.0, None) ** self.k
        y_pu = np.zeros(len(y_true), dtype=int)
        if n_labeled > 0:
            chosen = _lbe_sample(scores, n_labeled, rng, weights)
            y_pu[chosen] = 1
        n_pos = int(np.sum(y_true == 1))
        return y_pu, {
            "mechanism": "sar_lbe_b",
            "c_realized": n_labeled / n_pos if n_pos else 0.0,
            "shrink_coef": self.shrink_coef,
            "k": self.k,
            "posterior_fit_on": "real_labels",
            "posterior_version": "PU-Bench 2d95a19/lbfgs(max_iter=100)",
            "scores_file": None,
        }
