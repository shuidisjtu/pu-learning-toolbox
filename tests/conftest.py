"""Shared pytest fixtures for PU Learning Toolbox."""

# ruff: noqa: N806

import numpy as np
import pytest

from pu_toolbox.core.random import set_global_seed


@pytest.fixture(scope="session", autouse=True)
def _fixed_seed():
    """Ensure deterministic tests via a fixed global seed."""
    set_global_seed(42)


@pytest.fixture
def rng():
    """Return a fresh numpy RandomState for per-test use."""
    return np.random.RandomState(42)


# ═════════════════════════════════════════════════════════════════════
# Shared PU data factories
# ═════════════════════════════════════════════════════════════════════


def make_scar_data(rng, n=100, c=0.5, n_features=5, separation=4.0):
    """Generate synthetic SCAR data with known labeling propensity ``c``.

    Positive class centered at ``+separation/2``, negative at
    ``-separation/2``, each with ``n`` samples.

    Returns
    -------
    X : np.ndarray of shape (2*n, n_features)
    y_pu : np.ndarray of shape (2*n,)  — {1, 0}
    y_true : np.ndarray of shape (2*n,) — {1, 0}
    """
    delta = separation / 2.0
    X_pos = rng.randn(n, n_features) + delta
    X_neg = rng.randn(n, n_features) - delta
    X = np.vstack([X_pos, X_neg])
    y_true = np.hstack([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    y_pu = np.zeros(2 * n, dtype=int)
    pos_idx = np.where(y_true == 1)[0]
    n_labeled = max(1, int(n * c))
    labeled = rng.choice(pos_idx, size=n_labeled, replace=False)
    y_pu[labeled] = 1

    return X, y_pu, y_true


def make_gaussian_pu_data(rng, n_p=50, n_u=100, n_features=5, separation=2.0):
    """Generate 2-class Gaussian PU data.

    Positive class at ``+separation/2``, negative at ``-separation/2``.
    All positives are labeled.

    Returns
    -------
    X : np.ndarray of shape (n_p + n_u, n_features)
    y_pu : np.ndarray of shape (n_p + n_u,) — {1, 0}
    class_prior : float
    """
    delta = separation / 2.0
    X_p = rng.randn(n_p, n_features) + delta
    X_n = rng.randn(n_u, n_features) - delta
    X = np.vstack([X_p, X_n])
    y_pu = np.concatenate([np.ones(n_p, dtype=int), np.zeros(n_u, dtype=int)])
    class_prior = n_p / (n_p + n_u)
    return X, y_pu, class_prior


# ═════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_x_y_pu(rng):
    """Small SCAR dataset: 2×100 samples, separation=4.0, c=0.5."""
    return make_scar_data(rng, n=100, c=0.5, n_features=5, separation=4.0)
