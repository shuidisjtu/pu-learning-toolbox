"""Shared pytest fixtures for PU Learning Toolbox."""

# ruff: noqa: N806

import numpy as np
import pytest

from pu_toolbox.core.random import set_global_seed
from pu_toolbox.preprocessing import make_gaussian_pu_data as _pp_make_gaussian
from pu_toolbox.preprocessing import make_scar_dataset as _pp_make_scar_dataset


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
    return _pp_make_scar_dataset(
        n=n, c=c, n_features=n_features,
        separation=separation, random_state=rng,
    )


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
    return _pp_make_gaussian(
        n_p=n_p, n_u=n_u, n_features=n_features,
        separation=separation, random_state=rng,
    )


# ═════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_x_y_pu(rng):
    """Small SCAR dataset: 2×100 samples, separation=4.0, c=0.5."""
    return make_scar_data(rng, n=100, c=0.5, n_features=5, separation=4.0)
