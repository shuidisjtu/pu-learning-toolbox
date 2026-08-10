# ruff: noqa: N803, N806

"""Reusable data factories for tests.

Keep ordinary helpers outside ``conftest.py`` so tests do not import pytest's
fixture configuration module directly.
"""

import numpy as np

from pu_toolbox.preprocessing import make_scar_dataset as _make_scar_dataset
from pu_toolbox.preprocessing import make_scar_labels as _make_scar_labels


def make_scar_data(rng, n=100, c=0.5, n_features=5, separation=4.0):
    """Generate synthetic SCAR data with known labeling propensity ``c``."""
    return _make_scar_dataset(
        n=n,
        c=c,
        n_features=n_features,
        separation=separation,
        random_state=rng,
    )


def make_scar_data_unbalanced(
    rng,
    n=200,
    prior=0.5,
    c=0.5,
    n_features=5,
    separation=2.0,
):
    """Generate SCAR data with a controlled class prior (not necessarily 0.5)."""
    n_p = max(1, int(round(n * prior)))
    n_u = n - n_p
    delta = separation / 2.0
    X = np.vstack([rng.randn(n_p, n_features) + delta, rng.randn(n_u, n_features) - delta])
    y_true = np.hstack([np.ones(n_p, dtype=int), np.zeros(n_u, dtype=int)])
    y_pu = _make_scar_labels(y_true, c=c, random_state=rng)
    return X, y_pu, y_true
