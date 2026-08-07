"""Reusable data factories for tests.

Keep ordinary helpers outside ``conftest.py`` so tests do not import pytest's
fixture configuration module directly.
"""

from pu_toolbox.preprocessing import make_scar_dataset as _make_scar_dataset


def make_scar_data(rng, n=100, c=0.5, n_features=5, separation=4.0):
    """Generate synthetic SCAR data with known labeling propensity ``c``."""
    return _make_scar_dataset(
        n=n,
        c=c,
        n_features=n_features,
        separation=separation,
        random_state=rng,
    )
