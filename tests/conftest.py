"""Shared pytest fixtures for PU Learning Toolbox."""

# ruff: noqa: N806

import numpy as np
import pytest

from pu_toolbox.core.random import set_global_seed
from tests.helpers import make_scar_data


@pytest.fixture(scope="session", autouse=True)
def _fixed_seed():
    """Ensure deterministic tests via a fixed global seed."""
    set_global_seed(42)


@pytest.fixture
def rng():
    """Return a fresh numpy RandomState for per-test use."""
    return np.random.RandomState(42)


# ═════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_x_y_pu(rng):
    """Small SCAR dataset: 2×100 samples, separation=4.0, c=0.5."""
    return make_scar_data(rng, n=100, c=0.5, n_features=5, separation=4.0)
