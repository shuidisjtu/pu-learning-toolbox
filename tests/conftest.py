"""Shared pytest fixtures for PU Learning Toolbox."""

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


@pytest.fixture
def simple_x_y_pu(rng):
    """Small 2D Gaussian dataset with PU labels.

    Positive class (center=+2) and negative class (center=-2), each with
    100 samples.  50 % of positives are labeled (SCAR c=0.5).
    """
    n = 100
    x_pos = rng.randn(n, 5) + 2.0
    x_neg = rng.randn(n, 5) - 2.0
    x = np.vstack([x_pos, x_neg])
    y_true = np.hstack([np.ones(n), np.zeros(n)])

    # SCAR labeling: 50 % of positives get label +1
    y_pu = np.zeros(2 * n, dtype=int)
    pos_idx = np.where(y_true == 1)[0]
    labeled = rng.choice(pos_idx, size=n // 2, replace=False)
    y_pu[labeled] = 1

    # Return canonical {+1, 0}
    y_pu[pos_idx] = 1  # all positives → +1
    # labeled subset: half remain +1, other half become 0
    unlabeled_pos = list(set(pos_idx) - set(labeled))
    y_pu[unlabeled_pos] = 0
    # negatives: always 0
    y_pu[n:] = 0

    return x, y_pu, y_true
