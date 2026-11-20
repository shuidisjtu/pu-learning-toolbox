# ruff: noqa: N803, N806

"""Reusable data factories for tests.

Keep ordinary helpers outside ``conftest.py`` so tests do not import pytest's
fixture configuration module directly.
"""

import random

import numpy as np

from pu_toolbox.core.random import check_random_state
from pu_toolbox.preprocessing import (
    GaussianSourceConfig,
    PULabelingConfig,
    SourcePartition,
    construct_pu_dataset,
    simulate_pu_dataset,
)


def set_test_seed(seed: int) -> None:
    """Seed Python, NumPy, and optional torch for deterministic tests."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_scar_data(rng, n=100, c=0.5, n_features=5, separation=4.0):
    """Generate synthetic SCAR data with known labeling propensity ``c``."""
    rng = check_random_state(rng)
    result = simulate_pu_dataset(
        GaussianSourceConfig(2 * n, n_features, 0.5, separation, rng, row_order="class_blocked"),
        PULabelingConfig("scar", "fixed_count", rng, c=c),
    )
    return result.observations.features, result.observations.labels, result.audit.true_labels


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
    result = construct_pu_dataset(
        SourcePartition(X, y_true, "test-unbalanced", "generated"),
        PULabelingConfig("scar", "fixed_count", rng, c=c),
        positive_classes=(1,),
        negative_classes=(0,),
    )
    return X, result.observations.labels, y_true
