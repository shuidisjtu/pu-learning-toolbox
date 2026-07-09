"""Reproducible random seeding utilities."""

from __future__ import annotations

import random as _random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Set Python stdlib, numpy, and (if available) torch random seeds.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    _random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_random_state(seed: int | np.random.RandomState | None) -> np.random.RandomState:
    """Turn seed / None / RandomState into a RandomState instance.

    Mirrors ``sklearn.utils.check_random_state`` so sklearn-dependent code
    in the toolbox can use this function without importing sklearn directly.
    """
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise TypeError(
        f"seed must be int, RandomState, or None; got {type(seed).__name__}"
    )
