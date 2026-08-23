"""Random-state normalization for production code."""

from __future__ import annotations

import numpy as np


def check_random_state(seed: int | np.random.RandomState | None) -> np.random.RandomState:
    """Turn seed / None / RandomState into a RandomState instance.

    Mirrors ``sklearn.utils.check_random_state`` so sklearn-dependent code
    in the toolbox can use this function without importing sklearn directly.
    """
    if seed is None or isinstance(seed, int | np.integer):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise TypeError(f"seed must be int, RandomState, or None; got {type(seed).__name__}")
