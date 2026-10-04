"""Random-state normalization for production code."""

from __future__ import annotations

import numpy as np


def check_random_state(seed: int | np.random.RandomState | None) -> np.random.RandomState:
    """Turn seed / None / RandomState into a RandomState instance.

    Mirrors ``sklearn.utils.check_random_state`` so sklearn-dependent code
    in the toolbox can use this function without importing sklearn directly.

    Scope: this helper serves the ``preprocessing`` and ``experiment`` layers,
    whose ``random_state`` is declared ``int | np.random.RandomState | None``.
    Estimators deliberately construct ``np.random.RandomState(random_state)``
    inline instead: their contract is ``int | None``, and widening it would be
    unreachable for the classes that hand ``random_state`` straight to
    ``torch.manual_seed``.  See ``docs/dev/single_source_map.md`` (随机源).
    """
    if seed is None or isinstance(seed, int | np.integer):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise TypeError(f"seed must be int, RandomState, or None; got {type(seed).__name__}")
