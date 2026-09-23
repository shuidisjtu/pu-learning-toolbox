"""Numerically stable activation helpers.

Single source for activation functions used across PU loss modules
(e.g. the sigmoid surrogate in uPU / nnPU losses).
"""

from __future__ import annotations

import numpy as np


def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid: 1 / (1 + exp(−z)).

    Both branches are evaluated through ``exp(-|z|)``, whose exponent is never
    positive, so ``exp`` cannot overflow whatever the input dtype carries.
    Clipping instead has to pick a bound that is safe for that dtype: a bound
    chosen for float64 (``exp(500)`` is a finite 1.4e217) still overflows to
    ``inf`` on the float32 arrays the survey splits carry, and any bound wide
    enough to saturate float64 throws away the subnormal tail rather than
    returning it.  The underflow on the far side is the correct limit and is
    silent, so no bound is needed at all.
    """
    values = np.asarray(z)
    if values.dtype.kind != "f":
        values = values.astype(float)
    exp_neg_abs = np.exp(-np.abs(values))
    return np.where(values >= 0, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))
