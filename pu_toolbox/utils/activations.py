"""Numerically stable activation helpers.

Single source for activation functions used across PU loss modules
(e.g. the sigmoid surrogate in uPU / nnPU losses).
"""

from __future__ import annotations

import numpy as np


def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid: 1 / (1 + exp(−z))."""
    # Clip to avoid overflow in exp.
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))
