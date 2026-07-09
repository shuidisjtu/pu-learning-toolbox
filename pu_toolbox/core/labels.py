"""PU label normalisation utilities.

Canonical internal representation::

    +1  →  labeled positive (P)
     0  →  unlabeled (U)

The module accepts common conventions found in the PU literature and
normalises them to the canonical form.
"""

from __future__ import annotations

import numpy as np

from .config import POSITIVE_LABEL, UNLABELED_LABEL
from .exceptions import ValidationError


def normalize_pu_labels(y: np.ndarray) -> np.ndarray:
    """Convert a PU label vector to canonical ``{+1, 0}`` encoding.

    Accepted input conventions
    --------------------------
    - ``{+1, 0}``   — canonical form (returned as-is)
    - ``{+1, -1}``  — common in SVM / PU bagging literature
    - ``{1, 0}``    — standard binary convention
    - ``{1, -1}``   — some older papers

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
        Label vector.  Must be 1-D.

    Returns
    -------
    np.ndarray of shape (n_samples,) and dtype int
        Labels in canonical ``{+1, 0}`` form.

    Raises
    ------
    ValidationError
        If ``y`` contains values outside the recognised sets or is not 1-D.
    """
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValidationError(f"y must be 1-D, got ndim={y.ndim}")

    unique_vals = set(np.unique(y))

    # Already canonical
    if unique_vals <= {POSITIVE_LABEL, UNLABELED_LABEL}:
        return y.astype(int)

    # {+1, -1} — remap -1 → 0
    if unique_vals <= {1.0, -1.0}:
        return np.where(y == 1, POSITIVE_LABEL, UNLABELED_LABEL).astype(int)

    # {1, 0} is numerically identical to canonical {+1, 0}; the first
    # branch above already handles both.  No separate remap needed.

    raise ValidationError(
        f"Unrecognised label values {sorted(unique_vals)}. "
        f"Expected one of: {{+1, 0}}, {{+1, -1}}, {{1, 0}}, or {{1, -1}}."
    )
