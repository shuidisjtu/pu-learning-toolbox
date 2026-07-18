"""PU / PNU label normalisation utilities.

Canonical internal representations::

    +1  →  labeled positive (P)
     0  →  unlabeled (U)
    -1  →  labeled negative (N)   [PNU / semi-supervised only]

This module provides label normalisation for two modes:

* :func:`normalize_pu_labels` — P/U only (``{+1, 0}``).
* :func:`normalize_pnu_labels` — P/N/U (``{+1, -1, 0}``), used by
  semi-supervised PU classifiers such as PNU.

Both are built on shared validation primitives so that future multi-group
label conventions can be added without duplicating base checks.
"""

from __future__ import annotations

import numpy as np

from .config import NEGATIVE_LABEL, POSITIVE_LABEL, UNLABELED_LABEL
from .exceptions import ValidationError

# ═════════════════════════════════════════════════════════════════════
# Shared validation primitives
# ═════════════════════════════════════════════════════════════════════


def _check_y_1d(y: np.ndarray) -> None:
    """Raise :class:`ValidationError` if *y* is not 1-D."""
    if y.ndim != 1:
        raise ValidationError(
            f"y must be 1-D, got ndim={y.ndim}."
        )


def _check_label_values(
    y: np.ndarray,
    valid_sets: list[set[float]],
    *,
    require_all: bool = False,
) -> set[float]:
    """Validate that *y* values belong to a recognised convention.

    Parameters
    ----------
    y : np.ndarray
        Label vector (already converted to float).
    valid_sets : list of set of float
        Allowed label-value sets (e.g. ``[{1.0, 0.0}, {1.0, -1.0}]``).
    require_all : bool
        If ``True``, **every** value in the matched convention must be
        present in *y* (used by P/N/U to ensure all three groups exist).

    Returns
    -------
    set of float
        The actual unique values in *y*.

    Raises
    ------
    ValidationError
        If *y* contains values outside any recognised convention, or if
        ``require_all=True`` and a required class is missing.
    """
    unique_vals = set(np.unique(y))

    for allowed in valid_sets:
        if unique_vals <= allowed:
            if require_all and not allowed <= unique_vals:
                missing = allowed - unique_vals
                raise ValidationError(
                    f"Label vector must contain all of {sorted(allowed)}; "
                    f"missing {sorted(missing)}."
                )
            return unique_vals

    # No match found — build a helpful error message.
    all_valid = sorted(set().union(*valid_sets))
    raise ValidationError(
        f"Unrecognised label values {sorted(unique_vals)}. "
        f"Expected values from one of: {valid_sets}. "
        f"All valid individual values: {all_valid}."
    )


# ═════════════════════════════════════════════════════════════════════
# PU labels (P/U only)
# ═════════════════════════════════════════════════════════════════════


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
    _check_y_1d(y)

    pu_valid_sets = [
        {float(POSITIVE_LABEL), float(UNLABELED_LABEL)},  # {+1, 0}
        {1.0, -1.0},  # {1, -1} → remap below
    ]
    unique_vals = _check_label_values(y, pu_valid_sets)

    # Already canonical
    if unique_vals <= {float(POSITIVE_LABEL), float(UNLABELED_LABEL)}:
        return y.astype(int)

    # {+1, -1} — remap -1 → 0
    if unique_vals <= {1.0, -1.0}:
        return np.where(y == 1, POSITIVE_LABEL, UNLABELED_LABEL).astype(int)

    # {1, 0} is numerically identical to canonical {+1, 0}; the first
    # branch above already handles both.  No separate remap needed.
    # Unreachable — _check_label_values already rejects all bad sets.
    raise AssertionError("unreachable")  # pragma: no cover


# ═════════════════════════════════════════════════════════════════════
# PNU labels (P/N/U — three-way)
# ═════════════════════════════════════════════════════════════════════


def normalize_pnu_labels(y: np.ndarray) -> np.ndarray:
    """Convert a P/N/U label vector to canonical ``{+1, -1, 0}`` encoding.

    Accepted input conventions
    --------------------------
    - ``{+1, -1, 0}`` — canonical form (P, N, U)
    - ``{1, -1, 0}``  — standard binary variant

    All three classes (P, N, U) **must** be present — PNU training
    requires at least one sample from each group.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
        Label vector.  Must be 1-D.

    Returns
    -------
    np.ndarray of shape (n_samples,) and dtype int
        Labels in canonical ``{+1, -1, 0}`` form.

    Raises
    ------
    ValidationError
        If *y* is not 1-D, contains unrecognised values, or is missing
        any of the three required classes.
    """
    y = np.asarray(y, dtype=float)
    _check_y_1d(y)

    pnu_valid_sets = [
        {1.0, -1.0, 0.0},
    ]
    _check_label_values(y, pnu_valid_sets, require_all=True)

    # Canonical form {+1, -1, 0} — map +1 to POSITIVE_LABEL, keep others
    return np.where(
        y == 1.0,
        POSITIVE_LABEL,
        np.where(y == -1.0, NEGATIVE_LABEL, UNLABELED_LABEL),
    ).astype(int)
