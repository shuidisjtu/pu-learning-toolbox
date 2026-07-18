"""PU / PNU data validation routines.

Every public estimator should call the appropriate ``validate_*_X_y``
helper at the beginning of ``fit`` and ``fit_predict`` to ensure
consistent input validation and early detection of data problems.

Shared internal helpers (:func:`_validate_X_common`, etc.) factor out
checks that are common to all label conventions so that future
multi-group validators stay lean.
"""

# ruff: noqa: N802, N803, N806

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse

from .config import MAX_PU_RATIO, MIN_POSITIVE_SAMPLES, NEGATIVE_LABEL
from .exceptions import ValidationError
from .labels import normalize_pnu_labels, normalize_pu_labels

# ═════════════════════════════════════════════════════════════════════
# Shared X-validation helpers
# ═════════════════════════════════════════════════════════════════════


def _validate_X_common(
    X: np.ndarray | sparse.spmatrix,
    n_samples_y: int,
    *,
    accept_sparse: bool,
    allow_nd: bool,
    estimator_name: str,
    label_name: str = "y",
) -> None:
    """Base X checks shared by all ``validate_*_X_y`` functions.

    Parameters
    ----------
    X : np.ndarray or sparse matrix
        Feature matrix.
    n_samples_y : int
        Number of samples in the corresponding label vector.
    accept_sparse : bool
        If False, reject sparse matrices.
    allow_nd : bool
        If True, allow ``X.ndim > 2``.
    estimator_name : str
        Included in error messages for traceability.
    label_name : str
        Human-readable label parameter name for error messages
        (e.g. ``"y_pu"`` or ``"y_pnu"``).

    Raises
    ------
    ValidationError
    """
    # ndim check must come FIRST so 1-D arrays don't report misleading
    # sample-count mismatch (X.shape[0] on a 1-D array = n_features).
    if not allow_nd and (not sparse.issparse(X) and X.ndim != 2):
        raise ValidationError(
            f"[{estimator_name}] Expected X to be 2-D; got ndim={X.ndim}. "
            "Use allow_nd=True if higher-order tensors are intentional."
        )

    n_samples_x = X.shape[0]
    if n_samples_x != n_samples_y:
        raise ValidationError(
            f"[{estimator_name}] X has {n_samples_x} samples "
            f"but {label_name} has {n_samples_y}."
        )

    if not accept_sparse and sparse.issparse(X):
        raise ValidationError(
            f"[{estimator_name}] Sparse input is not supported "
            "for this estimator."
        )


# ═════════════════════════════════════════════════════════════════════
# Parameter validation helpers
# ═════════════════════════════════════════════════════════════════════


def check_scalar_in_range(
    value: float,
    low: float,
    high: float,
    name: str,
    *,
    inclusive: bool = True,
) -> None:
    """Raise ``ValueError`` if *value* is outside ``(low, high)``.

    Parameters
    ----------
    value : float
        The scalar value to check.
    low : float
        Lower bound.
    high : float
        Upper bound.
    name : str
        Human-readable parameter name for the error message.
    inclusive : bool
        If ``True`` (default), bounds are closed ``[low, high]``;
        otherwise open ``(low, high)``.

    Raises
    ------
    ValueError
    """
    valid = low <= value <= high if inclusive else low < value < high
    if not valid:
        brack = ("[", "]") if inclusive else ("(", ")")
        raise ValueError(
            f"{name} must be in {brack[0]}{low}, {high}{brack[1]}; "
            f"got {value}."
        )


def check_positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    """Raise ``ValueError`` if *value* is not strictly positive.

    Parameters
    ----------
    value : float
    name : str
    allow_zero : bool
        If ``True``, accept ``value >= 0``.

    Raises
    ------
    ValueError
    """
    cmp = value >= 0.0 if allow_zero else value > 0.0
    if not cmp:
        qual = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {qual}; got {value}.")


# ═════════════════════════════════════════════════════════════════════
# PU data validation (P/U only)
# ═════════════════════════════════════════════════════════════════════


def validate_pu_X_y(
    X: np.ndarray | sparse.spmatrix,
    y_pu: np.ndarray,
    *,
    accept_sparse: bool = True,
    allow_nd: bool = False,
    estimator_name: str | None = None,
) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray]:
    """Validate and canonicalise PU training data ``(X, y_pu)``.

    Checks performed
    ----------------
    1. ``y_pu`` is 1-D and contains only recognised PU label values.
    2. ``y_pu`` contains at least :data:`~pu_toolbox.core.config.MIN_POSITIVE_SAMPLES`
       labeled positives.
    3. ``X`` and ``y_pu`` have the same number of rows.
    4. ``X`` is a dense ``ndarray`` (or sparse matrix when ``accept_sparse=True``).
    5. Warns if the unlabeled-to-positive ratio exceeds
       :data:`~pu_toolbox.core.config.MAX_PU_RATIO`.

    Parameters
    ----------
    X : np.ndarray or sparse matrix of shape (n_samples, n_features)
        Feature matrix.
    y_pu : np.ndarray of shape (n_samples,)
        PU labels (any format accepted by :func:`normalize_pu_labels`).
    accept_sparse : bool
        If True (default), allow scipy sparse matrices.
    allow_nd : bool
        If True, allow ``X.ndim > 2``.  Default False.
    estimator_name : str or None
        Optional name included in error messages for better traceability.

    Returns
    -------
    X : np.ndarray or sparse matrix
        Feature matrix (may be converted to ndarray).
    y_pu : np.ndarray
        Canonical PU labels ``{+1, 0}``.

    Raises
    ------
    ValidationError
        If any validation check fails.
    """
    est = estimator_name or "?"
    y_pu = normalize_pu_labels(y_pu)
    n_positive = int(np.sum(y_pu == 1))

    if n_positive < MIN_POSITIVE_SAMPLES:
        raise ValidationError(
            f"[{est}] Need at least {MIN_POSITIVE_SAMPLES} labeled "
            f"positives; got {n_positive}."
        )

    _validate_X_common(
        X, y_pu.shape[0], accept_sparse=accept_sparse, allow_nd=allow_nd,
        estimator_name=est, label_name="y_pu",
    )

    # ── Ratio warning ──────────────────────────────────────────────
    n_unlabeled = y_pu.shape[0] - n_positive
    if n_unlabeled / max(n_positive, 1) > MAX_PU_RATIO:
        warnings.warn(
            f"[{est}] Unlabeled-to-positive ratio "
            f"({n_unlabeled / n_positive:.1f}:1) exceeds {MAX_PU_RATIO}:1. "
            "Results may be unstable.",
            UserWarning,
            stacklevel=2,
        )

    return X, y_pu


# ═════════════════════════════════════════════════════════════════════
# PNU data validation (P/N/U — three-way)
# ═════════════════════════════════════════════════════════════════════


def validate_pnu_X_y(
    X: np.ndarray | sparse.spmatrix,
    y_pnu: np.ndarray,
    *,
    accept_sparse: bool = True,
    allow_nd: bool = False,
    estimator_name: str | None = None,
) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray]:
    """Validate and canonicalise PNU training data ``(X, y_pnu)``.

    Checks performed
    ----------------
    1. ``y_pnu`` contains all three label values ``{+1, -1, 0}``
       (P, N, U), with at least one sample in each group.
    2. ``X`` and ``y_pnu`` have the same number of rows.
    3. ``X`` is a dense ``ndarray`` (or sparse matrix when
       ``accept_sparse=True``).

    Parameters
    ----------
    X : np.ndarray or sparse matrix of shape (n_samples, n_features)
        Feature matrix.
    y_pnu : np.ndarray of shape (n_samples,)
        P/N/U labels in ``{+1, -1, 0}`` or ``{1, -1, 0}`` format.
    accept_sparse : bool
        If True (default), allow scipy sparse matrices.
    allow_nd : bool
        If True, allow ``X.ndim > 2``.  Default False.
    estimator_name : str or None
        Optional name included in error messages.

    Returns
    -------
    X : np.ndarray or sparse matrix
    y_pnu : np.ndarray
        Canonical P/N/U labels ``{+1, -1, 0}``.

    Raises
    ------
    ValidationError
        If any validation check fails (missing class, shape mismatch,
        unrecognised label values).
    """
    est = estimator_name or "?"
    y_pnu = normalize_pnu_labels(y_pnu)
    n_P = int(np.sum(y_pnu == 1))
    n_N = int(np.sum(y_pnu == NEGATIVE_LABEL))
    n_U = int(np.sum(y_pnu == 0))

    # Each group must be non-empty (enforced by normalize_pnu_labels'
    # require_all=True, but we double-check for extra safety).
    if n_P == 0:
        raise ValidationError(
            f"[{est}] Need at least 1 positive sample (label = +1)."
        )
    if n_N == 0:
        raise ValidationError(
            f"[{est}] Need at least 1 negative sample (label = -1)."
        )
    if n_U == 0:
        raise ValidationError(
            f"[{est}] Need at least 1 unlabeled sample (label = 0)."
        )

    _validate_X_common(
        X, y_pnu.shape[0], accept_sparse=accept_sparse, allow_nd=allow_nd,
        estimator_name=est, label_name="y_pnu",
    )

    # ── Imbalance warning ───────────────────────────────────────
    min_labeled = min(max(n_P, 1), max(n_N, 1))
    if n_U / min_labeled > MAX_PU_RATIO:
        warnings.warn(
            f"[{est}] Unlabeled-to-labeled ratio "
            f"({n_U / min_labeled:.1f}:1) exceeds {MAX_PU_RATIO}:1. "
            "Results may be unstable with such few labeled samples.",
            UserWarning,
            stacklevel=2,
        )

    return X, y_pnu
