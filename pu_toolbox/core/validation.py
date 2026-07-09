"""PU data validation routines.

Every public estimator should call :func:`validate_pu_X_y` at the
beginning of ``fit`` and ``fit_predict`` to ensure consistent input
early detection of data problems.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse

from .config import MAX_PU_RATIO, MIN_POSITIVE_SAMPLES
from .exceptions import ValidationError
from .labels import normalize_pu_labels


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
    prefix = f"[{estimator_name}] " if estimator_name else ""

    # ── y_pu checks ────────────────────────────────────────────────
    y_pu = normalize_pu_labels(y_pu)
    n_positive = int(np.sum(y_pu == 1))

    if n_positive < MIN_POSITIVE_SAMPLES:
        raise ValidationError(
            f"{prefix}Need at least {MIN_POSITIVE_SAMPLES} labeled positives; "
            f"got {n_positive}."
        )

    # ── X checks ───────────────────────────────────────────────────
    n_samples_y = y_pu.shape[0]

    # ndim check must come FIRST so 1-D arrays don't report misleading
    # sample-count mismatch (X.shape[0] on a 1-D array = n_features).
    if not allow_nd and (not sparse.issparse(X) and X.ndim != 2):
        raise ValidationError(
            f"{prefix}Expected X to be 2-D; got ndim={X.ndim}. "
            "Use allow_nd=True if higher-order tensors are intentional."
        )

    n_samples_x = X.shape[0]

    if n_samples_x != n_samples_y:
        raise ValidationError(
            f"{prefix}X has {n_samples_x} samples but y_pu has {n_samples_y}."
        )

    if not accept_sparse and sparse.issparse(X):
        raise ValidationError(
            f"{prefix}Sparse input is not supported for this estimator."
        )

    # ── Ratio warning ──────────────────────────────────────────────
    n_unlabeled = n_samples_y - n_positive
    if n_unlabeled / max(n_positive, 1) > MAX_PU_RATIO:
        warnings.warn(
            f"{prefix}Unlabeled-to-positive ratio "
            f"({n_unlabeled / n_positive:.1f}:1) exceeds {MAX_PU_RATIO}:1. "
            "Results may be unstable.",
            UserWarning,
            stacklevel=2,
        )

    return X, y_pu
