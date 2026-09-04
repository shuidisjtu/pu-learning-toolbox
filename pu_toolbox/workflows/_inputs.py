# ruff: noqa: N803, N806

"""Input preparation and CV provenance for :mod:`pu_toolbox.workflows`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.config import POSITIVE_LABEL
from ..core.exceptions import PipelineError, ValidationError
from ..core.validation import validate_pu_X_y, validate_true_binary_labels


def prepare_pipeline_inputs(
    X: Any,
    y_pu: np.ndarray,
    y_true: np.ndarray | None,
    *,
    is_deep: bool,
    architecture: str,
    resolve_splitter: Callable[[Any, np.ndarray], Any],
) -> tuple[Any, np.ndarray, np.ndarray | None, Any, Any, int]:
    """Validate data once, build the splitter, and enforce fold feasibility."""
    X, y_pu = validate_pu_X_y(
        X,
        y_pu,
        allow_nd=True,
        estimator_name="PUPipeline",
    )
    if X.ndim not in (2, 4):
        raise ValidationError(f"X must be 2-D (table) or 4-D (NCHW images); got ndim={X.ndim}.")
    if X.ndim == 4 and not (is_deep and architecture == "cnn"):
        raise PipelineError(
            "4-D image inputs require an explicit deep classifier "
            "(wconpu or infomax_pu) with architecture='cnn'."
        )
    if X.ndim == 2 and architecture == "cnn":
        raise PipelineError(
            "architecture='cnn' requires 4-D NCHW image inputs; "
            "got 2-D data. Use architecture='mlp' for tables."
        )
    analysis_X = X.reshape(X.shape[0], -1) if X.ndim == 4 else X
    if y_true is not None:
        y_true = _validate_y_true(y_true, X.shape[0])

    splitter = resolve_splitter(X, y_pu)
    n_splits = resolved_n_splits(splitter, X, y_pu)
    n_pos = int((y_pu == POSITIVE_LABEL).sum())
    if n_pos < n_splits:
        raise ValidationError(
            f"n_labeled_positives ({n_pos}) < n_splits ({n_splits}). "
            "Reduce the number of CV folds or provide more labeled positives."
        )
    return X, y_pu, y_true, analysis_X, splitter, n_splits


def _validate_y_true(y_true: np.ndarray, n_samples: int) -> np.ndarray:
    y_true = np.asarray(y_true)
    if y_true.ndim != 1 or len(y_true) != n_samples:
        raise ValidationError(
            f"y_true must be 1-D with length {n_samples}; got shape {y_true.shape}."
        )
    validate_true_binary_labels(y_true, estimator_name="y_true")
    return y_true.astype(int, copy=False)


def resolved_n_splits(splitter: Any, X: Any, y_pu: np.ndarray) -> int:
    if hasattr(splitter, "get_n_splits"):
        return int(splitter.get_n_splits(X, y_pu))
    raise ValueError(
        f"cv splitter {type(splitter).__name__} must implement "
        "get_n_splits(X, y) so the pipeline can validate fold counts."
    )


def cv_provenance(splitter: Any, n_splits: int) -> dict[str, Any]:
    info: dict[str, Any] = {"n_splits": n_splits, "splitter": type(splitter).__name__}
    try:
        params = splitter.get_params()
        info["shuffle"] = params.get("shuffle")
        info["random_state"] = params.get("random_state")
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass
    return info
