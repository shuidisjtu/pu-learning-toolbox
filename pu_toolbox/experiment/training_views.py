# ruff: noqa: N803, N806

"""Mini-batch OS and TS-compatible training views for the survey protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

SamplingAssumption = Literal["os", "ts", "both"]
RunView = Literal["os", "ts"]
ViewRole = Literal["train", "pu_val", "clean_val", "test"]


@dataclass(frozen=True)
class TSOSBatchView:
    """Positive and loss-unlabeled inputs derived from one OS mini-batch."""

    positive_features: np.ndarray
    loss_unlabeled_features: np.ndarray
    positive_indices: np.ndarray
    loss_unlabeled_indices: np.ndarray
    run_view: str
    calibration_applied: bool
    manifest: dict[str, Any]


def calibrate_ts_os_batch(
    X_batch: np.ndarray,
    y_pu_batch: np.ndarray,
    *,
    os_or_ts: RunView = "os",
    native_sampling_assumption: SamplingAssumption,
    role: ViewRole = "train",
    method_name: str,
    indices: np.ndarray | None = None,
) -> TSOSBatchView:
    """Create the loss inputs for an OS or TS-compatible mini-batch.

    For ``os_or_ts="ts"``, the original OS unlabeled subset remains in the
    unlabeled-loss input and the labeled-positive subset is appended to it.
    The positive subset is returned independently and therefore still
    contributes to the positive loss. No resampling or persistent TS dataset
    is created.

    TS calibration is allowed only for the train role and for a method whose
    experiment ledger explicitly declares a native ``"ts"`` (or ``"both"``)
    sampling assumption.
    """
    _validate_options(
        os_or_ts=os_or_ts,
        native_sampling_assumption=native_sampling_assumption,
        role=role,
        method_name=method_name,
    )
    X_values = np.asarray(X_batch)
    labels = np.asarray(y_pu_batch)
    if X_values.ndim < 2 or X_values.shape[0] == 0:
        raise ValueError("TS-OS calibration expects a non-empty batch-first feature array.")
    if labels.ndim != 1 or len(labels) != len(X_values):
        raise ValueError("y_pu_batch must be one-dimensional and align with X_batch.")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("y_pu_batch must use canonical PU labels {0, 1}.")

    source_indices = np.arange(len(X_values)) if indices is None else np.asarray(indices)
    if source_indices.ndim != 1 or len(source_indices) != len(X_values):
        raise ValueError("indices must be one-dimensional and align with X_batch.")
    if len(set(source_indices.tolist())) != len(source_indices):
        raise ValueError("indices must be unique within the source OS mini-batch.")

    positive_mask = labels == 1
    unlabeled_mask = labels == 0
    if not np.any(positive_mask) or not np.any(unlabeled_mask):
        raise ValueError(
            "TS-OS calibration requires both labeled-positive and unlabeled rows in every batch."
        )
    positive_X = X_values[positive_mask]
    original_unlabeled_X = X_values[unlabeled_mask]
    positive_indices = source_indices[positive_mask]
    original_unlabeled_indices = source_indices[unlabeled_mask]

    calibration_applied = os_or_ts == "ts"
    if calibration_applied:
        loss_unlabeled_X = np.concatenate((original_unlabeled_X, positive_X), axis=0)
        loss_unlabeled_indices = np.concatenate(
            (original_unlabeled_indices, positive_indices), axis=0
        )
        run_view = "TS-compatible"
    else:
        loss_unlabeled_X = original_unlabeled_X
        loss_unlabeled_indices = original_unlabeled_indices
        run_view = "OS"

    index_payload = {
        "positive": _json_indices(positive_indices),
        "original_unlabeled": _json_indices(original_unlabeled_indices),
        "loss_unlabeled": _json_indices(loss_unlabeled_indices),
    }
    manifest = {
        "schema_version": "1.0",
        "method": method_name,
        "native_sampling_assumption": native_sampling_assumption,
        "requested_view": os_or_ts,
        "run_view": run_view,
        "calibration_applied": calibration_applied,
        "calibration": ("D_U_batch <- D_U_batch union D_P_batch" if calibration_applied else None),
        "role": role,
        "source_batch_size": len(X_values),
        "positive_count": len(positive_X),
        "original_unlabeled_count": len(original_unlabeled_X),
        "loss_unlabeled_count": len(loss_unlabeled_X),
        "positive_rows_added_to_unlabeled_loss": (len(positive_X) if calibration_applied else 0),
        "indices": index_payload,
        "indices_sha256": hashlib.sha256(
            json.dumps(index_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return TSOSBatchView(
        positive_features=positive_X,
        loss_unlabeled_features=loss_unlabeled_X,
        positive_indices=positive_indices,
        loss_unlabeled_indices=loss_unlabeled_indices,
        run_view=run_view,
        calibration_applied=calibration_applied,
        manifest=manifest,
    )


def _validate_options(
    *,
    os_or_ts: str,
    native_sampling_assumption: str,
    role: str,
    method_name: str,
) -> None:
    if os_or_ts not in {"os", "ts"}:
        raise ValueError("os_or_ts must be 'os' or 'ts'.")
    if native_sampling_assumption not in {"os", "ts", "both"}:
        raise ValueError("native_sampling_assumption must be 'os', 'ts', or 'both'.")
    if role not in {"train", "pu_val", "clean_val", "test"}:
        raise ValueError("role must be 'train', 'pu_val', 'clean_val', or 'test'.")
    if not isinstance(method_name, str) or not method_name.strip():
        raise ValueError("method_name must identify the method ledger entry.")
    if os_or_ts == "ts" and native_sampling_assumption not in {"ts", "both"}:
        raise ValueError("TS-OS calibration requires a method declared native to TS sampling.")
    if os_or_ts == "ts" and role != "train":
        raise ValueError("TS-OS calibration is train-only; validation and test remain OS.")


def _json_indices(indices: np.ndarray) -> list[int | float | str | bool | None]:
    values = []
    for value in indices.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(value, bool | int | float | str):
            raise ValueError("indices must contain JSON scalar values.")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("floating-point indices must be finite.")
        values.append(value)
    return values
