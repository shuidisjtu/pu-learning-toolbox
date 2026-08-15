"""Dependency-light upload parsing for the graphical interface."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def _read_csv_bytes(content: bytes, *, what: str) -> pd.DataFrame:
    if not content:
        raise ValueError(f"{what} file is empty.")
    frame = pd.read_csv(io.BytesIO(content))
    if frame.empty:
        raise ValueError(f"{what} file has no data rows.")
    # A fully numeric first row is consumed as column names by pandas. Keep
    # the CLI's safety contract and reject this ambiguous, headerless input.
    try:
        [float(column) for column in frame.columns]
    except (TypeError, ValueError):
        pass
    else:
        raise ValueError(f"{what} CSV needs a non-numeric header row.")
    return frame


def load_feature_data(content: bytes, filename: str) -> tuple[np.ndarray, list[str]]:
    """Load a UI upload as numeric CSV table or 4-D NCHW ``.npy`` data."""
    if filename.lower().endswith(".npy"):
        array = np.load(io.BytesIO(content), allow_pickle=False)
        if array.ndim != 4:
            raise ValueError(f"image data must be 4-D NCHW; got shape {array.shape}.")
        array = array.astype(np.float32, copy=False)
        columns = [f"channel_{index}" for index in range(array.shape[1])]
    else:
        frame = _read_csv_bytes(content, what="feature")
        try:
            array = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("feature CSV must contain only numeric values.") from exc
        columns = [str(column) for column in frame.columns]
    if not np.isfinite(array).all():
        raise ValueError("feature data contains NaN or Inf values; clean or impute it first.")
    return array, columns


def load_label_data(content: bytes, *, what: str = "labels") -> np.ndarray:
    """Load a single-column CSV label upload."""
    frame = _read_csv_bytes(content, what=what)
    if frame.shape[1] != 1:
        raise ValueError(f"{what} CSV must have exactly one column; got {frame.shape[1]}.")
    try:
        numeric = frame.iloc[:, 0].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must contain numeric labels in {{0, 1}}.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{what} contains NaN or Inf values.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{what} must contain integer labels in {{0, 1}}; decimals are invalid.")
    values = numeric.astype(int)
    invalid = sorted(set(np.unique(values)) - {0, 1})
    if invalid:
        raise ValueError(f"{what} must contain only labels {{0, 1}}; got invalid values {invalid}.")
    return values
