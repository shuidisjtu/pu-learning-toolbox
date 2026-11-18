"""Shared feature and label readers for file-based CLI commands."""

from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: Path, what: str) -> pd.DataFrame:
    """Read a CSV, rejecting headerless files instead of dropping a row.

    ``pd.read_csv(header=0)`` treats the first row as column names, so a
    headerless numeric CSV silently loses its first sample.  A genuine
    header row (column names) is non-numeric; when the first row is all
    numeric we cannot tell it apart from data and fail with a clear error.
    """
    if not path.exists():
        raise FileNotFoundError(f"{what} file not found: {path}")
    try:
        raw = pd.read_csv(path, header=None)
    except pd.errors.ParserError as exc:
        raise ValueError(f"cannot parse CSV {path}: {exc}") from exc
    try:
        raw.iloc[0].astype(float)
    except (ValueError, TypeError):
        # Non-numeric first row -> a real header; read with header=0.
        return pd.read_csv(path, header=0)
    raise ValueError(
        f"{what} CSV {path}: the first row is numeric, so it cannot be "
        "distinguished from data. Add a header row (non-numeric column "
        "names) -- otherwise the first sample is consumed as column names."
    )


def _load_features(path: Path) -> np.ndarray:
    """Read a feature matrix: CSV table (2-D) or .npy image array (4-D NCHW)."""
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if not isinstance(array, np.ndarray):
            # np.load on a .npz archive returns an NpzFile (no .ndim),
            # which would surface as a bare AttributeError outside the
            # CLI's error mapping.
            raise ValueError(
                f"image data file {path} must be a .npy array; got "
                f"{type(array).__name__}. Only .npy (not .npz) archives are supported."
            )
        if array.ndim != 4:
            raise ValueError(
                f"image data file {path} must be a 4-D NCHW float array; "
                f"got ndim={array.ndim}. Table data must use CSV."
            )
        array = array.astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"image data file {path} contains NaN or Inf values.")
        return array
    array = _read_csv(path, "data").to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(
            f"data CSV {path} contains NaN or Inf values. Impute or remove them before running."
        )
    return array


def _load_label_column(path: Path, what: str) -> np.ndarray:
    """Read a single-column labels CSV; header (if any) is tolerated."""
    df = _read_csv(path, what)
    if df.shape[1] != 1:
        raise ValueError(f"{what} CSV {path} must be a single column; got {df.shape[1]} columns.")
    return df.iloc[:, 0].to_numpy(dtype=float)
