"""Data-role contract for the four-way protocol (§2.4).

Design notes: inputs are ALWAYS clean views (real labels); the Generator
in the runner produces PU views (DatasetPart(view="pu")) so that PA
paths structurally cannot receive real labels. See
docs/dev/experiment_layer.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from pu_toolbox.core.training_views import ROLES
from pu_toolbox.core.validation import validate_true_binary_labels
from pu_toolbox.utils.serialization import json_scalars, strict_canonical_hash

LabelView = Literal["pu", "clean"]

__all__ = ["DatasetPart", "DatasetBundle", "validate_bundle"]


@dataclass(frozen=True)
class DatasetPart:
    """One dataset partition with an explicit label view.

    Parameters
    ----------
    X : (n_samples, n_features) or (n, c, h, w) array
    labels : (n_samples,) int array, interpreted per ``view``
    view : "clean" (real labels {+1,0}) or "pu" (PU labels {+1,0})
    indices : global sample identifiers from the user's split manifest
    for_selection : False is mandatory for ``test``
    """

    X: np.ndarray
    labels: np.ndarray
    view: LabelView
    indices: np.ndarray
    for_selection: bool = True


@dataclass(frozen=True)
class DatasetBundle:
    """Four-way data bundle: train, pu_val, clean_val, test (all clean views)."""

    train: DatasetPart
    pu_val: DatasetPart
    clean_val: DatasetPart
    test: DatasetPart


def validate_bundle(bundle: DatasetBundle) -> None:
    """Fail loudly if the four-way contract is violated.

    Raises
    ------
    ValueError
        On wrong view, invalid labels, overlapping indices, or
        ``test.for_selection`` not False.
    """
    for name in ROLES:
        part = getattr(bundle, name)
        if part.view != "clean":
            raise ValueError(f"{name} must use view='clean' (real labels); got {part.view!r}.")
        validate_true_binary_labels(part.labels, estimator_name=f"{name}.labels")

    if bundle.test.for_selection:
        raise ValueError("test must have for_selection=False (may not be read by selection).")

    _validate_role_indices({role: getattr(bundle, role).indices for role in ROLES})


def _validate_role_indices(indices: dict) -> None:
    """Require one-dimensional, disjoint sample identifiers for every role."""
    seen: dict[object, str] = {}
    for name, values in indices.items():
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} indices must be one-dimensional")
        for idx in array.tolist():
            if idx in seen:
                raise ValueError(f"indices overlap between {name} and {seen[idx]}: {idx}.")
            seen[idx] = name


def role_indices_digest(indices: dict) -> str:
    """Name the four-role split using the manifest's established hash contract."""
    return strict_canonical_hash(
        {role: json_scalars(np.asarray(indices[role]), name=f"{role} indices") for role in ROLES}
    )


def split_indices_digest(data_dir: str | Path, declared=None, *, include_support=False) -> str:
    """Hash actual sample identifiers, validating any supplied receipt claim."""
    roles = (*ROLES, "support") if include_support else ROLES
    indices = {}
    for role in roles:
        path = Path(data_dir) / f"{role}.npz"
        if not path.is_file():
            raise ValueError(f"missing split files for index identity: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if "indices" not in archive.files:
                raise ValueError(f"split array has no sample indices: {path}")
            indices[role] = archive["indices"]
    _validate_role_indices(indices)
    actual = strict_canonical_hash(
        {role: json_scalars(indices[role], name=f"{role} indices") for role in roles}
    )
    if declared is not None and declared != actual:
        raise ValueError("split manifest indices_sha256 disagrees with the actual sample indices")
    return role_indices_digest(indices)
