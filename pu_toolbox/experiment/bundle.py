"""Data-role contract for the four-way protocol (§2.4).

Design notes: inputs are ALWAYS clean views (real labels); the Generator
in the runner produces PU views (DatasetPart(view="pu")) so that PA
paths structurally cannot receive real labels. See
docs/research/pu_survey/implementation_plan.md §1.4 and
docs/dev/experiment_layer.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pu_toolbox.core.validation import validate_true_binary_labels

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
    for name in ("train", "pu_val", "clean_val", "test"):
        part = getattr(bundle, name)
        if part.view != "clean":
            raise ValueError(f"{name} must use view='clean' (real labels); got {part.view!r}.")
        validate_true_binary_labels(part.labels, estimator_name=f"{name}.labels")

    if bundle.test.for_selection:
        raise ValueError("test must have for_selection=False (may not be read by selection).")

    seen: dict[int, str] = {}
    for name in ("train", "pu_val", "clean_val", "test"):
        for idx in np.asarray(getattr(bundle, name).indices, dtype=object).tolist():
            if idx in seen:
                raise ValueError(f"indices overlap between {name} and {seen[idx]}: {idx}.")
            seen[idx] = name
