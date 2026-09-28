# ruff: noqa: N803, N806

"""Which rows play which role in one training partition.

Three roles are built from one OS-labelled partition:

* ``positive``            -- the labeled positives, ``D_P``;
* ``native_unlabeled``    -- the unlabeled rows as they arrived, ``D_U``;
* ``loss_unlabeled``      -- what the unlabeled risk term acts on: ``D_U``
  under the OS view, ``D_U ∪ D_P`` under the calibrated (ts) view.

A positive row is deliberately in two roles at once under the calibrated view.
That reuse is the calibration (case-control sampling puts the labeled positives
back into the population the unlabeled risk is estimated over), and it cannot
be simulated by rewriting the positives as unlabeled -- the positive term still
needs them as positives.

This module is **neutral**: it knows no method ledger, no method name and no
manifest, because estimators import it and estimators may not import the survey
experiment layer (see ``training_views`` in ``pu_toolbox.experiment`` for the
ledger gate, the survey manifest and the legacy batch view).  Which view a
method may take is decided there, by the ledger and the training interface; a
view built here is a decision already taken, not a capability claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

#: The view a caller has already decided on.  Lowercase, and the only
#: vocabulary this module knows: the survey layer maps it to its own strings
#: (``"OS"``/``"TS-compatible"`` in the batch view, ``"os-compatible"``/
#: ``"ts-compatible"`` in a run manifest) at the boundary, so history does not
#: reach the callers that never belonged to it.
RunView = Literal["os", "ts"]

#: The protocol's partition roles.  Calibration applies to the train role only;
#: a validation or test partition is never the place a population is restored.
ViewRole = Literal["train", "pu_val", "clean_val", "test"]

_LEGAL_VIEWS = ("os", "ts")
_LEGAL_ROLES = ("train", "pu_val", "clean_val", "test")


def _owned(values: np.ndarray) -> np.ndarray:
    """A copy this view owns and nobody writes through.

    ``@dataclass(frozen=True)`` stops an attribute being rebound; it does not
    make the array behind it read-only.  The positions and indices are the
    view's own findings, small, and what every downstream judgement is keyed
    on, so they are copied and closed.  The bulk arrays are borrowed instead --
    see :func:`build_training_view`.
    """
    owned = np.array(values, copy=True)
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True)
class TrainingView:
    """One partition's three roles, as positions into the source arrays.

    Positions rather than materialized arrays: every row's origin survives, a
    consumer that needs contiguous arrays can take them (``positive_features``
    and friends), and nothing forces a second copy of a whole training set.

    Ownership is split on purpose.  ``source_features`` and ``source_labels``
    are the caller's arrays, borrowed, and must not be written while the view
    is alive -- nor may the view outlive a step that rewrites the partition it
    was built from (see :func:`build_training_view`).  ``source_indices`` and
    the three position arrays belong to the view and are read-only.

    Every role is non-empty by construction: this object describes a training
    partition, where both the positive and the unlabeled term have to exist.
    """

    source_features: np.ndarray
    source_labels: np.ndarray
    source_indices: np.ndarray

    positive_positions: np.ndarray
    native_unlabeled_positions: np.ndarray
    loss_unlabeled_positions: np.ndarray

    requested_view: RunView
    role: ViewRole
    calibration_applied: bool

    @property
    def positive_features(self) -> np.ndarray:
        return self.source_features[self.positive_positions]

    @property
    def native_unlabeled_features(self) -> np.ndarray:
        return self.source_features[self.native_unlabeled_positions]

    @property
    def loss_unlabeled_features(self) -> np.ndarray:
        return self.source_features[self.loss_unlabeled_positions]

    @property
    def positive_indices(self) -> np.ndarray:
        return self.source_indices[self.positive_positions]

    @property
    def native_unlabeled_indices(self) -> np.ndarray:
        return self.source_indices[self.native_unlabeled_positions]

    @property
    def loss_unlabeled_indices(self) -> np.ndarray:
        return self.source_indices[self.loss_unlabeled_positions]


def build_training_view(
    X: np.ndarray,
    y_pu: np.ndarray,
    *,
    requested_view: RunView = "os",
    role: ViewRole = "train",
    indices: np.ndarray | None = None,
) -> TrainingView:
    """Build the three roles for one partition, under an already-decided view.

    ``requested_view`` is the answer, not the question: whether a method may be
    held to the calibrated view is settled by the ledger and the estimator's
    training interface before this is called.  Handing this function a raw
    capability flag would put survey policy in the layer estimators depend on,
    and would not enforce anything a caller could not assert anyway.

    ``indices`` names the rows in whatever space the caller cares about (a
    position in a larger matrix, a row id).  Omitted, the rows are numbered
    ``0..n-1``.  Either way the view copies them before closing them, so a
    caller's array is never frozen out from under it.

    Both groups must be non-empty, and for **every** role rather than only for
    calibration.  A partition missing either group describes no PU risk -- the
    positive term or the unlabeled term would have nothing to act on -- and
    ``loss_unlabeled`` only means something while training.  The non-train roles
    appear in the signature so that a calibrated request for one can be refused
    by name (below), not because this object builds evaluation views: a caller
    wanting to hold a single-class test partition is not the caller this
    serves, and it is refused here rather than silently handed a view with an
    empty role.

    The view **borrows** ``X`` and ``y_pu`` (see the class docstring), so it
    describes the partition as it was at this moment.  Keep it for the training
    step that asked for it; do not hold it across a reordering, an in-place
    normalisation or a relabelling, any of which would leave ``source_labels``
    disagreeing with the positions derived from it.  The positions and indices
    are the view's own copies and stay consistent regardless.
    """
    if requested_view not in _LEGAL_VIEWS:
        raise ValueError(f"requested_view must be one of {_LEGAL_VIEWS}, got {requested_view!r}.")
    if role not in _LEGAL_ROLES:
        raise ValueError(f"role must be one of {_LEGAL_ROLES}, got {role!r}.")
    calibration_applied = requested_view == "ts"
    if calibration_applied and role != "train":
        raise ValueError(
            f"a calibrated (ts) view applies to the train role only; got role={role!r}. "
            "Validation and test partitions keep the protocol-defined view."
        )

    X_values = np.asarray(X)
    labels = np.asarray(y_pu)
    if X_values.ndim < 2 or X_values.shape[0] == 0:
        raise ValueError("a training view expects a non-empty batch-first feature array.")
    if labels.ndim != 1 or len(labels) != len(X_values):
        raise ValueError("y_pu must be one-dimensional and align with X.")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("y_pu must use canonical PU labels {0, 1}.")

    positive_mask = labels == 1
    unlabeled_mask = labels == 0
    if not np.any(positive_mask) or not np.any(unlabeled_mask):
        raise ValueError(
            "a training view requires both labeled-positive and unlabeled rows: "
            "the positive term and the unlabeled risk term are not optional."
        )

    source_indices = np.arange(len(X_values)) if indices is None else np.asarray(indices)
    if source_indices.ndim != 1 or len(source_indices) != len(X_values):
        raise ValueError("indices must be one-dimensional and align with X.")
    if len(set(source_indices.tolist())) != len(source_indices):
        raise ValueError("indices must be unique within the source partition.")

    positive_positions = _owned(np.flatnonzero(positive_mask))
    native_unlabeled_positions = _owned(np.flatnonzero(unlabeled_mask))
    # Original unlabeled first, then the positives: the order is part of the
    # contract, so two consumers comparing views compare the same rows.
    loss_unlabeled_positions = (
        _owned(np.concatenate((native_unlabeled_positions, positive_positions)))
        if calibration_applied
        else native_unlabeled_positions
    )

    return TrainingView(
        source_features=X_values,
        source_labels=labels,
        source_indices=_owned(source_indices),
        positive_positions=positive_positions,
        native_unlabeled_positions=native_unlabeled_positions,
        loss_unlabeled_positions=loss_unlabeled_positions,
        requested_view=requested_view,
        role=role,
        calibration_applied=calibration_applied,
    )
