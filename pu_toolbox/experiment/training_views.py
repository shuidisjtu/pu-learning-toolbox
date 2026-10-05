# ruff: noqa: N803, N806

"""The survey layer over the neutral role view: gates, manifest, legacy strings.

The roles themselves are built in :mod:`pu_toolbox.core.training_views`, which
estimators import and which therefore may not know about this layer.  What
lives here is everything that *is* survey policy or survey history:

* the ledger gate -- a calibrated view needs a method whose ledger entry says so
  (``_validate_options``), and the routing decision that reads the ledger
  (:func:`resolve_training_view`);
* the run manifest and its ``run_view`` vocabulary (``"OS"``/``"TS-compatible"``
  here, ``"os-compatible"``/``"ts-compatible"`` in a run manifest), neither of
  which the core should have to carry forward;
* the legacy ``TSOSBatchView`` surface, kept as a boundary adapter.

The dependency direction is one-way: this module imports the core, and the core
never imports this one.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..core.training_views import ROLES, RUN_VIEWS, RunView, ViewRole, build_training_view
from ..utils.serialization import canonical_hash, json_scalars
from .method_ledger import native_sampling_assumption
from .protocols import accepts_training_view

#: The method ledger's vocabulary.  It stays on this side of the boundary: the
#: core view is built from a decision already taken, and this is where the
#: decision is read and enforced.
SamplingAssumption = Literal["os", "ts", "both"]


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
    # The ledger gate stays here, not one layer down: the core knows no ledger,
    # so a caller who wanted to bypass this could assert anything to it anyway.
    # What the core must not carry is the vocabulary, not the check.
    _validate_options(
        os_or_ts=os_or_ts,
        native_sampling_assumption=native_sampling_assumption,
        role=role,
        method_name=method_name,
    )
    view = build_training_view(
        X_batch, y_pu_batch, requested_view=os_or_ts, role=role, indices=indices
    )

    positive_indices = view.positive_indices
    original_unlabeled_indices = view.native_unlabeled_indices
    loss_unlabeled_indices = view.loss_unlabeled_indices
    calibration_applied = view.calibration_applied
    # History stays at the boundary: the core carries the request, and the
    # strings this artifact has always used are derived from it here.
    run_view = "TS-compatible" if calibration_applied else "OS"

    index_payload = {
        "positive": json_scalars(positive_indices, name="indices"),
        "original_unlabeled": json_scalars(original_unlabeled_indices, name="indices"),
        "loss_unlabeled": json_scalars(loss_unlabeled_indices, name="indices"),
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
        "source_batch_size": len(view.source_features),
        "positive_count": len(view.positive_positions),
        "original_unlabeled_count": len(view.native_unlabeled_positions),
        "loss_unlabeled_count": len(view.loss_unlabeled_positions),
        "positive_rows_added_to_unlabeled_loss": (
            len(view.positive_positions) if calibration_applied else 0
        ),
        "indices": index_payload,
        "indices_sha256": canonical_hash(index_payload),
    }
    return TSOSBatchView(
        positive_features=view.positive_features,
        loss_unlabeled_features=view.loss_unlabeled_features,
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
    if os_or_ts not in RUN_VIEWS:
        raise ValueError("os_or_ts must be 'os' or 'ts'.")
    if native_sampling_assumption not in {"os", "ts", "both"}:
        raise ValueError("native_sampling_assumption must be 'os', 'ts', or 'both'.")
    if role not in ROLES:
        raise ValueError("role must be 'train', 'pu_val', 'clean_val', or 'test'.")
    if not isinstance(method_name, str) or not method_name.strip():
        raise ValueError("method_name must identify the method ledger entry.")
    if os_or_ts == "ts" and native_sampling_assumption not in {"ts", "both"}:
        raise ValueError("TS-OS calibration requires a method declared native to TS sampling.")
    if os_or_ts == "ts" and role != "train":
        raise ValueError("TS-OS calibration is train-only; validation and test remain OS.")


#: The views a versioned pilot may record.  ``os-compatible`` covers both a
#: method native to OS and one declared native to TS but not yet wired for
#: calibration -- the field is the view a run took, not the ledger's assumption.
LEGAL_RUN_VIEWS = frozenset({"os-compatible", "ts-compatible"})

#: The supervised upper bound.  It trains on real labels, so it generates no PU
#: label view -- and no calibrated (ts) view to be run under.  ``--oracle
#: --os-or-ts ts`` is already refused at the command line; this is the
#: aggregation-side half of the same rule.
ORACLE_METHOD = "pn_oracle"


def validated_run_view(payload: dict[str, Any]) -> str:
    """The view one manifest ran under, checked against its calibration flag.

    The protocol pairs the two: a calibrated run is the only one that reports
    ``ts-compatible``.  A manifest whose flag contradicts its view is malformed
    rather than a third view, so it is refused rather than sorted somewhere.
    A missing flag is refused too -- the runner writes one on every manifest it
    produces, including the rejected ones.

    The oracle is refused a calibrated view outright: it trains on real labels,
    so the unlabeled loss that calibration feeds does not exist for it.

    Raised as ``ValueError`` so each caller can decide what a malformed artifact
    means to it: aggregating stops on one, while a resume scan counts it as not
    done and keeps going.
    """
    view = payload.get("run_view")
    if view not in LEGAL_RUN_VIEWS:
        raise ValueError(
            f"run_view must be one of {', '.join(sorted(LEGAL_RUN_VIEWS))}, got {view!r}"
        )
    calibrated = payload.get("calibration_applied")
    if calibrated is not (view == "ts-compatible"):
        raise ValueError(f"run_view {view!r} disagrees with calibration_applied {calibrated!r}")
    unit = payload.get("execution_unit")
    method = unit.get("method") if isinstance(unit, dict) else None
    if view == "ts-compatible" and method == ORACLE_METHOD:
        raise ValueError(
            "the PN oracle trains on real labels and generates no PU label view, "
            "so it has no calibrated (ts) view"
        )
    return view


def resolve_training_view(
    ledger: dict[str, Any],
    method: str,
    requested: str | None,
    *,
    is_oracle: bool,
    estimator_class: type | None = None,
) -> str:
    """Resolve the run's training view: ledger default, or the explicit request.

    Protocol §2.3 gates the calibrated view on **two** conditions — the method
    ledger declaring native TS/case-control sampling, *and* the training
    interface allowing the unlabeled-loss input to be replaced.  Both are
    checked here: the ledger supplies the default, and ``--os-or-ts`` overrides
    it.  A method that is native TS but whose estimator has no ``os_or_ts`` fit
    hook stays on ``os`` rather than failing every run; only an *explicit* ``ts``
    request on such a method is refused, naming the missing interface.

    ``estimator_class`` is what makes that second condition checkable, so it is
    required wherever the answer could be ``ts``: an explicit ``ts`` request
    without one is refused rather than granted unverified.  Answering ``os``
    there instead would be worse than useless -- the operator asked for the
    calibrated view and would get a run that never took it, which is the same
    silent downgrade this module exists to prevent.  The default path keeps the
    conservative reading: a missing class resolves to ``os``, where a later
    ``ts`` run merely stays pending instead of opening a hole in the matrix.

    Shared by the unit script and the pilot driver, so the view a resumed unit
    is held to is the view that unit will actually run under.  A second copy of
    this decision is how a driver ends up expecting something the runner never
    does.
    """
    if is_oracle:
        if requested == "ts":
            raise ValueError(
                "--oracle trains on real labels and generates no PU label view, so "
                "--os-or-ts ts does not apply; drop --os-or-ts."
            )
        return "os"
    entry = ledger["methods"].get(method)
    if entry is None:
        raise ValueError(f"method {method!r} is not in the survey ledger")
    native = native_sampling_assumption(entry)
    # A fit parameter may exist solely to reject TS (e.g. PULNS); signature
    # presence alone must not advertise a risk substitution that does not exist.
    # Absent declarations retain the frozen legacy routing behavior.
    if entry.get("calibration_hooked") is False:
        if requested == "ts":
            raise ValueError(
                f"method {method!r} explicitly declares calibration_hooked=false; "
                "the calibrated (ts) training-risk substitution is not implemented."
            )
        if requested is None:
            return "os"
    if requested is None:
        if native not in {"ts", "both"} or estimator_class is None:
            return "os"
        return "ts" if accepts_training_view(_fit_parameters(estimator_class)) else "os"
    if requested == "ts":
        if native not in {"ts", "both"}:
            raise ValueError(
                f"method {method!r} is declared native to {native!r} sampling in the "
                "method ledger, so the calibrated (ts) view does not apply to it."
            )
        if estimator_class is None:
            raise ValueError(
                f"method {method!r} cannot be held to the calibrated (ts) view: no "
                "estimator class was supplied, so the second §2.3 condition -- a "
                "training interface that can replace the unlabeled-loss input -- "
                "cannot be checked."
            )
        if not accepts_training_view(_fit_parameters(estimator_class)):
            raise ValueError(
                f"method {method!r} is native to TS sampling but "
                f"{estimator_class.__name__}.fit() declares no os_or_ts parameter, "
                "so the training interface cannot replace the unlabeled-loss input."
            )
    return requested


def _fit_parameters(estimator_class: type) -> Any:
    """The ``fit`` parameters of an estimator class, for the view gate."""
    return inspect.signature(estimator_class.fit).parameters
