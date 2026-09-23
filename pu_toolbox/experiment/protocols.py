"""Strategy interfaces of the experiment layer.

Design notes: one ABC per variation point instead of a Bridge hierarchy —
a concrete strategy is injected into ExperimentRunner, no subclassing the
runner. DIY = implement an ABC and inject. See
docs/dev/experiment_layer.md §2.
"""

# ruff: noqa: N803

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from .bundle import DatasetPart, LabelView
from .tracking import RunTrajectory, SelectionArtifact


class Generator(ABC):
    """Labeling strategy: clean labels -> label view (+ meta).

    ``output_view`` declares which view the produced labels carry: PU
    generators emit ``"pu"``; the PN-oracle generator passes real labels
    through and declares ``"clean"``, so PA stays structurally excluded
    (``ProtocolPA`` rejects non-PU views) instead of relying on the caller.
    See docs/research/pu_survey/pn_oracle_integration.md §5 (D-A).
    """

    output_view: LabelView = "pu"

    @abstractmethod
    def generate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        c: float,
        seed: int | None,
    ) -> tuple[np.ndarray, dict]: ...


class Trainer(ABC):
    """Training strategy: one candidate run -> trajectory + checkpoint.

    ``trains_on_real_labels`` declares the label semantics ``fit`` expects:
    ``False`` (default) for PU trainers, whose objective reads label ``0`` as
    "unlabeled"; ``True`` for the PN-oracle trainers, which train supervised on
    the real labels.  The runner gates clean views on this declaration, so a PU
    trainer cannot be pointed at an oracle view (and vice versa) — a declaration
    rather than a class list, so custom trainers stay injectable.
    """

    trains_on_real_labels: bool = False

    @abstractmethod
    def fit(
        self,
        estimator: object,
        X: np.ndarray,
        y: np.ndarray,
        *,
        class_prior: float | None = None,
        val_pu: tuple[np.ndarray, np.ndarray] | None = None,
        os_or_ts: str | None = None,
    ) -> RunTrajectory: ...


def accepts_training_view(parameters: Mapping[str, Any]) -> bool:
    """Whether a ``fit`` signature can carry ``os_or_ts``.

    Protocol §2.3 gates the calibrated view on the training interface allowing
    the unlabeled-loss input to be replaced, so both the view resolver and the
    router ask this question the same way: a named parameter, or an explicit
    ``**kwargs``.
    """
    return "os_or_ts" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def route_training_view(
    kwargs: dict, parameters: Mapping[str, Any], os_or_ts: str | None, target: object
) -> None:
    """Add ``os_or_ts`` to a fit-kwargs mapping, refusing a silent drop.

    Only a calibrated (``"ts"``) request changes training, so only that value is
    routed.  A target whose ``fit`` never declares the parameter would run the OS
    view while the run's manifest claimed otherwise, so the mismatch is raised
    here rather than swallowed.  Acceptance follows ``runner._select_kwargs``: a
    named parameter or an explicit ``**kwargs``, detected by signature
    inspection, never by catching ``TypeError``.
    """
    if os_or_ts != "ts":
        return
    if not accepts_training_view(parameters):
        raise ValueError(
            f"{type(target).__name__}.fit() does not accept os_or_ts, so the "
            f"requested {os_or_ts!r} training view cannot be applied."
        )
    kwargs["os_or_ts"] = "ts"


class SelectionProtocol(ABC):
    """Model-selection protocol: trajectories + its val view -> artifact.

    ``class_prior`` is the population class prior pi (protocol §3.1), passed by
    keyword.  A protocol whose criterion needs pi (PA) MUST fail loudly when it
    is None; one whose criterion does not (OA) must ignore it, so every protocol
    in the tree shares a single call shape.  Keyword-only and optional: this is
    an addition to the interface, not a reordering of it.
    """

    @abstractmethod
    def select(
        self,
        trajectories: list[RunTrajectory],
        val_part: DatasetPart,
        threshold_candidates: np.ndarray | None = None,
        *,
        class_prior: float | None = None,
    ) -> SelectionArtifact: ...
