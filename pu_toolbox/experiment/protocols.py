"""Strategy interfaces of the experiment layer.

Design notes: one ABC per variation point instead of a Bridge hierarchy —
a concrete strategy is injected into ExperimentRunner, no subclassing the
runner. DIY = implement an ABC and inject. See
docs/dev/experiment_layer.md §2 and implementation_plan.md §1.4.
"""

# ruff: noqa: N803

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .bundle import DatasetPart
from .tracking import RunTrajectory, SelectionArtifact


class Generator(ABC):
    """Labeling strategy: clean labels -> PU label view (+ meta)."""

    @abstractmethod
    def generate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        c: float,
        seed: int | None,
    ) -> tuple[np.ndarray, dict]: ...


class Trainer(ABC):
    """Training strategy: one candidate run -> trajectory + checkpoint."""

    @abstractmethod
    def fit(
        self,
        estimator: object,
        X: np.ndarray,
        y: np.ndarray,
        *,
        class_prior: float | None = None,
        val_pu: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> RunTrajectory: ...


class SelectionProtocol(ABC):
    """Model-selection protocol: trajectories + its val view -> artifact."""

    @abstractmethod
    def select(
        self,
        trajectories: list[RunTrajectory],
        val_part: DatasetPart,
        threshold_candidates: np.ndarray | None = None,
    ) -> SelectionArtifact: ...
