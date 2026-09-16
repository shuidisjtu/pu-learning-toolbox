"""Run-level records: trajectories, selection artifacts, results.

Design notes: dataclasses only — no logic — so every layer can hold and
inspect them without coupling. Checkpoint identity is ``model`` (the
fitted estimator, or the estimator restored to its internal best state).
Optional checkpoint references retain completed epoch weights separately
from the estimator's internally selected/final model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpochRecord:
    """One epoch record from a training history.

    ``epoch`` is a display label copied from the estimator's own history
    (source indexing — may be 0-based). To locate the best epoch within a
    trajectory, use the 1-based ``RunTrajectory.best_epoch`` position.
    """

    epoch: int
    metrics: dict[str, float]


@dataclass
class RunTrajectory:
    """Fitted run trace: per-epoch records plus the selected best epoch.

    ``best_epoch`` is the 1-based position inside ``epochs``
    (``epochs[best_epoch - 1]`` holds the legacy internally selected record);
    independent snapshot selection does not share this value across protocols.
    It is ``None`` when internal tracking is unavailable. ``EpochRecord.epoch`` is a
    display label only and may use a different base.
    """

    epochs: list[EpochRecord]
    model: Any
    best_epoch: int | None = None
    checkpoints: list[Any] = field(default_factory=list)
    checkpoint_owner: Any = field(default=None, repr=False, compare=False)


@dataclass
class SelectionArtifact:
    """Selection outcome: protocol + where the run was resumed from.

    ``epoch`` is a 1-based position in the selected run's ``epochs`` list
    (``epochs[epoch - 1]`` is the selected record), or ``None`` for a single
    point. ``checkpoint_index`` independently identifies the selected weights;
    legacy trajectories without snapshots retain ``best_epoch`` semantics.
    """

    protocol: str
    run_index: int
    epoch: int | None
    threshold: float | None
    metrics: dict[str, float | None]
    checkpoint_index: int | None = None


@dataclass
class RunResult:
    selections: dict[str, SelectionArtifact]
    test_metrics: dict[str, dict[str, float | str | None]]
    manifest: dict
    failures: list[dict] = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    selected_models: dict[str, Any] = field(default_factory=dict, repr=False)
