"""Run-level records: trajectories, selection artifacts, results.

Design notes: dataclasses only — no logic — so every layer can hold and
inspect them without coupling. Checkpoint identity is ``model`` (the
fitted estimator, or the estimator restored to its internal best state).
Per-epoch snapshots are intentionally out of P0 scope (see spec §9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpochRecord:
    epoch: int
    metrics: dict[str, float]


@dataclass
class RunTrajectory:
    epochs: list[EpochRecord]
    model: Any
    best_epoch: int | None = None


@dataclass
class SelectionArtifact:
    protocol: str
    run_index: int
    epoch: int | None
    threshold: float | None
    metrics: dict[str, float]


@dataclass
class RunResult:
    selections: dict[str, SelectionArtifact]
    test_metrics: dict[str, dict[str, float]]
    manifest: dict
    failures: list[dict] = field(default_factory=list)
