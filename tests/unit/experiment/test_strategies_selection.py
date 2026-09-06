# tests/unit/experiment/test_strategies_selection.py

# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA, select_threshold
from pu_toolbox.experiment.tracking import EpochRecord, RunTrajectory

pytestmark = pytest.mark.unit


class FakeModel:
    def decision_function(self, X):
        return X[:, 0]


def _traj(value):
    return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=value)


def test_select_threshold_picks_best_accuracy():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.0, -0.1, -0.2, -0.3])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    best_thr, best_acc = select_threshold(scores, labels, np.array([0.2, 0.5, 0.9]))
    assert best_thr == 0.2  # fully separable below 0.9; tie at 0.2/0.5 -> lowest picks first
    assert best_acc == 1.0


def test_basic_protocoloa_uses_real_labels():
    X_val = np.array([[0.1], [0.9], [-0.3]])
    val_part = DatasetPart(X=X_val, labels=np.array([0, 1, 0]), view="clean", indices=np.arange(3))
    traj = _traj(FakeModel())
    art = ProtocolOA().select([traj], val_part)
    assert art.protocol == "OA"
    assert art.threshold is not None
    assert art.metrics["val_accuracy"] >= 0.5


def test_protocolpa_marks_pu_view_usage():
    X_val = np.array([[0.1], [0.9], [-0.3]])
    val_part = DatasetPart(X=X_val, labels=np.array([1, 0, 0]), view="pu", indices=np.arange(3))
    traj = _traj(FakeModel())
    art = ProtocolPA().select([traj], val_part)
    assert art.protocol == "PA"


def test_param_protocolpa_rejects_clean_view():
    val_part = DatasetPart(
        X=np.array([[0.1]]), labels=np.array([1]), view="clean", indices=np.arange(1)
    )
    traj = _traj(FakeModel())
    with pytest.raises(ValueError):
        ProtocolPA().select([traj], val_part)


def test_edge_protocoloa_empty_trajectories():
    val_part = DatasetPart(
        X=np.array([[0.1], [0.9], [-0.3]]),
        labels=np.array([0, 1, 0]),
        view="clean",
        indices=np.arange(3),
    )
    with pytest.raises(ValueError, match="at least one trajectory"):
        ProtocolOA().select([], val_part)
