# tests/unit/experiment/test_trainers.py

# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.experiment.strategies import DeepFitTrainer, FitTrainer, SupervisedTrainer

pytestmark = pytest.mark.unit


def test_fit_trainer_single_point():
    X = np.random.RandomState(0).randn(30, 3)
    y = np.array([1] * 6 + [0] * 24)
    est = UPUClassifier(0.3, random_state=0)
    traj = FitTrainer().fit(est, X, y, class_prior=0.3)
    assert len(traj.epochs) == 1
    assert traj.best_epoch is None
    assert traj.model is est  # fitted in place


def test_fit_trainer_class_prior_fallback():
    # Estimator whose fit rejects class_prior kwarg -> plain fit.
    X = np.random.RandomState(0).randn(30, 3)
    y = np.array([1] * 6 + [0] * 24)
    est = UPUClassifier(0.5, random_state=0)

    class NoPriorFit:
        def fit(self, X, y):
            return est.fit(X, y)

    est_dup = NoPriorFit()
    traj = FitTrainer().fit(est_dup, X, y, class_prior=0.5)
    assert len(traj.epochs) == 1


def test_supervised_trainer_uses_true_labels():
    X = np.random.RandomState(0).randn(30, 3)
    y = np.array([1] * 6 + [0] * 24)
    est = UPUClassifier(0.5, random_state=0)
    traj = SupervisedTrainer().fit(est, X, y)
    assert len(traj.epochs) >= 1


def test_deep_trainer_reads_history_after_nnpu_fix():
    X = np.random.RandomState(0).randn(12, 2).astype("float32")
    y_pu = np.array([1, 1, 0] * 4)
    est = NonNegativePUClassifier(
        class_prior=0.3, max_epochs=3, patience=2, random_state=0, device="cpu"
    )
    traj = DeepFitTrainer().fit(est, X, y_pu, class_prior=0.3, val_pu=(X, y_pu))
    assert len(traj.epochs) == 3
    assert "val_risk" in traj.epochs[0].metrics


def test_deep_trainer_reads_self_pu_history_without_clean_label_leakage():
    rng = np.random.RandomState(4)
    X = rng.randn(24, 3).astype("float32")
    y_pu = np.r_[np.ones(6, dtype=int), np.zeros(18, dtype=int)]
    estimator = SelfPUClassifier(
        0.3,
        hidden_dim=4,
        warmup_epochs=0,
        self_paced_start=0,
        self_paced_end=1,
        distill_start=1,
        max_epochs=2,
        batch_size=12,
        random_state=0,
        device="cpu",
    )

    with pytest.warns(UserWarning, match="explicit Self-PU ablation"):
        trajectory = DeepFitTrainer().fit(
            estimator,
            X,
            y_pu,
            class_prior=0.3,
            val_pu=(X, y_pu),
        )

    assert len(trajectory.epochs) == 2
    assert trajectory.best_epoch == int(np.argmin(estimator.history_["val_risk"])) + 1
    assert all("val_risk" in epoch.metrics for epoch in trajectory.epochs)
    assert estimator.calibration_mode_ == "ablation"
    assert estimator.teacher_selection_basis_ == "pu_validation_nnpu_risk"
