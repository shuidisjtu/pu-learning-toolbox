# tests/unit/experiment/test_trainers.py

# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer
from pu_toolbox.experiment.strategies import DeepFitTrainer, FitTrainer, SupervisedTrainer

pytestmark = pytest.mark.unit


def test_basic_trainer_declares_label_semantics():
    """The runner gates clean views on this flag (see the runner's view guards)."""
    assert SupervisedTrainer.trains_on_real_labels is True
    assert FitTrainer.trains_on_real_labels is False
    assert DeepFitTrainer.trains_on_real_labels is False


def test_fit_trainer_single_point():
    X = np.random.RandomState(0).randn(30, 3)
    y = np.array([1] * 6 + [0] * 24)
    est = UPUClassifier(0.25, random_state=0)
    traj = FitTrainer().fit(est, X, y, risk_prior=0.4)
    assert len(traj.epochs) == 1
    assert traj.best_epoch is None
    assert traj.model is est  # fitted in place
    assert est.risk_prior_ == 0.4


def test_basic_fit_trainer_omits_unsupplied_prior():
    # A method with no prior input still fits.
    X = np.random.RandomState(0).randn(30, 3)
    y = np.array([1] * 6 + [0] * 24)
    est = UPUClassifier(0.5, random_state=0)

    class NoPriorFit:
        def fit(self, X, y):
            return est.fit(X, y)

    est_dup = NoPriorFit()
    traj = FitTrainer().fit(est_dup, X, y)
    assert len(traj.epochs) == 1


def test_supervised_trainer_uses_true_labels():
    """The estimator must receive the real labels unmodified (no PU remapping).

    Asserting the captured labels rather than "it ran": the previous version
    only checked ``len(traj.epochs) >= 1``, which held for any input and is why
    the runner's PU-label mis-wiring escaped this file (see
    docs/dev/experiment_layer.md).
    """
    X = np.random.RandomState(0).randn(30, 3)
    y = np.array([1] * 6 + [0] * 24)
    seen = {}

    class RecordingEstimator:
        def fit(self, X_arg, y_arg, **kwargs):
            seen["y"] = np.array(y_arg)
            return self

    traj = SupervisedTrainer().fit(RecordingEstimator(), X, y)

    assert np.array_equal(seen["y"], y)
    assert len(traj.epochs) == 1


def test_deep_trainer_reads_history_after_nnpu_fix():
    X = np.random.RandomState(0).randn(12, 2).astype("float32")
    y_pu = np.array([1, 1, 0] * 4)
    est = NonNegativePUClassifier(
        risk_prior=0.3, max_epochs=3, patience=2, random_state=0, device="cpu"
    )
    traj = DeepFitTrainer().fit(
        est, X, y_pu, risk_prior=0.3, val_pu=(X, y_pu), validation_risk_prior=0.55
    )
    assert len(traj.epochs) == 3
    assert "val_risk" in traj.epochs[0].metrics
    assert est.validation_risk_prior_ == 0.55


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
            risk_prior=0.3,
            val_pu=(X, y_pu),
        )

    assert len(trajectory.epochs) == 2
    assert trajectory.best_epoch == int(np.argmin(estimator.history_["val_risk"])) + 1
    assert all("val_risk" in epoch.metrics for epoch in trajectory.epochs)
    assert estimator.calibration_mode_ == "ablation"
    assert estimator.teacher_selection_basis_ == "pu_validation_nnpu_risk"


@pytest.mark.parametrize("trainer_type", [FitTrainer, DeepFitTrainer])
def test_edge_trainers_propagate_an_internal_type_error_once(trainer_type):
    class BrokenEstimator:
        calls = 0

        def fit(self, X, y, *, risk_prior=None, validation_data=None):
            self.calls += 1
            raise TypeError("solver failed")

    estimator = BrokenEstimator()
    with pytest.raises(TypeError, match="solver failed"):
        trainer_type().fit(estimator, [[0], [1]], [0, 1], risk_prior=0.5, val_pu=([[0]], [1]))
    assert estimator.calls == 1


@pytest.mark.parametrize("trainer_type", [DeepFitTrainer, EpochCheckpointTrainer])
def test_param_builtin_trainers_forward_both_named_priors(trainer_type):
    X = np.random.RandomState(2).randn(16, 2).astype("float32")
    y = np.tile([1, 0, 0, 0], 4)
    estimator = NonNegativePUClassifier(risk_prior=0.2, max_epochs=1, device="cpu", random_state=1)
    trainer_type().fit(estimator, X, y, risk_prior=0.3, validation_risk_prior=0.55, val_pu=(X, y))
    train, validation = estimator._prior_consumptions_
    assert (train["value"], train["target"]) == (0.3, "native_unlabeled")
    assert (validation["value"], validation["target"]) == (0.55, "native_unlabeled")
