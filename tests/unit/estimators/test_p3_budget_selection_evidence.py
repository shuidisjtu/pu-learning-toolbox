"""P1-1 budget facts and intentional source adaptations; no formal candidate choice."""

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import (  # noqa: E402
    GradPUClassifier,
    PUExtraTreesClassifier,
    PULDAClassifier,
    RobustPUClassifier,
    SplitPUClassifier,
)
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit
CLASSES = (
    PULDAClassifier,
    PUExtraTreesClassifier,
    GradPUClassifier,
    RobustPUClassifier,
    SplitPUClassifier,
)


def test_basic_determ_pulda_unequal_phases_keep_tail_and_report_real_steps(tmp_path, monkeypatch):
    features = np.random.RandomState(4).normal(size=(16, 3)).astype("float32")
    labels = np.r_[np.ones(5, int), np.zeros(11, int)]
    steps = []
    original = torch.optim.Adam.step

    def observed(optimizer, *args, **kwargs):
        steps.append(1)
        return original(optimizer, *args, **kwargs)

    horizons = []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    def observed_schedule(optimizer, horizon, **kwargs):
        horizons.append(horizon)
        return scheduler(optimizer, horizon, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", observed)
    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", observed_schedule)
    model = PULDAClassifier(
        0.4,
        warmup_epochs=1,
        pu_epochs=2,
        hidden_dim=4,
        positive_batch_size=4,
        unlabeled_batch_size=4,
        random_state=3,
        device="cpu",
    )
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, features, labels)
    assert len(steps) == model.optimizer_steps_ == 9
    assert horizons == [2, 2]
    assert model.history_["optimizer_steps"] == [3, 6, 9]
    assert model.history_["phase"] == ["warmup", "pu_mixup", "pu_mixup"]
    assert len(trajectory.checkpoints) == model.checkpoint_epoch_count == 3
    assert all(
        checkpoint.reference()["prediction_batch_size"] == 4
        for checkpoint in trajectory.checkpoints
    )
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(features),
        model.decision_function(features),
    )
    steps.clear()
    model.fit(features, labels)
    assert len(steps) == model.optimizer_steps_ == 9


def test_edge_puet_ties_and_zero_gain_keep_deterministic_negative_leaf():
    # The author samples a leaf tie; this intentional toolbox adapter does not.
    features = np.zeros((8, 2))
    labels = np.r_[np.ones(4, int), np.zeros(4, int)]
    fitted = PUExtraTreesClassifier(0.5, n_estimators=2, random_state=7).fit(features, labels)
    np.testing.assert_array_equal(fitted.predict(features), np.zeros(8))
    # Every cut of alternating P/U pairs has zero gain, so no split is admitted.
    features = np.repeat(np.arange(4), 2)[:, None]
    labels = np.tile([1, 0], 4)
    fitted = PUExtraTreesClassifier(0.5, n_estimators=2, max_candidates=5, random_state=7).fit(
        features, labels
    )
    np.testing.assert_array_equal(fitted.n_leaves_, [1, 1])
    np.testing.assert_array_equal(fitted.feature_importances_, [0])


@pytest.mark.parametrize("klass", CLASSES)
def test_param_fit_interfaces_cannot_take_selection_or_test_labels(klass):
    params = inspect.signature(klass.fit).parameters
    assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert not (
        {"y_true", "X_val", "y_val", "validation_data", "test_data", "clean_val"} & set(params)
    )


def test_determ_cpu_tree_has_no_neural_epoch_or_cuda_budget():
    model = PUExtraTreesClassifier(0.4, n_estimators=3, max_depth=2, random_state=7)
    assert "device" not in model.get_params() and not hasattr(model, "checkpoint_epoch_count")
    assert "epoch_callback" not in inspect.signature(model.fit).parameters
    features = np.arange(24).reshape(8, 3)
    labels = np.r_[np.ones(4, int), np.zeros(4, int)]
    first, second = (
        model.fit(features, labels),
        PUExtraTreesClassifier(**model.get_params()).fit(features, labels),
    )
    assert len(first.trees_) == 3
    np.testing.assert_array_equal(
        first.decision_function(features), second.decision_function(features)
    )
