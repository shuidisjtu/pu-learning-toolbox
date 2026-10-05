"""Observe real optimizer calls and stage snapshots, not just budget formulas."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import RobustPUClassifier, SplitPUClassifier  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def data():
    features = np.random.RandomState(4).normal(size=(24, 3)).astype("float32")
    return features, np.r_[np.ones(8, int), np.zeros(16, int)]


def candidate(method):
    common = dict(class_prior=0.4, hidden_dim=4, batch_size=8, random_state=3, device="cpu")
    if method == "robust_pu":
        return RobustPUClassifier(pretrain_epochs=2, episodes=3, inner_epochs=2, **common)
    return SplitPUClassifier(
        teacher_epochs=1,
        split_epochs=3,
        student_epochs=1,
        rounds=2,
        agreement_threshold=0.01,
        **common,
    )


@pytest.mark.parametrize("method", ["robust_pu", "split_pu"])
def test_basic_optimizer_counter_matches_instrumented_all_stage_updates_and_resets(
    monkeypatch, method
):
    features, labels = data()
    calls = []
    for optimizer_type in (torch.optim.Adam, torch.optim.SGD):
        original = optimizer_type.step

        def tracked(optimizer, *args, _step=original, **kwargs):
            calls.append(type(optimizer).__name__)
            return _step(optimizer, *args, **kwargs)

        monkeypatch.setattr(optimizer_type, "step", tracked)
    model = candidate(method).fit(features, labels, os_or_ts="ts")
    assert model.optimizer_steps_ == len(calls) > 0
    if method == "robust_pu":
        assert model.optimizer_steps_ == 22  # pretrain 2*2 + episodes 3*2*3
    else:
        assert "Adam" in calls and "SGD" in calls
    calls.clear()
    model.fit(features, labels, os_or_ts="ts")
    assert model.optimizer_steps_ == len(calls) > 0  # refit resets, never accumulates


@pytest.mark.parametrize("method", ["robust_pu", "split_pu"])
def test_determ_snapshots_respect_budget_bound_and_early_stop_without_changing_fit(
    tmp_path, method
):
    features, labels = data()
    model = candidate(method)
    plain = candidate(method).fit(features, labels, os_or_ts="ts")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        model, features, labels, os_or_ts="ts"
    )
    if method == "robust_pu":
        assert len(trajectory.checkpoints) == model.checkpoint_epoch_count == 5
    else:
        assert len(model.history_["split_agreement"]) == 1
        assert len(trajectory.checkpoints) == 4 < model.checkpoint_epoch_count == 6
    assert model.optimizer_steps_ == plain.optimizer_steps_
    np.testing.assert_array_equal(
        model.decision_function(features), plain.decision_function(features)
    )
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(features),
        model.decision_function(features),
        atol=1e-6,
    )


@pytest.mark.parametrize("method", ["robust_pu", "split_pu"])
def test_edge_param_failed_snapshot_is_not_swallowed_or_retried(tmp_path, monkeypatch, method):
    features, labels = data()
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise TypeError("synthetic snapshot error")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(TypeError, match="synthetic snapshot error"):
        EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(candidate(method), features, labels)
    assert calls == [1]
