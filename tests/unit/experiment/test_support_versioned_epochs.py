"""Versioned support selection includes every paid epoch, including warmup."""

from dataclasses import replace

import numpy as np
import pytest
from _support_checkpoint_helpers import adapter, inputs, model

from pu_toolbox.experiment.bundle import DatasetBundle
from pu_toolbox.experiment.runner import _checkpoint_context
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPACleanSupport
from pu_toolbox.experiment.support_checkpoints import SelfPUTSCleanSupportCheckpointTrainer

pytestmark = pytest.mark.unit


def trajectory(method, directory, *, view="os"):
    parts, support = inputs()
    trainer = adapter(method, support)
    if view == "ts":
        trainer = SelfPUTSCleanSupportCheckpointTrainer(**vars(trainer))
    estimator = model(method)
    if method == "self_pu":
        estimator.set_params(warmup_epochs=1, self_paced_start=1)
    config = {
        "survey_protocol": {},
        "clean_support_provided": True,
        "checkpoint_bytes_per_component": 65536,
        "os_or_ts": view,
    }
    trainer.validate_experiment(
        estimator, DatasetBundle(*parts), config, [ProtocolPACleanSupport(), ProtocolOA()]
    )
    trainer.bind_checkpoint_directory(directory)
    return trainer.fit(
        estimator, parts[0].X, parts[0].labels, class_prior=0.5, os_or_ts=view
    ), parts


@pytest.mark.parametrize("method,view", [("lagam", "os"), ("self_pu", "os"), ("self_pu", "ts")])
def test_basic_all_paid_epochs_have_both_selection_metrics(method, view, tmp_path):
    recorded, parts = trajectory(method, tmp_path, view=view)
    expected = [0, 1, 2] if method == "lagam" else [1, 2, 3]
    assert [epoch.epoch for epoch in recorded.epochs] == expected
    assert recorded.model.training_provenance_["captured_epochs"] == expected
    assert recorded.model.training_provenance_["selection_scope"] == "completed_epochs"
    ProtocolOA().select([recorded], parts[2])
    ProtocolPACleanSupport().select([recorded], replace(parts[1], view="pu"), class_prior=0.5)
    components = 1 if method == "lagam" else 2
    assert len(recorded.checkpoints) == 3 * components
    for checkpoint in recorded.checkpoints:
        assert set(checkpoint.validation_metrics) == {"PA+clean-support", "OA"}


@pytest.mark.parametrize("method", ["lagam", "self_pu"])
def test_edge_dropping_warmup_keeps_the_checkpoint_blocker(method, tmp_path):
    recorded, _ = trajectory(method, tmp_path)
    blocker = "per_epoch_independent_PA_OA_checkpoint_selection"
    context = {
        "execution_mode": "versioned_pilot",
        "budget": {"epochs": 3},
        "formal_blockers": [blocker],
    }
    assert _checkpoint_context(context, [recorded])["selection_checkpoint_scope"] == (
        "independent_per_epoch"
    )
    shortened = replace(recorded, epochs=recorded.epochs[1:])
    assert blocker in _checkpoint_context(context, [shortened])["formal_blockers"]


def test_determ_warmup_snapshots_replay_the_same_scores(tmp_path):
    one, parts = trajectory("lagam", tmp_path / "one")
    two, _ = trajectory("lagam", tmp_path / "two")
    first = one.checkpoints[0].restore()
    repeated = two.checkpoints[0].restore()
    np.testing.assert_array_equal(
        first.decision_function(parts[3].X), repeated.decision_function(parts[3].X)
    )


@pytest.mark.parametrize("method", ["lagam", "self_pu"])
def test_param_warmup_can_win_selection(method, tmp_path, monkeypatch):
    from pu_toolbox.experiment.checkpoints import EpochCheckpoint

    recorded, parts = trajectory(method, tmp_path)
    truth = parts[2].labels

    class Scores:
        def __init__(self, warmup):
            self.warmup = warmup

        def decision_function(self, features):
            return truth.astype(float) if self.warmup else np.zeros(len(features))

    monkeypatch.setattr(
        EpochCheckpoint,
        "restore",
        lambda checkpoint, **kwargs: Scores(checkpoint.epoch_position == 1),
    )
    selected = ProtocolOA().select([recorded], parts[2])
    assert selected.epoch == 1 and selected.checkpoint_index == 0
    assert recorded.epochs[0].epoch == (0 if method == "lagam" else 1)
    assert len(recorded.epochs) == 3 and selected.metrics["val_accuracy"] == 1.0
