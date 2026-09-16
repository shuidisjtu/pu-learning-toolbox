# ruff: noqa: N803, N806
"""Snapshot isolation, safe persistence, callback behavior and deep method hooks."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier
from pu_toolbox.estimators.risk.dist_pu import DistPUClassifier
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer, load_epoch_checkpoint
from pu_toolbox.experiment.survey_execution import PilotOracleMLP

pytestmark = pytest.mark.unit


def _data():
    X = np.random.RandomState(4).normal(size=(12, 3)).astype(np.float32)
    return X, np.array([1, 1, 0] * 4)


def _nnpu(tmp_path, *, device="cpu"):
    X, y = _data()
    model = NonNegativePUClassifier(class_prior=0.3, max_epochs=3, random_state=0, device=device)
    return X, EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)


def test_basic_nnpu_records_all_completed_epochs_and_shared_cpu_template(tmp_path):
    X, trajectory = _nnpu(tmp_path)
    assert [checkpoint.epoch_position for checkpoint in trajectory.checkpoints] == [1, 2, 3]
    assert [record.epoch for record in trajectory.epochs] == [0, 1, 2]
    assert len({id(checkpoint.template) for checkpoint in trajectory.checkpoints}) == 1
    assert all(
        parameter.device.type == "cpu"
        for parameter in trajectory.checkpoints[0].template.parameters()
    )
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(X),
        trajectory.model.decision_function(X),
    )


def test_determ_recorded_weights_ignore_later_estimator_mutation(tmp_path):
    X, trajectory = _nnpu(tmp_path)
    checkpoint = trajectory.checkpoints[0]
    scores = checkpoint.restore().decision_function(X)
    with torch.no_grad():
        next(trajectory.model.model_.parameters()).add_(100)
    np.testing.assert_array_equal(scores, checkpoint.restore().decision_function(X))


def test_basic_persisted_reference_loads_with_explicit_architecture(tmp_path):
    X, trajectory = _nnpu(tmp_path)
    checkpoint = trajectory.checkpoints[-1]
    reference = checkpoint.reference()
    loaded = load_epoch_checkpoint(reference, nn.Linear(3, 1), device="cpu")
    np.testing.assert_array_equal(
        loaded.decision_function(X), checkpoint.restore().decision_function(X)
    )
    assert reference["persistent"] and not reference["training_resume_supported"]


def test_param_modified_weight_bytes_rejected_before_load(tmp_path):
    _, trajectory = _nnpu(tmp_path)
    checkpoint = trajectory.checkpoints[0]
    Path(checkpoint.path).write_bytes(b"damaged weights")
    with pytest.raises(ValueError, match="digest mismatch"):
        checkpoint.restore()


def test_edge_missing_checkpoint_never_falls_back_to_final_model(tmp_path):
    _, trajectory = _nnpu(tmp_path)
    checkpoint = trajectory.checkpoints[0]
    Path(checkpoint.path).unlink()
    with pytest.raises(FileNotFoundError):
        checkpoint.restore()


def test_param_requires_explicit_callback_hook():
    X, y = _data()

    class Unsupported:
        def fit(self, X, y):
            pytest.fail("unsupported fit must not start")

    with pytest.raises(ValueError, match="explicit epoch_callback"):
        EpochCheckpointTrainer().fit(Unsupported(), X, y)


def test_edge_no_completed_callback_cannot_claim_checkpoint_coverage():
    X, y = _data()

    class Silent:
        def fit(self, X, y, *, epoch_callback=None):
            return self

    with pytest.raises(ValueError, match="no completed checkpoints"):
        EpochCheckpointTrainer().fit(Silent(), X, y)


def test_param_callback_typeerror_does_not_trigger_bare_refit(tmp_path, monkeypatch):
    X, y = _data()
    calls = []

    def failing_save(*args, **kwargs):
        calls.append(1)
        raise TypeError("snapshot failure")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(TypeError, match="snapshot failure"):
        EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
            NonNegativePUClassifier(class_prior=0.3, max_epochs=3, random_state=0), X, y
        )
    assert len(calls) == 1


def test_determ_capture_does_not_change_training_seed_or_optimizer_updates(tmp_path):
    X, y = _data()
    first = NonNegativePUClassifier(
        class_prior=0.3, max_epochs=3, random_state=0, device="cpu"
    ).fit(X, y)
    _, trajectory = _nnpu(tmp_path)
    np.testing.assert_array_equal(first.decision_function(X), trajectory.model.decision_function(X))


def test_basic_distpu_records_fullbatch_epochs_without_repeated_fit(tmp_path):
    X, y = _data()
    model = DistPUClassifier(0.3, epochs=2, hidden_dim=4, device="cpu")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)
    assert len(trajectory.checkpoints) == len(model.loss_history_) == 2
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(X), model.decision_function(X)
    )


def test_basic_oracle_callback_receives_training_only_and_no_prior(tmp_path):
    X, _ = _data()
    y = np.array([1, 0] * 6)
    model = PilotOracleMLP(max_epochs=2, batch_size=4, random_state=0)
    trajectory = EpochCheckpointTrainer(supervised=True, checkpoint_dir=tmp_path).fit(model, X, y)
    assert len(trajectory.checkpoints) == 2
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(X), model.decision_function(X)
    )
    with pytest.raises(ValueError, match="must not receive a PU class prior"):
        EpochCheckpointTrainer(supervised=True).fit(model, X, y, class_prior=0.3)


def test_basic_selfpu_keeps_both_teachers_at_every_epoch(tmp_path):
    X, y = _data()
    model = SelfPUClassifier(
        0.3,
        hidden_dim=4,
        max_epochs=2,
        batch_size=6,
        threshold=0.7,
        random_state=0,
        device="cpu",
        warmup_epochs=0,
        self_paced_start=0,
        self_paced_end=1,
        distill_start=1,
    )
    with pytest.warns(UserWarning, match="explicit Self-PU ablation"):
        trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)
    assert [
        (checkpoint.epoch_position, checkpoint.component) for checkpoint in trajectory.checkpoints
    ] == [(1, "teacher_1"), (1, "teacher_2"), (2, "teacher_1"), (2, "teacher_2")]
    assert len({id(checkpoint.template) for checkpoint in trajectory.checkpoints}) == 2
    assert all(
        checkpoint.cutoff == pytest.approx(np.log(0.7 / 0.3))
        for checkpoint in trajectory.checkpoints
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_basic_gpu_snapshot_restores_to_cpu_and_cuda_with_same_scores(tmp_path):
    X, trajectory = _nnpu(tmp_path, device="cuda")
    checkpoint = trajectory.checkpoints[-1]
    gpu, cpu = checkpoint.restore(), checkpoint.restore(device="cpu")
    assert next(gpu.model_.parameters()).device.type == "cuda"
    assert next(cpu.model_.parameters()).device.type == "cpu"
    np.testing.assert_allclose(
        gpu.decision_function(X), cpu.decision_function(X), rtol=1e-5, atol=1e-6
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("kind", ["dist_pu", "self_pu", "oracle"])
def test_basic_deep_gpu_completed_epoch_snapshots(kind, tmp_path):
    X, y = _data()
    if kind == "dist_pu":
        model = DistPUClassifier(0.3, epochs=2, hidden_dim=4, device="cuda")
    elif kind == "self_pu":
        model = SelfPUClassifier(
            0.3,
            max_epochs=2,
            hidden_dim=4,
            warmup_epochs=0,
            self_paced_start=0,
            self_paced_end=1,
            distill_start=1,
            device="cuda",
        )
    else:
        model = PilotOracleMLP(max_epochs=2, batch_size=4, device="cuda")
    trajectory = EpochCheckpointTrainer(supervised=kind == "oracle", checkpoint_dir=tmp_path).fit(
        model, X, y
    )
    assert len(trajectory.epochs) == 2
    assert len(trajectory.checkpoints) == (4 if kind == "self_pu" else 2)
    for checkpoint in trajectory.checkpoints:
        restored = checkpoint.restore()
        assert next(restored.model_.parameters()).device.type == "cuda"
        assert np.isfinite(restored.decision_function(X)).all()
