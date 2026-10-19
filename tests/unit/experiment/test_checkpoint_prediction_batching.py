# ruff: noqa: N803, N806
"""Inference chunking survives weights-only replay without relaxing score assertions."""

import numpy as np
import pytest
import torch
from torch import nn

from pu_toolbox.experiment.checkpoints import (
    EpochCheckpointTrainer,
    SnapshotPredictor,
    load_epoch_checkpoint,
)

pytestmark = pytest.mark.unit


class BatchedEstimator:
    checkpoint_prediction_batch_size = 1

    def fit(self, X, y, *, epoch_callback=None):
        self.model_ = nn.Linear(2, 1)
        with torch.no_grad():
            self.model_.weight.copy_(torch.tensor([[0.0, 1.0]]))
            self.model_.bias.fill_(-0.5)
        epoch_callback(0, self)
        return self

    def decision_function(self, X):
        with torch.no_grad():
            return np.concatenate(
                [self.model_(torch.as_tensor(row[None])).reshape(-1).numpy() for row in X]
            )


def snapshot(tmp_path):
    X = np.array([[0, 1], [1, 0]], dtype=np.float32)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        BatchedEstimator(), X, np.array([1, 0])
    )
    return X, trajectory.model, trajectory.checkpoints[-1]


def test_basic_determ_declared_prediction_batches_survive_persisted_replay(tmp_path):
    X, fitted, checkpoint = snapshot(tmp_path)
    reference = checkpoint.reference()
    assert reference["prediction_batch_size"] == 1
    for restored in (checkpoint.restore(), load_epoch_checkpoint(reference, nn.Linear(2, 1))):
        sizes = []
        restored.model_.register_forward_pre_hook(
            lambda module, inputs, sizes=sizes: sizes.append(len(inputs[0]))
        )
        np.testing.assert_array_equal(restored.decision_function(X), fitted.decision_function(X))
        assert restored.batch_size == 1 and sizes == [1, 1]


def test_edge_legacy_reference_keeps_historical_256_prediction_batch_size(tmp_path):
    _, _, checkpoint = snapshot(tmp_path)
    reference = checkpoint.reference()
    del reference["prediction_batch_size"]
    assert load_epoch_checkpoint(reference, nn.Linear(2, 1)).batch_size == 256


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2", None])
def test_param_invalid_persisted_prediction_batch_size_is_refused(tmp_path, bad):
    _, _, checkpoint = snapshot(tmp_path)
    reference = checkpoint.reference()
    reference["prediction_batch_size"] = bad
    with pytest.raises(ValueError, match="prediction batch_size"):
        load_epoch_checkpoint(reference, nn.Linear(2, 1))


@pytest.mark.parametrize(
    "method", ["pulda", "pan", "gradpu", "robust_pu", "split_pu", "holistic_pu", "gen_pu"]
)
def test_basic_candidate_declares_its_actual_inference_chunk_size(method):
    from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

    register_all_builtin_methods()
    kwargs = (
        {"class_prior": 0.4, "positive_batch_size": 2, "unlabeled_batch_size": 7}
        if method == "pulda"
        else {"batch_size": 7}
    )
    assert get_algorithm(method)(**kwargs).checkpoint_prediction_batch_size == 7


@pytest.mark.parametrize("sample_shape", [(4,), (1, 2, 2)])
def test_edge_inplace_snapshot_forward_cannot_rewrite_validation_inputs(sample_shape):
    features = (np.arange(-10, 10, dtype=np.float32) / 8).reshape((5, *sample_shape))
    before = features.copy()
    network = nn.Sequential(nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(4, 1))
    with torch.no_grad():
        network[-1].weight.fill_(1)
        network[-1].bias.zero_()
    predictor = SnapshotPredictor(network, device="cpu", batch_size=2)
    sizes = []
    network.register_forward_pre_hook(lambda _module, inputs: sizes.append(len(inputs[0])))
    expected = np.maximum(before, 0).reshape(5, -1).sum(axis=1)

    np.testing.assert_array_equal(predictor.decision_function(features), expected)
    np.testing.assert_array_equal(features, before)
    assert sizes == [2, 2, 1]
    np.testing.assert_array_equal(predictor.decision_function(features), expected)
    np.testing.assert_array_equal(features, before)


def test_edge_failed_inplace_snapshot_forward_leaves_caller_array_unchanged():
    failure = RuntimeError("snapshot forward interrupted")

    class MutatingFailure(nn.Module):
        def forward(self, features):
            features.relu_()
            raise failure

    features = np.array([[-2, 3], [-4, 1]], dtype=np.float32)
    before = features.copy()
    predictor = SnapshotPredictor(MutatingFailure(), device="cpu", batch_size=1)
    with pytest.raises(RuntimeError) as caught:
        predictor.decision_function(features)
    assert caught.value is failure
    np.testing.assert_array_equal(features, before)
