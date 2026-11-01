"""Opt-in saved-tensor offload preserves GradPU's second-order full-batch update."""

import copy
import os
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import GradPUClassifier  # noqa: E402
from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.deep.vision import build_wconpu_backbone  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def template(device="cpu"):
    torch.manual_seed(7)
    # Same topology as the project CNN, small channel count only for unit tests.
    encoder = build_wconpu_backbone(
        "cnn13_no_bn",
        in_channels=1,
        base_channels=2,
        normalization_mean=(0.4,),
        normalization_std=(0.3,),
    )
    return GradPUClassifier(
        encoder=encoder,
        hidden_dim=4,
        batch_size=4,
        max_epochs=2,
        random_state=11,
        device=device,
    )


def data():
    return np.random.RandomState(1).normal(size=(12, 1, 8, 8)).astype("float32"), np.r_[
        np.ones(5, int), np.zeros(7, int)
    ]


@pytest.mark.parametrize("view", ["os", "ts"])
@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_determ_edge_full_trajectory_second_order_gradients_rng_and_short_batches(view, alpha):
    features, labels = data()
    base = template().set_params(alpha=alpha)
    reference = clone(base).fit(features, labels, os_or_ts=view)
    rng = torch.get_rng_state().clone()
    offloaded = (
        clone(base).set_params(activation_memory="cpu_offload").fit(features, labels, os_or_ts=view)
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert reference.history_ == offloaded.history_
    assert reference.optimizer_steps_ == offloaded.optimizer_steps_ == 4
    for left, right in zip(
        reference.model_.parameters(), offloaded.model_.parameters(), strict=False
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    assert reference.training_provenance_["batch"] == offloaded.training_provenance_["batch"]
    assert offloaded.training_provenance_["activation_memory"]["policy"] == "cpu_offload"


def test_determ_dropout_forward_rng_and_snapshot_roundtrip(tmp_path):
    features = np.random.RandomState(5).normal(size=(12, 3)).astype("float32")
    labels = np.r_[np.ones(5, int), np.zeros(7, int)]
    torch.manual_seed(19)
    network = torch.nn.Sequential(
        torch.nn.Linear(3, 6), torch.nn.Tanh(), torch.nn.Dropout(0.25), torch.nn.Linear(6, 1)
    )
    base = GradPUClassifier(model=network, batch_size=4, max_epochs=2, random_state=9, device="cpu")
    reference = clone(base).fit(features, labels, os_or_ts="ts")
    rng = torch.get_rng_state().clone()
    offloaded = clone(base).set_params(activation_memory="cpu_offload")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        offloaded, features, labels, os_or_ts="ts"
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert reference.history_ == offloaded.history_
    expected = reference.decision_function(features)
    np.testing.assert_array_equal(expected, offloaded.decision_function(features))
    np.testing.assert_array_equal(
        expected, pickle.loads(pickle.dumps(offloaded)).decision_function(features)
    )
    np.testing.assert_allclose(
        expected, trajectory.checkpoints[-1].restore().decision_function(features)
    )


@pytest.mark.parametrize("invalid", [None, [], True, "auto", "fp16"])
def test_param_invalid_memory_policy_invalidates_previous_fit(invalid):
    features, labels = data()
    estimator = template().fit(features, labels)
    estimator.set_params(activation_memory=invalid)
    with pytest.raises(ValueError, match="activation_memory"):
        estimator.fit(features, labels)
    with pytest.raises(NotFittedError):
        estimator.predict(features)


@pytest.mark.gpu
def test_cuda_offload_same_updates_rng_and_restored_predictions():
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    features, labels = data()
    base = template("cuda")
    reference = clone(base).fit(features, labels, os_or_ts="ts")
    rng = torch.cuda.get_rng_state().clone()
    before = copy.deepcopy(reference.model_.state_dict())
    offloaded = (
        clone(base).set_params(activation_memory="cpu_offload").fit(features, labels, os_or_ts="ts")
    )
    assert torch.equal(rng, torch.cuda.get_rng_state())
    for name, value in offloaded.model_.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=1e-6, atol=1e-7)
    for key in reference.history_:
        np.testing.assert_allclose(
            reference.history_[key], offloaded.history_[key], rtol=1e-6, atol=1e-7
        )
    np.testing.assert_allclose(
        reference.decision_function(features),
        offloaded.decision_function(features),
        rtol=1e-6,
        atol=1e-7,
    )
