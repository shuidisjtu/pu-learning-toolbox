# ruff: noqa: N806
"""PAN Eq. 7 signs, alternating gradients, serialization and OS-only routing."""

import copy
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.deep.pan import (  # noqa: E402
    PANClassifier,
    pan_classifier_loss,
    pan_discriminator_loss,
)
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods  # noqa: E402

pytestmark = pytest.mark.unit


def data():
    rng = np.random.RandomState(9)
    return rng.normal(size=(24, 3)).astype("float32"), np.r_[np.ones(8, int), np.zeros(16, int)]


def model(**kwargs):
    params = dict(hidden_dim=4, max_epochs=2, batch_size=8, device="cpu", random_state=2)
    params.update(kwargs)
    return PANClassifier(**params)


@pytest.mark.math
def test_losses_match_probability_objective_with_correct_minmax_signs():
    dp = torch.tensor([0.2, -0.7], requires_grad=True)
    du = torch.tensor([-0.5, 1.0], requires_grad=True)
    cu = torch.tensor([0.3, -0.8], requires_grad=True)
    game = (torch.log(1 - cu.sigmoid()) - torch.log(cu.sigmoid())) * (2 * du.sigmoid() - 1)
    expected = (
        -torch.log(dp.sigmoid()).mean() - torch.log(1 - du.sigmoid()).mean() - 0.4 * game.mean()
    )
    loss = pan_discriminator_loss(dp, du, cu, weight=0.4)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert dp.grad is not None and du.grad is not None and cu.grad is None
    dp.grad = du.grad = None
    c_loss = pan_classifier_loss(cu, du, weight=0.4)
    torch.testing.assert_close(c_loss, 0.4 * game.mean())
    c_loss.backward()
    assert cu.grad is not None and du.grad is None


def test_extreme_logits_remain_finite():
    raw = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    loss = pan_discriminator_loss(raw, raw, raw, weight=1)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(raw.grad).all()


def test_fit_epoch_recovery_clone_and_pickle(tmp_path):
    X, y = data()
    fitted = clone(model())
    with pytest.raises(NotFittedError):
        fitted.predict(X)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, X, y)
    assert fitted.optimizer_steps_ == 8
    assert len(trajectory.checkpoints) == 2
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(X),
        fitted.decision_function(X),
        atol=1e-6,
    )
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.decision_function(X), fitted.decision_function(X))
    np.testing.assert_allclose(fitted.predict_proba(X).sum(axis=1), 1)
    register_all_builtin_methods()
    assert get_algorithm("pan") is PANClassifier
    assert get_algorithm("predictive_adversarial_pu") is PANClassifier


def test_seed_determinism_prior_ignored_and_injected_models_not_mutated():
    X, y = data()
    network = torch.nn.Linear(3, 1)
    before = copy.deepcopy(network.state_dict())
    first = model(model=network).fit(X, y)
    second = model(model=network).fit(X, y, class_prior=0.4)
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
    for key in before:
        assert torch.equal(before[key], network.state_dict()[key])
    assert first.history_ == second.history_
    assert not first.calibration_applied_
    with pytest.raises(ValueError, match="native OS"):
        model().fit(X, y, os_or_ts="ts")
    with pytest.raises(NotImplementedError, match="sample_weight"):
        model().fit(X, y, sample_weight=np.ones(len(y)))


@pytest.mark.parametrize(
    "kwargs",
    [{"max_epochs": 0}, {"batch_size": True}, {"learning_rate": np.nan}, {"adversarial_weight": 0}],
)
def test_invalid_parameters(kwargs):
    X, y = data()
    with pytest.raises(ValueError):
        model(**kwargs).fit(X, y)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_short_fit_roundtrip(tmp_path):
    X, y = data()
    fitted = model(max_epochs=1, device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, X, y)
    assert next(fitted.model_.parameters()).is_cuda
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(X),
        fitted.decision_function(X),
        atol=1e-5,
    )
