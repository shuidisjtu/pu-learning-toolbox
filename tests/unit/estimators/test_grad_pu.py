"""GradPU paper equations, classifier behavior and implementation boundaries."""

# ruff: noqa: N806

import copy
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError, ValidationError  # noqa: E402
from pu_toolbox.estimators.deep.grad_pu import (  # noqa: E402
    GradPUClassifier,
    gradpu_objective,
    gradpu_positive_weight,
)
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods  # noqa: E402
from pu_toolbox.workflows import PUPipeline  # noqa: E402

pytestmark = pytest.mark.unit


def _data():
    rng = np.random.RandomState(9)
    positive = rng.normal(1.0, 0.4, (8, 3)).astype(np.float32)
    unlabeled = rng.normal(-0.5, 0.8, (16, 3)).astype(np.float32)
    return np.vstack([positive, unlabeled]), np.r_[np.ones(8, int), np.zeros(16, int)]


def _model(**kwargs):
    params = {"hidden_dim": 6, "batch_size": 5, "max_epochs": 2, "random_state": 3}
    params.update(kwargs)
    return GradPUClassifier(**params)


@pytest.mark.math
def test_basic_positive_weight_is_unnormalized_and_harder_positives_weigh_more():
    raw = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    weights = gradpu_positive_weight(raw, beta=0.5)
    expected = 1.0 - 0.5 * torch.log((1.0 + torch.tanh(raw)) / 2.0)
    torch.testing.assert_close(weights, expected)
    assert bool(torch.all(weights >= 1.0))
    assert weights[0] > weights[1] > weights[2]
    assert weights.sum() > len(raw)  # no normalized-to-one positive weights
    weights.sum().backward()
    assert bool(torch.all(raw.grad < 0))


@pytest.mark.math
def test_basic_objective_matches_equations_five_to_seven():
    positive = torch.tensor([0.0, 0.5], requires_grad=True)
    unlabeled = torch.tensor([-0.5, 0.5], requires_grad=True)
    mixed = torch.tensor([[0.2, -0.1], [0.5, 0.3]], requires_grad=True)
    scale = torch.tensor(2.0, requires_grad=True)
    raw_mixed = scale * mixed[:, 0] - 3 * mixed[:, 1]
    total, parts = gradpu_objective(
        positive,
        unlabeled,
        beta=0.4,
        alpha=0.2,
        interpolated_inputs=mixed,
        raw_interpolated=raw_mixed,
    )
    expected_positive = (
        (1 - 0.4 * torch.log((1 + torch.tanh(positive)) / 2)) * (1 - torch.tanh(positive))
    ).mean()
    expected_unlabeled = (1 + torch.tanh(unlabeled)).mean()
    torch.testing.assert_close(parts["positive_loss"], expected_positive)
    torch.testing.assert_close(parts["unlabeled_loss"], expected_unlabeled)
    torch.testing.assert_close(parts["gradient_penalty"], torch.tensor(13.0))
    torch.testing.assert_close(total, expected_positive + expected_unlabeled + 2.6)
    total.backward()
    assert positive.grad is not None and scale.grad is not None


def test_basic_fit_predict_checkpoint_and_registry(tmp_path):
    X, y = _data()
    model = _model()
    with pytest.raises(NotFittedError):
        model.predict(X)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)
    scores = model.decision_function(X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all() and np.max(np.abs(scores)) <= 1
    assert set(model.predict(X)) <= {0, 1}
    assert model.optimizer_steps_ == 8  # 16 U / batch 5 => 4 updates per epoch
    assert len(trajectory.checkpoints) == model.max_epochs
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(X), scores, atol=1e-6
    )
    register_all_builtin_methods()
    assert get_algorithm("gradpu") is GradPUClassifier
    assert get_algorithm("grad_pu") is GradPUClassifier


def test_determ_seeded_fits_and_prior_compatibility_are_stable():
    X, y = _data()
    first = _model().fit(X, y)
    second = _model().fit(X, y, class_prior=0.4)
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
    assert first.history_ == second.history_
    assert second.get_pu_metadata()["class_prior"] is None  # no prior in Eq. 7


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"batch_size": 0}, "batch_size"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"alpha": -1}, "alpha"),
        ({"beta_max": float("nan")}, "beta_max"),
        ({"learning_rate": 0}, "learning_rate"),
    ],
)
def test_param_invalid_training_parameters_fail_before_updates(kwargs, message):
    X, y = _data()
    with pytest.raises(ValueError, match=message):
        _model(**kwargs).fit(X, y)


def test_param_zero_strengths_disable_both_added_terms():
    X, y = _data()
    fitted = _model(alpha=0.0, beta_max=0.0).fit(X, y)
    assert fitted.history_["gradient_penalty"] == [0.0] * fitted.max_epochs
    assert fitted.history_["beta"] == [0.0] * fitted.max_epochs
    assert np.isfinite(fitted.decision_function(X)).all()


def test_edge_input_and_unsupported_weight_paths_reject():
    X, y = _data()
    with pytest.raises(NotImplementedError, match="sample_weight"):
        _model().fit(X, y, sample_weight=np.ones(len(X)))
    with pytest.raises(ValueError, match="class_prior"):
        _model().fit(X, y, class_prior=1.0)
    with pytest.raises(ValidationError, match="2-D"):
        _model().fit(X.reshape(len(X), 1, 1, 3), y)
    with pytest.raises(ValueError, match="finite"):
        corrupted = X.copy()
        corrupted[0, 0] = np.nan
        _model().fit(corrupted, y)
    fitted = _model().fit(X, y)
    with pytest.raises(ValueError, match="fitted 2-D"):
        fitted.predict(X[:, :2])
    with pytest.raises(NotImplementedError, match="predict_proba"):
        fitted.predict_proba(X)


def test_edge_custom_model_shape_and_batchnorm_fail_closed():
    X, y = _data()
    with pytest.raises(ValueError, match="one raw score"):
        _model(model=torch.nn.Linear(3, 2)).fit(X, y)
    batchnorm = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 1))
    with pytest.raises(ValueError, match="BatchNorm"):
        _model(model=batchnorm).fit(X, y)
    frozen = torch.nn.Linear(3, 1)
    frozen.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        _model(model=frozen).fit(X, y)
    double_precision = _model(model=torch.nn.Linear(3, 1).double()).fit(X, y)
    assert np.isfinite(double_precision.decision_function(X)).all()


def test_basic_weights_only_roundtrip_preserves_scores():
    X, y = _data()
    fitted = _model().fit(X, y)
    restored = copy.deepcopy(fitted.model_)
    restored.load_state_dict(fitted.model_.state_dict())
    with torch.no_grad():
        values = restored(torch.as_tensor(X)).reshape(-1).numpy()
    np.testing.assert_allclose(values, fitted.decision_function(X), atol=1e-6)


@pytest.mark.integration
def test_basic_pipeline_resolves_gradpu_without_prior():
    X, y = _data()
    report = PUPipeline(
        classifier="gradpu",
        classifier_params={"hidden_dim": 6, "batch_size": 5},
        prior_estimator=None,
        max_epochs=2,
        cv=2,
        random_state=3,
    ).fit_evaluate(X, y)
    assert isinstance(report.final_model, GradPUClassifier)
    assert report.final_model.predict(X).shape == (len(X),)


@pytest.mark.gpu
def test_basic_gpu_gradient_penalty_and_prediction_use_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA; GPU gate must not silently skip")
        pytest.skip("CUDA unavailable in the default test environment")
    rng = np.random.RandomState(5)
    X = rng.normal(size=(12, 3)).astype(np.float32)
    y = np.array([1] * 4 + [0] * 8)
    model = GradPUClassifier(
        hidden_dim=4, max_epochs=1, batch_size=4, random_state=5, device="cuda"
    ).fit(X, y)
    assert next(model.model_.parameters()).device.type == "cuda"
    assert model.history_["gradient_penalty"][0] > 0
    assert np.isfinite(model.decision_function(X)).all()
