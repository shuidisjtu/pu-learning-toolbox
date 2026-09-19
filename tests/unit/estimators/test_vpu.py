"""VPU objective, prior-free estimator behavior, and toolbox contracts."""

# ruff: noqa: N806

import copy
import math
import os

import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError, ValidationError  # noqa: E402
from pu_toolbox.estimators.risk.vpu import VPUClassifier, vpu_objective  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods  # noqa: E402
from pu_toolbox.workflows import PUPipeline  # noqa: E402

pytestmark = pytest.mark.unit


def _data():
    rng = np.random.RandomState(31)
    positive = rng.normal(1.0, 0.5, (9, 3)).astype(np.float32)
    unlabeled = rng.normal(-0.3, 0.8, (15, 3)).astype(np.float32)
    return np.vstack([positive, unlabeled]), np.r_[np.ones(9, int), np.zeros(15, int)]


def _model(**kwargs):
    params = {"hidden_dim": 8, "depth": 1, "max_epochs": 2, "batch_size": 5, "random_state": 7}
    params.update(kwargs)
    return VPUClassifier(**params)


@pytest.mark.math
def test_variational_objective_matches_paper_equations_and_target_has_gradient():
    log_p = torch.tensor([-0.2, -0.6], requires_grad=True)
    log_x = torch.tensor([-0.3, -0.7], requires_grad=True)
    log_mix = torch.tensor([-0.4, -0.8], requires_grad=True)
    mixing = torch.tensor(0.25)
    total, variational, consistency = vpu_objective(
        log_p, log_x, log_mix, mixing, regularization_weight=0.3
    )
    expected_var = torch.log(torch.exp(log_x).mean()) - log_p.mean()
    target = torch.log(mixing + (1 - mixing) * torch.exp(log_x))
    expected_consistency = ((target - log_mix) ** 2).mean()
    torch.testing.assert_close(variational, expected_var)
    torch.testing.assert_close(consistency, expected_consistency)
    torch.testing.assert_close(total, expected_var + 0.3 * expected_consistency)
    total.backward()
    assert log_p.grad is not None and log_x.grad is not None and log_mix.grad is not None
    # The official MixUp target is differentiated; detaching it changes this gradient.
    expected_x_grad = torch.softmax(log_x.detach(), dim=0) + (
        0.3
        * (target.detach() - log_mix.detach())
        * (1 - mixing)
        * torch.exp(log_x.detach())
        / (mixing + (1 - mixing) * torch.exp(log_x.detach()))
    )
    torch.testing.assert_close(log_x.grad, expected_x_grad)


def test_fit_normalization_checkpoint_and_registry(tmp_path):
    X, y = _data()
    model = _model()
    with pytest.raises(NotFittedError):
        model.predict(X)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)
    scores = model.decision_function(X)
    proba = model.predict_proba(X)
    assert scores.shape == (len(X),)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(proba[:, 1], 0.5 * np.exp(scores), atol=1e-6)
    assert proba[:, 1].max() == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(scores).all()
    assert set(model.predict(X)) <= {0, 1}
    assert model.optimizer_steps_ == 10  # 24 full-pool rows / batch 5, two epochs
    assert len(trajectory.checkpoints) == 2
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(X), scores, atol=1e-6
    )
    register_all_builtin_methods()
    assert get_algorithm("vpu") is VPUClassifier
    assert get_algorithm("variational_pu") is VPUClassifier


def test_seed_prior_independence_and_pu_validation():
    X, y = _data()
    validation = (X, y)
    first = _model().fit(X, y, pu_validation_data=validation)
    second = _model().fit(X, y, class_prior=0.4, pu_validation_data=validation)
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
    assert first.history_ == second.history_
    assert len(first.history_["val_risk"]) == first.max_epochs
    assert np.isfinite(first.history_["val_risk"]).all()
    assert first.get_pu_metadata()["class_prior"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"depth": -1}, "depth"),
        ({"batch_size": 0}, "batch_size"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"regularization_weight": float("nan")}, "regularization_weight"),
        ({"mixup_alpha": 0}, "mixup_alpha"),
    ],
)
def test_invalid_training_parameters(kwargs, message):
    X, y = _data()
    with pytest.raises(ValueError, match=message):
        _model(**kwargs).fit(X, y)


def test_edge_input_guards_and_weight_rejection():
    X, y = _data()
    with pytest.raises(NotImplementedError, match="sample_weight"):
        _model().fit(X, y, sample_weight=np.ones(len(X)))
    with pytest.raises(ValueError, match="class_prior"):
        _model().fit(X, y, class_prior=1.0)
    with pytest.raises(ValidationError, match="2-D"):
        _model().fit(X[:, :, None], y)
    with pytest.raises(ValidationError, match="Sparse"):
        _model().fit(sparse.csr_matrix(X), y)
    with pytest.raises(ValueError, match="finite"):
        bad = X.copy()
        bad[0, 0] = np.nan
        _model().fit(bad, y)
    with pytest.raises(ValueError, match="unlabeled"):
        _model().fit(X, np.ones(len(X), dtype=int))
    with pytest.raises(ValueError, match="PU view"):
        from pu_toolbox.experiment import DatasetPart

        _model().fit(
            X,
            y,
            pu_validation_data=DatasetPart(X=X, labels=y, view="clean", indices=np.arange(len(X))),
        )
    fitted = _model().fit(X, y)
    with pytest.raises(ValueError, match="feature dimension"):
        fitted.predict(X[:, :2])
    assert fitted.predict(X[:0]).shape == (0,)
    assert fitted.predict_proba(X[:0]).shape == (0, 2)


def test_weights_only_roundtrip_preserves_calibrated_scores():
    X, y = _data()
    fitted = _model().fit(X, y)
    restored = copy.deepcopy(fitted.model_)
    restored.load_state_dict(fitted.model_.state_dict())
    with torch.no_grad():
        values = restored(torch.as_tensor(X)).flatten().numpy()
    np.testing.assert_allclose(values, fitted.decision_function(X), atol=1e-6)
    assert math.isfinite(fitted.max_log_phi_)


@pytest.mark.integration
def test_pipeline_resolves_vpu_without_prior():
    X, y = _data()
    report = PUPipeline(
        classifier="vpu",
        classifier_params={"hidden_dim": 8, "batch_size": 5},
        prior_estimator=None,
        max_epochs=2,
        cv=2,
        random_state=7,
    ).fit_evaluate(X, y)
    assert isinstance(report.final_model, VPUClassifier)
    assert report.final_model.predict(X).shape == (len(X),)


@pytest.mark.gpu
def test_gpu_training_and_prediction_use_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA; GPU gate must not silently skip")
        pytest.skip("CUDA unavailable in the default test environment")
    X, y = _data()
    model = _model(device="cuda", max_epochs=1).fit(X, y)
    assert next(model.model_.parameters()).device.type == "cuda"
    assert np.isfinite(model.decision_function(X)).all()
