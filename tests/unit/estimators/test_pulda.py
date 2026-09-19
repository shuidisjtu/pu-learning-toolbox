"""PULDA loss equations, two-stage training, and toolbox contracts."""

# ruff: noqa: N806

import copy
import os

import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError, ValidationError  # noqa: E402
from pu_toolbox.estimators.risk.pulda import (  # noqa: E402
    PULDAClassifier,
    pulda_distribution_alignment,
    pulda_two_way_margin,
    symmetric_softplus_distance,
)
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods  # noqa: E402
from pu_toolbox.workflows import PUPipeline  # noqa: E402

pytestmark = pytest.mark.unit


def _data():
    rng = np.random.RandomState(41)
    positive = rng.normal(1.2, 0.4, (10, 3)).astype(np.float32)
    unlabeled = rng.normal(-0.4, 0.9, (18, 3)).astype(np.float32)
    return np.vstack((positive, unlabeled)), np.r_[np.ones(10, int), np.zeros(18, int)]


def _model(**kwargs):
    params = {
        "class_prior": 0.35,
        "hidden_dim": 7,
        "depth": 1,
        "warmup_epochs": 1,
        "pu_epochs": 1,
        "positive_batch_size": 4,
        "unlabeled_batch_size": 7,
        "random_state": 5,
    }
    params.update(kwargs)
    return PULDAClassifier(**params)


@pytest.mark.math
def test_distribution_alignment_matches_released_loss():
    logits = torch.tensor([0.4, -0.2, 0.7, -0.8], requires_grad=True)
    labels = torch.tensor([1, 1, 0, 0])
    total, positive, unlabeled = pulda_distribution_alignment(
        logits, labels, class_prior=0.3, temperature=3.5
    )
    scores = torch.sigmoid(logits)
    expected_positive = 1 - scores[:2].mean()
    expected_unlabeled = symmetric_softplus_distance(
        scores[2:].mean(), torch.tensor(0.3), temperature=3.5
    )
    torch.testing.assert_close(positive, expected_positive)
    torch.testing.assert_close(unlabeled, expected_unlabeled)
    torch.testing.assert_close(total, 0.6 * expected_positive + expected_unlabeled)
    total.backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())


@pytest.mark.math
def test_two_way_margin_matches_released_loss():
    logits = torch.tensor([0.4, -0.2, 0.7, -0.8], requires_grad=True)
    labels = torch.tensor([1, 1, 0, 0])
    total, positive, distribution = pulda_two_way_margin(
        logits, labels, class_prior=0.3, margin=0.6, temperature=1.0
    )
    pos = logits[:2]
    unl = logits[2:]
    expected_positive = (torch.sigmoid(pos) * torch.sigmoid(-(pos - 0.6))).mean()
    positive_negative = (torch.sigmoid(pos + 0.6) * torch.sigmoid(-pos)).mean()
    unlabeled_negative = (torch.sigmoid(unl + 0.6) * torch.sigmoid(-unl)).mean()
    expected_distribution = symmetric_softplus_distance(
        unlabeled_negative, 0.3 * positive_negative, temperature=1.0
    )
    torch.testing.assert_close(positive, expected_positive)
    torch.testing.assert_close(distribution, expected_distribution)
    torch.testing.assert_close(total, 0.3 * expected_positive + expected_distribution)


def test_basic_two_stage_fit_checkpoint_registry_and_probabilities(tmp_path):
    X, y = _data()
    model = _model()
    with pytest.raises(NotFittedError):
        model.predict(X)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)
    assert model.history_["phase"] == ["warmup", "pu_mixup"]
    assert model.history_["mixup_loss"][0] == 0
    assert model.history_["mixup_loss"][1] > 0
    assert len(trajectory.checkpoints) == 2
    scores = model.decision_function(X)
    probabilities = model.predict_proba(X)
    assert scores.shape == (len(X),)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(X), scores, atol=1e-6
    )
    register_all_builtin_methods()
    assert get_algorithm("pulda") is PULDAClassifier
    assert get_algorithm("label_distribution_alignment") is PULDAClassifier


def test_determ_seeded_fit_and_prior_override_are_stable():
    X, y = _data()
    first = _model().fit(X, y)
    second = _model(class_prior=0.2).fit(X, y, class_prior=0.35)
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
    assert first.history_ == second.history_
    assert second.get_pu_metadata()["class_prior"] == 0.35


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"depth": -1}, "depth"),
        ({"warmup_epochs": 0, "pu_epochs": 0}, "at least one"),
        ({"positive_batch_size": 0}, "positive_batch_size"),
        ({"unlabeled_batch_size": 0}, "unlabeled_batch_size"),
        ({"temperature": 0}, "temperature"),
        ({"unlabeled_ema": 1}, "unlabeled_ema"),
        ({"margin_ema": float("nan")}, "margin_ema"),
        ({"mixup_alpha": 0}, "mixup_alpha"),
        ({"mixup_weight": -1}, "mixup_weight"),
    ],
)
def test_param_invalid_training_parameters(kwargs, message):
    X, y = _data()
    with pytest.raises(ValueError, match=message):
        _model(**kwargs).fit(X, y)


def test_edge_invalid_inputs_and_weights_fail_loudly():
    X, y = _data()
    with pytest.raises(NotImplementedError, match="sample_weight"):
        _model().fit(X, y, sample_weight=np.ones(len(X)))
    with pytest.raises(ValueError, match="class_prior"):
        _model(class_prior=1).fit(X, y)
    with pytest.raises(ValidationError, match="2-D"):
        _model().fit(X[:, :, None], y)
    with pytest.raises(ValidationError, match="Sparse"):
        _model().fit(sparse.csr_matrix(X), y)
    with pytest.raises(ValueError, match="finite"):
        broken = X.copy()
        broken[0, 0] = np.inf
        _model().fit(broken, y)
    fitted = _model().fit(X, y)
    with pytest.raises(ValueError, match="feature dimension"):
        fitted.predict(X[:, :2])
    assert fitted.predict(X[:0]).shape == (0,)


def test_basic_weights_roundtrip_preserves_scores():
    X, y = _data()
    fitted = _model().fit(X, y)
    restored = copy.deepcopy(fitted.model_)
    restored.load_state_dict(fitted.model_.state_dict())
    # fit resolves the default device, so the weights can be on CUDA; a bare
    # module call does no placement of its own and the input has to follow.
    device = next(restored.parameters()).device
    with torch.no_grad():
        scores = restored(torch.as_tensor(X, device=device)).flatten().cpu().numpy()
    np.testing.assert_allclose(scores, fitted.decision_function(X), atol=1e-6)


@pytest.mark.integration
def test_basic_pipeline_resolves_pulda_with_prior():
    X, y = _data()
    report = PUPipeline(
        classifier="pulda",
        classifier_params={
            "hidden_dim": 7,
            "warmup_epochs": 1,
            "pu_epochs": 1,
            "positive_batch_size": 4,
            "unlabeled_batch_size": 7,
        },
        prior_estimator=None,
        cv=2,
        random_state=5,
    ).fit_evaluate(X, y, class_prior=0.35)
    assert isinstance(report.final_model, PULDAClassifier)
    assert report.final_model.predict(X).shape == (len(X),)


@pytest.mark.gpu
def test_basic_gpu_two_stage_training_uses_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA; GPU gate must not silently skip")
        pytest.skip("CUDA unavailable in the default test environment")
    X, y = _data()
    model = _model(device="cuda").fit(X, y)
    assert next(model.model_.parameters()).device.type == "cuda"
    assert np.isfinite(model.decision_function(X)).all()
