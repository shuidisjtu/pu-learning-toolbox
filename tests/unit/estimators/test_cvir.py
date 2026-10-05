"""CVIR's negative selection, alpha semantics and technical adapter gates."""

# ruff: noqa: N806

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.risk.cvir import CVIRClassifier, select_cvir_negatives  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402
from pu_toolbox.experiment.method_ledger import load_ledger  # noqa: E402
from pu_toolbox.experiment.training_views import resolve_training_view  # noqa: E402
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods  # noqa: E402

pytestmark = pytest.mark.unit


def _data():
    rng = np.random.RandomState(9)
    X = np.vstack((rng.normal(1, 0.5, (8, 3)), rng.normal(-1, 0.6, (20, 3))))
    return X.astype(np.float32), np.r_[np.ones(8, int), np.zeros(20, int)]


def _model(**kwargs):
    params = dict(
        unlabeled_positive_prior=0.25,
        hidden_dim=8,
        warm_start_epochs=1,
        max_epochs=2,
        batch_size=6,
        random_state=7,
        device="cpu",
    )
    params.update(kwargs)
    return CVIRClassifier(**params)


@pytest.mark.math
def test_negative_selection_is_low_score_stable_and_never_empty():
    scores = np.array([0.5, -1.0, 0.5, 2.0, -2.0])
    np.testing.assert_array_equal(select_cvir_negatives(scores, 0.4), [4, 1, 0])
    np.testing.assert_array_equal(select_cvir_negatives(scores, 0.99), [4])
    with pytest.raises(ValueError, match="unlabeled_positive_prior"):
        select_cvir_negatives(scores, 1)
    with pytest.raises(ValueError, match="scores"):
        select_cvir_negatives(np.array([np.nan]), 0.3)


def test_fit_seed_determinism_checkpoint_registry_and_prior_gate(tmp_path):
    X, y = _data()
    model = _model()
    with pytest.raises(NotFittedError):
        model.predict(X)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)
    assert len(trajectory.checkpoints) == 3
    assert model.history_["selected_negative_count"] == [15, 15]
    assert len(model.selected_negative_indices_) == 15
    assert set(model.selected_negative_indices_) <= set(np.flatnonzero(y == 0))
    scores = model.decision_function(X)
    assert np.isfinite(scores).all() and scores.shape == (len(y),)
    np.testing.assert_array_equal(_model().fit(X, y).decision_function(X), scores)
    np.testing.assert_allclose(trajectory.checkpoints[-1].restore().decision_function(X), scores)
    register_all_builtin_methods()
    assert get_algorithm("cvir") is CVIRClassifier
    assert get_algorithm("conditional_value_ignoring_risk") is CVIRClassifier


def test_invalid_prior_or_unsupported_view_fails_loud():
    X, y = _data()
    with pytest.raises(ValueError, match="unlabeled_positive_prior"):
        _model(unlabeled_positive_prior=None).fit(X, y)
    with pytest.raises(ValueError, match="class_prior is the population prior"):
        _model().fit(X, y, class_prior=0.4)
    with pytest.raises(NotImplementedError, match="sample_weight"):
        _model().fit(X, y, sample_weight=np.ones(len(y)))
    assert (
        resolve_training_view(
            load_ledger(), "cvir", None, is_oracle=False, estimator_class=CVIRClassifier
        )
        == "os"
    )
    with pytest.raises(ValueError, match="no os_or_ts parameter"):
        resolve_training_view(
            load_ledger(), "cvir", "ts", is_oracle=False, estimator_class=CVIRClassifier
        )


@pytest.mark.parametrize("params", [{"max_epochs": 0}, {"batch_size": 0}, {"learning_rate": 0}])
def test_invalid_hyperparameters(params):
    X, y = _data()
    with pytest.raises(ValueError):
        _model(**params).fit(X, y)


@pytest.mark.gpu
def test_cuda_smoke():
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    X, y = _data()
    model = _model(device="cuda").fit(X, y)
    assert next(model.model_.parameters()).device.type == "cuda"
    assert np.isfinite(model.decision_function(X)).all()
