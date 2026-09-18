"""PUET nnPU node-risk equations, randomized trees, and toolbox contracts."""

# ruff: noqa: N806

import pickle

import numpy as np
import pytest
from scipy import sparse

from pu_toolbox.core.exceptions import NotFittedError, ValidationError
from pu_toolbox.estimators.risk.puet import (
    PUExtraTreesClassifier,
    _nnpu_quadratic_node_risk,
)
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods
from pu_toolbox.workflows import PUPipeline

pytestmark = pytest.mark.unit


def _data():
    rng = np.random.RandomState(12)
    positive = rng.normal(2.0, 0.2, size=(12, 3))
    unlabeled = rng.normal(-2.0, 0.3, size=(24, 3))
    return np.vstack([positive, unlabeled]), np.r_[np.ones(12, int), np.zeros(24, int)]


def _model(**kwargs):
    params = {"class_prior": 0.4, "n_estimators": 5, "max_depth": 4, "random_state": 8}
    params.update(kwargs)
    return PUExtraTreesClassifier(**params)


@pytest.mark.math
def test_basic_nnpu_quadratic_node_risk_matches_proposition_two():
    # Wp = 2*0.1 = 0.2, Wu = 4*0.125 = 0.5, Wn = 0.3.
    assert _nnpu_quadratic_node_risk(2, 4, 0.1, 0.125) == pytest.approx(0.48)
    assert _nnpu_quadratic_node_risk(2, 0, 0.1, 0.125) == 0.0
    assert _nnpu_quadratic_node_risk(0, 4, 0.1, 0.125) == 0.0
    assert _nnpu_quadratic_node_risk(6, 1, 0.1, 0.125) == 0.0  # nnPU clamp


def test_basic_fit_vote_margin_registry_and_importances():
    X, y = _data()
    model = _model(max_candidates=8)
    with pytest.raises(NotFittedError):
        model.predict(X)
    model.fit(X, y)
    scores = model.decision_function(X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all() and np.max(np.abs(scores)) <= 1
    assert model.predict(np.array([[3.0, 3.0, 3.0]]))[0] == 1
    assert model.predict(np.array([[-3.0, -3.0, -3.0]]))[0] == 0
    assert model.feature_importances_.shape == (X.shape[1],)
    assert np.all(model.feature_importances_ >= 0)
    assert model.feature_importances_.sum() > 0
    assert len(model.trees_) == model.n_estimators
    assert np.all(model.n_leaves_ >= 1)
    assert np.all(model.tree_depths_ <= model.max_depth)
    register_all_builtin_methods()
    assert get_algorithm("puet") is PUExtraTreesClassifier
    assert get_algorithm("pu_extra_trees") is PUExtraTreesClassifier


@pytest.mark.math
def test_basic_single_split_gain_equals_closed_form_root_risk():
    X = np.r_[np.ones((4, 1)), np.zeros((4, 1))]
    y = np.r_[np.ones(4, int), np.zeros(4, int)]
    fitted = _model(n_estimators=1, max_depth=1, max_features="all").fit(X, y)
    # Root Wp=0.4, Wu=1, Wn=0.6 => risk=4*0.4*0.6=0.96;
    # both children have zero nnPU risk.
    assert fitted.feature_importances_[0] == pytest.approx(0.96)
    np.testing.assert_array_equal(fitted.n_leaves_, [2])
    np.testing.assert_array_equal(fitted.tree_depths_, [1])
    np.testing.assert_array_equal(fitted.predict(X), y)


def test_determ_same_seed_repeats_with_or_without_group_bootstrap():
    X, y = _data()
    for bootstrap in (False, True):
        first = _model(bootstrap=bootstrap).fit(X, y)
        second = _model(bootstrap=bootstrap).fit(X, y)
        np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
        np.testing.assert_array_equal(first.feature_importances_, second.feature_importances_)
        np.testing.assert_array_equal(first.n_leaves_, second.n_leaves_)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_estimators": 0}, "n_estimators"),
        ({"max_depth": 0}, "max_depth"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"max_features": "log2"}, "max_features"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"bootstrap": 1}, "bootstrap"),
        ({"class_prior": 1.0}, "class_prior"),
    ],
)
def test_param_bad_hyperparameters_fail_loudly(kwargs, message):
    X, y = _data()
    with pytest.raises(ValueError, match=message):
        _model(**kwargs).fit(X, y)


def test_param_fit_prior_override_is_recorded():
    X, y = _data()
    fitted = _model(class_prior=0.2).fit(X, y, class_prior=0.4)
    assert fitted.get_pu_metadata()["class_prior"] == 0.4


def test_edge_constant_features_and_empty_prediction():
    X = np.zeros((8, 2))
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    fitted = _model(n_estimators=2).fit(X, y)
    np.testing.assert_array_equal(fitted.n_leaves_, [1, 1])
    assert fitted.feature_importances_.sum() == 0
    assert fitted.predict(X).shape == (len(X),)
    assert fitted.decision_function(X[:0]).shape == (0,)


def test_edge_invalid_input_and_sample_weight_rejected():
    X, y = _data()
    with pytest.raises(NotImplementedError, match="sample_weight"):
        _model().fit(X, y, sample_weight=np.ones(len(X)))
    with pytest.raises(ValidationError, match="2-D"):
        _model().fit(X[:, :, None], y)
    with pytest.raises(ValidationError, match="Sparse"):
        _model().fit(sparse.csr_matrix(X), y)
    with pytest.raises(ValueError, match="finite"):
        invalid = X.copy()
        invalid[0, 0] = np.nan
        _model().fit(invalid, y)
    fitted = _model().fit(X, y)
    with pytest.raises(ValueError, match="fitted 2-D"):
        fitted.predict(X[:, :2])
    with pytest.raises(NotImplementedError, match="predict_proba"):
        fitted.predict_proba(X)


def test_basic_pickle_roundtrip_preserves_vote_margins():
    X, y = _data()
    fitted = _model().fit(X, y)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.decision_function(X), fitted.decision_function(X))


@pytest.mark.integration
def test_basic_pipeline_resolves_puet_with_explicit_prior():
    X, y = _data()
    report = PUPipeline(
        classifier="puet",
        classifier_params={"n_estimators": 3, "max_depth": 3},
        prior_estimator=None,
        cv=2,
        random_state=8,
    ).fit_evaluate(X, y, class_prior=0.4)
    assert isinstance(report.final_model, PUExtraTreesClassifier)
    assert report.final_model.predict(X).shape == (len(X),)
