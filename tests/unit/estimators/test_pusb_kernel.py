# ruff: noqa: N806

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from pu_toolbox.estimators.bias_aware import PUSBKernelClassifier
from pu_toolbox.estimators.bias_aware.pusb_kernel import (
    _pu_objective_and_gradient,
    _rbf_design,
    _squared_distances,
    prior_quantile_predict,
)


def _data(seed=7):
    rng = np.random.RandomState(seed)
    positive = rng.normal(1.0, 0.5, size=(18, 3))
    unlabeled = rng.normal(-0.2, 0.9, size=(30, 3))
    X = np.vstack((positive, unlabeled))
    y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(unlabeled), dtype=int)]
    return X, y


def _small_model(random_state=11):
    return PUSBKernelClassifier(
        n_basis=8,
        cv=3,
        sigma_grid=(0.5, 1.5),
        reg_grid=(0.01, 0.1),
        random_state=random_state,
        max_iter=80,
    )


@pytest.mark.unit
def test_basic_pusb_kernel_fit_exposes_selected_hyperparameters():
    X, y = _data()
    model = _small_model().fit(X, y, class_prior=0.4)

    assert model.sigma_ in model.sigma_grid
    assert model.reg_lambda_ in model.reg_grid
    assert model.cv_scores_.shape == (2, 2)
    assert model.coef_.shape == (9,)
    assert np.isfinite(model.decision_function(X)).all()
    assert set(model.predict(X)) <= {0, 1}


@pytest.mark.unit
@pytest.mark.parametrize("class_prior", [None, 0.0, 1.0, -0.1])
def test_param_pusb_kernel_rejects_invalid_class_prior(class_prior):
    X, y = _data()
    with pytest.raises(ValueError, match="class_prior"):
        _small_model().fit(X, y, class_prior=class_prior)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("n_basis", 0),
        ("cv", 1),
        ("sigma_grid", (0.0,)),
        ("reg_grid", ()),
        ("max_iter", 0),
        ("tol", 0.0),
    ],
)
def test_param_pusb_kernel_rejects_invalid_hyperparameters(parameter, value):
    X, y = _data()
    model = _small_model().set_params(**{parameter: value})
    with pytest.raises(ValueError):
        model.fit(X, y, class_prior=0.4)


@pytest.mark.unit
def test_edge_prior_quantile_uses_strict_official_threshold():
    predictions, threshold = prior_quantile_predict(np.arange(10.0), class_prior=0.3)

    assert threshold == 7.0
    assert predictions.sum() == 2
    assert predictions[7] == 0


@pytest.mark.unit
def test_edge_public_prediction_is_independent_of_batch_composition():
    X, y = _data()
    model = _small_model().fit(X, y, class_prior=0.4)

    single = model.predict(X[[0]])[0]
    batch = model.predict(np.vstack((X[[0]], X[-10:])))[0]
    assert single == batch


@pytest.mark.unit
def test_determ_objective_gradient_matches_finite_difference():
    X, y = _data()
    design = _rbf_design(_squared_distances(X, X[:5]), sigma=0.8)
    coef = np.linspace(-0.2, 0.3, design.shape[1])
    objective, gradient = _pu_objective_and_gradient(coef, design, y, 0.4, 0.07)
    epsilon = 1e-6
    numerical = np.empty_like(coef)
    for index in range(len(coef)):
        offset = np.zeros_like(coef)
        offset[index] = epsilon
        upper = _pu_objective_and_gradient(coef + offset, design, y, 0.4, 0.07)[0]
        lower = _pu_objective_and_gradient(coef - offset, design, y, 0.4, 0.07)[0]
        numerical[index] = (upper - lower) / (2.0 * epsilon)

    assert np.isfinite(objective)
    assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_determ_same_seed_reproduces_basis_folds_scores_and_predictions():
    X, y = _data()
    first = _small_model().fit(X, y, class_prior=0.4)
    second = clone(_small_model()).fit(X, y, class_prior=0.4)

    assert_allclose(first.centers_, second.centers_)
    assert_allclose(first.cv_scores_, second.cv_scores_)
    assert_allclose(first.coef_, second.coef_)
    assert np.array_equal(first.fold_ids_, second.fold_ids_)
    assert np.array_equal(first.predict(X), second.predict(X))
