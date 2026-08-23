# ruff: noqa: N803, N806

"""Synthetic protocol tests for the research joint-shift classifier."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.estimators.research import JointShiftPUClassifier, relative_joint_weight

pytestmark = pytest.mark.integration


def _joint_shift(seed=12):
    rng = np.random.default_rng(seed)

    def domain(n, prior, negative_mean):
        y = (rng.random(n) < prior).astype(int)
        X = rng.normal(size=(n, 2))
        X[y == 1] += np.array([1.8, 0.5])
        X[y == 0] += np.array([negative_mean, -0.3])
        s = ((y == 1) & (rng.random(n) < 0.5)).astype(int)
        return X, s, y

    return (*domain(100, 0.35, -0.5), *domain(90, 0.55, 0.4))


def test_basic_joint_shift_fit_produces_bounded_finite_weights():
    Xs, ys, _, Xt, yt, _ = _joint_shift()
    model = JointShiftPUClassifier(max_iter=2, n_cv_folds=2, alpha=0.2).fit(
        Xs, ys, X_target=Xt, y_target_pu=yt, class_prior=0.35, target_class_prior=0.55
    )
    assert model.n_iter_ <= 2
    assert np.isfinite(model.relative_joint_weights_).all()
    assert model.relative_joint_weights_.max() <= 5.0 / model.relative_joint_weights_.mean() + 1e-8
    assert model.predict(Xt).shape == (len(Xt),)
    assert model.get_pu_metadata()["maturity"] == "research"


def test_param_relative_transform_has_exact_bound_and_validation():
    values = relative_joint_weight(np.array([0.0, 1.0, 1e12]), alpha=0.25)
    assert values[0] == 0
    assert values[-1] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="alpha"):
        relative_joint_weight(np.ones(2), alpha=0)


def test_edge_missing_domain_or_priors_is_rejected():
    Xs, ys, _, Xt, yt, _ = _joint_shift()
    model = JointShiftPUClassifier(n_cv_folds=2)
    with pytest.raises(ValueError, match="X_target"):
        model.fit(Xs, ys, class_prior=0.35, target_class_prior=0.55)
    with pytest.raises(ValueError, match="both required"):
        model.fit(Xs, ys, X_target=Xt, y_target_pu=yt)


def test_determ_same_seed_reproduces_trace_and_predictions():
    Xs, ys, _, Xt, yt, _ = _joint_shift()
    outputs = []
    for _ in range(2):
        model = JointShiftPUClassifier(max_iter=2, n_cv_folds=2, random_state=9).fit(
            Xs, ys, X_target=Xt, y_target_pu=yt, class_prior=0.35, target_class_prior=0.55
        )
        outputs.append((model.training_trace_, model.predict(Xt)))
    assert outputs[0][0] == outputs[1][0]
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


def test_basic_target_scores_are_finite_under_prior_and_class_conditional_shift():
    Xs, ys, _, Xt, yt, y_true_target = _joint_shift()
    model = JointShiftPUClassifier(max_iter=2, n_cv_folds=2).fit(
        Xs, ys, X_target=Xt, y_target_pu=yt, class_prior=0.35, target_class_prior=0.55
    )
    scores = model.decision_function(Xt)
    assert np.isfinite(scores).all()
    assert scores[y_true_target == 1].mean() > scores[y_true_target == 0].mean()
