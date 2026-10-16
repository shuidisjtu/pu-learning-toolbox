"""Branch symmetry, independent SCAR gradient and preplanned diagnostic fits."""

import json

import numpy as np
import pytest
from _lbe_comparison_helpers import write_split
from _lbe_diagnostic_reference import SCARFixedPropensityDiagnostic, scar_objective
from run_lbe_diagnostics import diagnostic_plan, execute

from pu_toolbox import LBEClassifier
from pu_toolbox.core.exceptions import NotFittedError

pytestmark = pytest.mark.unit


def test_basic_scar_gradient_matches_independent_central_difference():
    rng = np.random.RandomState(0)
    features = rng.normal(size=(12, 3))
    observed = np.r_[np.ones(4), np.zeros(8)]
    point = np.array([0.2, -0.5, 0.1, 0.4])
    _, actual = scar_objective(point, features, observed, 0.3, 2.0)
    expected = []
    for i in range(4):
        step = np.eye(4)[i] * 1e-5
        left = scar_objective(point - step, features, observed, 0.3, 2.0)[0]
        right = scar_objective(point + step, features, observed, 0.3, 2.0)[0]
        expected.append((right - left) / 2e-5)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_basic_lbe_diagnostics_do_not_impose_prior_or_assign_branch_truth():
    features = np.random.RandomState(0).normal(size=(40, 3))
    observed = np.r_[np.ones(8), np.zeros(32)]
    fitted = LBEClassifier(n_em_iter=3).fit(features, observed, class_prior=0.4)
    assert len(fitted.posterior_max_delta_history_) == fitted.n_iter_ == 3
    assert len(fitted.m_step_iterations_) == 3
    assert fitted.initial_class_prior_ == 0.4
    diagnostics = fitted.decomposition_diagnostics_
    assert diagnostics["initial_prior_is_constraint"] is False
    assert not diagnostics["branch_semantics_identified_by_likelihood"]
    assert diagnostics["observed_likelihood_has_branch_exchange_symmetry"]
    p = fitted.predict_proba(features)[:, 1]
    c = fitted.propensity_model_.predict_proba(features)[:, 1]
    np.testing.assert_array_equal(p * c, c * p)
    # Equal L2 penalties are also unchanged when the two fitted models exchange roles.
    assert np.sum(fitted.classifier_[-1].coef_ ** 2) + np.sum(
        fitted.propensity_model_[-1].coef_ ** 2
    ) == (
        np.sum(fitted.propensity_model_[-1].coef_ ** 2) + np.sum(fitted.classifier_[-1].coef_ ** 2)
    )


def test_determ_scar_reference_is_repeatable_cloneable_and_not_registered():
    from sklearn.base import clone

    from pu_toolbox.registry import list_algorithms

    features = np.random.RandomState(1).normal(size=(50, 3))
    observed = np.r_[np.ones(10), np.zeros(40)]
    first = SCARFixedPropensityDiagnostic()
    with pytest.raises(NotFittedError):
        first.predict(features)
    first.fit(features, observed, class_prior=0.4)
    second = clone(first).fit(features, observed, class_prior=0.4)
    assert first.propensity_ == 0.5
    np.testing.assert_array_equal(first.predict(features), second.predict(features))
    assert "scar_fixed_propensity" not in list_algorithms()


@pytest.mark.parametrize("prior", [None, 0, 1, 0.1, float("nan")])
def test_param_invalid_scar_rate_or_prior_is_rejected(prior):
    features = np.zeros((20, 2))
    observed = np.r_[np.ones(5), np.zeros(15)]
    with pytest.raises(ValueError):
        SCARFixedPropensityDiagnostic().fit(features, observed, class_prior=prior)


def test_edge_plan_identity_and_output_nonoverwrite(tmp_path):
    root = write_split(tmp_path)
    plan = diagnostic_plan(root, "spambase", [0], ["0.5"])
    output = tmp_path / "output"
    result = execute(plan, output)
    assert result["runs"] == 3 and not result["formal_admission"]
    assert set(plan["variants"]) == {"linear_em20", "linear_em200", "scar_fixed_propensity"}
    assert json.loads((output / "summary.json").read_text())["runs"] == 3
    assert all(row["sample_sd"] is None for row in result["metrics"])
    assert all(
        row["training"].get("branch_exchange_likelihood_delta", 0) == 0 for row in result["rows"]
    )
    with pytest.raises(FileExistsError):
        execute(plan, output)
