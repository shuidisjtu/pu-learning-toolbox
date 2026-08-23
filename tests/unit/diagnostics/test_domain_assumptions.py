# ruff: noqa: N803, N806

"""Tests for cross-domain prior and labeling-mechanism analysis."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pu_toolbox.core.base import BasePriorEstimator
from pu_toolbox.diagnostics import DomainAssumptionReport, analyze_domain_assumptions

pytestmark = pytest.mark.unit


class _RatePriorEstimator(BasePriorEstimator):
    def fit(self, X, y_pu):
        self.value_ = min(0.95, float(np.mean(y_pu == 1)) + 0.25)
        return self

    def estimate(self):
        return self.value_


def _domains():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(100, 2))
    target = rng.normal(0.2, size=(100, 2))
    source_labels = np.zeros(100, dtype=int)
    target_labels = np.zeros(100, dtype=int)
    source_labels[:20] = 1
    target_labels[:30] = 1
    return source, source_labels, target, target_labels


def test_basic_separates_prior_and_propensity_shift():
    Xs, ys, Xt, yt = _domains()
    report = analyze_domain_assumptions(
        Xs, ys, Xt, yt, source_class_prior=0.4, target_class_prior=0.6
    )
    assert isinstance(report, DomainAssumptionReport)
    assert report.conclusion == "class_prior_shift"
    assert report.source["mean_label_propensity"] == pytest.approx(0.5)
    assert report.target["mean_label_propensity"] == pytest.approx(0.5)


def test_param_propensity_change_is_distinct_from_prior_change():
    Xs, ys, Xt, yt = _domains()
    report = analyze_domain_assumptions(
        Xs, ys, Xt, yt, source_class_prior=0.5, target_class_prior=0.5
    )
    assert report.conclusion == "labeling_mechanism_shift"
    assert report.differences["mean_label_propensity"] == pytest.approx(0.2)


def test_edge_infeasible_prior_is_explicitly_inconclusive():
    Xs, ys, Xt, yt = _domains()
    report = analyze_domain_assumptions(
        Xs, ys, Xt, yt, source_class_prior=0.1, target_class_prior=0.2
    )
    assert report.conclusion == "inconclusive"
    assert report.issues[0].severity == "error"


def test_determ_sensitivity_grid_and_serialization(tmp_path):
    Xs, ys, Xt, yt = _domains()
    report = analyze_domain_assumptions(
        Xs, ys, Xt, yt, source_class_prior=0.4, target_class_prior=0.6
    )
    assert len(report.sensitivity) == 9
    path = report.save(tmp_path / "domain.json")
    payload = json.loads(path.read_text())
    assert payload["analysis_type"] == "cross_domain_pu_assumption_analysis"
    report.save(tmp_path / "domain.md")


def test_edge_bad_threshold_and_row_mismatch_fail():
    Xs, ys, Xt, yt = _domains()
    with pytest.raises(ValueError, match="prior_shift_threshold"):
        analyze_domain_assumptions(
            Xs, ys, Xt, yt, source_class_prior=0.4, target_class_prior=0.6, prior_shift_threshold=-1
        )
    with pytest.raises(ValueError, match="match"):
        analyze_domain_assumptions(
            Xs[:-1], ys, Xt, yt, source_class_prior=0.4, target_class_prior=0.6
        )


def test_basic_bootstrap_reestimates_both_priors_and_propagates_differences():
    Xs, ys, Xt, yt = _domains()
    report = analyze_domain_assumptions(
        Xs,
        ys,
        Xt,
        yt,
        prior_estimator=_RatePriorEstimator(),
        bootstrap_replicates=40,
        random_state=9,
    )
    uncertainty = report.uncertainty
    assert uncertainty["n_valid"] == 40
    assert (
        uncertainty["source"]["class_prior"]["low"] < uncertainty["source"]["class_prior"]["high"]
    )
    assert (
        uncertainty["differences"]["mean_label_propensity"]["low"]
        <= uncertainty["differences"]["mean_label_propensity"]["high"]
    )


def test_param_explicit_prior_has_degenerate_prior_ci_but_rate_uncertainty():
    Xs, ys, Xt, yt = _domains()
    report = analyze_domain_assumptions(
        Xs,
        ys,
        Xt,
        yt,
        source_class_prior=0.5,
        target_class_prior=0.6,
        bootstrap_replicates=30,
        random_state=3,
    )
    source_ci = report.uncertainty["source"]
    assert source_ci["class_prior"]["low"] == source_ci["class_prior"]["high"] == 0.5
    assert source_ci["observed_label_rate"]["low"] < source_ci["observed_label_rate"]["high"]


def test_edge_invalid_bootstrap_configuration_is_rejected():
    Xs, ys, Xt, yt = _domains()
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        analyze_domain_assumptions(
            Xs, ys, Xt, yt, source_class_prior=0.5, target_class_prior=0.6, bootstrap_replicates=1
        )
    with pytest.raises(ValueError, match="confidence"):
        analyze_domain_assumptions(
            Xs, ys, Xt, yt, source_class_prior=0.5, target_class_prior=0.6, confidence=1.0
        )


def test_determ_bootstrap_is_reproducible_with_fixed_seed():
    Xs, ys, Xt, yt = _domains()
    options = dict(
        source_class_prior=0.5,
        target_class_prior=0.6,
        bootstrap_replicates=20,
        random_state=11,
    )
    first = analyze_domain_assumptions(Xs, ys, Xt, yt, **options)
    second = analyze_domain_assumptions(Xs, ys, Xt, yt, **options)
    assert first.uncertainty == second.uncertainty
