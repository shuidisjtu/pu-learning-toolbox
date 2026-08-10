# ruff: noqa: N802, N803, N806, S101, S113, E501

"""Tests for PUPipeline — the end-to-end PU workflow.

Covers: full-run report contents and serialization, class-prior
resolution precedence (user > constructor > estimation), auto mode via
the recommender, registry-name parsing and fail-fast errors, metric
availability semantics (missing y_true / scores / prior), CV fold
prechecks, and run-to-run determinism.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pu_toolbox.core.base import BasePriorEstimator, BasePUClassifier
from pu_toolbox.core.config import POSITIVE_LABEL
from pu_toolbox.core.exceptions import ValidationError
from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.prior import ClassPriorEstimator
from pu_toolbox.registry.registry import get_algorithm
from pu_toolbox.workflows import DEFAULT_METRICS, PipelineError, PipelineReport, PUPipeline
from tests.helpers import make_scar_data


@pytest.mark.integration
class TestPipelineBasic:
    """End-to-end runs and prior resolution."""

    def test_full_run_returns_complete_report(self, rng):
        """A full run produces a report with all expected sections."""
        X, y_pu, y_true = make_scar_data(rng, n=150, separation=4.0)
        report = PUPipeline(classifier="upu", prior_estimator="recpe").fit_evaluate(
            X, y_pu, y_true=y_true
        )
        assert isinstance(report, PipelineReport)
        assert set(report.cv_metrics) == set(DEFAULT_METRICS)
        assert report.prior.source == "estimated"
        assert report.prior.auto_selected is False
        assert report.final_model is not None
        report.final_model.predict(X[:5])
        assert report.diagnostic is not None
        assert report.has_errors is False
        summary = report.summary()
        # Either issues were found (Summary: N error(s)...) or none were.
        assert "Summary:" in summary or "No issues detected" in summary

        # Strict JSON serialization: parseable, no NaN literals.
        payload = json.loads(report.to_json())
        assert payload["schema_version"] == "1.0"
        assert set(payload["cv_metrics"]) == set(DEFAULT_METRICS)
        assert payload["prior"]["source"] == "estimated"
        assert payload["cv_metrics"]["pu_recall"]["n_computed"] == 5

    def test_explicit_class_prior_wins(self, rng):
        """fit_evaluate(class_prior=...) takes precedence and skips estimation."""
        X, y_pu, _ = make_scar_data(rng, n=150, separation=4.0)
        report = PUPipeline(classifier="upu", prior_estimator="recpe").fit_evaluate(
            X, y_pu, class_prior=0.4
        )
        assert report.prior.source == "user"
        assert report.prior.value == pytest.approx(0.4)
        assert report.prior.estimator is None

    def test_auto_mode_uses_recommendation(self, rng):
        """auto selects a classifier via the recommender and estimates a prior."""
        X, y_pu, _ = make_scar_data(rng, n=150, separation=4.0)
        report = PUPipeline().fit_evaluate(X, y_pu)
        assert report.recommendation is not None
        assert report.provenance["classifier_mode"] == "auto"
        assert report.prior.source == "estimated"
        assert report.prior.auto_selected is True
        assert report.final_model is not None
        # v1.2.1 guard: the auto prior must land in the acceptance band so the
        # selected risk-method classifier does not degenerate (recall=0) as
        # it did with the collapsed recpe estimate (0.036 on probe data).
        assert 0.25 <= report.prior.value <= 0.75, f"prior={report.prior.value}"
        candidate_classes = {
            get_algorithm(c.name).__name__ for c in report.recommendation.candidates
        }
        assert report.provenance["classifier"] in candidate_classes

    def test_classifier_instance_uses_constructor_prior(self, rng):
        """A user-supplied instance carries its own constructor prior."""
        X, y_pu, _ = make_scar_data(rng, n=150, separation=4.0)
        pipe = PUPipeline(classifier=UPUClassifier(class_prior=0.4))
        report = pipe.fit_evaluate(X, y_pu)
        assert report.prior.source == "constructor"
        assert report.prior.value == pytest.approx(0.4)
        assert report.provenance["classifier_mode"] == "instance"


@pytest.mark.integration
class TestPipelineParameterErrors:
    """Fail-fast validation of constructor arguments."""

    def test_invalid_classifier_name_raises(self):
        with pytest.raises(PipelineError, match="Unknown classifier"):
            PUPipeline(classifier="nope")

    def test_param_prior_params_forwarded_to_estimator(self, rng):
        """prior_params reach the estimator constructor (string name path)."""
        X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
        via_params = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": 3.0})
        via_instance = PUPipeline(prior_estimator=ClassPriorEstimator(sigma=3.0))
        report_a = via_params.fit_evaluate(X, y_pu)
        report_b = via_instance.fit_evaluate(X, y_pu)
        assert report_a.prior.value == pytest.approx(report_b.prior.value)

    def test_param_prior_params_with_instance_raises(self):
        with pytest.raises(TypeError, match="prior_params cannot be combined"):
            PUPipeline(prior_estimator=ClassPriorEstimator(sigma=1.0), prior_params={"sigma": 2.0})

    def test_param_invalid_prior_param_raises(self, rng):
        X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
        pipe = PUPipeline(prior_estimator="pen_l1", prior_params={"not_a_param": 1})
        with pytest.raises(PipelineError, match="invalid prior parameters"):
            pipe.fit_evaluate(X, y_pu)

    def test_non_instantiable_classifier_raises(self):
        with pytest.raises(PipelineError, match="flip_probability"):
            PUPipeline(classifier="ldce")

    def test_missing_prior_raises(self, rng):
        X, y_pu, _ = make_scar_data(rng, n=150, separation=4.0)
        pipe = PUPipeline(classifier="upu", prior_estimator=None)
        with pytest.raises(PipelineError, match="class_prior") as excinfo:
            pipe.fit_evaluate(X, y_pu)
        assert "y_true" in str(excinfo.value)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            PUPipeline(metrics=["nope"])
        pipe = PUPipeline(metrics=["auc"])
        assert pipe.metrics == ["pu_auc_roc"]

    def test_invalid_cv_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            PUPipeline(cv=1)
        with pytest.raises(TypeError, match="split"):
            PUPipeline(cv=object())


@pytest.mark.integration
class TestPipelineEdgeCases:
    """Availability semantics and boundary validation."""

    def test_no_y_true_skips_oracle_metric(self, rng):
        X, y_pu, _ = make_scar_data(rng, n=150, separation=4.0)
        report = PUPipeline(classifier="upu").fit_evaluate(X, y_pu, class_prior=0.4)
        auc = report.cv_metrics["pu_auc_roc"]
        assert auc.available is False
        assert auc.reason is not None and "y_true" in auc.reason
        assert report.cv_metrics["pu_recall"].available is True

    def test_too_few_positives_raises(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        y_pu = np.zeros(50, dtype=int)
        y_pu[:3] = POSITIVE_LABEL
        pipe = PUPipeline(classifier="upu")
        with pytest.raises(ValidationError, match="n_splits"):
            pipe.fit_evaluate(X, y_pu, class_prior=0.4)

    def test_zero_positives_raises(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        y_pu = np.zeros(50, dtype=int)
        with pytest.raises(ValidationError):
            PUPipeline(classifier="upu").fit_evaluate(X, y_pu, class_prior=0.4)

    def test_no_decision_function_skips_score_metrics(self, rng):
        X, y_pu, y_true = make_scar_data(rng, n=150, separation=4.0)
        pipe = PUPipeline(classifier=_NoScoresClassifier())
        report = pipe.fit_evaluate(X, y_pu, y_true=y_true, class_prior=0.4)
        risk = report.cv_metrics["pu_zero_one_risk"]
        assert risk.available is False
        assert "decision" in risk.reason
        assert report.cv_metrics["pu_recall"].available is True

    def test_prior_estimation_failure_degrades_in_auto_but_raises_explicit(self, rng):
        """Auto mode degrades on estimator failure; explicit mode raises."""
        X, y_pu, _ = make_scar_data(rng, n=150, separation=4.0)
        failing = _FailingPriorEstimator()
        report = PUPipeline(prior_estimator=failing).fit_evaluate(X, y_pu)
        assert report.prior.degraded is not None
        assert report.prior.source == "none"
        assert any(i.code == "prior_estimation_failed" for i in report.issues)
        assert report.recommendation is not None
        # No-prior recommendation excludes prior-requiring methods.
        assert not any(c.metadata.requires_class_prior for c in report.recommendation.candidates)
        # Explicit classifier with a real prior requirement still raises.
        pipe = PUPipeline(classifier="upu", prior_estimator=_FailingPriorEstimator())
        with pytest.raises(PipelineError, match="failed"):
            pipe.fit_evaluate(X, y_pu)


@pytest.mark.integration
class TestPipelineDeterminism:
    """Same configuration produces identical results."""

    def test_deterministic_repeated_runs_match(self, rng):
        X, y_pu, _ = make_scar_data(rng, n=150, separation=4.0)
        first = PUPipeline(classifier="upu").fit_evaluate(X, y_pu, class_prior=0.4)
        second = PUPipeline(classifier="upu").fit_evaluate(X, y_pu, class_prior=0.4)
        for name in DEFAULT_METRICS:
            assert first.cv_metrics[name].mean == pytest.approx(second.cv_metrics[name].mean)
            assert first.cv_metrics[name].std == pytest.approx(second.cv_metrics[name].std)
        assert first.prior.value == second.prior.value


class _NoScoresClassifier(BasePUClassifier):
    """Minimal classifier without a usable decision function."""

    requires_class_prior = False

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        self._is_fitted = True
        return self

    def _predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

    def _decision_function(self, X):
        raise NotImplementedError("no scores available")


class _FailingPriorEstimator(BasePriorEstimator):
    """Prior estimator whose fit always fails."""

    def fit(self, X, y_pu):
        raise RuntimeError("boom")

    def estimate(self):
        raise AssertionError("unreachable")
