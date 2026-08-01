# ruff: noqa: N803, N806

"""Tests for composable PU diagnostic reports."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from pu_toolbox.diagnostics import PUDiagnosticReport, build_diagnostic_report


class _FixedEstimator:
    def __init__(self, predictions, scores):
        self.predictions = np.asarray(predictions)
        self.scores = np.asarray(scores)
        self.predict_calls = 0

    def fit(self, X, y):
        raise AssertionError("The report generator must never fit an estimator.")

    def predict(self, X):
        self.predict_calls += 1
        return self.predictions

    def decision_function(self, X):
        return self.scores

    def get_pu_metadata(self):
        return {"implementation_status": "native", "iteration": np.int64(3)}


class _ProbabilityEstimator:
    def __init__(self, predictions, probabilities):
        self.predictions = np.asarray(predictions)
        self.probabilities = np.asarray(probabilities)

    def predict(self, X):
        return self.predictions

    def predict_proba(self, X):
        return np.c_[1.0 - self.probabilities, self.probabilities]


@pytest.fixture
def diagnostic_data():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y_pu = np.r_[np.ones(5, dtype=int), np.zeros(15, dtype=int)]
    return X, y_pu


@pytest.mark.unit
class TestReportBasics:
    def test_basic_data_only_report_has_explicit_unavailable_metrics(self, diagnostic_data):
        X, y_pu = diagnostic_data
        report = build_diagnostic_report(X, y_pu)

        assert isinstance(report, PUDiagnosticReport)
        assert report.model["input_mode"] == "data_only"
        assert report.metrics["labeled_positive_recall"].available is False
        assert report.metrics["accuracy"].basis == "unavailable"
        assert "Predictions were not supplied" in report.metrics["labeled_positive_recall"].reason
        json.loads(report.to_json())

    def test_basic_explicit_outputs_match_golden_metrics(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        y_pu = np.array([1, 1, 0, 0])
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])

        report = build_diagnostic_report(
            X,
            y_pu,
            y_pred=y_pred,
            scores=scores,
            y_true=y_true,
            class_prior=0.5,
        )

        expected = {
            "labeled_positive_recall": 0.5,
            "unlabeled_negative_rate": 0.5,
            "predicted_positive_rate": 0.5,
            "pu_estimated_precision": 0.5,
            "pu_zero_one_risk": 0.5,
            "accuracy": 0.5,
            "f1": 0.5,
            "roc_auc": 1.0,
        }
        assert {name: metric.value for name, metric in report.metrics.items()} == expected
        assert report.metrics["pu_zero_one_risk"].basis == "class_prior_dependent"
        assert report.metrics["roc_auc"].basis == "supervised_oracle"

    def test_estimator_mode_reads_outputs_without_fitting(self, diagnostic_data):
        X, y_pu = diagnostic_data
        estimator = _FixedEstimator(np.zeros(20, dtype=int), np.linspace(-1, 1, 20))

        report = build_diagnostic_report(X, y_pu, estimator=estimator)

        assert estimator.predict_calls == 1
        assert report.model["input_mode"] == "estimator"
        assert report.model["score_source"] == "decision_function"
        assert report.model["metadata"]["iteration"] == 3
        assert "constant_predictions" in {issue.code for issue in report.issues}

    def test_param_probability_fallback(self, diagnostic_data):
        X, y_pu = diagnostic_data
        probabilities = np.linspace(0.1, 0.9, 20)
        estimator = _ProbabilityEstimator(probabilities >= 0.5, probabilities)

        report = build_diagnostic_report(X, y_pu, estimator=estimator)

        assert report.model["score_source"] == "predict_proba"
        assert report.prediction_statistics["score_min"] == pytest.approx(0.1)
        assert report.prediction_statistics["score_max"] == pytest.approx(0.9)


@pytest.mark.unit
class TestReportValidation:
    def test_param_mutually_exclusive_and_shape_validation(self, diagnostic_data):
        X, y_pu = diagnostic_data
        estimator = _FixedEstimator(np.zeros(20), np.zeros(20))

        with pytest.raises(ValueError, match="cannot be combined"):
            build_diagnostic_report(X, y_pu, estimator=estimator, y_pred=np.zeros(20))
        with pytest.raises(ValueError, match="y_pred must have 20"):
            build_diagnostic_report(X, y_pu, y_pred=np.zeros(19))
        with pytest.raises(ValueError, match=r"only \{0, 1\}"):
            build_diagnostic_report(X, y_pu, y_pred=np.full(20, -1))

    def test_edge_unfitted_estimator_has_friendly_error(self, diagnostic_data):
        X, y_pu = diagnostic_data
        with pytest.raises(ValueError, match="must be fitted"):
            build_diagnostic_report(X, y_pu, estimator=LogisticRegression())

    def test_edge_nonfinite_and_constant_scores_are_diagnosed(self, diagnostic_data):
        X, y_pu = diagnostic_data
        predictions = np.r_[np.ones(5), np.zeros(15)]
        bad = np.linspace(-1, 1, 20)
        bad[3] = np.nan
        bad_report = build_diagnostic_report(X, y_pu, y_pred=predictions, scores=bad)
        assert bad_report.has_errors
        assert "nonfinite_scores" in {issue.code for issue in bad_report.issues}

        constant_report = build_diagnostic_report(
            X,
            y_pu,
            y_pred=predictions,
            scores=np.ones(20),
        )
        assert "constant_scores" in {issue.code for issue in constant_report.issues}

    def test_edge_metrics_without_required_label_groups_are_unavailable(self):
        X = np.arange(20, dtype=float).reshape(10, 2)
        report = build_diagnostic_report(
            X,
            np.zeros(10, dtype=int),
            y_pred=np.zeros(10, dtype=int),
            scores=np.zeros(10),
            y_true=np.zeros(10, dtype=int),
            class_prior=0.2,
        )
        assert report.metrics["labeled_positive_recall"].available is False
        assert report.metrics["pu_estimated_precision"].available is False
        assert report.metrics["f1"].available is False
        assert report.metrics["roc_auc"].available is False

    def test_sparse_input_is_supported(self, diagnostic_data):
        X, y_pu = diagnostic_data
        report = build_diagnostic_report(sparse.csr_matrix(X), y_pu)
        assert report.data_profile.summary["is_sparse"] is True


@pytest.mark.unit
class TestReportRendering:
    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [(".json", '"schema_version": "1.0"'), (".md", "# PU Diagnostic Report")],
    )
    def test_param_save_formats(self, diagnostic_data, tmp_path, suffix, expected):
        X, y_pu = diagnostic_data
        report = build_diagnostic_report(X, y_pu)
        path = tmp_path / f"report{suffix}"

        returned = report.save(path)

        assert returned == path
        assert expected in path.read_text(encoding="utf-8")

    def test_edge_unknown_save_format(self, diagnostic_data, tmp_path):
        X, y_pu = diagnostic_data
        report = build_diagnostic_report(X, y_pu)
        with pytest.raises(ValueError, match="Cannot infer"):
            report.save(tmp_path / "report.txt")

    def test_deterministic_rendering(self, diagnostic_data):
        X, y_pu = diagnostic_data
        first = build_diagnostic_report(X, y_pu, random_state=9)
        second = build_diagnostic_report(X, y_pu, random_state=9)
        assert first.to_json() == second.to_json()
        assert first.to_markdown() == second.to_markdown()
