# ruff: noqa: N803, N806

"""Tests for PU uncertainty, abstention, and active review."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pu_toolbox.diagnostics import PUUncertaintyReport, analyze_pu_uncertainty

pytestmark = pytest.mark.unit


class _ProbabilityModel:
    def __init__(self, probability):
        self.probability = np.asarray(probability)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probability, self.probability])


def test_basic_abstention_reports_coverage_and_oracle_accuracy():
    probability = np.array([0.05, 0.45, 0.55, 0.95])
    report = analyze_pu_uncertainty(
        _ProbabilityModel(probability),
        np.arange(8).reshape(4, 2),
        min_confidence=0.5,
        y_true=np.array([0, 0, 1, 1]),
    )
    assert isinstance(report, PUUncertaintyReport)
    np.testing.assert_array_equal(report.selective_predictions, [0, -1, -1, 1])
    assert report.summary["coverage"] == 0.5
    assert report.summary["selective_accuracy"] == 1.0


def test_param_queries_exclude_labeled_positives_and_are_deterministic():
    probability = np.array([0.49, 0.51, 0.3, 0.8])
    y_pu = np.array([1, 0, 0, 0])
    report = analyze_pu_uncertainty(
        _ProbabilityModel(probability), np.arange(8).reshape(4, 2), y_pu=y_pu, query_budget=2
    )
    np.testing.assert_array_equal(report.query_indices, [1, 2])
    assert 0 not in report.query_indices


def test_param_shift_weighted_changes_review_priority():
    probability = np.array([0.49, 0.4, 0.3])
    report = analyze_pu_uncertainty(
        _ProbabilityModel(probability),
        np.arange(6).reshape(3, 2),
        query_budget=1,
        query_strategy="shift_weighted",
        importance_weight=np.array([0.1, 5.0, 1.0]),
    )
    assert report.query_indices.tolist() == [1]


def test_basic_diverse_queries_return_budgeted_unique_rows():
    X = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]])
    report = analyze_pu_uncertainty(
        _ProbabilityModel([0.49, 0.48, 0.51, 0.52]),
        X,
        query_budget=2,
        query_strategy="diverse_uncertainty",
        random_state=2,
    )
    assert len(report.query_indices) == 2
    assert len(set(report.query_indices)) == 2


def test_edge_invalid_options_and_missing_weights_fail():
    model = _ProbabilityModel([0.5, 0.6])
    X = np.ones((2, 1))
    with pytest.raises(ValueError, match="min_confidence"):
        analyze_pu_uncertainty(model, X, min_confidence=2)
    with pytest.raises(ValueError, match="importance_weight"):
        analyze_pu_uncertainty(model, X, query_strategy="shift_weighted")


def test_determ_report_saves_json_markdown_and_rows(tmp_path):
    report = analyze_pu_uncertainty(
        _ProbabilityModel([0.2, 0.5, 0.9]), np.ones((3, 2)), query_budget=1
    )
    report.save(tmp_path / "uncertainty.json")
    report.save(tmp_path / "uncertainty.md")
    report.save(tmp_path / "uncertainty.csv")
    payload = json.loads((tmp_path / "uncertainty.json").read_text())
    assert payload["analysis_type"] == "pu_prediction_uncertainty"
    assert len((tmp_path / "uncertainty.csv").read_text().splitlines()) == 4
