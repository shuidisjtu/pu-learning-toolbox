# ruff: noqa: N803

"""Tests for PU class-prior and labeling-propensity sensitivity."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pu_toolbox.diagnostics import PUSensitivityAnalysis, analyze_pu_sensitivity


@pytest.fixture
def sensitivity_data():
    y_pu = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 0])
    return y_pu, y_pred


@pytest.mark.unit
class TestSensitivityMath:
    @pytest.mark.math
    def test_basic_class_prior_golden_values(self, sensitivity_data):
        y_pu, y_pred = sensitivity_data
        result = analyze_pu_sensitivity(
            y_pu,
            y_pred,
            class_priors=[0.2, 0.25, 0.5],
        )

        assert isinstance(result, PUSensitivityAnalysis)
        assert result.observed_label_rate == pytest.approx(0.25)
        first, second, third = result.points
        assert first.label_propensity == pytest.approx(1.25)
        assert first.is_consistent is False
        assert second.label_propensity == pytest.approx(1.0)
        assert third.label_propensity == pytest.approx(0.5)
        assert third.pu_estimated_precision == pytest.approx(2.0 / 3.0)
        assert third.pu_zero_one_risk == pytest.approx(1.0 / 3.0)

    @pytest.mark.math
    def test_basic_label_propensity_golden_values(self, sensitivity_data):
        y_pu, y_pred = sensitivity_data
        result = analyze_pu_sensitivity(
            y_pu,
            y_pred,
            label_propensities=[0.2, 0.25, 0.5, 1.0],
        )

        inconsistent, boundary, middle, full = result.points
        assert inconsistent.class_prior == pytest.approx(1.25)
        assert inconsistent.is_consistent is False
        assert inconsistent.pu_estimated_precision is None
        assert boundary.class_prior == pytest.approx(1.0)
        assert boundary.is_consistent is True
        assert boundary.pu_zero_one_risk is None
        assert middle.class_prior == pytest.approx(0.5)
        assert full.class_prior == pytest.approx(0.25)

    def test_basic_metric_ranges_use_only_consistent_available_points(self, sensitivity_data):
        y_pu, y_pred = sensitivity_data
        result = analyze_pu_sensitivity(
            y_pu,
            y_pred,
            class_priors=[0.2, 0.25, 0.5],
            label_propensities=[0.2, 0.5, 1.0],
        )

        prior_precision = result.metric_ranges["class_prior"]["pu_estimated_precision"]
        assert prior_precision["n_available"] == 2
        assert prior_precision["minimum"] == pytest.approx(1.0 / 3.0)
        assert prior_precision["maximum"] == pytest.approx(2.0 / 3.0)
        assert result.has_inconsistent_assumptions
        assert len(result.to_frame(axis="label_propensity")) == 3

    @pytest.mark.math
    def test_param_explicit_scores_control_risk_only(self, sensitivity_data):
        y_pu, y_pred = sensitivity_data
        scores = np.array([1.0, 0.5, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        result = analyze_pu_sensitivity(
            y_pu,
            y_pred,
            class_priors=[0.5],
            scores=scores,
        )

        point = result.points[0]
        assert point.pu_estimated_precision == pytest.approx(2.0 / 3.0)
        assert point.pu_zero_one_risk == pytest.approx(-1.0 / 3.0)
        assert result.provenance["score_source"] == "explicit_scores"


@pytest.mark.unit
class TestSensitivityValidation:
    def test_param_invalid_grids_raise_friendly_errors(self, sensitivity_data):
        y_pu, y_pred = sensitivity_data
        with pytest.raises(ValueError, match="Supply class_priors"):
            analyze_pu_sensitivity(y_pu, y_pred)
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            analyze_pu_sensitivity(y_pu, y_pred, class_priors=[1.0])
        with pytest.raises(ValueError, match="duplicate"):
            analyze_pu_sensitivity(y_pu, y_pred, label_propensities=[0.5, 0.5])
        with pytest.raises(ValueError, match="cannot be empty"):
            analyze_pu_sensitivity(y_pu, y_pred, class_priors=[])

    def test_param_invalid_predictions_and_scores_raise(self, sensitivity_data):
        y_pu, y_pred = sensitivity_data
        with pytest.raises(ValueError, match="same length"):
            analyze_pu_sensitivity(y_pu, y_pred[:-1], class_priors=[0.5])
        with pytest.raises(ValueError, match=r"only \{0, 1\}"):
            analyze_pu_sensitivity(y_pu, np.full(8, -1), class_priors=[0.5])
        scores = np.ones(8)
        scores[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            analyze_pu_sensitivity(y_pu, y_pred, class_priors=[0.5], scores=scores)

    @pytest.mark.parametrize("y_pu", [np.zeros(8, dtype=int), np.ones(8, dtype=int)])
    def test_edge_single_observed_group_raises(self, y_pu):
        with pytest.raises(ValueError, match="both labeled-positive and unlabeled"):
            analyze_pu_sensitivity(y_pu, np.zeros(8), class_priors=[0.5])

    def test_edge_unknown_axis_and_save_format_raise(self, sensitivity_data, tmp_path):
        y_pu, y_pred = sensitivity_data
        result = analyze_pu_sensitivity(y_pu, y_pred, class_priors=[0.5])
        with pytest.raises(ValueError, match="axis must be"):
            result.to_frame(axis="unknown")
        with pytest.raises(ValueError, match="Cannot infer"):
            result.save(tmp_path / "result.txt")


@pytest.mark.unit
class TestSensitivityRendering:
    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            (".json", '"schema_version": "1.0"'),
            (".md", "# PU Assumption Sensitivity Analysis"),
            (".csv", "axis,value,class_prior"),
        ],
    )
    def test_param_save_formats(self, sensitivity_data, tmp_path, suffix, expected):
        y_pu, y_pred = sensitivity_data
        result = analyze_pu_sensitivity(
            y_pu,
            y_pred,
            class_priors=[0.25, 0.5],
            label_propensities=[0.5, 1.0],
        )
        path = tmp_path / f"sensitivity{suffix}"

        assert result.save(path) == path
        assert expected in path.read_text(encoding="utf-8")

    def test_deterministic_json_markdown_and_frame(self, sensitivity_data):
        y_pu, y_pred = sensitivity_data
        kwargs = {
            "class_priors": [0.25, 0.5],
            "label_propensities": [0.5, 1.0],
        }
        first = analyze_pu_sensitivity(y_pu, y_pred, **kwargs)
        second = analyze_pu_sensitivity(y_pu, y_pred, **kwargs)

        assert json.loads(first.to_json()) == json.loads(second.to_json())
        assert first.to_markdown() == second.to_markdown()
        assert first.to_frame().equals(second.to_frame())
