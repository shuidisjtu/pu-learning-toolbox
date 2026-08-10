# ruff: noqa: N806

"""Tests for the structured PU data profiler."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from pu_toolbox.preprocessing import (
    PUDataProfile,
    make_sar_dataset,
    profile_pu_data,
    scar_diagnostic,
)


@pytest.mark.unit
class TestProfileStructure:
    def test_basic_report_is_serializable_and_readable(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(120, 4))
        y = np.r_[np.ones(40, dtype=int), np.zeros(80, dtype=int)]

        report = profile_pu_data(X, y, class_prior=0.5)

        assert isinstance(report, PUDataProfile)
        assert report.summary["n_samples"] == 120
        assert report.summary["implied_label_frequency"] == pytest.approx(2 / 3)
        assert report.selection_diagnostic["evidence"] == "observed_mixture"
        assert report.selection_diagnostic["is_identifying"] is False
        # v1.3.0: the old "is_scar_plausible" name suggested SCAR was tested;
        # the renamed key is a pure screening flag for observed dependence.
        assert "is_observed_dependence_absent" in report.selection_diagnostic
        assert "is_scar_plausible" not in report.selection_diagnostic
        assert "PU data profile" in report.format_text()
        json.dumps(report.to_dict(), allow_nan=False)

    def test_param_threshold_rules(self):
        X = np.c_[np.arange(20), np.ones(20), np.arange(20) * 1e-9]
        y = np.r_[np.ones(2, dtype=int), np.zeros(18, dtype=int)]

        report = profile_pu_data(
            X,
            y,
            min_labeled_positives=5,
            max_unlabeled_to_positive=5,
            low_variance_threshold=1e-15,
        )
        codes = {issue.code for issue in report.issues}
        assert {"few_labeled_positives", "extreme_pu_imbalance"} <= codes
        assert report.feature_statistics["constant_feature_indices"] == [1]
        assert report.feature_statistics["low_variance_feature_indices"] == [2]

    def test_edge_nonfinite_features_are_reported_without_cv_failure(self):
        X = np.zeros((20, 2))
        X[0, 0] = np.nan
        X[1, 1] = np.inf
        y = np.r_[np.ones(5, dtype=int), np.zeros(15, dtype=int)]

        report = profile_pu_data(X, y)

        codes = {issue.code for issue in report.issues}
        assert {"missing_features", "infinite_features"} <= codes
        assert report.has_errors
        assert report.selection_diagnostic["status"] == "inconclusive"
        assert report.selection_diagnostic["evidence"] == "not_evaluated"

    def test_sparse_input_and_high_dimensional_warning(self):
        rng = np.random.RandomState(1)
        X = sparse.csr_matrix(rng.normal(size=(30, 40)))
        y = np.r_[np.ones(10, dtype=int), np.zeros(20, dtype=int)]

        report = profile_pu_data(X, y)

        assert report.summary["is_sparse"] is True
        assert "high_dimensional_data" in {issue.code for issue in report.issues}


@pytest.mark.unit
class TestProfileLabelChecks:
    @pytest.mark.parametrize(
        ("y", "expected_code"),
        [
            (np.zeros(20, dtype=int), "no_labeled_positives"),
            (np.ones(20, dtype=int), "no_unlabeled_samples"),
        ],
    )
    def test_edge_missing_label_group(self, y, expected_code):
        report = profile_pu_data(np.zeros((20, 2)), y)
        assert expected_code in {issue.code for issue in report.issues}
        assert report.selection_diagnostic["status"] == "inconclusive"

    def test_param_class_prior_consistency_and_validation(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.r_[np.ones(8, dtype=int), np.zeros(12, dtype=int)]

        report = profile_pu_data(X, y, class_prior=0.3)
        assert report.summary["implied_label_frequency"] == pytest.approx(4 / 3)
        assert "inconsistent_class_prior" in {issue.code for issue in report.issues}

        with pytest.raises(ValueError, match="class_prior must be in"):
            profile_pu_data(X, y, class_prior=1.0)
        with pytest.raises(ValueError, match="must be 2-D"):
            profile_pu_data(np.arange(20), y)
        with pytest.raises(ValueError, match="y_pu has 19"):
            profile_pu_data(X, y[:-1])


@pytest.mark.unit
class TestSelectionDiagnostic:
    def test_audited_scar_is_direct_and_plausible(self):
        X, y_pu, y_true, _ = make_sar_dataset(
            n_samples=2000,
            n_features=5,
            class_prior=0.5,
            separation=2.0,
            mechanism="scar",
            label_frequency=0.5,
            random_state=7,
        )

        result = scar_diagnostic(X, y_pu, y_true=y_true, cv=5)

        assert result["evidence"] == "audited_positives"
        assert result["is_identifying"] is True
        assert result["status"] == "plausible"
        assert result["separability_auc"] < 0.65

    def test_audited_sar_is_direct_and_flagged(self):
        X, y_pu, y_true, _ = make_sar_dataset(
            n_samples=2000,
            n_features=5,
            class_prior=0.5,
            separation=2.0,
            mechanism="linear",
            label_frequency=0.5,
            strength=3.0,
            random_state=7,
        )

        report = profile_pu_data(X, y_pu, y_true=y_true)

        assert report.selection_diagnostic["status"] == "at_risk"
        assert report.selection_diagnostic["is_identifying"] is True
        assert "sar_signal" in {issue.code for issue in report.issues}

    def test_observed_mode_is_explicitly_non_identifying(self):
        X, y_pu, _, _ = make_sar_dataset(
            n_samples=1000,
            n_features=4,
            class_prior=0.3,
            separation=3.0,
            mechanism="scar",
            random_state=4,
        )

        result = scar_diagnostic(X, y_pu)

        assert result["evidence"] == "observed_mixture"
        assert result["is_identifying"] is False
        assert "cannot establish SAR" in result["message"]

    def test_edge_inconclusive_and_invalid_audit(self):
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.r_[1, np.zeros(9, dtype=int)]
        result = scar_diagnostic(X, y)
        assert result["status"] == "inconclusive"
        assert np.isnan(result["separability_auc"])
        report = profile_pu_data(X, y)
        assert report.to_dict()["selection_diagnostic"]["separability_auc"] is None

        y_true = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="must be positive in y_true"):
            scar_diagnostic(X, y, y_true=y_true)
        with pytest.raises(ValueError, match="cv must be"):
            scar_diagnostic(X, y, cv=1)

    def test_deterministic_cross_validation(self):
        X, y_pu, y_true, _ = make_sar_dataset(
            n_samples=600,
            n_features=3,
            class_prior=0.5,
            mechanism="linear",
            random_state=12,
        )

        first = scar_diagnostic(X, y_pu, y_true=y_true, random_state=9)
        second = scar_diagnostic(X, y_pu, y_true=y_true, random_state=9)
        assert first["separability_auc"] == second["separability_auc"]
