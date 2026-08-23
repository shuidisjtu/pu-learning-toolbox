# ruff: noqa: N803, N806

"""Integration coverage for paired shift adaptation comparisons."""

from __future__ import annotations

import json

import pytest

from pu_toolbox.workflows import ShiftAwarePUPipeline, ShiftComparisonReport
from tests.helpers import make_scar_data

pytestmark = pytest.mark.integration


def _domains(rng, *, shift=0.2):
    X, y_pu, y_true = make_scar_data(rng, n=90, separation=2.0)
    return X, y_pu, y_true, X + rng.normal(shift, 0.05, X.shape)


def test_basic_comparison_runs_paired_arms_with_oracle_selection(rng):
    X, y_pu, y_true, target = _domains(rng)
    report = ShiftAwarePUPipeline(classifier="elkan_noto", cv=2, shift_cv=2).compare(
        X,
        y_pu,
        target,
        y_target_pu=y_pu,
        y_true_source=y_true,
        y_true_target=y_true,
        class_prior=0.5,
        target_class_prior=0.5,
    )
    assert isinstance(report, ShiftComparisonReport)
    assert report.weighted is not None
    assert report.primary_metric == "pu_auc_roc"
    assert report.recommendation in {"reweight_recommended", "no_clear_benefit"}
    assert report.metric_deltas["pu_auc_roc"]["available"]


def test_param_target_prior_selects_risk_without_truth(rng):
    X, y_pu, _, target = _domains(rng)
    report = ShiftAwarePUPipeline(classifier="elkan_noto", cv=2, shift_cv=2).compare(
        X, y_pu, target, y_target_pu=y_pu, class_prior=0.5, target_class_prior=0.5
    )
    assert report.primary_metric == "pu_zero_one_risk"


def test_edge_no_selection_evidence_stays_audit_only(rng):
    X, y_pu, _, target = _domains(rng)
    report = ShiftAwarePUPipeline(classifier="elkan_noto", cv=2, shift_cv=2).compare(
        X, y_pu, target, y_target_pu=y_pu
    )
    assert report.primary_metric is None
    assert report.recommendation == "audit_only"


def test_edge_pu_observed_primary_metric_is_rejected(rng):
    X, y_pu, _, target = _domains(rng)
    with pytest.raises(ValueError, match="PU-observed"):
        ShiftAwarePUPipeline(classifier="elkan_noto", cv=2, shift_cv=2).compare(
            X, y_pu, target, y_target_pu=y_pu, primary_metric="pu_recall"
        )


def test_determ_report_serialization_is_stable(rng, tmp_path):
    X, y_pu, _, target = _domains(rng)
    report = ShiftAwarePUPipeline(classifier="elkan_noto", cv=2, shift_cv=2).compare(
        X, y_pu, target, y_target_pu=y_pu, class_prior=0.5, target_class_prior=0.5
    )
    report.save(tmp_path / "comparison.json")
    report.save(tmp_path / "comparison.md")
    payload = json.loads((tmp_path / "comparison.json").read_text())
    assert payload["analysis_type"] == "shift_adaptation_comparison"
    assert "Paired Target Metrics" in (tmp_path / "comparison.md").read_text()
