# ruff: noqa: N803, N806

"""Integration tests for covariate-shift-aware PU orchestration."""

from __future__ import annotations

import json

import pytest

from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.workflows import (
    PipelineError,
    PUPipeline,
    ShiftAwarePipelineReport,
    ShiftAwarePUPipeline,
)
from tests.helpers import make_scar_data

pytestmark = pytest.mark.integration


def _domains(rng, n=80, shift=0.1):
    X_source, y_source, y_true_source = make_scar_data(rng, n=n, separation=2.0)
    X_target = X_source + rng.normal(loc=shift, scale=0.05, size=X_source.shape)
    return X_source, y_source, y_true_source, X_target, y_source.copy(), y_true_source.copy()


def test_basic_weighted_workflow_separates_source_and_target_metrics(rng):
    X_source, y_source, y_true_source, X_target, y_target, y_true_target = _domains(rng)
    workflow = ShiftAwarePUPipeline(classifier="elkan_noto", cv=3, shift_cv=3)
    report = workflow.fit_evaluate(
        X_source,
        y_source,
        X_target,
        y_target_pu=y_target,
        y_true_source=y_true_source,
        y_true_target=y_true_target,
        target_class_prior=0.5,
    )

    assert isinstance(report, ShiftAwarePipelineReport)
    assert report.adaptation_applied
    assert report.source_pipeline.provenance["sample_weight"]["supplied"]
    assert report.target_metrics["pu_auc_roc"].available
    assert len(report.target_predictions) == len(X_target)


def test_param_audit_only_mode_runs_without_target_pu(rng):
    X_source, y_source, _, X_target, _, _ = _domains(rng)
    report = ShiftAwarePUPipeline(classifier="elkan_noto", cv=2, shift_cv=2).fit_evaluate(
        X_source,
        y_source,
        X_target,
        adapt=False,
    )

    assert not report.adaptation_applied
    assert not report.shift.adaptation_ready
    assert not report.source_pipeline.provenance["sample_weight"]["supplied"]
    assert all(not metric.available for metric in report.target_metrics.values())


def test_edge_missing_target_labels_and_ignored_weight_classifier_fail(rng):
    X_source, y_source, _, X_target, y_target, _ = _domains(rng)
    workflow = ShiftAwarePUPipeline(classifier="elkan_noto", cv=2, shift_cv=2)
    with pytest.raises(PipelineError, match="requires target-domain PU labels"):
        workflow.fit_evaluate(X_source, y_source, X_target)

    ignored = ShiftAwarePUPipeline(
        pipeline=PUPipeline(classifier=UPUClassifier(class_prior=0.5), cv=2),
        shift_cv=2,
    )
    with pytest.raises(PipelineError, match="declared support is 'ignored'"):
        ignored.fit_evaluate(X_source, y_source, X_target, y_target_pu=y_target)


def test_edge_unstable_overlap_requires_explicit_override(rng):
    X_source, y_source, _, _, y_target, _ = _domains(rng)
    X_target = X_source + 20.0
    workflow = ShiftAwarePUPipeline(classifier="elkan_noto", cv=2, shift_cv=2)
    with pytest.raises(PipelineError, match="unstable importance weights"):
        workflow.fit_evaluate(X_source, y_source, X_target, y_target_pu=y_target)


def test_determ_combined_report_saves_strict_json_and_markdown(rng, tmp_path):
    X_source, y_source, _, X_target, y_target, _ = _domains(rng)
    report = ShiftAwarePUPipeline(classifier="elkan_noto", cv=2, shift_cv=2).fit_evaluate(
        X_source,
        y_source,
        X_target,
        y_target_pu=y_target,
    )
    json_path = tmp_path / "combined.json"
    markdown_path = tmp_path / "combined.md"
    report.save(json_path)
    report.save(markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["analysis_type"] == "covariate_shift_aware_pu_pipeline"
    assert payload["provenance"]["guarantee"] == "covariate_shift_only"
    assert "Source Cross-validation Metrics" in markdown_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="not CSV"):
        report.save(tmp_path / "combined.csv")
