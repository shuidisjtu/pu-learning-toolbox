# ruff: noqa: N803

"""Tests for source-to-target PU distribution-shift auditing."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from pu_toolbox.diagnostics import PUShiftReport, analyze_pu_shift

pytestmark = pytest.mark.unit


def _pu_labels(n_samples: int, n_positive: int) -> np.ndarray:
    labels = np.zeros(n_samples, dtype=int)
    labels[:n_positive] = 1
    return labels


@pytest.mark.math
def test_basic_identical_distributions_have_stable_normalized_weights():
    rng = np.random.default_rng(12)
    source = rng.normal(size=(240, 4))
    target = rng.normal(size=(240, 4))
    report = analyze_pu_shift(
        source,
        _pu_labels(240, 48),
        target,
        y_target_pu=_pu_labels(240, 48),
        random_state=7,
    )

    assert isinstance(report, PUShiftReport)
    assert 0.5 <= report.domain_auc < 0.60
    assert report.severity == "low"
    assert report.source_importance_weights.mean() == pytest.approx(1.0)
    assert report.weight_summary["effective_sample_fraction"] > 0.9
    assert report.adaptation_ready


def test_basic_mean_shift_is_detected_and_reported():
    rng = np.random.default_rng(4)
    source = rng.normal(loc=-1.5, size=(180, 3))
    target = rng.normal(loc=1.5, size=(120, 3))
    report = analyze_pu_shift(source, _pu_labels(180, 30), target, random_state=3)

    assert report.domain_auc > 0.95
    assert report.severity == "high"
    assert not report.adaptation_ready
    assert {issue.code for issue in report.issues} >= {"domain_shift_high", "target_pu_missing"}


@pytest.mark.math
def test_basic_effective_sample_size_matches_definition():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(90, 2))
    target = rng.normal(loc=0.8, size=(70, 2))
    report = analyze_pu_shift(source, _pu_labels(90, 15), target, cv=3)
    weights = report.source_importance_weights
    expected = weights.sum() ** 2 / np.square(weights).sum()

    assert report.weight_summary["effective_sample_size"] == pytest.approx(expected)
    assert report.weight_summary["raw_relative_upper_bound"] == pytest.approx(10.0)


def test_basic_sparse_and_4d_inputs_are_supported():
    rng = np.random.default_rng(9)
    source = sparse.csr_matrix(rng.normal(size=(50, 6)))
    target = sparse.csr_matrix(rng.normal(size=(40, 6)))
    sparse_report = analyze_pu_shift(source, _pu_labels(50, 10), target, cv=2)
    image_report = analyze_pu_shift(
        rng.normal(size=(30, 1, 2, 3)),
        _pu_labels(30, 6),
        rng.normal(size=(24, 1, 2, 3)),
        cv=2,
    )

    assert len(sparse_report.source_importance_weights) == 50
    assert image_report.sample_summary["n_features"] == 6


def test_basic_target_label_rate_warning_preserves_pu_semantics():
    rng = np.random.default_rng(10)
    source = rng.normal(size=(80, 2))
    target = rng.normal(size=(80, 2))
    report = analyze_pu_shift(
        source,
        _pu_labels(80, 8),
        target,
        y_target_pu=_pu_labels(80, 32),
        cv=4,
    )

    issue = next(item for item in report.issues if item.code == "observed_label_rate_shift")
    assert "label rate is not π" in issue.action


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"probability_clip": 0.5}, "probability_clip"),
        ({"cv": 1}, "cv"),
        ({"moderate_auc": 0.8, "high_auc": 0.7}, "AUC thresholds"),
    ],
)
def test_param_invalid_configuration_raises(kwargs, message):
    source = np.arange(40, dtype=float).reshape(20, 2)
    target = source.copy()
    with pytest.raises(ValueError, match=message):
        analyze_pu_shift(source, _pu_labels(20, 4), target, **kwargs)


def test_edge_mismatched_features_labels_and_nonfinite_raise():
    source = np.ones((20, 2))
    target = np.ones((20, 3))
    with pytest.raises(ValueError, match="same number"):
        analyze_pu_shift(source, _pu_labels(20, 4), target)
    with pytest.raises(ValueError, match="has 19 rows"):
        analyze_pu_shift(source, _pu_labels(19, 4), source)
    source[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analyze_pu_shift(source, _pu_labels(20, 4), np.ones((20, 2)))


@pytest.mark.parametrize("labels", [np.zeros(20, dtype=int), np.ones(20, dtype=int)])
def test_edge_each_pu_label_vector_needs_both_groups(labels):
    features = np.ones((20, 2))
    with pytest.raises(ValueError, match="both labeled-positive and unlabeled"):
        analyze_pu_shift(features, labels, features)


def test_determ_serialization_and_artifacts_are_strict(tmp_path):
    rng = np.random.default_rng(5)
    source = rng.normal(size=(60, 3))
    target = rng.normal(loc=0.2, size=(50, 3))
    report = analyze_pu_shift(source, _pu_labels(60, 12), target, cv=2)
    json_path = tmp_path / "shift.json"
    markdown_path = tmp_path / "shift.md"
    csv_path = tmp_path / "weights.csv"

    report.save(json_path)
    report.save(markdown_path)
    report.save(csv_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["analysis_type"] == "marginal_distribution_shift_audit"
    assert "source_importance_weights" not in payload
    assert "Interpretation Boundary" in markdown_path.read_text(encoding="utf-8")
    assert csv_path.read_text(encoding="utf-8").startswith("source_row,importance_weight")
