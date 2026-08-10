# ruff: noqa: N802, N803, N806, E501

"""Tests for PipelineReport.summary(): prior reliability context and
assumption notes (v1.3.0 user-facing transparency fixes)."""

from __future__ import annotations

import pytest

from pu_toolbox.preprocessing.data_profiler import PUDataProfile
from pu_toolbox.workflows.report import _PRIOR_ESTIMATION_NOTE, PipelineReport, PriorInfo

pytestmark = [pytest.mark.unit]


def _report(*, prior: PriorInfo, diagnostic: dict | None = None) -> PipelineReport:
    profile = PUDataProfile(
        summary={"n_samples": 400, "n_features": 5, "positive_fraction": 0.25},
        feature_statistics={},
        selection_diagnostic=(
            diagnostic
            if diagnostic is not None
            else {"separability_auc": None, "is_identifying": False, "status": "inconclusive"}
        ),
        issues=(),
        assumption_hints=(),
    )
    return PipelineReport(
        profile=profile,
        recommendation=None,
        prior=prior,
        cv_metrics={},
        cv_provenance={"n_splits": 5},
        final_model=None,
        diagnostic=None,
        issues=(),
        provenance={"classifier": "UPUClassifier"},
    )


@pytest.mark.unit
def test_summary_estimated_prior_carries_estimator_and_boundary_note():
    """Estimated priors show the estimator and the boundary note (not just
    a bare value the user cannot trust or interpret)."""
    report = _report(
        prior=PriorInfo(
            value=0.44,
            source="estimated",
            method_requires_prior=True,
            estimator="ClassPriorEstimator",
            auto_selected=True,
        )
    )
    out = report.summary()
    assert "source: estimated, estimator: ClassPriorEstimator" in out
    assert "auto-selected: yes" in out
    assert _PRIOR_ESTIMATION_NOTE in out


@pytest.mark.unit
def test_summary_user_prior_has_no_estimator_note():
    """Explicit user priors skip the estimation boundary note."""
    report = _report(
        prior=PriorInfo(value=0.5, source="user", method_requires_prior=True)
    )
    out = report.summary()
    assert "(source: user)" in out
    assert _PRIOR_ESTIMATION_NOTE not in out


@pytest.mark.unit
def test_summary_shows_separability_auc_when_present():
    """Assumption notes surface the screening separability AUC."""
    report = _report(
        prior=PriorInfo(value=0.44, source="estimated", method_requires_prior=True),
        diagnostic={"separability_auc": 0.859, "is_identifying": False, "status": "at_risk"},
    )
    out = report.summary()
    assert "Labeled-vs-unlabeled separability AUC: 0.859" in out
    assert "not identifiable" in out


@pytest.mark.unit
def test_summary_identifying_diagnostic_omits_identifiability_caveat():
    """With true labels (identifying diagnostic) the note drops the
    'not identifiable' caveat."""
    report = _report(
        prior=PriorInfo(value=0.44, source="estimated", method_requires_prior=True),
        diagnostic={"separability_auc": 0.32, "is_identifying": True, "status": "plausible"},
    )
    out = report.summary()
    assert "Labeled-vs-unlabeled separability AUC: 0.320" in out
    assert "not identifiable" not in out
