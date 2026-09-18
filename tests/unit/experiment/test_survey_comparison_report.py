# ruff: noqa: N803, N806
"""Aggregation-side comparison artifact: manifest context and investigation record."""

import copy

import pytest

from pu_toolbox.experiment.survey_comparison import (
    COMPARISON_CONCLUSIONS,
    ELIGIBILITY_CLASSES,
    build_comparison_report,
    comparison_context,
    comparison_digest,
    load_comparison_protocol,
    record_investigation,
)
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def survey():
    return load_protocol()


@pytest.fixture(scope="module")
def comparison(survey):
    return load_comparison_protocol(survey=survey)


def _identity(**overrides):
    identity = {
        "method": "nnpu",
        "dataset": "imdb",
        "labeling_mechanism": "scar",
        "c_token": "0.1",
        "selection_protocol": "oa",
        "metric": "accuracy",
        "training_path": "native_2d",
    }
    identity.update(overrides)
    return identity


def _result(identity=None, **overrides):
    result = {
        "result_identity": _identity() if identity is None else identity,
        "mean": 0.802,
        "std": 0.021,
        "n_repeats": 5,
        "metric_unit": "fraction",
    }
    result.update(overrides)
    return result


def _numeric_comparison(survey, **anchor_overrides):
    """A minimal numeric matrix; the shipped v1 matrix is deliberately all non-numeric."""
    anchor = {
        "anchor_id": "synthetic_numeric_anchor",
        "source_id": "synthetic_source",
        "paper_location": {"table": "1", "page": 1},
        "method_identity": "nnpu",
        "dataset": "imdb",
        "labeling_mechanism": "scar",
        "c_token": "0.1",
        "selection_protocol": "oa",
        "metric": "accuracy",
        "mean_percent": 80.0,
        "spread_percent": 2.0,
        "uncertainty_kind": "std",
        "n_repeats": 5,
        "value_producer": "original_authors",
        "protocol_provenance": "certain",
        "transcription_evidence": "synthetic",
        "review_state": "pending_review",
        "verified_by": [],
    }
    anchor.update(anchor_overrides)
    return {
        "schema_version": "1.0",
        "comparison_version": "survey-comparison-v1",
        "review_status": "pending_collaborator_review",
        "bound_survey_protocol": {
            "protocol_version": survey["protocol_version"],
            "protocol_sha256": comparison_digest(survey),
        },
        "metric_unit": "percentage_points",
        "result_input_unit": "fraction",
        "eligibility_classes": list(ELIGIBILITY_CLASSES),
        "decision_rules": {"limit_floor_pp": 3.0, "limit_multiplier": 2.0},
        "sources": [
            {
                "source_id": "synthetic_source",
                "kind": "paper",
                "paper_title": "synthetic",
                "paper_version": "v0",
                "repository": "",
                "repository_commit": "",
                "evidence_reference": "synthetic",
                "notes": "",
            }
        ],
        "anchors": [anchor],
        "mappings": [
            {
                "mapping_id": "synthetic_unit",
                "scope": "unit",
                "result_selector": _identity(),
                "eligibility": "numeric",
                "anchor_ids": ["synthetic_numeric_anchor"],
                "protocol_differences": [],
                "rationale": "synthetic numeric case",
                "review_state": "pending_review",
                "verified_by": [],
            }
        ],
        "contradictions": [],
        "formal_blockers": ["collaborator_review"],
    }


# --- run manifest context ----------------------------------------------------


def test_basic_context_records_version_digest_and_mapping(comparison):
    context = comparison_context(_identity(), comparison=comparison)
    assert context["comparison_version"] == "survey-comparison-v1"
    assert context["comparison_sha256"] == comparison_digest(comparison)
    assert context["mapping_id"]
    # nnPU/IMDB at c=0.1 is covered by PU-Bench, but only in magnitude and trend.
    assert context["eligibility"] == "magnitude_and_trend"
    assert context["review_status"] == "pending_collaborator_review"


def test_edge_context_for_an_unmapped_unit_fails_loud(comparison):
    with pytest.raises(ValueError, match="no comparison mapping"):
        comparison_context(_identity(method="not_a_method"), comparison=comparison)


# --- aggregation report ------------------------------------------------------


def test_basic_non_numeric_units_report_not_comparable_with_reasons(comparison):
    report = build_comparison_report(
        _result(_identity(selection_protocol="pa")), comparison=comparison
    )
    assert report["status"] == "not_comparable"
    assert report["eligibility"] == "blocked_pending_pa_criterion"
    assert report["rationale"]
    assert "anchor_comparisons" not in report


def test_basic_absent_anchor_is_stated_rather_than_omitted(comparison):
    """c=0.3 has no published counterpart, and that must be written down."""
    report = build_comparison_report(_result(_identity(c_token="0.3")), comparison=comparison)
    assert report["status"] == "not_comparable"
    assert report["eligibility"] == "no_direct_anchor"
    assert report["anchor_ids"] == []
    assert report["rationale"]


def test_basic_numeric_report_carries_per_anchor_verdicts(survey):
    matrix = _numeric_comparison(survey)
    report = build_comparison_report(_result(), comparison=matrix)
    assert report["status"] == "consistent"
    assert [item["anchor_id"] for item in report["anchor_comparisons"]] == [
        "synthetic_numeric_anchor"
    ]
    assert report["anchor_comparisons"][0]["delta_pp"] == pytest.approx(0.2)


def test_basic_report_never_mutates_the_result_summary(survey):
    result = _result()
    before = copy.deepcopy(result)
    build_comparison_report(result, comparison=_numeric_comparison(survey))
    assert result == before


# --- investigation record ----------------------------------------------------


def test_edge_only_the_three_declared_conclusions_are_accepted(survey):
    assert COMPARISON_CONCLUSIONS == (
        "implementation_error",
        "protocol_difference_explains",
        "unexplained_warning",
    )
    report = build_comparison_report(_result(mean=0.70), comparison=_numeric_comparison(survey))
    assert report["status"] == "investigate"
    with pytest.raises(ValueError, match="conclusion"):
        record_investigation(report, conclusion="probably_fine", evidence="x")


def test_basic_only_an_unexplained_warning_raises_an_alert(survey):
    matrix = _numeric_comparison(survey)
    for conclusion, escalates in (
        ("implementation_error", False),
        ("protocol_difference_explains", False),
        ("unexplained_warning", True),
    ):
        report = build_comparison_report(_result(mean=0.70), comparison=matrix)
        recorded = record_investigation(report, conclusion=conclusion, evidence="probe")
        assert recorded["investigation"]["conclusion"] == conclusion
        assert recorded["investigation"]["escalates"] is escalates


def test_edge_investigation_only_attaches_to_an_open_investigation(survey):
    matrix = _numeric_comparison(survey)
    consistent = build_comparison_report(_result(), comparison=matrix)
    with pytest.raises(ValueError, match="investigat"):
        record_investigation(consistent, conclusion="unexplained_warning", evidence="x")
    not_comparable = build_comparison_report(
        _result(_identity(selection_protocol="pa")),
        comparison=load_comparison_protocol(survey=survey),
    )
    with pytest.raises(ValueError, match="investigat"):
        record_investigation(not_comparable, conclusion="unexplained_warning", evidence="x")


def test_edge_investigation_needs_evidence_and_leaves_metrics_untouched(survey):
    result = _result(mean=0.70)
    before = copy.deepcopy(result)
    report = build_comparison_report(result, comparison=_numeric_comparison(survey))
    with pytest.raises(ValueError, match="evidence"):
        record_investigation(report, conclusion="unexplained_warning", evidence="")
    recorded = record_investigation(
        report, conclusion="protocol_difference_explains", evidence="backbone differs"
    )
    assert recorded["investigation"]["state"] == "recorded"
    # The annotation layer never writes back into the measured result, and it
    # carries the converted percentage points rather than the raw fraction.
    assert result == before
    assert recorded["mean_ours_pp"] == pytest.approx(70.0)


def test_determ_report_is_reproducible_and_bound_to_the_matrix_digest(survey):
    """Same inputs, same artifact -- and the digest ties it to the frozen matrix."""
    matrix = _numeric_comparison(survey)
    first = build_comparison_report(_result(), comparison=matrix)
    second = build_comparison_report(_result(), comparison=matrix)
    assert first == second
    assert first["comparison_sha256"] == comparison_digest(matrix)
    changed = _numeric_comparison(survey, mean_percent=81.0)
    assert (
        build_comparison_report(_result(), comparison=changed)["comparison_sha256"]
        != (first["comparison_sha256"])
    )
