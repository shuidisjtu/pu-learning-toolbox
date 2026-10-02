"""The matrix attachment: which rows may be adjudicated, and which must not be read as if.

The matrix classifies every unit, and only one of the six classes has a numeric rule
behind it.  This file pins the two gates that stand between a computed mean and a
literature verdict: the matrix's class, and our own protocol's judgement on the row.
A row that fails either is recorded with the class it has and the reason it was not
adjudicated -- never as a comparison with an empty result, which reads as agreement.

The rules themselves are asserted by ``test_survey_comparison_report.py``; the
subject here is this entry point's walk, gates, tally and rendering.

The matrix below is synthetic and minimal on purpose: what is under test is the
handling of a class, not the shipped matrix's contents.
"""

import importlib
import sys

import pytest
from _survey_summary_helpers import SCRIPTS_DIR, manifest

from pu_toolbox.experiment.survey_comparison import ELIGIBILITY_CLASSES
from pu_toolbox.experiment.survey_summary import RESULT_IDENTITY_FIELDS, matrix_selector

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
attach = importlib.import_module("compare_survey_results")

pytestmark = pytest.mark.unit


def _identity(**overrides):
    selector = matrix_selector(manifest(), selection_protocol="OA")
    selector.update(overrides)
    return {field: selector[field] for field in RESULT_IDENTITY_FIELDS}


def _row(*, status="formal", n_observed=5, identity=None, mean=0.6, spread=0.02):
    return {
        "result_identity": identity if identity is not None else _identity(),
        "row_key": {"selection_protocol": "OA"},
        "status": status,
        "metric": {
            "metric_name": "accuracy",
            "mean": mean,
            "std": spread,
            "n_observed": n_observed,
            "metric_unit": "fraction",
        },
    }


def _matrix(*, eligibility="numeric", review_state="accepted"):
    """A one-mapping matrix whose single anchor is as certain as it can be.

    Shaped like the shipped file for the fields the entry point reads, so the walk is
    exercised over a matrix of the right kind rather than over a stub whose missing
    keys would fail before reaching the gate under test.
    """
    return {
        "comparison_version": "survey-comparison-synthetic",
        "review_status": "accepted",
        "formal_blockers": [],
        "decision_rules": {"limit_floor_pp": 3.0, "limit_multiplier": 2.0},
        "mappings": [
            {
                "mapping_id": "M-synthetic",
                "eligibility": eligibility,
                "scope": "unit",
                "result_selector": _identity(),
                "anchor_ids": ["A-synthetic"],
                "rationale": "synthetic mapping for the attachment tests",
                "protocol_differences": [],
                "review_state": review_state,
            }
        ],
        "anchors": [
            {
                "anchor_id": "A-synthetic",
                "protocol_provenance": "certain",
                "uncertainty_kind": "std",
                "n_repeats": 5,
                "mean_percent": 55.0,
                "spread_percent": 2.0,
                "metric": "accuracy",
                "review_state": review_state,
            }
        ],
    }


def test_basic_a_row_the_matrix_does_not_cover_is_an_unresolved_item():
    # No mapping matches the row's selector: a coverage defect, and specifically not
    # a comparison whose anchors came back empty.
    entry = attach.compare_row(
        _row(identity=_identity(method="no_such_method")), comparison=_matrix()
    )

    assert entry["eligibility"] == attach.UNRESOLVED
    assert entry["context"] is None
    assert "no comparison mapping covers" in entry["note"]


def test_basic_a_blocked_row_is_reported_as_not_comparable_under_its_own_class():
    entry = attach.compare_row(
        _row(), comparison=_matrix(eligibility="blocked_pending_pa_criterion")
    )

    assert entry["eligibility"] == "blocked_pending_pa_criterion"
    assert entry["report"]["status"] == "not_comparable"
    assert entry["report"]["rationale"]


def test_param_a_numeric_row_that_is_not_formal_is_not_adjudicated():
    # The mean exists; what the row lost is precisely the standing a numeric
    # comparison needs, so the class is kept and the number is not computed.
    entry = attach.compare_row(_row(status="partial"), comparison=_matrix())

    assert entry["eligibility"] == "numeric"
    assert entry["report"] is None
    assert "row status is 'partial'" in entry["note"]


def test_basic_a_numeric_formal_row_is_adjudicated_through_the_matrix_rule():
    entry = attach.compare_row(_row(mean=0.56), comparison=_matrix())

    verdict = entry["report"]
    assert verdict["status"] == "consistent"
    assert verdict["mean_ours_pp"] == pytest.approx(56.0)
    assert verdict["anchor_comparisons"][0]["delta_pp"] == pytest.approx(1.0)


def test_param_a_numeric_row_with_one_seed_is_recorded_as_not_adjudicated():
    entry = attach.compare_row(_row(n_observed=1, mean=0.56, spread=None), comparison=_matrix())

    assert entry["report"] is None
    assert entry["note"].startswith("not evaluated:")


def test_basic_the_tally_separates_adjudicated_from_not_comparable():
    summary = {"rows": [_row(mean=0.56), _row(identity=_identity(method="no_such_method"))]}
    report = attach.build_comparison(summary, comparison=_matrix())

    assert report["coverage"]["rows"] == 2
    assert report["coverage"]["adjudicated"] == 1
    assert report["coverage"]["not_comparable"] == 0
    assert report["coverage"]["by_verdict"] == {"consistent": 1}
    # The uncovered row is what a reviewer must see, and it reaches the file rather
    # than only the tally.  This matrix is fully accepted, so it adds no pending row.
    assert [item["method"] for item in attach.unresolved_items(report)] == ["no_such_method"]


def test_basic_the_markdown_heading_counts_verdicts_not_reports(tmp_path):
    # A row the matrix calls numeric and our protocol calls partial gets a withheld
    # report, and a row it does not cover gets none at all: neither is a verdict, and
    # the heading is the one place a reader counts them.
    summary = {
        "rows": [
            _row(mean=0.56),
            _row(status="partial"),
            _row(identity=_identity(method="no_such_method")),
        ]
    }
    report = attach.build_comparison(summary, comparison=_matrix())
    attach.write_report(report, tmp_path)

    markdown = (tmp_path / "comparison_summary.md").read_text(encoding="utf-8")
    # Two renderings of one fact, read from the same predicate, so they cannot drift.
    assert report["coverage"]["adjudicated"] == 1
    assert f"## 数值裁决（{report['coverage']['adjudicated']} 行）" in markdown
    assert f"已裁决 {report['coverage']['adjudicated']}" in markdown


def test_edge_an_empty_summary_yields_zeroes_and_still_names_the_matrix_state():
    report = attach.build_comparison(
        {"rows": []}, comparison=_matrix(review_state="pending_review")
    )

    assert report["coverage"]["rows"] == 0
    assert report["coverage"]["adjudicated"] == 0
    assert report["matrix_state"]["anchors_pending_review"] == 1
    # The matrix's own pending review is a fact about the attachment regardless of
    # how many rows there are, so the unresolved list is never empty while it stands.
    assert attach.unresolved_items(report)[-1]["eligibility"] == "matrix_review_pending"


def test_basic_the_eligibility_classes_are_read_from_the_matrix_module():
    report = attach.build_comparison({"rows": []}, comparison=_matrix())

    assert report["eligibility_classes"] == list(ELIGIBILITY_CLASSES)
    assert set(report["coverage"]["by_eligibility"]) == set(ELIGIBILITY_CLASSES) | {
        attach.UNRESOLVED
    }


def test_determ_the_attachment_is_a_pure_function_of_the_summary_it_is_handed():
    summary = {"rows": [_row(mean=0.56), _row(identity=_identity(method="no_such_method"))]}
    matrix = _matrix()

    first = attach.build_comparison(summary, comparison=matrix)
    second = attach.build_comparison(summary, comparison=matrix)

    assert first == second
    assert first["coverage"]["by_eligibility"]["numeric"] == 1
    assert first["coverage"]["by_eligibility"][attach.UNRESOLVED] == 1
