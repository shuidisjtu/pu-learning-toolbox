"""The bridge to the pre-registered matrix: its contract, not ours.

Two things are asserted against the *shipped* matrix rather than a fixture,
because both are promises about a file we do not own:

* **The adapter's field names and unit.**  The matrix reads ``result_identity`` /
  ``mean`` / ``std`` / ``n_repeats`` / ``metric_unit`` and declares the unit it
  expects our results in.  Emitting a locally convenient name or unit would fail
  at the matrix boundary, long after the row that caused it was built.
* **That nothing is adjudicable yet.**  Every one of the matrix's 194 mappings is
  qualitative or blocked: the 54 PA mappings await the PA criterion, 111 have no
  direct anchor, 22 admit only a magnitude-and-trend reading, 4 are non-runnable
  and 3 are background.  Zero are ``numeric``, so the pre-registered
  ``|Δ| ≤ max(3pp, 2·SE_pooled)`` rule has nothing to fire on.  That is the
  correct state of the matrix, not a defect in this tool -- and it is pinned here
  so that P2.2's comparison appendix is not quietly expected to produce verdicts
  it cannot produce.

This module only *passes through*: it never reclassifies, adds an anchor or
relaxes a class.  The matrix owns every one of those decisions.
"""

import pytest

from pu_toolbox.experiment.survey_comparison import (
    ELIGIBILITY_CLASSES,
    load_comparison_protocol,
    resolve_comparison_unit,
)
from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.experiment.survey_summary import (
    METRIC_UNIT,
    RESULT_IDENTITY_FIELDS,
    SummaryError,
    to_result_summary,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def comparison():
    return load_comparison_protocol(survey=load_protocol())


def _identity(**overrides):
    identity = {
        "method": "dist_pu",
        "dataset": "spambase",
        "labeling_mechanism": "scar",
        "c_token": "0.1",
        "selection_protocol": "oa",
        "metric": "accuracy",
        "training_path": "native_2d",
    }
    identity.update(overrides)
    return identity


def _row(*, identity=None, mean=0.9, std=0.02, n_observed=5, metric_name="accuracy"):
    return {
        "result_identity": dict(identity or _identity()),
        "metric": {
            "metric_name": metric_name,
            "mean": mean,
            "std": std,
            "n_observed": n_observed,
            "metric_unit": METRIC_UNIT,
        },
    }


def test_basic_the_adapter_emits_exactly_the_matrix_input_contract():
    summary = to_result_summary(_row())

    assert sorted(summary) == ["mean", "metric_unit", "n_repeats", "result_identity", "std"]
    assert sorted(summary["result_identity"]) == sorted(RESULT_IDENTITY_FIELDS)
    assert summary["metric_unit"] == "fraction"
    assert summary["n_repeats"] == 5


def test_basic_the_unit_the_adapter_emits_is_the_one_the_matrix_expects(comparison):
    # Not a restatement: the matrix declares the unit it wants our results in, so
    # the adapter has to agree with a value it does not own.
    assert comparison["result_input_unit"] == METRIC_UNIT


def test_basic_a_scar_pa_identity_resolves_to_blocked_pending_the_pa_criterion(comparison):
    resolved = resolve_comparison_unit(_identity(selection_protocol="pa"), comparison=comparison)

    assert resolved["eligibility"] == "blocked_pending_pa_criterion"
    assert resolved["anchors"] == []


def test_basic_an_oracle_identity_uses_its_own_token_and_stays_background_only(comparison):
    identity = _identity(
        method="pn_oracle",
        labeling_mechanism="c_independent",
        c_token="c_independent",
        selection_protocol="oa",
    )
    resolved = resolve_comparison_unit(identity, comparison=comparison)

    assert resolved["eligibility"] == "background_only"


def test_basic_no_mapping_in_the_shipped_matrix_is_numeric_yet(comparison):
    """Pins the current state: the numeric rule has nothing to run on.

    If a mapping ever becomes ``numeric``, this test should be changed
    deliberately -- it is the signal that a numeric verdict became possible.
    """
    by_class = {}
    for mapping in comparison["mappings"]:
        by_class[mapping["eligibility"]] = by_class.get(mapping["eligibility"], 0) + 1

    assert by_class.get("numeric", 0) == 0
    assert sum(by_class.values()) == len(comparison["mappings"])
    # The matrix is still awaiting collaborator review, so even a qualitative
    # appendix is not a signed-off finding.
    assert comparison["review_status"] == "pending_collaborator_review"
    assert "collaborator_review" in comparison["formal_blockers"]


def test_basic_every_eligibility_the_matrix_uses_is_one_of_its_six_classes(comparison):
    used = {mapping["eligibility"] for mapping in comparison["mappings"]}

    assert used <= set(ELIGIBILITY_CLASSES)
    assert len(ELIGIBILITY_CLASSES) == 6
    # A class the matrix uses but the constant omits would make every reader that
    # validates against the constant reject the shipped file.
    assert used


def test_param_rejects_a_row_with_fewer_than_two_repeats():
    for n_observed in (0, 1):
        with pytest.raises(SummaryError, match="at least 2"):
            to_result_summary(_row(std=None, n_observed=n_observed))


def test_param_rejects_a_row_whose_metric_is_not_the_one_asked_for():
    with pytest.raises(SummaryError, match="not 'auc'"):
        to_result_summary(_row(), metric="auc")


def test_param_rejects_an_identity_missing_a_selector_field():
    identity = _identity()
    del identity["training_path"]

    with pytest.raises(SummaryError, match="result_identity is missing"):
        to_result_summary(_row(identity=identity))


def test_edge_a_row_claiming_repeats_but_carrying_no_number_is_refused():
    # Five repeats with no mean and no spread: the count is fine, the row is not.
    with pytest.raises(SummaryError, match="mean and a spread"):
        to_result_summary(_row(mean=None, std=None, n_observed=5))


def test_determ_the_adapter_is_a_pure_function_of_the_row():
    row = _row()
    first = to_result_summary(row)

    assert to_result_summary(row) == first
    assert row["metric"]["mean"] == 0.9
    assert "mean" not in row


def test_determ_the_adapter_orders_identity_fields_the_way_the_matrix_does(comparison):
    summary = to_result_summary(_row())

    assert list(summary["result_identity"]) == list(RESULT_IDENTITY_FIELDS)
    # Same field set the matrix's own selectors use, so a resolve() finds it.
    assert set(summary["result_identity"]) == set(comparison["mappings"][0]["result_selector"])
