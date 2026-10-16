"""Readme diagnostics inspect definitions, not negated historical quotations."""

from copy import deepcopy

import pytest

from scripts.audit_survey_workbook import readme_correction_requests

pytestmark = pytest.mark.unit


def test_basic_legacy_two_column_dictionary_is_consumed():
    rows = [
        ("Class-prior grid", None),
        ("c", "Requested class-prior token for the run."),
        ("run_view", "os-compatible or ts-compatible. os = priors/oracle-free operating view."),
        ("pa_accuracy", "Test accuracy under the PA (prior-aware) selection criterion."),
    ]
    assert [x["field"] for x in readme_correction_requests(rows)] == ["c", "run_view", "PA"]


@pytest.mark.parametrize(
    "row,field",
    [
        (("c", "Requested class-prior token."), "c"),
        (("run_view", "OS = priors/oracle-free operating view."), "run_view"),
        (("PA = prior-aware selection", None), "PA"),
        (("pa_accuracy", "Test accuracy under PA (prior-aware) selection."), "PA"),
    ],
)
def test_param_affirmative_stale_definitions_are_reported(row, field):
    assert [x["field"] for x in readme_correction_requests([row])] == [field]


def test_edge_corrected_prose_does_not_trigger_on_negated_old_names():
    rows = [
        ("原 Class-prior grid / Requested class-prior token 的命名错误已纠正。", None),
        ("PA = proxy accuracy；不能仅叫 prior-aware。", None),
        ("OS/TS不表示 priors/oracle-free 或 transfer。", None),
        ("c", "Label frequency P(S=1|Y=1), not class prior."),
        (None, None),
    ]
    assert readme_correction_requests(rows) == []


def test_determ_diagnostics_are_ordered_and_do_not_mutate_input():
    rows = [
        ("pa_accuracy", "Test accuracy under PA (prior-aware) selection."),
        ("c", "Requested class-prior token."),
    ]
    before = deepcopy(rows)
    first = readme_correction_requests(rows)
    assert first == readme_correction_requests(rows)
    assert rows == before
    assert [x["field"] for x in first] == ["c", "PA"]
