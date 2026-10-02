"""What a finding is allowed to say about *where* a problem is.

The aggregation gate raises on the first bad group and names the rule that broke,
never the file that broke it.  A finding that picked one member out of a failing
group would manufacture a culprit, so ``members`` is mandatory for unit- and
group-scoped checks and the whole affected group is named instead.  Conversely a
manifest-scoped check may name exactly one file: claiming one file while listing
several is the same misattribution in miniature.

Refusals roll up into a single check rather than one per group, because two
findings sharing an id cannot be cited unambiguously.  Nothing is lost: each
group's rule and members still appear.

The preflight has its own file, ``test_survey_audit.py``.
"""

import json

import pytest

from pu_toolbox.experiment.survey_audit import (
    AuditError,
    make_check,
    overall_status,
    rollup_gate_check,
)

pytestmark = pytest.mark.unit


def _check(result):
    return make_check(check_id="A", result=result, severity="info", scope="global", message="x")


def test_basic_gate_refusals_roll_up_into_one_check_naming_every_member():
    first = ["b1/manifest.json", "b1/other/manifest.json"]
    second = ["b2/manifest.json"]
    check = rollup_gate_check(
        check_id="A08",
        failures=[
            (first, "unit spans training paths ['a', 'b']; a comparability group must pin one"),
            (second, "leaderboard fairness mismatch for dataset 'x', method 'y': split_sha256."),
        ],
    )

    assert check["result"] == "fail"
    assert check["scope"] == "group"
    # One check, one id: two findings sharing an id cannot be cited unambiguously.
    assert check["members"] == first + second
    assert check["reasons"] == ["fairness_gate_blocked", "group_key_mismatch", "split_mismatch"]
    assert len(check["evidence"]) == 2
    assert "2 member(s)" in check["evidence"][0]


def test_basic_overall_status_fails_on_a_failure_and_is_partial_when_one_did_not_run():
    assert overall_status([_check("pass")]) == "pass"
    assert overall_status([_check("pass"), _check("warn")]) == "pass"
    assert overall_status([_check("pass"), _check("fail")]) == "fail"
    assert overall_status([_check("pass"), _check("not_run")]) == "partial"
    assert overall_status([]) == "pass"


def test_edge_members_and_evidence_are_stored_as_strings_so_a_report_can_be_written():
    # Callers naturally pass the Path objects they discovered with; a check that
    # cannot be serialized is a report that cannot be written.
    from pathlib import Path

    path = Path("b1") / "manifest.json"
    check = rollup_gate_check(check_id="A08", failures=[([path], "some rule broke")])

    assert check["members"] == [str(path)]
    json.dumps(check)


def test_param_rejects_an_unknown_check_result():
    with pytest.raises(AuditError, match="unknown check result"):
        make_check(check_id="A01", result="ok", severity="info", scope="global", message="x")


def test_param_rejects_a_group_scoped_check_without_members():
    with pytest.raises(AuditError, match="must name its members"):
        make_check(check_id="A08", result="fail", severity="error", scope="group", message="x")


def test_param_rejects_a_manifest_scoped_check_naming_two_manifests():
    with pytest.raises(AuditError, match="exactly one manifest"):
        make_check(
            check_id="A04",
            result="fail",
            severity="error",
            scope="manifest",
            message="x",
            members=["a/manifest.json", "b/manifest.json"],
        )


def test_param_rejects_a_check_carrying_an_unknown_reason():
    with pytest.raises(AuditError, match="unknown reason code"):
        make_check(
            check_id="A08",
            result="fail",
            severity="error",
            scope="global",
            message="x",
            reasons=["because_i_said_so"],
        )


def test_param_a_gate_rollup_without_a_failure_is_refused():
    with pytest.raises(AuditError, match="at least one failure"):
        rollup_gate_check(check_id="A08", failures=[])


def test_param_a_gate_failure_without_members_is_refused():
    with pytest.raises(AuditError, match="must name the members"):
        rollup_gate_check(check_id="A08", failures=[([], "some rule broke")])


def test_param_rejects_an_unknown_check_scope():
    with pytest.raises(AuditError, match="unknown check scope"):
        make_check(check_id="A01", result="pass", severity="info", scope="run", message="x")


def test_determ_overall_status_does_not_depend_on_check_order():
    forward = [_check("pass"), _check("not_run"), _check("fail")]

    assert overall_status(forward) == overall_status(list(reversed(forward)))
