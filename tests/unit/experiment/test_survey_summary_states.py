"""A row's identity and its status: the two closed vocabularies of a report.

Both vocabularies exist because the aggregation gate speaks a different language
than a delivery record does, and mixing them is the failure these tests pin:

* the gate calls a ``(seed, c)`` unit's ``state`` ``comparable`` or ``blocked``.
  Neither is a status.  A gate-blocked unit whose metrics exist is ``partial``,
  and the word ``blocked`` must be rejected outright if it reaches ``status``.
* a c-independent row is ``c_independent`` and nothing else.  The oracle manifest
  records a nominal ``c_requested`` of ``0.1`` alongside its flag, so a reader
  that looked at ``c`` first would file the oracle among the ordinary 0.1 runs --
  the one mistake this field is most likely to attract, and the reason the flag
  is consulted before any ``c`` value at all.
"""

import pytest

from pu_toolbox.experiment.survey_summary import (
    GATE_REFUSAL_REASONS,
    STATUSES,
    SummaryError,
    choose_status,
    normalize_c_token,
    reasons_for,
    resolve_status,
    row_key,
    status_for_completeness,
    status_for_gate_block,
    status_for_gate_error,
    status_for_refusal,
)

pytestmark = pytest.mark.unit


def _manifest(**overrides):
    """A manifest carrying only the fields identity resolution reads."""
    payload = {
        "execution_unit": {
            "method": "nnpu",
            "dataset": "spambase",
            "comparability_group": "spambase/native_2d/minibatch",
        },
        "run_view": "ts-compatible",
        "c_requested_token": "0.1",
        "generation": {"train": {"mechanism": "scar", "c_requested": 0.1}},
    }
    payload.update(overrides)
    return payload


def _oracle(**overrides):
    """The shape B1's five oracle manifests actually carry.

    ``c_requested`` is a nominal ``0.1`` even though the run is c-independent --
    which is why the flag has to be read first.
    """
    payload = {
        "execution_unit": {
            "method": "pn_oracle",
            "dataset": "spambase",
            "comparability_group": "spambase/native_2d/minibatch",
        },
        "run_view": "os-compatible",
        "c_independent": True,
        "broadcast_c_values": [0.1],
        "c_requested_token": None,
        "generation": {"train": {"mechanism": "pn_oracle", "c_requested": 0.1}},
    }
    payload.update(overrides)
    return payload


def test_basic_oracle_is_its_own_token_despite_recording_a_c():
    manifest = _oracle()

    # The trap: reading c_requested first would answer "0.1" and put the oracle
    # in the same row as the ordinary 0.1 runs.
    assert normalize_c_token(manifest) == "c_independent"
    assert normalize_c_token(manifest) != "0.1"
    assert row_key(manifest, selection_protocol="OA")[-1] == "c_independent"


def test_basic_the_recorded_token_is_what_a_row_carries():
    manifest = _manifest(
        c_requested_token="0.05", generation={"train": {"mechanism": "scar", "c_requested": 0.05}}
    )

    assert normalize_c_token(manifest) == "0.05"
    key = row_key(manifest, selection_protocol="PA")
    assert key[-1] == "0.05"
    assert key[-2] == "PA"


def test_param_rejects_a_token_that_disagrees_with_the_recorded_c():
    # c_requested_token is what the runner recorded and c_requested is what the
    # aggregation gate keys on; a disagreement would file one run under two keys.
    with pytest.raises(SummaryError, match="disagrees with"):
        normalize_c_token(
            _manifest(c_requested_token="0.3", generation={"train": {"c_requested": 0.1}})
        )


def test_edge_falls_back_to_the_recorded_c_when_the_token_is_absent():
    manifest = _manifest(
        c_requested_token=None, generation={"train": {"mechanism": "scar", "c_requested": 0.05}}
    )

    assert normalize_c_token(manifest) == "0.05"


def test_param_rejects_a_manifest_with_neither_token_nor_c():
    with pytest.raises(SummaryError, match="neither c_independent nor a c token"):
        normalize_c_token(_manifest(c_requested_token=None, generation={"train": {}}))


def test_basic_priority_orders_refused_above_not_reproducible_above_partial():
    assert choose_status(["partial", "not_reproducible", "refused"]) == "refused"
    assert choose_status(["partial", "not_reproducible"]) == "not_reproducible"
    assert choose_status(["formal", "partial"]) == "partial"
    assert choose_status(["formal"]) == "formal"


def test_param_rejects_blocked_as_a_status_value():
    assert "blocked" not in STATUSES
    with pytest.raises(SummaryError, match="blocked is not a status"):
        choose_status(["comparable", "blocked"])


def test_param_formal_row_may_not_carry_reasons():
    assert reasons_for(status="formal", reasons=[]) == []
    with pytest.raises(SummaryError, match="formal rows carry no reasons"):
        reasons_for(status="formal", reasons=["missing_seed"])


def test_param_non_formal_row_must_carry_at_least_one_reason():
    assert reasons_for(status="partial", reasons=["missing_seed"]) == ["missing_seed"]
    with pytest.raises(SummaryError, match="needs at least one reason"):
        reasons_for(status="partial", reasons=[])
    with pytest.raises(SummaryError, match="unknown reason code"):
        reasons_for(status="partial", reasons=["because_i_said_so"])


def test_basic_an_unknown_blocker_still_reports_formal_blockers_present():
    status, reasons = status_for_gate_block(["protocol_deviation"])

    assert status == "partial"
    assert reasons == ["formal_blockers_present", "protocol_deviation"]
    # A blocker added later must not vanish from the report just because the
    # mapping table was not extended with it.
    assert status_for_gate_block(["a_blocker_nobody_mapped"]) == (
        "partial",
        ["formal_blockers_present"],
    )


def test_basic_a_gate_refusal_becomes_a_refused_row_with_its_own_code():
    for reason in sorted(GATE_REFUSAL_REASONS):
        assert status_for_refusal(reason) == ("refused", [reason])
    with pytest.raises(SummaryError, match="not a gate refusal reason"):
        status_for_refusal("something_else")


def test_basic_a_recognized_gate_message_selects_the_finer_reason():
    status, reasons = status_for_gate_error(
        "leaderboard fairness mismatch for dataset 'cifar10', method 'lbe': split_sha256."
    )

    assert status == "partial"
    assert reasons == ["fairness_gate_blocked", "split_mismatch"]


def test_edge_an_unrecognized_gate_message_falls_back_to_the_coarse_reason():
    status, reasons = status_for_gate_error("something the gate has never said before")

    assert status == "partial"
    assert reasons == ["fairness_gate_blocked"]


def test_determ_seed_completeness_decides_formal_partial_or_incomplete():
    assert status_for_completeness(missing_seeds=[], total_seeds=5) == ("formal", [])
    assert status_for_completeness(missing_seeds=[2], total_seeds=5) == (
        "partial",
        ["missing_seed"],
    )
    assert status_for_completeness(missing_seeds=[0, 1, 2, 3, 4], total_seeds=5) == (
        "incomplete",
        ["missing_seed"],
    )
    assert status_for_completeness(missing_seeds=[0], total_seeds=5, run_in_progress=True) == (
        "incomplete",
        ["run_in_progress"],
    )


def test_param_resolve_status_refuses_a_formal_row_that_collected_reasons():
    assert resolve_status(["partial", "formal"], ["missing_seed"]) == ("partial", ["missing_seed"])
    assert resolve_status(["formal"], []) == ("formal", [])
    with pytest.raises(SummaryError, match="formal rows carry no reasons"):
        resolve_status(["formal"], ["missing_seed"])
