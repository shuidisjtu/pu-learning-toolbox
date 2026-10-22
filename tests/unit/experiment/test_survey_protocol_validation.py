"""Fail-closed matrix validation for collaborator review of the pilot."""

import copy
import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import (
    bind_runtime_parameters,
    load_protocol,
    resolve_unit,
)

pytestmark = pytest.mark.unit


def _load_changed(tmp_path, change):
    payload = copy.deepcopy(load_protocol())
    change(payload)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_protocol(path)


def test_basic_kldce_is_explicitly_non_runnable_for_all_three_datasets():
    protocol = load_protocol()
    rows = [row for row in protocol["execution_units"] if row["method"] == "kldce"]
    assert {row["dataset"] for row in rows} == {"spambase", "imdb", "cifar10"}
    assert all(row["runnable"] is False for row in rows)
    assert all("flip_probability" in row["non_runnable_reason"] for row in rows)
    for row in rows:
        with pytest.raises(ValueError, match="flip_probability"):
            resolve_unit(protocol, row["dataset"], "kldce")


def test_basic_smoke_verified_cifar_native_cnn_oracle_is_runnable_but_protocol_stays_draft():
    draft_path = (
        Path(__file__).resolve().parents[3]
        / "pu_toolbox"
        / "experiment"
        / "survey_protocol_v3_draft.json"
    )
    protocol = load_protocol(draft_path)

    row = resolve_unit(protocol, "cifar10", "pn_oracle", "native_cnn")

    assert row["runnable"] is True
    assert row["oracle_alignment"] == "shared_resnet18_end_to_end_and_minibatch_budget"
    assert "non_runnable_reason" not in row
    assert protocol["review_status"] != "accepted"
    budget_blocker = "P4.1-B method-specific budgets and candidate recipes are not frozen."
    assert budget_blocker in protocol["formal_blockers"]


@pytest.mark.parametrize(
    "key",
    [
        "review_status",
        "seeds",
        "candidate_pool",
        "formal_blockers",
        "selection_spec",
        "selection_spec_kind",
    ],
)
def test_param_required_top_level_review_fields_fail_closed(tmp_path, key):
    with pytest.raises(ValueError, match=key):
        _load_changed(tmp_path, lambda payload: payload.pop(key))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("review_status", "approved", "review_status"),
        ("seeds", [0, 0], "seeds"),
        ("seeds", [True], "seeds"),
        ("candidate_pool", [1], "candidate_pool"),
        ("formal_blockers", ["same", "same"], "formal_blockers"),
        ("selection_spec_kind", "runner_contract", "selection_spec_kind"),
    ],
)
def test_param_malformed_review_fields_fail_closed(tmp_path, key, value, message):
    with pytest.raises(ValueError, match=message):
        _load_changed(tmp_path, lambda payload: payload.__setitem__(key, value))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runnable", "false", "JSON boolean"),
        ("runnable", 0, "JSON boolean"),
        ("non_runnable_reason", None, "non_runnable_reason"),
    ],
)
def test_param_execution_rows_reject_ambiguous_runnability(tmp_path, field, value, message):
    def change(payload):
        row = next(row for row in payload["execution_units"] if not row["runnable"])
        row[field] = value

    with pytest.raises(ValueError, match=message):
        _load_changed(tmp_path, change)


def test_edge_missing_row_field_and_runnable_reason_rejected(tmp_path):
    def missing_field(payload):
        payload["execution_units"][0].pop("comparability_group")

    with pytest.raises(ValueError, match="comparability_group"):
        _load_changed(tmp_path, missing_field)

    def missing_reason(payload):
        row = next(row for row in payload["execution_units"] if not row["runnable"])
        row.pop("non_runnable_reason")

    with pytest.raises(ValueError, match="non_runnable_reason"):
        _load_changed(tmp_path, missing_reason)


def test_basic_self_pu_budget_and_selection_spec_prose_are_explicit():
    protocol = load_protocol()
    assert protocol["selection_spec_kind"] == "descriptive_documentation"
    assert protocol["budgets"]["two_student_sampled"]["optimizer_steps_per_epoch"] == 2
    for row in protocol["execution_units"]:
        if row["method"] == "self_pu":
            assert row["budget"] == "two_student_sampled"
            assert row["comparability_group"].endswith("/two_student_sampled")
            assert "no_matched" in row["oracle_alignment"]


def test_determ_review_matrix_and_kldce_rejection_are_stable():
    first, second = load_protocol(), load_protocol()
    assert first == second
    for protocol in (first, second):
        with pytest.raises(ValueError, match="flip_probability"):
            resolve_unit(protocol, "spambase", "kldce")


def test_kldce_runtime_binding_is_scar_only_and_uses_realized_c():
    draft_path = (
        Path(__file__).resolve().parents[3]
        / "pu_toolbox"
        / "experiment"
        / "survey_protocol_v3_draft.json"
    )
    profile = load_protocol(draft_path)["method_profiles"]["kldce"]

    assert bind_runtime_parameters("kldce", {}, profile, mechanism="scar", c_realized=0.1) == {
        "flip_probability": pytest.approx(0.9)
    }
    with pytest.raises(ValueError, match="supports labeling mechanisms"):
        bind_runtime_parameters("kldce", {}, profile, mechanism="sar_lbe_a", c_realized=0.5)
    with pytest.raises(ValueError, match="cannot be overridden"):
        bind_runtime_parameters(
            "kldce", {"flip_probability": 0.4}, profile, mechanism="scar", c_realized=0.6
        )
    with pytest.raises(ValueError, match="supports labeling mechanisms"):
        bind_runtime_parameters(
            "kldce", {"flip_probability": 0.4}, profile, mechanism="sar_lbe_b", c_realized=0.5
        )
