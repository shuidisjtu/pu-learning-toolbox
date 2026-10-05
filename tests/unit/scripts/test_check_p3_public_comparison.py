"""Draft public readings must not become numeric verdicts or formal admission."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.method_ledger import load_ledger
from pu_toolbox.experiment.survey_protocol import load_protocol

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p3_public_check", ROOT / "scripts/check_p3_public_comparison.py"
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)
pytestmark = pytest.mark.unit


def facts():
    return json.loads((ROOT / check.DRAFT_REF).read_text(encoding="utf-8"))


def test_basic_determ_inventory_receipt_is_read_only_and_never_a_verdict(capsys):
    draft, ledger, protocol = facts(), load_ledger(), load_protocol()
    before = copy.deepcopy((draft, ledger, protocol))
    receipt = check.validate_draft(draft, ledger, protocol)
    assert receipt["readings_checked"] == 12 and receipt["coverage_cells"] == 15
    assert receipt["dispositions"] == {
        "source_unavailable": 3,
        "no_observed_reading": 7,
        "protocol_mismatch": 5,
    }
    assert receipt["numeric_verdicts"] == [] and receipt["formal_admission"] is False
    assert (draft, ledger, protocol) == before
    for _ in range(2):
        assert check.main([]) == 0
        output = json.loads(capsys.readouterr().out)
        assert output == {**receipt, "paper_files_checked": 0, "source_files_checked": {}}


@pytest.mark.parametrize(
    "fault",
    [
        "accepted",
        "signature",
        "policy",
        "bool_policy",
        "threshold",
        "result",
        "verdict",
        "binding",
        "missing_binding",
        "nan",
        "bool_mean",
        "unit",
        "sem",
        "repeat",
        "column",
        "producer",
        "c_axis",
        "duplicate_id",
        "ghost_ref",
        "coverage",
        "eligible",
        "missing_reading",
        "PULDA_bytes",
        "ledger_ref",
        "old_admission",
    ],
)
def test_param_false_acceptance_units_identity_and_missing_evidence_fail_closed(fault):
    draft, ledger, protocol = facts(), load_ledger(), load_protocol()
    row = draft["readings"][0]
    if fault == "accepted":
        row["review_state"] = "accepted"
    elif fault == "signature":
        draft["owner_signatures"] = ["synthetic"]
    elif fault in {"policy", "bool_policy"}:
        draft["numeric_policy"]["enabled"] = True if fault == "policy" else 0
    elif fault == "threshold":
        draft["numeric_policy"]["tolerance_pp"] = 2
    elif fault in {"result", "verdict"}:
        draft["new_method_results" if fault == "result" else "numeric_verdicts"] = [{}]
    elif fault == "binding":
        draft["bound_files_sha256"]["uv.lock"] = "0" * 64
    elif fault == "missing_binding":
        del draft["bound_files_sha256"]["uv.lock"]
    elif fault in {"nan", "bool_mean"}:
        row["mean"] = float("nan") if fault == "nan" else True
    elif fault == "unit":
        row["spread_unit"] = "fraction"
    elif fault == "sem":
        row["uncertainty_kind"] = "sem"
    elif fault == "repeat":
        row["n_repeats"] = 10
    elif fault == "column":
        row["paper_location"]["column"] = "Logistic Loss"
    elif fault == "producer":
        row["value_producer"] = "third_party_reproduction"
    elif fault == "c_axis":
        row["axis"]["name"] = "c"
    elif fault == "duplicate_id":
        draft["readings"].append(copy.deepcopy(row))
    elif fault == "ghost_ref":
        draft["coverage"][0]["reading_ids"] = ["ghost"]
    elif fault == "coverage":
        draft["coverage"].pop()
    elif fault == "eligible":
        draft["coverage"][0]["numeric_eligible"] = True
    elif fault == "missing_reading":
        draft["readings"].pop()
    elif fault == "PULDA_bytes":
        draft["source_profiles"]["pulda"]["paper_sha256"] = "0" * 64
    elif fault == "ledger_ref":
        ledger["methods"]["pulda"]["public_comparison_ref"] = "another.json"
    else:
        protocol["method_profiles"]["pulda"] = {}
    with pytest.raises(ValueError):
        check.validate_draft(draft, ledger, protocol)


def test_basic_table_columns_and_mixed_units_have_independent_golden_values():
    rows = facts()["readings"]
    grad = [r for r in rows if r["method"] == "gradpu"]
    assert [r["mean"] for r in grad] == [90.1, 91.9]  # Not adjacent Sup. 90.5/92.9.
    assert all(r["uncertainty_kind"] == "unspecified_deviation" for r in grad)
    split = [r for r in rows if r["method"] == "split_pu"]
    assert [r["mean"] for r in split] == [89.18, 90.51, 92.51]
    puet = rows[0]
    assert puet["mean"] == 79.74 and puet["spread_token"] == "0.37"
    robust = [r for r in rows if r["method"] == "robust_pu"]
    normalized = [check.normalize_reading(r) for r in robust]
    assert [r["accuracy_mean_pp"] for r in normalized] == [92.42, 89.74, 89.27, 93.6, 90.21, 88.75]
    assert [r["spread_pp"] for r in normalized] == [0.4, 0.6, 0.5, 0.6, 1.4, 0.9]
    assert all(r["numeric_eligible"] is False for r in normalized)


@pytest.mark.parametrize("fault", ["negative", "infinite", "bool", "unit", "metric", "range"])
def test_edge_normalization_rejects_invalid_units_and_non_finite_inputs(fault):
    row = facts()["readings"][0]
    if fault == "negative":
        row["spread_token"] = "-1"
    elif fault == "infinite":
        row["spread_token"] = "Infinity"
    elif fault == "bool":
        row["mean"] = False
    elif fault == "unit":
        row["mean_unit"] = "fraction"
    elif fault == "metric":
        row["metric"] = "f1"
    else:
        row["mean"] = 101
    with pytest.raises(ValueError):
        check.normalize_reading(row)


def test_basic_local_paper_check_verifies_bytes_without_loading_pdf_code(tmp_path):
    profiles = copy.deepcopy(facts()["source_profiles"])
    data = b"%PDF-1.7\nsynthetic inert fixture, not a real paper\n"
    for profile in profiles.values():
        if profile["paper_file"] is not None:
            (tmp_path / profile["paper_file"]).write_bytes(data)
            profile["paper_sha256"] = hashlib.sha256(data).hexdigest()
    assert check.check_papers(profiles, tmp_path) == 4
    (tmp_path / "puet.pdf").write_bytes(data + b"changed")
    with pytest.raises(ValueError, match="paper byte"):
        check.check_papers(profiles, tmp_path)
    profiles["puet"]["paper_file"] = "../outside.pdf"
    with pytest.raises(ValueError, match="reference"):
        check.check_papers(profiles, tmp_path)


@pytest.mark.parametrize("argument", ["unknown=/tmp", "pulda", "pulda=", "gradpu=/tmp"])
def test_param_invalid_source_arguments_return_json_without_traceback(argument, capsys):
    assert check.main(["--source-dir", argument]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_edge_missing_local_papers_and_draft_do_not_pass(tmp_path, capsys):
    assert check.main(["--paper-dir", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert check.main(["--draft", str(tmp_path / "missing.json")]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_edge_bad_source_git_checkout_returns_json_not_traceback(tmp_path, capsys):
    assert check.main(["--source-dir", f"pulda={tmp_path}"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
