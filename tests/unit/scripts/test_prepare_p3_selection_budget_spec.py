"""Current engineering facts must never become approved method-specific recipes."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.method_ledger import load_ledger
from pu_toolbox.experiment.survey_protocol import load_protocol

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p3_selection_spec", ROOT / "scripts/prepare_p3_selection_budget_spec.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
pytestmark = pytest.mark.unit


def _recorded():
    return json.loads((ROOT / prepare.DRAFT_REF).read_text(encoding="utf-8"))


def test_basic_determ_spec_matches_current_ast_and_preserves_input(capsys):
    ledger, protocol = load_ledger(), load_protocol()
    before = copy.deepcopy((ledger, protocol))
    generated = prepare.build_spec(ledger, protocol)
    assert generated == _recorded()
    assert prepare.build_spec(ledger, protocol) == generated
    assert (ledger, protocol) == before
    assert prepare.main(["--check", str(ROOT / prepare.DRAFT_REF)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {
        "ok": True,
        "methods_checked": list(prepare.METHODS),
        "formal_admission": False,
    }
    assert prepare.main([]) == 0
    assert json.loads(capsys.readouterr().out) == generated


def test_basic_budget_units_stage_scope_and_approvals_are_separate():
    generated = _recorded()
    assert generated["owner_signatures"] == []
    assert generated["review_status"] == "pending_method_owner_review"
    assert generated["storage_contract"]["reclaim_reduces_training_peak"] is False
    expected = {"pulda": 120, "puet": None, "gradpu": 200, "robust_pu": 30, "split_pu": 40}
    for method, row in generated["methods"].items():
        budget = row["budget_fact"]
        assert budget["default_snapshot_count"] == expected[method]
        assert budget["bound_only_due_to_early_stop"] is (method == "split_pu")
        assert budget["defaults_are_approved_budget"] is False
        assert row["checkpoint_fact"]["stages"] == list(prepare.STAGES[method])
        assert (
            row["current_technical_selection"]["current_behavior_approves_formal_stage_eligibility"]
            is False
        )
        decisions = row["formal_decisions"]
        assert decisions.pop("admitted") is False
        assert decisions.pop("GPU_exemption_approved") is False
        assert set(decisions.values()) == {None}
    tree = generated["methods"]["puet"]
    assert tree["budget_fact"]["trees_per_default_fit"] == 100
    assert tree["budget_fact"]["optimizer_steps_expression"] is None
    assert tree["checkpoint_fact"]["single_final_fit_without_epoch_callback"] is True
    contract = generated["fit_failure_contract"]
    assert contract["scope"] == list(prepare.METHODS)
    assert contract["helper_ref"] == prepare.FIT_GUARD_REF
    assert contract["helper_sha256"] == prepare.source_digest(ROOT / prepare.FIT_GUARD_REF)
    assert contract["invalidate_before_argument_binding"] is True
    assert contract["invalidate_on"] == "BaseException_then_reraise_same_object"
    assert contract["successful_fit_and_callback_visibility_owned_by_estimator"] is True
    assert contract["learned_attributes_or_external_artifacts_rolled_back"] is False
    assert contract["training_retry_or_resume_added"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "signature",
        "status",
        "budget",
        "eligible_PA",
        "eligible_OA",
        "rounds",
        "threshold",
        "candidate",
        "admitted",
        "GPU_exemption",
        "stage",
        "counter",
        "source",
        "binding",
        "ghost",
        "fit_guard_source",
        "fit_rollback",
    ],
)
def test_param_modified_specs_are_rejected_without_false_approval(tmp_path, fault, capsys):
    recorded = _recorded()
    row = recorded["methods"]["split_pu"]
    if fault == "signature":
        recorded["owner_signatures"] = ["synthetic"]
    elif fault == "status":
        recorded["review_status"] = "approved"
    elif fault in {
        "budget",
        "eligible_PA",
        "eligible_OA",
        "rounds",
        "threshold",
        "candidate",
        "admitted",
        "GPU_exemption",
    }:
        field = {
            "budget": "approved_budget",
            "eligible_PA": "eligible_stages_PA",
            "eligible_OA": "eligible_stages_OA",
            "rounds": "eligible_student_rounds",
            "threshold": "threshold_policy_ref",
            "candidate": "candidate_pool",
            "admitted": "admitted",
            "GPU_exemption": "GPU_exemption_approved",
        }[fault]
        row["formal_decisions"][field] = True
    elif fault == "stage":
        row["checkpoint_fact"]["stages"] = ["student"]
    elif fault == "counter":
        row["budget_fact"]["default_snapshot_count"] = True
    elif fault == "source":
        row["implementation_sha256"] = "0" * 64
    elif fault == "binding":
        recorded["bound_files_sha256"]["uv.lock"] = "0" * 64
    elif fault == "fit_guard_source":
        recorded["fit_failure_contract"]["helper_sha256"] = "0" * 64
    elif fault == "fit_rollback":
        recorded["fit_failure_contract"]["learned_attributes_or_external_artifacts_rolled_back"] = (
            True
        )
    else:
        recorded["methods"]["ghost"] = copy.deepcopy(row)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(recorded), encoding="utf-8")
    assert prepare.main(["--check", str(path)]) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["ok"] is False and "draft drift" in receipt["error"]


@pytest.mark.parametrize("fault", ["spec_ref", "old_admission", "calibration", "class", "fit_ref"])
def test_param_bad_upstream_facts_cannot_generate_a_consistent_draft(fault):
    ledger, protocol = load_ledger(), load_protocol()
    if fault == "spec_ref":
        ledger["methods"]["pulda"]["selection_budget_spec_ref"] = "wrong.json"
    elif fault == "old_admission":
        protocol["method_profiles"]["pulda"] = {}
    elif fault == "calibration":
        ledger["methods"]["pulda"]["calibration_applied"] = False
    elif fault == "fit_ref":
        ledger["methods"]["pulda"]["fit_failure_review_ref"] = "wrong.md"
    else:
        ledger["methods"]["pulda"]["class"] = "WrongClass"
    with pytest.raises(ValueError):
        prepare.build_spec(ledger, protocol)


@pytest.mark.parametrize(
    "text,match",
    [
        ("class M:\n def fit(self, X, y, **kwargs): pass\n", "silently"),
        ("class M:\n def fit(self, X, y, *, clean_val=None): pass\n", "role leak"),
        ("class M:\n checkpoint_stages = ['warmup']\n def fit(self, X, y): pass\n", "nonliteral"),
        ("class M:\n def fit(self, X, y): pass\n", "missing failed-fit"),
    ],
)
def test_param_ast_role_and_stage_interface_checks_are_fail_closed(tmp_path, text, match):
    path = tmp_path / "synthetic.py"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        prepare._interface(path, "M")


def test_edge_missing_check_input_returns_json_not_traceback(tmp_path, capsys):
    assert prepare.main(["--check", str(tmp_path / "missing.json")]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_edge_shared_helper_changes_invalidate_recorded_spec_without_training(monkeypatch):
    original = prepare.source_digest

    def modified_helper(path):
        return "0" * 64 if path == ROOT / prepare.FIT_GUARD_REF else original(path)

    monkeypatch.setattr(prepare, "source_digest", modified_helper)
    generated = prepare.build_spec(load_ledger(), load_protocol())
    assert generated["fit_failure_contract"]["helper_sha256"] == "0" * 64
    assert generated != _recorded()
