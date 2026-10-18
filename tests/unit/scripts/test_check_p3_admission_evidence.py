"""Handoff checks catch drift and overclaims, not human-source truth or approval."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pu_toolbox.experiment.method_ledger import load_ledger
from pu_toolbox.experiment.survey_protocol import load_protocol

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p3_evidence_check", ROOT / "scripts/check_p3_admission_evidence.py"
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)
pytestmark = pytest.mark.unit


def facts():
    return json.loads((ROOT / check.EVIDENCE_REF).read_text(encoding="utf-8"))


def test_basic_determ_live_evidence_matches_code_ledger_and_repeated_cli(capsys):
    evidence, ledger, protocol = facts(), load_ledger(), load_protocol()
    before = copy.deepcopy((evidence, ledger, protocol))
    receipt = check.validate_evidence(evidence, ledger, protocol)
    assert receipt["ok"] and receipt["formal_admission"] is False
    assert receipt["methods_checked"] == list(check.METHODS)
    assert (evidence, ledger, protocol) == before
    for _ in range(2):
        assert check.main([]) == 0
        emitted = json.loads(capsys.readouterr().out)
        assert emitted == {**receipt, "source_files_checked": {}}


@pytest.mark.parametrize(
    "fault",
    [
        "default",
        "bool_default",
        "roles",
        "peak",
        "signature",
        "budget",
        "snapshot_property",
        "snapshot_bound",
        "missing_method",
        "commit",
        "digest",
        "test_ref",
        "old_admission",
    ],
)
def test_param_mutated_defaults_roles_source_and_admission_fail_closed(fault):
    evidence, ledger, protocol = facts(), load_ledger(), load_protocol()
    pulda = evidence["methods"]["pulda"]
    if fault == "default":
        pulda["constructor_defaults"]["pu_epochs"] += 1
    elif fault == "bool_default":
        evidence["methods"]["puet"]["constructor_defaults"]["bootstrap"] = 0
    elif fault == "roles":
        evidence["selection_contract"]["PA_role"] = "clean_val"
    elif fault == "peak":
        evidence["storage_contract"]["reclaim_reduces_training_peak"] = True
    elif fault == "signature":
        evidence["owner_signatures"] = ["synthetic approval"]
    elif fault == "budget":
        pulda["budget"]["approved_storage_bytes"] = 0
    elif fault == "snapshot_property":
        pulda["budget"]["snapshot_property"] = None
    elif fault == "snapshot_bound":
        evidence["methods"]["gradpu"]["budget"]["snapshot_bound"] = None
    elif fault == "missing_method":
        del evidence["methods"]["split_pu"]
    elif fault == "commit":
        pulda["source"]["commit"] = "main"
    elif fault == "digest":
        pulda["source"]["files"][0]["sha256"] = "not a digest"
    elif fault == "test_ref":
        pulda["source_behavior_tests"][0] = (
            "tests/unit/estimators/test_p3_source_behavior.py::test_missing"
        )
    else:
        protocol["method_profiles"]["pulda"] = {}
    with pytest.raises(ValueError):
        check.validate_evidence(evidence, ledger, protocol)


def test_edge_repo_reference_traversal_missing_and_empty_are_rejected(tmp_path):
    for reference in ("../outside.py", "missing.py", "", None):
        with pytest.raises(ValueError):
            check.repo_file(reference, tmp_path)


def test_basic_source_check_uses_commit_and_bytes_without_importing_code(tmp_path, monkeypatch):
    payload = b"raise RuntimeError('never import author code')\n"
    (tmp_path / "source.py").write_bytes(payload)
    commit = "a" * 40
    source = {
        "commit": commit,
        "files": [{"path": "source.py", "sha256": hashlib.sha256(payload).hexdigest()}],
    }
    calls = []

    def read_commit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=commit + "\n")

    monkeypatch.setattr(check.subprocess, "run", read_commit)
    assert check.validate_source_checkout(source, tmp_path) == 1
    assert calls == [["git", "-C", str(tmp_path), "rev-parse", "HEAD"]]
    (tmp_path / "source.py").write_bytes(payload + b"# modified\n")
    with pytest.raises(ValueError, match="byte mismatch"):
        check.validate_source_checkout(source, tmp_path)
    source["commit"] = "b" * 40
    with pytest.raises(ValueError, match="commit mismatch"):
        check.validate_source_checkout(source, tmp_path)


@pytest.mark.parametrize("source_dir", ["unknown=/tmp", "pulda", "pulda=", "gradpu=/tmp"])
def test_param_cli_bad_source_or_unverified_author_code_exits_one(source_dir, capsys):
    assert check.main(["--source-dir", source_dir]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_edge_missing_evidence_returns_json_error_not_traceback(tmp_path, capsys):
    assert check.main(["--evidence", str(tmp_path / "absent.json")]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "evidence_mass",
        "ledger_mass",
        "both_mass",
        "missing_mass",
        "bool_mass",
        "sample_count",
        "drop_mass_test",
    ],
)
def test_param_puet_mass_and_regression_reference_cannot_drift(fault):
    evidence, ledger, protocol = facts(), load_ledger(), load_protocol()
    row = evidence["methods"]["puet"]
    fact = row["calibration"]["bootstrap_risk_contract"]
    if fault in ("evidence_mass", "both_mass"):
        fact["unlabeled_total_weight"] = 0.5
    if fault in ("ledger_mass", "both_mass"):
        ledger["methods"]["puet"]["bootstrap_risk_contract"]["unlabeled_total_weight"] = 0.5
    elif fault == "missing_mass":
        del row["calibration"]["bootstrap_risk_contract"]
    elif fault == "bool_mass":
        fact["unlabeled_total_weight"] = True
    elif fault == "sample_count":
        fact["unlabeled_resample_count"] = "original_n_U_even_in_TS"
    elif fault == "drop_mass_test":
        row["source_behavior_tests"].remove(check.PUET_MASS_TEST)
    with pytest.raises(ValueError, match="PUET"):
        check.validate_evidence(evidence, ledger, protocol)


@pytest.mark.parametrize(
    "fault",
    [
        "drop_utils",
        "drop_models",
        "drop_readme",
        "paper_ref",
        "ledger_ref",
        "digest",
        "row_prior",
        "ledger_prior",
        "both_prior",
        "automatic_conversion",
        "drop_gradient_test",
    ],
)
def test_param_robust_requested_source_behavior_and_prior_evidence_cannot_disappear(fault):
    evidence, ledger, protocol = facts(), load_ledger(), load_protocol()
    row, entry = evidence["methods"]["robust_pu"], ledger["methods"]["robust_pu"]
    if fault.startswith("drop_") and fault != "drop_gradient_test":
        name = {"drop_utils": "utils.py", "drop_models": "models.py", "drop_readme": "README.md"}[
            fault
        ]
        row["source"]["files"] = [r for r in row["source"]["files"] if r["path"] != name]
    elif fault == "paper_ref":
        row["source"]["paper_review_ref"] = None
    elif fault == "ledger_ref":
        entry["source_behavior_review_ref"] = None
    elif fault == "digest":
        row["source"]["files"][-1]["sha256"] = "a" * 64
    elif fault in ("row_prior", "both_prior"):
        row["calibration"]["prior_input_contract"]["explicit_os_input"] = "population_pi"
        if fault == "both_prior":
            entry["runtime_prior_contract"]["explicit_os_input"] = "population_pi"
    elif fault == "ledger_prior":
        entry["runtime_prior_contract"]["prior_used_in"] = "all_training"
    elif fault == "automatic_conversion":
        row["calibration"]["prior_input_contract"]["automatic_prior_conversion"] = True
    else:
        row["source_behavior_tests"].remove(
            next(t for t in check.ROBUST_BEHAVIOR_TESTS if "actual_episode_bce_gradient" in t)
        )
    with pytest.raises(ValueError, match="Robust-PU"):
        check.validate_evidence(evidence, ledger, protocol)


@pytest.mark.parametrize(
    "fault",
    [
        "license",
        "permission",
        "execution",
        "signature",
        "formal",
        "digest",
        "prior",
        "test_ref",
    ],
)
def test_param_robust_review_cannot_grant_permission_or_self_sign(tmp_path, monkeypatch, fault):
    evidence, ledger, protocol = facts(), load_ledger(), load_protocol()
    review = json.loads((ROOT / check.ROBUST_REVIEW_REF).read_text(encoding="utf-8"))
    if fault == "license":
        review["source"]["license_observation"]["status"] = "MIT"
    elif fault == "permission":
        review["source"]["license_observation"]["permission_inferred_from_public_repository"] = True
    elif fault == "execution":
        review["source"]["executed_upstream_code"] = True
    elif fault == "signature":
        review["owner_signatures"] = ["invented acceptance"]
    elif fault == "formal":
        review["engineering_scope"]["formal_admission"] = True
    elif fault == "digest":
        review["source"]["file_sha256"]["utils.py"] = "a" * 64
    elif fault == "prior":
        review["prior_input_contract"]["post_pretrain_pseudo_negative_pool"] = "U_plus_P"
    else:
        review["behavior_test_refs"] = []
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    original = check.repo_file

    def repo_file(reference, root):
        return path if reference == check.ROBUST_REVIEW_REF else original(reference, root)

    monkeypatch.setattr(check, "repo_file", repo_file)
    with pytest.raises(ValueError, match="Robust-PU"):
        check.validate_evidence(evidence, ledger, protocol)


def test_edge_source_text_digest_is_additional_not_raw_byte_relaxation(tmp_path, monkeypatch):
    payload = b"raise RuntimeError('never execute')\r\n"
    path = tmp_path / "source.py"
    path.write_bytes(payload)
    commit = "a" * 40
    raw_digest = hashlib.sha256(payload).hexdigest()
    text_digest = hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()
    source = {
        "commit": commit,
        "files": [{"path": "source.py", "sha256": raw_digest}],
        "normalized_text_digest_policy": "utf8_text_universal_newlines_sha256",
        "normalized_text_sha256": {"source.py": text_digest},
    }
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=commit))
    assert check.validate_source_checkout(source, tmp_path) == 1
    source["normalized_text_sha256"]["source.py"] = "0" * 64
    with pytest.raises(ValueError, match="normalized text"):
        check.validate_source_checkout(source, tmp_path)
    source["normalized_text_sha256"]["source.py"] = text_digest
    path.write_bytes(payload.replace(b"\r\n", b"\n"))
    with pytest.raises(ValueError, match="source byte"):
        check.validate_source_checkout(source, tmp_path)
