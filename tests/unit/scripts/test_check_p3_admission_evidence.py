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
