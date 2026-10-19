"""Fail-closed candidate guard declarations, source drift and offline auditing."""

import copy
import json
import shutil
import subprocess
import sys

import pytest

from scripts import check_candidate_fit_contracts as checker

pytestmark = pytest.mark.unit
RECORD = "docs/research/pu_survey/data/candidate_fit_contracts_20261019_draft.json"


@pytest.fixture
def checkout(tmp_path):
    report = checker.build_contracts()
    references = {checker.LEDGER, checker.GUARD, checker.CHECKER}
    references.update(row["implementation_ref"] for row in report["methods"].values())
    references.update(row["fit_failure_review_ref"] for row in report["methods"].values())
    for reference in references:
        target = tmp_path / reference
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checker.ROOT / reference, target)
    return tmp_path


def test_basic_determ_record_matches_current_sources_without_authorization():
    report = checker.build_contracts()
    assert report == checker.build_contracts()
    assert report == json.loads((checker.ROOT / RECORD).read_text(encoding="utf-8"))
    assert len(report["methods"]) == 16
    assert report["methods"]["p3mix"]["class"] == "P3MixClassifier"
    assert report["methods"]["p3mix"]["calibration_applied"] is False
    assert report["formal_admission"] is False and report["owner_signatures"] == []
    assert report["rollback_of_models_or_external_artifacts"] is False
    assert report["methods"]["tedn"]["ledger_ref"].endswith("independent_extensions.tedn")
    assert report["methods"]["pulns"]["calibration_applied"] is False
    for method, row in report["methods"].items():
        assert row["formal_admission"] is False
        assert len(row["implementation_sha256"]) == 64
        assert row["fit_parameters"][0]["required"]
        names = [p["name"] for p in row["fit_parameters"]]
        assert "y_true" not in names and "test_data" not in names
        if method in {"pulns", "lagam"}:
            assert "support_data" in names  # extra supervision stays explicit


@pytest.mark.parametrize("method", list(checker.METHODS))
def test_edge_missing_guard_on_any_candidate_is_rejected(checkout, method):
    path = checkout / "pu_toolbox/estimators" / checker.METHODS[method]
    source = path.read_text(encoding="utf-8")
    assert "@invalidate_failed_fit\n" in source
    path.write_text(source.replace("    @invalidate_failed_fit\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="guard"):
        checker.build_contracts(root=checkout)


@pytest.mark.parametrize("fault", ["import", "clean_role", "kwargs", "duplicate", "reference"])
def test_param_invalid_contracts_are_not_silently_accepted(checkout, fault):
    path = checkout / "pu_toolbox/estimators/deep/gen_pu.py"
    source = path.read_text(encoding="utf-8")
    if fault == "import":
        source = source.replace("from ...core.fit_state import invalidate_failed_fit", "")
    elif fault == "clean_role":
        source = source.replace("epoch_callback=None", "test_data=None")
    elif fault == "kwargs":
        source = source.replace("epoch_callback=None", "epoch_callback=None, **kwargs")
    elif fault == "duplicate":
        source = source.replace(
            "@invalidate_failed_fit", "@invalidate_failed_fit\n    @invalidate_failed_fit"
        )
    else:
        ledger_path = checkout / checker.LEDGER
        facts = json.loads(ledger_path.read_text(encoding="utf-8"))
        facts["methods"]["genpu"]["fit_failure_review_ref"] = "../outside.md"
        ledger_path.write_text(json.dumps(facts), encoding="utf-8")
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        checker.build_contracts(root=checkout)


def test_edge_semantic_source_drift_is_visible_but_crlf_is_portable(checkout):
    before = checker.build_contracts(root=checkout)
    path = checkout / "pu_toolbox/estimators/deep/gen_pu.py"
    data = path.read_text(encoding="utf-8").encode("utf-8")
    path.write_bytes(data.replace(b"\n", b"\r\n"))
    assert checker.build_contracts(root=checkout) == before
    path.write_text(
        path.read_text(encoding="utf-8") + "\n# changed implementation\n", encoding="utf-8"
    )
    after = checker.build_contracts(root=checkout)
    assert after != before
    assert (
        after["methods"]["genpu"]["implementation_sha256"]
        != before["methods"]["genpu"]["implementation_sha256"]
    )


@pytest.mark.parametrize("fault", ["guard", "signature", "source", "admission"])
def test_param_cli_rejects_stale_or_overclaimed_receipt(tmp_path, capsys, fault):
    report = copy.deepcopy(checker.build_contracts())
    if fault == "guard":
        report["guard_sha256"] = "0" * 64
    elif fault == "signature":
        report["methods"]["rp"]["fit_parameters"].pop()
    elif fault == "source":
        report["methods"]["pulns"]["implementation_sha256"] = "0" * 64
    else:
        report["formal_admission"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert checker.main(["--check", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_basic_cli_check_and_no_torch_imports(capsys):
    assert checker.main(["--check", str(checker.ROOT / RECORD)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    script = """
import importlib.abc
import runpy
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch deliberately unavailable')
sys.meta_path.insert(0, NoTorch())
sys.argv = ['check_candidate_fit_contracts.py']
runpy.run_path('scripts/check_candidate_fit_contracts.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=checker.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(result.stdout)["methods"]) == 16
