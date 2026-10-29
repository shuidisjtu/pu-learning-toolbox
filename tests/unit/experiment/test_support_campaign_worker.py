"""Pinned correction worker ABI: own evidence root, unchanged budgets, no overwrite."""

import importlib.util
import json
from pathlib import Path

import pytest
import test_support_campaign as campaign_fixtures

from pu_toolbox.experiment import batch_evidence, support_campaign

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
prepared = campaign_fixtures.prepared


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location(
        "support_fit_worker_test", ROOT / "scripts/run_survey_queue_fit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request(prepared):
    _, jobs = support_campaign.load_support_campaign(prepared[0])
    job = jobs[0]
    path = prepared[0].parent / "worker.json"
    path.write_text(json.dumps(job))
    return path, job


def test_basic_determ_worker_routes_packet_and_single_fit_without_budget_overrides(
    prepared, worker, monkeypatch
):
    path, job = request(prepared)
    monkeypatch.setattr(worker.sys, "path", list(worker.sys.path))
    monkeypatch.setattr(
        batch_evidence,
        "build_evidence",
        lambda **kwargs: dict(code_commit=job["source_code_commit"], code_commit_dirty=False),
    )
    written = []
    monkeypatch.setattr(
        batch_evidence,
        "write",
        lambda directory, packet: written.append(directory) or directory / "packet.json",
    )
    executed = []
    monkeypatch.setattr(
        worker.os, "execv", lambda executable, argv: executed.append((executable, argv))
    )
    worker.main(["--worker", str(path)])
    assert written == [Path(job["campaign_root"]) / "batch_evidence"]
    executable, argv = executed[0]
    assert executable == worker.sys.executable
    assert argv[2] == str(Path(job["source"]) / "scripts/run_survey_experiment.py")
    for flag, value in {
        "--device": "cuda",
        "--seeds": "0",
        "--c": "0.1",
        "--out-dir": job["root"],
        "--protocol": "survey-v3.0",
        "--batch-evidence": str(written[0] / "packet.json"),
    }.items():
        assert argv[argv.index(flag) + 1] == value
    assert "--reclaim-unselected-checkpoints" in argv
    assert not {"--max-epochs", "--epochs", "--batch-size", "--params"}.intersection(argv)


@pytest.mark.parametrize("problem", ["receipt", "lock", "commit", "protocol", "existing_output"])
def test_param_edge_changed_identity_or_existing_output_never_executes(
    prepared, worker, monkeypatch, problem
):
    path, job = request(prepared)
    if problem == "receipt":
        job["support_receipt_sha256"] = "changed"
    elif problem == "lock":
        job["dependency_lock_sha256"] = "changed"
    elif problem == "commit":
        job["source_code_commit"] = "changed"
    elif problem == "protocol":
        job["protocol_sha256"] = "changed"
    else:
        root = Path(job["root"])
        root.mkdir(parents=True)
        (root / "manifest.json").write_text("keep")
    path.write_text(json.dumps(job))
    monkeypatch.setattr(worker.sys, "path", list(worker.sys.path))
    monkeypatch.setattr(worker.os, "execv", lambda *args: pytest.fail("launched invalid fit"))
    with pytest.raises(ValueError):
        worker.main(["--worker", str(path)])
