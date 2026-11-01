"""Recovery preserves parent history, rejects strangers and checks offload receipts."""

import json
import os

import pytest

from scripts import recover_survey_backlog_canary as script
from scripts import run_survey_backlog_campaign as backlog

pytestmark = pytest.mark.unit


def parent(tmp_path, monkeypatch):
    folder = tmp_path / "parent"
    folder.mkdir()
    job = dict(
        label="gradpu-adni", method="gradpu", canary=True, status="pending", data_dir="/data"
    )
    campaign = dict(jobs=[job], inputs={"/data": {}}, gpu_allowlist=[0, 5, 6, 7])
    (folder / "campaign.json").write_text(json.dumps(campaign))
    (folder / "status.json").write_text(
        json.dumps(dict(updated_at=backlog.now(), jobs=[dict(job, status="failed", reason="OOM")]))
    )
    monkeypatch.setattr(backlog, "clean_commit", lambda: "fixed-source")
    monkeypatch.setattr(backlog, "verify_inputs", lambda *args, **kwargs: None)
    return folder


def test_basic_determ_recovery_preserves_parent_and_pins_new_source(tmp_path, monkeypatch):
    folder = parent(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    output = tmp_path / "recovery"
    result = script.prepare(output, folder, "gradpu-adni", 6)
    assert result["commit"] == "fixed-source"
    assert result["gradpu_activation_memory"] == "cpu_offload"
    assert result["jobs"][0]["status"] == "pending"
    assert result["recovery"]["parent_failed_job"]["status"] == "failed"
    assert result["recovery"]["additional_scientific_fits"] == 0
    assert result["total_fits"] == 1 and result["gpu_allowlist"] == [6]
    assert script.prepare(tmp_path / "second", folder, "gradpu-adni", 6) == result
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(ValueError, match="exists"):
        script.prepare(output, folder, "gradpu-adni", 6)


@pytest.mark.parametrize("gpu,status", [(1, "failed"), (6, "completed"), (6, "running")])
def test_param_recovery_cannot_expand_cards_or_rerun_nonfailures(
    tmp_path, monkeypatch, gpu, status
):
    folder = parent(tmp_path, monkeypatch)
    metadata = backlog.read(folder / "status.json")
    metadata["jobs"][0]["status"] = status
    (folder / "status.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        script.prepare(tmp_path / "out", folder, "gradpu-adni", gpu)


@pytest.mark.parametrize("case", ["own", "unknown", "stale", "low_gpu", "low_ram", "no_sharing"])
def test_edge_shared_card_admission_is_fail_closed(tmp_path, monkeypatch, case):
    folder = parent(tmp_path, monkeypatch)
    campaign = script.prepare(tmp_path / "out", folder, "gradpu-adni", 6)
    metadata = dict(updated_at=backlog.now(), jobs=[dict(status="running", gpu=6, pid=os.getpid())])
    if case == "stale":
        metadata["updated_at"] = "2020-01-01T00:00:00+00:00"
    (folder / "status.json").write_text(json.dumps(metadata))
    pids = {os.getpid() + 1} if case == "unknown" else {os.getpid()}
    monkeypatch.setattr(
        script, "gpu_processes", lambda gpu: (100 if case == "low_gpu" else 40000, pids)
    )
    monkeypatch.setattr(
        backlog, "memory_available", lambda: (10 if case == "low_ram" else 300) * 1024**3
    )
    accepted, _ = script.admission(campaign, allow_parent_sharing=case != "no_sharing")
    assert accepted is (case == "own")


def test_edge_existing_status_never_implicitly_restarts(tmp_path):
    (tmp_path / "status.json").write_text("{}")
    with pytest.raises(ValueError, match="reconciliation"):
        script.run(tmp_path)
