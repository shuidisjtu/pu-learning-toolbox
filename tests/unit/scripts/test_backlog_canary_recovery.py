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
    monkeypatch.setattr(script, "owned_process", lambda pid: pid == os.getpid())
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


def completed_recovery(tmp_path, monkeypatch, status="completed"):
    folder = parent(tmp_path, monkeypatch)
    original = backlog.read(folder / "campaign.json")
    job = original["jobs"][0]
    job.update(
        family="gradpu-adni-scar",
        dataset="adni",
        path="native_cnn",
        mechanism="scar",
        seed=0,
        c="0.1",
        lane="gpu",
    )
    original.update(protocol_sha256="protocol", registry_sha256="registry")
    original["jobs"].extend([dict(job, label=f"sibling-{i}", seed=i, canary=False) for i in (1, 2)])
    (folder / "campaign.json").write_text(json.dumps(original))
    state = dict(
        jobs=[
            dict(j, status="failed" if j["canary"] else "held_family_failure")
            for j in original["jobs"]
        ]
    )
    (folder / "status.json").write_text(json.dumps(state))
    recovery = tmp_path / "recovery"
    fixed = script.prepare(recovery, folder, "gradpu-adni", 6)
    (recovery / "status.json").write_text(
        json.dumps(dict(jobs=[dict(fixed["jobs"][0], status=status)]))
    )
    monkeypatch.setattr(
        backlog,
        "accept",
        lambda *args: dict(manifest="accepted.json", manifest_sha256="accepted-hash"),
    )
    return folder, recovery


def test_basic_followup_keeps_offload_and_parent_history_after_full_acceptance(
    tmp_path, monkeypatch
):
    folder, recovery = completed_recovery(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    fixed = script.prepare_followup(tmp_path / "followup", recovery)
    assert fixed["gradpu_activation_memory"] == "cpu_offload"
    assert fixed["gpu_allowlist"] == [6] and fixed["total_fits"] == 2
    assert fixed["followup"]["additional_scientific_fits"] == 0
    assert [j["status"] for j in fixed["jobs"]] == ["pending", "waiting_canary"]
    backlog.verify_followup(fixed)
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    fixed["gradpu_activation_memory"] = "device"
    with pytest.raises(ValueError, match="offload policy"):
        backlog.verify_followup(fixed)


@pytest.mark.parametrize("status", ["running", "failed", "pending"])
def test_param_partial_or_failed_gradpu_never_releases_siblings(tmp_path, monkeypatch, status):
    _, recovery = completed_recovery(tmp_path, monkeypatch, status)
    with pytest.raises(ValueError, match="has not completed"):
        script.prepare_followup(tmp_path / "followup", recovery)
    assert not (tmp_path / "followup").exists()


def test_edge_completed_status_without_accepted_artifacts_never_releases(tmp_path, monkeypatch):
    _, recovery = completed_recovery(tmp_path, monkeypatch)

    def invalid(*args):
        raise ValueError("incomplete epoch budget")

    monkeypatch.setattr(backlog, "accept", invalid)
    with pytest.raises(ValueError, match="incomplete epoch"):
        script.prepare_followup(tmp_path / "followup", recovery)
    assert not (tmp_path / "followup").exists()


def test_edge_followup_rechecks_parent_and_canary_after_preparation(tmp_path, monkeypatch):
    folder, recovery = completed_recovery(tmp_path, monkeypatch)
    fixed = script.prepare_followup(tmp_path / "followup", recovery)
    state = backlog.read(folder / "status.json")
    state["jobs"][1]["status"] = "running"
    (folder / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="not held"):
        backlog.verify_followup(fixed)
