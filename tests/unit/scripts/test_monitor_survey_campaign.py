"""Read-only dashboard telemetry, truthful progress and conservative ETA tests."""

import json
import os
import subprocess
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts import monitor_survey_campaign as monitor

pytestmark = pytest.mark.unit


def job(**changes):
    value = dict(
        label="fit",
        method="pan",
        dataset="cifar10",
        path="native_cnn",
        status="running",
        gpu=0,
        seed=0,
        c="0.1",
        mechanism="scar",
        started_at="1970-01-01T00:10:00+00:00",
        pid=123,
        checkpoint_epochs=20,
    )
    value.update(changes)
    return value


def snapshots(root, *, count=6, components=("model",), attempt="attempt-one", start=700):
    directory = root / "runs/fit" / attempt
    directory.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, count + 1):
        for component in components:
            path = directory / f"epoch_{epoch:04d}_{component}.pt"
            path.write_bytes(b"not loaded by monitor")
            modified = start + 10 * epoch
            os.utime(path, (modified, modified))


def state(root, jobs, *, revision="survey-v3.1", updated=765):
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign.json").write_text(json.dumps(dict(protocol_version=revision)))
    (root / "status.json").write_text(
        json.dumps(
            dict(
                updated_at=datetime.fromtimestamp(updated, timezone.utc).isoformat(),
                jobs=jobs,
                gpu_allowlist=[0, 5, 6, 7],
            )
        )
    )


def test_basic_determ_epoch_eta_uses_recent_completed_intervals(tmp_path):
    snapshots(tmp_path)
    first = monitor.epoch_progress(tmp_path, job(), 765)
    assert first == monitor.epoch_progress(tmp_path, job(), 765)
    assert "6/20 epochs (30%)" in first[0]
    assert "ETA~3m" in first[1] and "10.0s/epoch" in first[1]
    assert "last 5 intervals" in first[1]


def test_basic_two_teachers_are_one_epoch_and_partial_teacher_is_not_complete(tmp_path):
    snapshots(tmp_path, components=("teacher1", "teacher2"))
    j = job(method="self_pu_full_meta")
    assert "6/20" in monitor.epoch_progress(tmp_path, j, 765)[0]
    (tmp_path / "runs/fit/attempt-one/epoch_0006_teacher2.pt").unlink()
    assert "5/20" in monitor.epoch_progress(tmp_path, j, 765)[0]


@pytest.mark.parametrize(
    "case,reason",
    [
        ("few", "<3 epoch intervals"),
        ("stale", "uncertain"),
        ("attempts", "attempts must not be pooled"),
        ("manifest", "acceptance"),
        ("reclaimed", "non-contiguous"),
        ("finished", "training finished"),
    ],
)
def test_edge_incomplete_stale_reclaimed_or_finished_has_no_fake_countdown(tmp_path, case, reason):
    snapshots(tmp_path, count=2 if case == "few" else 6)
    now, j = 765, job()
    if case == "stale":
        now = 1000
    elif case == "attempts":
        snapshots(tmp_path, attempt="attempt-two")
    elif case == "manifest":
        (tmp_path / "runs/fit/manifest.json").write_text("{}")
    elif case == "reclaimed":
        (tmp_path / "runs/fit/attempt-one/epoch_0001_model.pt").unlink()
    elif case == "finished":
        j["checkpoint_epochs"] = 6
    detail = monitor.epoch_progress(tmp_path, j, now)[1]
    assert reason in detail and "ETA~" not in detail


def test_basic_cpu_estimate_requires_three_same_c_same_family_completed_runs():
    j = job(method="puet", path="native_2d", checkpoint_epochs=None)
    sample = job(**{**j, "status": "completed", "finished_at": "1970-01-01T00:12:00+00:00"})
    assert "2/3" in monitor.classical_progress(j, [sample] * 2, 20)[1]
    assert "ETA~2m" in monitor.classical_progress(j, [sample] * 3, 20)[1]
    changed = {**sample, "c": "0.5"}
    assert "0/3" in monitor.classical_progress(j, [changed] * 3, 20)[1]
    assert "exceeds median" in monitor.classical_progress(j, [sample] * 3, 150)[1]
    image = {**j, "path": "cnn_feature_adapter"}
    assert "no live counter" in monitor.classical_progress(image, [sample] * 3, 20)[1]


def test_basic_gpu_telemetry_and_unavailable_fields(monkeypatch):
    monkeypatch.setattr(
        monitor.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="0, 95, 1024, 49140\n1, [N/A], [N/A], 49140\n"),
    )
    devices, error = monitor.gpu_telemetry()
    assert error is None and devices[0]["util"] == 95
    assert devices[1]["util"] is None and devices[1]["used"] is None


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 4)])
def test_param_gpu_query_failure_is_visible_not_zero_utilization(monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(monitor.subprocess, "run", unavailable)
    devices, message = monitor.gpu_telemetry()
    assert devices == {} and "unavailable" in message


def test_basic_render_combined_campaigns_is_read_only_and_has_elapsed_and_eta(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    state(one, [job()])
    snapshots(one)
    state(
        two,
        [job(method="rp", path="cnn_feature_adapter", gpu=None, checkpoint_epochs=None)],
        revision="survey-v3.2",
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    text = monitor.render(
        [one, two], now=765, devices={0: dict(util=95, used=1024, total=49140)}, width=160
    )
    for token in ("95%", "1.0/48.0", "elapsed=3m", "6/20", "ETA~", "CPU", "survey-v3.2"):
        assert token in text
    assert "no live counter" in text and "Queue finish ETA: unavailable" in text
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert all(len(line) <= 80 for line in monitor.render([one], now=765, width=80).splitlines())


def test_edge_missing_corrupt_and_stale_state_are_not_reported_as_idle(tmp_path):
    (tmp_path / "campaign.json").write_text('{"protocol_version":"survey-v3.2"}')
    assert "waiting for first controller status" in monitor.render([tmp_path], now=1000)
    (tmp_path / "status.json").write_text("{")
    assert "status unavailable" in monitor.render([tmp_path], now=1000)
    (tmp_path / "status.json").write_text('{"jobs":[{"status":"running"}]}')
    assert "status unavailable" in monitor.render([tmp_path], now=1000)
    state(tmp_path, [job()], updated=100)
    snapshots(tmp_path)
    text = monitor.render([tmp_path], now=1000)
    assert "STALE STATUS!" in text and "stale controller status" in text


@pytest.mark.parametrize("recovery_status", ["running", "failed", "completed"])
@pytest.mark.parametrize("stale", [False, True])
def test_basic_recovery_is_separate_from_original_failure_and_never_hides_it(
    tmp_path, recovery_status, stale
):
    parent, recovery = tmp_path / "parent", tmp_path / "recovery"
    state(parent, [job(status="failed", gpu=7, reason="OOM")])
    state(recovery, [job(status=recovery_status, gpu=6)], updated=100 if stale else 765)
    (recovery / "campaign.json").write_text(
        json.dumps(
            dict(
                protocol_version="survey-v3.1",
                recovery=dict(parent=str(parent), replaces_label="fit"),
            )
        )
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    text = monitor.render([parent, recovery], now=765, width=200)
    assert "ORIGINAL FAILED GPU7 fit: OOM" in text
    assert f"recovery={recovery_status}{' STALE' if stale else ''} GPU6" in text
    assert "RECOVERY ATTEMPT" in text and "not an additional scientific fit" in text
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    # Same label in another campaign must not be treated as this one's recovery.
    other = tmp_path / "other"
    state(other, [job(status="failed", gpu=0)])
    text = monitor.render([other, recovery], now=765)
    assert "ORIGINAL FAILED" not in text


def test_edge_replacement_grid_does_not_describe_blocked_fits_as_gpu_waiting(tmp_path):
    state(tmp_path, [job(status="blocked_preflight", reason="equality_outside_box")])
    (tmp_path / "campaign.json").write_text(
        json.dumps(
            dict(protocol_version="survey-v3.1", reconciliation=dict(additional_scientific_fits=0))
        )
    )
    text = monitor.render([tmp_path], now=765, width=220)
    assert "blocked_preflight=1" in text
    assert "REPLACEMENT GRID" in text and "not additional experiments" in text
    assert "not waiting for a GPU" in text
