"""Controller handoff guards; signals are mocked and never target training."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc controller"),
]


@pytest.fixture
def service():
    path = Path(__file__).resolve().parents[3] / "scripts/run_survey_parallel_queue.py"
    spec = importlib.util.spec_from_file_location("survey_parallel_service_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_basic_determ_current_process_identity_is_stable(service):
    a = service.process_identity(os.getpid())
    assert a["start_ticks"] == service.process_identity(os.getpid())["start_ticks"] > 0
    assert service.command_line(os.getpid())


def test_edge_missing_process_and_missing_terminal_log(service, tmp_path):
    assert service.process_identity(999999999) is None
    assert service.command_line(999999999) == []
    assert not service.terminal_marker({"log": str(tmp_path / "absent")})


@pytest.mark.parametrize(
    "good_artifacts,marker,expected",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_param_adopted_exit_requires_both_valid_artifacts_and_success_marker(
    service, monkeypatch, good_artifacts, marker, expected
):
    monkeypatch.setattr(service, "process_identity", lambda pid: None)
    monkeypatch.setattr(service, "successful_artifacts", lambda job: good_artifacts)
    monkeypatch.setattr(service, "terminal_marker", lambda job: marker)
    assert service.poll_job({"pid": 123, "process_start_ticks": 1}) == (
        expected,
        None,
        "terminal_marker_and_artifacts",
    )


def test_edge_recycled_pid_is_not_mistaken_for_our_live_worker(service, monkeypatch):
    monkeypatch.setattr(service, "process_identity", lambda pid: dict(start_ticks=2, state="R"))
    monkeypatch.setattr(service, "successful_artifacts", lambda job: False)
    assert service.poll_job({"pid": 123, "process_start_ticks": 1})[0] is False


@pytest.mark.parametrize(
    "code,artifacts,expected",
    [(None, True, None), (0, True, True), (0, False, False), (1, True, False)],
)
def test_param_child_exit_and_artifact_validation(service, monkeypatch, code, artifacts, expected):
    monkeypatch.setattr(service, "successful_artifacts", lambda job: artifacts)
    result = service.poll_job({}, SimpleNamespace(poll=lambda: code))
    assert result is None if expected is None else result[0] is expected


def test_basic_terminal_marker_reads_success_at_end_of_large_log(service, tmp_path):
    log = tmp_path / "worker.log"
    log.write_text(
        "x" * 10000
        + "\nVersioned pilot evidence only: check manifest formal_blockers before aggregation.\n"
    )
    assert service.terminal_marker({"log": str(log)})


def test_edge_output_collision_precedes_process_creation(service, tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("keep")
    monkeypatch.setattr(
        service.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned a duplicate fit")
    )
    with pytest.raises(ValueError, match="overwrite"):
        service.spawn({"root": str(tmp_path)}, None, tmp_path, Path("worker.py"))
    assert (tmp_path / "manifest.json").read_text() == "keep"


def takeover_fixture(service, tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.py"
    state = dict(scheduler=dict(supervisor_pid=123), jobs=[])
    (tmp_path / "queue_status.json").write_text(json.dumps(state))
    identity = dict(start_ticks=7, state="R", wait_status=0)
    signals = []

    def kill(pid, sig):
        signals.append((pid, sig))
        if sig == service.signal.SIGSTOP:
            identity["state"] = "T"
        if sig == service.signal.SIGTERM:
            identity["state"] = "Z"

    monkeypatch.setattr(service.os, "kill", kill)
    monkeypatch.setattr(service, "process_identity", lambda pid: dict(identity))
    monkeypatch.setattr(service, "command_line", lambda pid: ["python", str(legacy), "--run"])
    monkeypatch.setattr(service, "devices", lambda: {})
    monkeypatch.setattr(service, "reconcile_children", lambda *args: [])
    return legacy, signals


def test_basic_handoff_signals_only_controller_and_archives_state(service, tmp_path, monkeypatch):
    legacy, signals = takeover_fixture(service, tmp_path, monkeypatch)
    assert service.takeover(tmp_path, legacy) == []
    assert signals == [
        (123, service.signal.SIGSTOP),
        (123, service.signal.SIGTERM),
        (123, service.signal.SIGCONT),
    ]
    assert list(tmp_path.glob("pre_parallel_handoff_*.json"))
    assert (tmp_path / "parallel_split_plan.json").exists()


def test_edge_failed_handoff_resumes_original_controller(service, tmp_path, monkeypatch):
    legacy, signals = takeover_fixture(service, tmp_path, monkeypatch)

    def reject(*args):
        raise ValueError("invalid ownership")

    monkeypatch.setattr(service, "split_pending_jobs", reject)
    with pytest.raises(ValueError, match="invalid ownership"):
        service.takeover(tmp_path, legacy)
    assert signals == [(123, service.signal.SIGSTOP), (123, service.signal.SIGCONT)]
    assert not list(tmp_path.glob("pre_parallel_handoff_*.json"))


def test_param_unverified_controller_never_receives_a_signal(service, tmp_path, monkeypatch):
    (tmp_path / "queue_status.json").write_text(
        json.dumps(dict(scheduler=dict(supervisor_pid=123)))
    )
    monkeypatch.setattr(service, "process_identity", lambda pid: None)
    monkeypatch.setattr(service, "command_line", lambda pid: [])
    monkeypatch.setattr(service.os, "kill", lambda *args: pytest.fail("signaled an unverified PID"))
    with pytest.raises(ValueError, match="identity is not verified"):
        service.takeover(tmp_path, tmp_path / "legacy.py")


def test_basic_real_process_handoff_preserves_independent_worker(service, tmp_path, monkeypatch):
    """Use only disposable CPU children owned by this test, never a real queue."""
    legacy = tmp_path / "legacy.py"
    root = str(tmp_path / "fit")
    request = dict(
        label="probe",
        root=root,
        status="running",
        dataset="probe",
        method="probe",
        path="native_2d",
        mechanism="scar",
        oracle=False,
        seeds=[0],
        c_tokens=["0.1"],
        expected_runs=1,
        expected_keys=[["probe", "probe", "native_2d", "scar", "0.1", 0]],
        gpu={},
        lane="cpu",
        log=str(tmp_path / "worker.log"),
    )
    legacy.write_text(
        "import json,os,subprocess,sys,time\n"
        f"job={request!r}\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)',"
        f"'--out-dir',{root!r}],start_new_session=True,stdin=subprocess.DEVNULL)\n"
        "job['pid']=child.pid\n"
        f"open({str(tmp_path / 'queue_status.json')!r},'w').write(json.dumps("
        "dict(scheduler=dict(supervisor_pid=os.getpid()),jobs=[job])))\n"
        "while True: time.sleep(.1)\n"
    )
    controller = subprocess.Popen(
        [sys.executable, str(legacy), "--run"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid = None
    child_identity = None
    try:
        for _ in range(100):
            path = tmp_path / "queue_status.json"
            if path.exists() and path.stat().st_size:
                try:
                    state = json.loads(path.read_text())
                    break
                except ValueError:
                    pass
            time.sleep(0.02)
        else:
            pytest.fail("probe controller did not initialize")
        child_pid = state["jobs"][0]["pid"]
        child_identity = service.process_identity(child_pid)
        monkeypatch.setattr(service, "devices", lambda: {})
        jobs = service.takeover(tmp_path, legacy)
        controller.wait(timeout=3)
        assert controller.returncode == -service.signal.SIGTERM
        assert jobs[0]["pid"] == child_pid and jobs[0]["adopted"]
        current = service.process_identity(child_pid)
        assert current["start_ticks"] == child_identity["start_ticks"] and current["state"] != "Z"
    finally:
        if controller.poll() is None:
            controller.terminate()
            controller.wait(timeout=3)
        current = service.process_identity(child_pid) if child_pid is not None else None
        if current and child_identity and current["start_ticks"] == child_identity["start_ticks"]:
            os.kill(child_pid, service.signal.SIGTERM)
