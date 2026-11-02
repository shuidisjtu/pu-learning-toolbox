"""Explicit co-tenancy admission preserves process ownership and family boundaries."""

import subprocess

import pytest

from scripts import run_survey_backlog_campaign as script

pytestmark = pytest.mark.unit


def test_basic_determ_default_remains_exclusive(monkeypatch):
    monkeypatch.setattr(script.service, "available_gpus", lambda indices: [5])
    monkeypatch.setattr(
        script.subprocess, "check_output", lambda *a, **kw: pytest.fail("no sharing query")
    )
    assert script.dispatch_gpus([0, 5, 6, 7]) == ({5}, set())
    assert script.dispatch_gpus([0, 5, 6, 7]) == ({5}, set())


@pytest.mark.parametrize(
    "util,free,eligible",
    [
        (0, 45000, True),
        (10, 40960, True),
        (11, 45000, False),
        (0, 40959, False),
        (-1, 45000, False),
    ],
)
def test_param_shared_admission_keeps_headroom_and_allowlist(monkeypatch, util, free, eligible):
    monkeypatch.setattr(script.service, "available_gpus", lambda indices: [5])
    monkeypatch.setattr(
        script.subprocess,
        "check_output",
        lambda *a, **kw: f"1, 0, 48000\n5, 0, 48000\n6, {util}, {free}\n",
    )
    exclusive, shared = script.dispatch_gpus([0, 5, 6, 7], ["profiled"])
    assert exclusive == {5}
    assert shared == ({6} if eligible else set())


@pytest.mark.parametrize("row", ["6, N/A, 45000", "bad", "6, 0"])
def test_edge_invalid_gpu_metrics_fail_closed(monkeypatch, row):
    monkeypatch.setattr(script.service, "available_gpus", lambda indices: [])
    monkeypatch.setattr(script.subprocess, "check_output", lambda *a, **kw: row)
    with pytest.raises(ValueError):
        script.dispatch_gpus([6], ["profiled"])


def test_edge_failed_query_never_returns_shareable_gpu(monkeypatch):
    monkeypatch.setattr(script.service, "available_gpus", lambda indices: [])

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(script.subprocess, "check_output", fail)
    with pytest.raises(subprocess.CalledProcessError):
        script.dispatch_gpus([6], ["profiled"])


@pytest.mark.parametrize(
    "lane,status,canary",
    [
        ("cpu", "completed", True),
        ("gpu", "failed", True),
        ("gpu", "pending", True),
        ("gpu", "completed", False),
    ],
)
def test_edge_sharing_requires_completed_gpu_canary(lane, status, canary):
    job = dict(family="f", lane=lane, status=status, canary=canary)
    with pytest.raises(ValueError, match="completed canary"):
        script.sharing_families(dict(jobs=[job]), dict(jobs=[job]), ["f"])


def test_basic_determ_opt_in_does_not_extend_to_other_families():
    jobs = [dict(family=f, lane="gpu", status="completed", canary=True) for f in ("f", "g")]
    assert script.sharing_families(dict(jobs=jobs), dict(jobs=jobs), ["f", "f"]) == ("f",)
    assert script.sharing_families(dict(jobs=jobs), dict(jobs=jobs), []) == ()
    with pytest.raises(ValueError):
        script.sharing_families(dict(jobs=jobs), dict(jobs=jobs), ["missing"])


@pytest.mark.parametrize("load_returns", [False, True])
def test_basic_dispatch_shares_only_profiled_pending_jobs_once_per_card(
    tmp_path, monkeypatch, load_returns
):
    def job(label, family="f", status="pending", canary=False):
        return dict(
            label=label,
            family=family,
            status=status,
            canary=canary,
            lane="gpu",
            method="p3mix",
            dataset="cifar10",
            path="native_cnn",
            mechanism="scar",
        )

    jobs = [
        job("canary", status="completed", canary=True),
        job("one"),
        job("two"),
        job("three"),
        job("transferred", status="paused_by_user"),
        job("other-canary", "g", "completed", True),
        job("other", "g"),
    ]
    campaign = dict(
        jobs=jobs,
        gpu_allowlist=[6, 7],
        source=str(script.ROOT),
        commit="fixed",
        dependency_lock_sha256="lock",
        inputs={},
    )
    monkeypatch.setattr(script, "lock_controller", lambda path: None)
    monkeypatch.setattr(script, "read", lambda path: campaign)
    monkeypatch.setattr(script, "clean_commit", lambda: "fixed")
    monkeypatch.setattr(script, "verify_reconciliation", lambda c: None)
    monkeypatch.setattr(script, "verify_followup", lambda c: None)
    monkeypatch.setattr(script, "memory_available", lambda: 1024**4)
    monkeypatch.setattr(script.service, "sha", lambda path: "lock")
    monkeypatch.setattr(script.service, "write", lambda *args, **kwargs: None)
    queries = []

    def gpu_query(indices, families):
        queries.append(1)
        # Initial two samples pass, but load can change before the Popen check.
        return set(), (set() if load_returns and len(queries) >= 3 else {6, 7})

    monkeypatch.setattr(script, "dispatch_gpus", gpu_query)
    launches = []

    class Process:
        def __init__(self, argv, **kwargs):
            launches.append((argv, kwargs))
            self.pid = len(launches)

        def poll(self):
            return None

    monkeypatch.setattr(script.subprocess, "Popen", Process)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise InterruptedError("three dispatch cycles")

    monkeypatch.setattr(script.time, "sleep", sleep)
    with pytest.raises(InterruptedError, match="three dispatch"):
        script.run(tmp_path, shared_families=["f"])
    assert len(launches) == (0 if load_returns else 2)
    assert next(j for j in jobs if j["label"] == "transferred")["status"] == "paused_by_user"
    assert next(j for j in jobs if j["label"] == "other")["status"] == "pending"
    if launches:
        assert {kw["env"]["CUDA_VISIBLE_DEVICES"] for _, kw in launches} == {"6", "7"}
        assert all(j["shared_gpu_admission"] for j in jobs if j["status"] == "running")
        assert all(kw["start_new_session"] for _, kw in launches)
