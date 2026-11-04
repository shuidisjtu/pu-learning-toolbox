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
@pytest.mark.parametrize("gpu_slots", [1, 2])
def test_basic_dispatch_shares_only_profiled_pending_jobs_once_per_card(
    tmp_path, monkeypatch, load_returns, gpu_slots
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
    jobs += [dict(job("t-canary", "t", "completed", True), path="native_2d")]
    jobs += [dict(job(f"table-{i}", "t"), path="native_2d") for i in range(4)]
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
    monkeypatch.setattr(script, "overlap_gpus", lambda indices: set() if load_returns else {6, 7})
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
        if len(sleeps) == 5:
            raise InterruptedError("three dispatch cycles")

    monkeypatch.setattr(script.time, "sleep", sleep)
    with pytest.raises(InterruptedError, match="three dispatch"):
        script.run(tmp_path, shared_families=["f", "t"], gpu_slots=gpu_slots)
    assert len(launches) == (0 if load_returns else 2 * gpu_slots)
    assert next(j for j in jobs if j["label"] == "transferred")["status"] == "paused_by_user"
    assert next(j for j in jobs if j["label"] == "other")["status"] == "pending"
    if launches:
        assert {kw["env"]["CUDA_VISIBLE_DEVICES"] for _, kw in launches} == {"6", "7"}
        assert all(j["shared_gpu_admission"] for j in jobs if j["status"] == "running")
        assert all(kw["start_new_session"] for _, kw in launches)
        for gpu in (6, 7):
            active = [j for j in jobs if j.get("gpu") == gpu]
            assert len(active) == gpu_slots
            assert sum(j["path"] == "native_cnn" for j in active) == 1


@pytest.mark.parametrize(
    "paths,new,allowed",
    [
        ([], "native_2d", False),
        (["native_cnn"], "native_cnn", False),
        (["native_cnn"], "native_2d", True),
        (["native_2d"], "native_cnn", True),
        (["native_2d"], "native_2d", True),
        (["native_2d", "native_cnn"], "native_2d", False),
    ],
)
def test_param_overlap_enforces_two_slots_and_one_image(paths, new, allowed):
    assert script.fits_overlap(dict(path=new), [dict(path=p) for p in paths]) is allowed


def test_edge_overlap_requires_memory_and_explicit_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        script.subprocess, "check_output", lambda *a, **k: "1, 48000\n6, 32767\n7, 32768\n"
    )
    assert script.overlap_gpus([6, 7]) == {7}
    for slots, families in [(0, ["f"]), (3, ["f"]), (2, [])]:
        with pytest.raises(ValueError, match="GPU slots"):
            script.run(tmp_path, gpu_slots=slots, shared_families=families)


@pytest.mark.parametrize("value", [0, 61, -1, 2.5, True])
def test_edge_invalid_poll_interval_rejected_before_start(tmp_path, value):
    with pytest.raises(ValueError, match="poll seconds"):
        script.run(tmp_path, poll_seconds=value)


@pytest.mark.parametrize("changed", [False, True])
def test_basic_reaped_worker_replaced_same_cycle_with_fresh_admission(
    tmp_path, monkeypatch, changed
):
    jobs = [
        dict(
            label=name,
            family="f",
            lane="gpu",
            status="pending",
            canary=False,
            method="pan",
            dataset="mnist",
            path="native_cnn",
            mechanism="scar",
        )
        for name in ("one", "two")
    ]
    campaign = dict(
        jobs=jobs,
        gpu_allowlist=[6],
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
    monkeypatch.setattr(script, "accept", lambda *a: {})
    monkeypatch.setattr(script.service, "sha", lambda p: "lock")
    states = []
    # Copy snapshots: jobs are mutated in place on subsequent cycles.
    monkeypatch.setattr(
        script.service, "write", lambda p, s: states.append(s["dispatch_status"].copy())
    )
    queries = []

    def query(*args):
        queries.append(1)
        # Initial two samples, launch recheck, next cycle, replacement recheck.
        return (set() if changed and len(queries) == 5 else {6}), set()

    monkeypatch.setattr(script, "dispatch_gpus", query)
    launches = []

    class Process:
        def __init__(self, *args, **kwargs):
            launches.append(kwargs)
            self.pid = len(launches)

        def poll(self):
            return 0

    monkeypatch.setattr(script.subprocess, "Popen", Process)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise InterruptedError("done")

    monkeypatch.setattr(script.time, "sleep", sleep)
    with pytest.raises(InterruptedError, match="done"):
        script.run(tmp_path)
    assert sleeps == [2, 2, 2]
    assert jobs[0]["status"] == "completed"
    assert len(queries) == 5
    assert len(launches) == (1 if changed else 2)
    assert states[0]["6"].startswith("confirming")
    assert states[-1]["6"] == (
        "GPU availability changed before launch" if changed else "started two"
    )
