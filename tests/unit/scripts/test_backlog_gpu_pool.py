"""Runtime GPU pool selection never expands the physical four-card budget."""

import pytest

from scripts import run_survey_backlog_campaign as script

pytestmark = pytest.mark.unit


def test_basic_determ_explicit_pool_keeps_frozen_campaign_and_cap():
    campaign = dict(gpu_allowlist=[0, 5, 6, 7], max_gpu_jobs=4)
    assert script.runtime_gpu_pool(campaign) == ((0, 5, 6, 7), 4)
    assert script.runtime_gpu_pool(campaign, range(8)) == (tuple(range(8)), 4)
    assert campaign == dict(gpu_allowlist=[0, 5, 6, 7], max_gpu_jobs=4)
    assert script.runtime_gpu_pool(dict(gpu_allowlist=[6], max_gpu_jobs=1), [1, 2]) == ((1, 2), 1)


@pytest.mark.parametrize("pool", [[], [1, 1], [-1], [True], [1.5]])
def test_param_invalid_pool_rejected(pool):
    with pytest.raises(ValueError, match="GPU pool"):
        script.runtime_gpu_pool(dict(gpu_allowlist=[0, 5, 6, 7]), pool)


def test_edge_cpu_campaign_cannot_acquire_gpus():
    assert script.runtime_gpu_pool(dict(gpu_allowlist=[], max_gpu_jobs=0)) == ((), 0)
    with pytest.raises(ValueError):
        script.runtime_gpu_pool(dict(gpu_allowlist=[], max_gpu_jobs=0), [1])


@pytest.mark.parametrize("free", [{1, 2, 3, 4}, set(range(8))])
def test_basic_pool_dispatches_four_jobs_not_eight_and_preserves_holds(tmp_path, monkeypatch, free):
    jobs = [
        dict(
            label=f"job{i}",
            family="f",
            lane="gpu",
            status="pending",
            canary=False,
            method="pan",
            dataset="mnist",
            path="native_cnn",
            mechanism="scar",
        )
        for i in range(9)
    ]
    jobs[0]["status"] = "paused_by_user"
    campaign = dict(
        jobs=jobs,
        gpu_allowlist=[0, 5, 6, 7],
        source=str(script.ROOT),
        commit="fixed",
        dependency_lock_sha256="lock",
        inputs={},
    )
    monkeypatch.setattr(script, "lock_controller", lambda p: None)
    monkeypatch.setattr(script, "read", lambda p: campaign)
    monkeypatch.setattr(script, "clean_commit", lambda: "fixed")
    monkeypatch.setattr(script, "verify_reconciliation", lambda c: None)
    monkeypatch.setattr(script, "verify_followup", lambda c: None)
    monkeypatch.setattr(script, "memory_available", lambda: 1024**4)
    monkeypatch.setattr(script.service, "sha", lambda p: "lock")
    states = []
    monkeypatch.setattr(script.service, "write", lambda p, state: states.append(state))
    monkeypatch.setattr(script, "dispatch_gpus", lambda *a: (free, set()))
    launches = []

    class Process:
        def __init__(self, argv, **kwargs):
            launches.append(kwargs)
            self.pid = len(launches)

        def poll(self):
            return None

    monkeypatch.setattr(script.subprocess, "Popen", Process)
    sleeps = []

    def stop(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 4:
            raise InterruptedError("done")

    monkeypatch.setattr(script.time, "sleep", stop)
    with pytest.raises(InterruptedError, match="done"):
        script.run(tmp_path, gpu_pool=range(8))
    assert len(launches) == 4
    assert {int(p["env"]["CUDA_VISIBLE_DEVICES"]) for p in launches} == set(sorted(free)[:4])
    assert jobs[0]["status"] == "paused_by_user"
    assert states[-1]["max_active_gpus"] == 4
    assert states[-1]["gpu_allowlist"] == list(range(8))
    assert campaign["gpu_allowlist"] == [0, 5, 6, 7]
