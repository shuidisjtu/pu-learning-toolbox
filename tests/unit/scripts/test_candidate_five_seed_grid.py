"""Five-seed expansion is gated per method/view and never duplicates canaries."""

import pytest

from scripts import run_candidate_five_seed_grid as grid

pytestmark = pytest.mark.unit


def parent():
    return dict(
        jobs=[
            dict(label=f"{m}-{v}", method=m, view=v, parameters={"device": "cuda"})
            for m, v in [
                ("gradpu", "ts"),
                ("cvir", "os"),
                ("cvir", "ts"),
                ("vpu", "ts"),
                ("pan", "os"),
                ("p3mix", "os"),
            ]
        ]
    )


def test_basic_determ_grid_is_exact_and_preserves_parameters():
    p = parent()
    jobs = grid.expand_jobs(p)
    assert jobs == grid.expand_jobs(p)
    assert len(jobs) == len({j["label"] for j in jobs}) == 84
    assert not any(j["seed"] == 0 and j["c"] == "0.1" for j in jobs)
    assert {j["seed"] for j in jobs} == set(range(5))
    assert {j["c"] for j in jobs} == {"0.1", "0.3", "0.5"}
    for j in jobs:
        assert j["parameters"] == next(
            x["parameters"] for x in p["jobs"] if x["label"] == j["canary"]
        )


def test_edge_missing_canary_family_refused():
    with pytest.raises(ValueError, match="six canaries"):
        grid.expand_jobs(dict(jobs=parent()["jobs"][:-1]))


@pytest.mark.parametrize("indices", [[], [6, 6], [1], [0, 5, 6, 7, 8]])
def test_param_gpu_allowance(indices, tmp_path):
    with pytest.raises(ValueError, match="maximum four"):
        grid.run(tmp_path, indices)


def test_basic_worker_binds_each_seed_and_rate(tmp_path):
    job = grid.expand_jobs(parent())[-1]
    c = dict(inputs={"4": {"directory": "/split_4"}})
    cmd = grid.worker_command(c, job, tmp_path)
    for flag, value in [("--seed", "4"), ("--c", "0.5"), ("--data-dir", "/split_4")]:
        assert cmd[cmd.index(flag) + 1] == value


def test_edge_only_accepted_method_view_releases(monkeypatch, tmp_path):
    p = parent()
    jobs = grid.expand_jobs(p)
    c = dict(jobs=jobs, parent_directory=str(tmp_path), inputs={"0": {}})
    state = dict(
        jobs=[
            dict(j, status="completed" if j["label"] == "cvir-os" else "running") for j in p["jobs"]
        ]
    )
    checks = []
    monkeypatch.setattr(grid, "accepted", lambda *args: checks.append(args) or "verified-sha")
    cache = {}
    grid.release_canaries(c, state, cache)
    grid.release_canaries(c, state, cache)
    assert len(checks) == 1
    assert sum(j["status"] == "pending" for j in jobs) == 14
    assert all(j["status"] == "waiting_canary" for j in jobs if j["canary"] != "cvir-os")
    assert checks[0][1]["seed"] == 0 and checks[0][1]["c"] == "0.1"


def test_edge_bad_canary_never_releases(monkeypatch, tmp_path):
    p = parent()
    c = dict(jobs=grid.expand_jobs(p), parent_directory=str(tmp_path), inputs={"0": {}})
    state = dict(jobs=[dict(j, status="completed") for j in p["jobs"]])

    def reject(*args):
        raise ValueError("artifact changed")

    monkeypatch.setattr(grid, "accepted", reject)
    grid.release_canaries(c, state, {})
    assert all(j["status"] == "held_canary_invalid" for j in c["jobs"])


@pytest.mark.parametrize("corruption", [None, "warning", "seed", "artifact", "source"])
def test_param_result_identity_and_integrity(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(grid, "ROOT", tmp_path)
    (tmp_path / "source.py").write_text("original")
    job = dict(method="cvir", view="ts", seed=1, c="0.3", parameters={"device": "cuda"})
    plan = dict(
        method="cvir",
        training_view="ts",
        seed=1,
        c_token="0.3",
        parameters_requested=job["parameters"],
        input_files_sha256={"train": "input-sha"},
        plan_sha256="plan-id",
        source_files_sha256={"source.py": grid.base.sha(tmp_path / "source.py")},
    )
    if corruption == "seed":
        plan["seed"] = 0
    grid.base.write(tmp_path / "plan.json", plan)
    for name in ("manifest.json", "selected_PA.pkl", "selected_OA.pkl"):
        (tmp_path / name).write_text("artifact")
    summary = dict(
        accepted_engineering_run=True,
        selected_models_replayed=True,
        warning_count=0,
        retry_count=0,
        plan_sha256="plan-id",
        files_sha256={
            n: grid.base.sha(tmp_path / n)
            for n in ("plan.json", "manifest.json", "selected_PA.pkl", "selected_OA.pkl")
        },
    )
    if corruption == "warning":
        summary["warning_count"] = 1
    grid.base.write(tmp_path / "summary.json", summary)
    if corruption == "artifact":
        (tmp_path / "selected_PA.pkl").write_text("corrupted")
    if corruption == "source":
        (tmp_path / "source.py").write_text("changed")
    inputs = {"evidence": {"files_sha256": {"train": "input-sha"}}}
    if corruption:
        with pytest.raises(ValueError):
            grid.accepted(tmp_path, job, inputs)
    else:
        assert grid.accepted(tmp_path, job, inputs) == grid.base.sha(tmp_path / "summary.json")


@pytest.mark.parametrize("fail,launches", [(False, 3), (True, 2)])
def test_param_service_holds_only_failed_family(tmp_path, monkeypatch, fail, launches):
    p = tmp_path / "parent"
    p.mkdir()
    grid.base.write(p / "campaign.json", {})
    grid.base.write(
        p / "status.json",
        dict(jobs=[dict(label=j["label"], status="completed") for j in parent()["jobs"]]),
    )
    (tmp_path / "uv.lock").write_text("lock")
    monkeypatch.setattr(grid, "ROOT", tmp_path)
    monkeypatch.setattr(grid, "source_commit", lambda: "commit")
    monkeypatch.setattr(grid.base, "validate_split", lambda *a, **k: {})
    monkeypatch.setattr(grid, "accepted", lambda *a: "sha")
    monkeypatch.setattr(grid.base, "available_gpus", lambda scope: list(scope))
    monkeypatch.setattr(grid.signal, "signal", lambda *a: None)
    monkeypatch.setattr(grid.time, "sleep", lambda *a: None)
    jobs = grid.expand_jobs(parent())
    jobs = [jobs[0], jobs[6], jobs[1]]
    c = dict(
        jobs=jobs,
        source=str(tmp_path),
        commit="commit",
        parent_directory=str(p),
        parent_campaign_sha256=grid.base.sha(p / "campaign.json"),
        dependency_lock_sha256=grid.base.sha(tmp_path / "uv.lock"),
        inputs={str(s): {"directory": str(tmp_path), "evidence": {}} for s in range(5)},
    )
    grid.base.write(tmp_path / "campaign.json", c)
    for j in jobs:
        grid.base.write(tmp_path / f"{j['label']}.parameters.json", j["parameters"])
    launched = []

    class Worker:
        def __init__(self, argv, **kwargs):
            self.pid = 100 + len(launched)
            self.code = 1 if fail and not launched else 0
            launched.append(argv)
            assert kwargs["start_new_session"] is True
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "6"

        def poll(self):
            return self.code

    monkeypatch.setattr(grid.subprocess, "Popen", Worker)
    grid.run(tmp_path, [6])
    state = grid.read(tmp_path / "status.json")
    assert len(launched) == launches
    assert sum(j["status"] == "held_family_failure" for j in state["jobs"]) == int(fail)
    assert not state["formal_admission"]
