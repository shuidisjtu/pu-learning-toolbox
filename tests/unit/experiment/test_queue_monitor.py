"""Read-only monitor phases, single-fit identity and bounded terminal layout."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def monitor():
    path = Path(__file__).resolve().parents[3] / "scripts/monitor_survey_queue.py"
    spec = importlib.util.spec_from_file_location("survey_queue_monitor_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job(root, **changes):
    result = dict(
        root=str(root),
        label="test",
        dataset="cifar10",
        method="genpu",
        path="native_cnn",
        oracle=False,
        mechanism="scar",
        status="running",
        expected_runs=2,
        expected_keys=[
            ["cifar10", "genpu", "native_cnn", "scar", "0.1", 0],
            ["cifar10", "genpu", "native_cnn", "scar", "0.3", 0],
        ],
        gpu=dict(index=6),
        pid=10,
    )
    result.update(changes)
    return result


def test_basic_determ_previous_selected_snapshot_is_not_current_progress(monitor, tmp_path):
    previous = tmp_path / "ts/c_0.1/seed_0"
    previous.mkdir(parents=True)
    (previous / "manifest.json").write_text(
        json.dumps(dict(seed=0, c_requested_token="0.1", selection={"OA": {}}, failures=[]))
    )
    (previous / "epoch_0099_model.pt").write_text("retained")
    result = monitor.progress(job(tmp_path))
    assert result == monitor.progress(job(tmp_path))
    assert result["epoch"] == "GAN/prePN" and result["done"] == "1/2"
    assert (result["seed"], result["c"]) == ("0", "0.3")


def test_basic_pn_phase_uses_active_snapshot(monitor, tmp_path):
    active = tmp_path / "ts/c_0.1/seed_0/checkpoints"
    active.mkdir(parents=True)
    (active / "epoch_0003_model.pt").write_text("new")
    result = monitor.progress(job(tmp_path))
    assert result["epoch"] == "PN3/100" and result["c"] == "0.1"


@pytest.mark.parametrize("method,phase", [("nnpu", "load/prep"), ("genpu", "GAN/prePN")])
def test_param_single_fit_shows_seed_and_c_before_first_checkpoint(
    monitor, tmp_path, method, phase
):
    item = job(
        tmp_path,
        method=method,
        expected_runs=1,
        expected_keys=[["cifar10", method, "native_cnn", "scar", "0.5", 4]],
    )
    facts = monitor.progress(item)
    assert facts["seed"] == "4" and facts["c"] == "0.5" and facts["epoch"] == phase


@pytest.mark.parametrize("view", ["running", "pending", "completed", "failed"])
@pytest.mark.parametrize("width", [35, 60, 85, 100, 120, 180])
@pytest.mark.parametrize("height", [14, 24, 40])
def test_edge_layout_remains_bounded_across_views(monitor, tmp_path, view, width, height):
    item = job(tmp_path)
    state = dict(updated_at="2020-01-01T00:00:00+00:00", jobs=[item])
    output = monitor.render(view=view, width=width, height=height, states=[state], devices={})
    assert len(output.splitlines()) <= height
    assert all(len(line) <= width for line in output.splitlines())
    assert "STALE" in output or width < 60


def test_basic_own_concurrency_and_foreign_sharing_are_distinct(monitor, tmp_path):
    jobs = [job(tmp_path / str(pid), pid=pid) for pid in (10, 11)]
    state = dict(updated_at="2020-01-01T00:00:00+00:00", jobs=jobs)
    device = {6: dict(pids={10, 11, 99}, util=99, memory=16000)}
    output = monitor.render(width=180, height=40, states=[state], devices=device)
    assert "GPU6=2" in output and "6*" in output
    assert "PHASE" in output and "GAN/prePN" in output


def test_edge_malformed_manifest_does_not_break_live_view(monitor, tmp_path):
    (tmp_path / "manifest.json").write_text("{")
    assert monitor.progress(job(tmp_path))["done"] == "0/2"


def test_basic_correction_canary_and_held_jobs_are_explicit(monitor, tmp_path):
    item = job(
        tmp_path,
        campaign_id="d53_test",
        campaign_canary=True,
        status="held",
        expected_runs=1,
        expected_keys=[["cifar10", "genpu", "native_cnn", "scar", "0.1", 0]],
    )
    state = dict(updated_at="2020-01-01T00:00:00+00:00", jobs=[item])
    output = monitor.render(view="held", width=180, height=40, states=[state], devices={})
    assert "1 HOLD" in output and "R!" in output
    assert "HOLD waits" in output and "SRC" in output
