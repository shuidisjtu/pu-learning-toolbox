"""Queue inventory counts fits once and keeps correction campaigns separate."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def inventory():
    path = Path(__file__).resolve().parents[3] / "scripts/audit_survey_queue.py"
    spec = importlib.util.spec_from_file_location("queue_inventory_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key(seed):
    return ("mnist", "nnpu", "native_cnn", "scar", "0.1", seed)


def job(tmp_path, seed, status="pending", campaign=None):
    label = f"{campaign or 'baseline'}-{seed}"
    return dict(
        label=label,
        root=str(tmp_path / label),
        expected_runs=1,
        expected_keys=[list(key(seed))],
        status=status,
        campaign_id=campaign,
    )


def test_basic_determ_corrections_are_not_new_fits(inventory, tmp_path):
    jobs = [
        job(tmp_path, 1, "completed"),
        job(tmp_path, 2, "running"),
        job(tmp_path, 1, "completed", "repair"),
    ]
    expected = {key(n) for n in range(3)}
    report = inventory.reconcile(expected, [key(0)], jobs, lambda j: True)
    assert report == inventory.reconcile(expected, [key(0)], jobs, lambda j: True)
    assert report["planned_fits"] == 3
    assert report["completed_fit_lower_bound"] == 2
    assert report["remaining_fit_upper_bound"] == 1
    assert report["correction_status_fits"]["artifact_verified_completed"] == 1
    assert report["coverage_complete"] and report["completed_artifacts_valid"]
    assert not report["formal_admission_verified"]


def test_edge_missing_extra_and_invalid_completion_are_explicit(inventory, tmp_path):
    jobs = [job(tmp_path, 1, "completed"), job(tmp_path, 4)]
    report = inventory.reconcile({key(n) for n in range(3)}, [key(0)], jobs, lambda j: False)
    assert report["missing_fit_keys"] == [key(2)]
    assert report["unexpected_fit_keys"] == [key(4)]
    assert report["invalid_completed_jobs"] == ["baseline-1"]
    assert report["completed_fit_lower_bound"] == 1
    assert not report["coverage_complete"] and not report["completed_artifacts_valid"]


@pytest.mark.parametrize("bad", ["duplicate_previous", "overlap", "duplicate_job", "status"])
def test_param_bad_identity_ownership_fails_closed(inventory, tmp_path, bad):
    previous, jobs = [key(0)], [job(tmp_path, 1)]
    if bad == "duplicate_previous":
        previous.append(key(0))
    elif bad == "overlap":
        previous.append(key(1))
    elif bad == "duplicate_job":
        jobs.append(job(tmp_path, 1))
    else:
        jobs[0]["status"] = "unknown"
    with pytest.raises(ValueError):
        inventory.reconcile({key(0), key(1)}, previous, jobs, lambda j: True)


def test_edge_cli_never_overwrites_output(inventory, tmp_path):
    output = tmp_path / "report.json"
    output.write_text("user content")
    with pytest.raises(SystemExit) as error:
        inventory.main(["--queue", str(tmp_path), "--protocol", "missing", "--output", str(output)])
    assert error.value.code == 1
    assert output.read_text() == "user content"


@pytest.mark.parametrize("complete,code", [(True, 0), (False, 2)])
def test_param_cli_status_and_report(inventory, tmp_path, monkeypatch, complete, code):
    report = dict(coverage_complete=complete, completed_artifacts_valid=True)
    monkeypatch.setattr(inventory, "audit", lambda q, p: report)
    output = tmp_path / "report.json"
    assert (
        inventory.main(["--queue", str(tmp_path), "--protocol", "unused", "--output", str(output)])
        == code
    )
    assert '"coverage_complete"' in output.read_text()
