"""Planning must be read-only, torch-free and never an admission/profile claim."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import prepare_candidate_image_resource_plan as planner

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "method,snapshots,steps",
    [("pulda", 120, 360), ("gradpu", 200, 400), ("pan", 100, 1000), ("vpu", 100, 600)],
)
def test_basic_determ_default_budget_counts_and_missing_measurement_fail_closed(
    tmp_path, monkeypatch, method, snapshots, steps
):
    monkeypatch.setattr(planner.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**15))
    first = planner.build_plan(method, target_dir=tmp_path, n_positive=400, n_unlabeled=257)
    assert first == planner.build_plan(method, target_dir=tmp_path, n_positive=400, n_unlabeled=257)
    assert first["default_snapshot_count"] == snapshots
    assert first["default_optimizer_updates_per_fit"] == steps
    assert first["sample_checkpoint"] is None
    assert first["weights_only_estimate_bytes"] is None
    assert first["estimated_weights_fit_available_space"] is None
    assert first["approved_representation_ref"] is None
    assert not first["ready_for_formal_run"] and not first["formal_admission"]
    assert not first["training_started"] and not first["cuda_initialized"]
    assert not first["formal_resource_profile_complete"]
    assert not list(tmp_path.iterdir())
    assert len(first["source_files_sha256"]) == 4
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("free_bytes,expected_fit", [(1_440_000, True), (1_439_999, False)])
def test_edge_measured_file_size_counts_all_snapshots_candidates_attempts_and_concurrency(
    tmp_path, monkeypatch, free_bytes, expected_fit
):
    # Deliberately not a torch file: the planner must not load or attest its format.
    sample = tmp_path / "untrusted-sample.pt"
    sample.write_bytes(b"x" * 1000)
    monkeypatch.setattr(planner.shutil, "disk_usage", lambda path: SimpleNamespace(free=free_bytes))
    result = planner.build_plan(
        "gradpu",
        target_dir=tmp_path,
        n_positive=2,
        n_unlabeled=3,
        sample_checkpoint=sample,
        candidates=3,
        retained_attempts=2,
        concurrent_runs=1,
    )
    assert result["weights_only_estimate_bytes"] == 1000 * 200 * 3 * 2 * 120 // 100
    assert result["estimated_weights_fit_available_space"] is expected_fit
    assert result["sample_checkpoint"]["bytes"] == 1000
    assert result["sample_checkpoint"]["sha256"] == planner._file_digest(sample)
    assert (
        result["sample_checkpoint"]["format_architecture_and_training_provenance_verified"] is False
    )
    assert not result["checkpoint_or_dataset_loaded"]
    assert not result["ready_for_formal_run"] and not result["capacity_is_reserved"]
    assert not result["post_selection_reclaim_reduces_estimate"]
    assert sample.read_bytes() == b"x" * 1000
    assert list(tmp_path.iterdir()) == [sample]


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "nnpu"},
        {"n_positive": True},
        {"n_unlabeled": 0},
        {"candidates": 0},
        {"retained_attempts": 3},
        {"retained_attempts": False},
        {"concurrent_runs": -1},
        {"safety_margin_percent": -1},
        {"safety_margin_percent": 0.5},
    ],
)
def test_param_invalid_requests_are_rejected_before_capacity_check(
    tmp_path, monkeypatch, overrides
):
    kwargs = dict(method="pulda", target_dir=tmp_path, n_positive=10, n_unlabeled=20)
    kwargs.update(overrides)

    def forbidden(path):
        pytest.fail("invalid input must be rejected before filesystem capacity check")

    monkeypatch.setattr(planner.shutil, "disk_usage", forbidden)
    with pytest.raises(ValueError):
        planner.build_plan(**kwargs)


def test_edge_missing_directory_empty_or_nonregular_sample_are_not_created_or_loaded(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        planner.build_plan("pulda", target_dir=missing, n_positive=1, n_unlabeled=1)
    assert not missing.exists()
    empty = tmp_path / "empty.pt"
    empty.touch()
    for target, sample in [(empty, None), (tmp_path, empty), (tmp_path, tmp_path)]:
        with pytest.raises(ValueError):
            planner.build_plan(
                "pulda",
                target_dir=target,
                n_positive=1,
                n_unlabeled=1,
                sample_checkpoint=sample,
            )


def test_basic_concurrency_and_integer_round_up_do_not_credit_reclaim(tmp_path, monkeypatch):
    sample = tmp_path / "one-byte"
    sample.write_bytes(b"x")
    monkeypatch.setattr(planner.shutil, "disk_usage", lambda path: SimpleNamespace(free=10000))
    result = planner.build_plan(
        "pulda",
        target_dir=tmp_path,
        n_positive=1,
        n_unlabeled=1,
        sample_checkpoint=sample,
        concurrent_runs=3,
        safety_margin_percent=1,
    )
    assert result["weights_only_estimate_bytes"] == 728  # ceil(120 * 2 * 3 * 1.01)
    assert result["default_optimizer_updates_per_fit"] == 120


def test_basic_cli_errors_and_json_output(tmp_path, capsys):
    args = [
        "--method",
        "pulda",
        "--target-dir",
        str(tmp_path),
        "--n-positive",
        "2",
        "--n-unlabeled",
        "3",
    ]
    assert planner.main(args) == 0
    assert json.loads(capsys.readouterr().out)["default_snapshot_count"] == 120
    with pytest.raises(SystemExit) as error:
        planner.main(args + ["--candidates", "0"])
    assert error.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("method,epochs", [("gradpu", 200), ("pan", 100), ("vpu", 100)])
def test_determ_cli_remains_importable_when_torch_is_forbidden(tmp_path, method, epochs):
    code = """
import importlib.abc, runpy, sys
class DenyTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision'}:
            raise AssertionError('resource planning must not import torch')
sys.meta_path.insert(0, DenyTorch())
sys.argv = ['planner', '--method', sys.argv[2], '--target-dir', sys.argv[1],
            '--n-positive', '2', '--n-unlabeled', '3']
runpy.run_module('scripts.prepare_candidate_image_resource_plan', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), method],
        cwd=planner.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(planner.ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["default_snapshot_count"] == epochs
    assert not list(tmp_path.iterdir())
