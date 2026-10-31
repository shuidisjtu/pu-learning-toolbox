"""KLDCE dense named-array lower bound checked against real tiny QP allocations."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit
PATH = Path(__file__).resolve().parents[3] / "scripts/prepare_kldce_resource_plan.py"
SPEC = importlib.util.spec_from_file_location("kldce_resource_plan_test", PATH)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def plan(**overrides):
    return planner.build_plan(
        **(dict(n_positive=4, n_unlabeled=6, feature_count=3, gram_itemsize=8) | overrides)
    )


def test_basic_determ_read_only_default_budget_and_named_array_arithmetic(tmp_path):
    planned = plan(prediction_rows=7)
    assert planned == plan(prediction_rows=7)
    assert planned["training_gram_shape"] == [10, 10]
    assert planned["qp_float64_named_blocks"]["Q"]["shape"] == [16, 16]
    assert planned["qp_named_array_payload_bytes"] == 8 * (100 + 100 + 60 + 36 + 256)
    assert planned["known_simultaneous_gram_and_qp_payload_bytes"] == 5216
    assert planned["prediction_gram_payload_bytes"] == 7 * 10 * 8
    assert planned["covariance_single_matrix_payload_bytes"] == 3 * 3 * 8
    assert planned["max_acs_iter"] == 50 and planned["max_inner_iter"] == 2000
    assert planned["qp_solves_upper_bound"] == 99
    assert planned["smo_pair_updates_upper_bound"] == 198000
    assert not planned["data_loaded"] and not planned["resource_profile_completed"]
    assert not planned["ready_for_formal_run"] and not list(tmp_path.iterdir())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_basic_actual_named_qp_payload_matches_plan_without_running_solver(dtype):
    from pu_toolbox.estimators.risk.kldce import _build_dual_qp, _centroid_covariance, _rbf_kernel

    features = np.random.RandomState(19).normal(size=(10, 3)).astype(dtype)
    kernel = _rbf_kernel(features, features, 0.5)
    captured = {}
    previous = sys.getprofile()

    def observe(frame, event, argument):
        if frame.f_code is _build_dual_qp.__code__ and event == "return":
            captured.update(frame.f_locals)

    try:
        sys.setprofile(observe)
        _build_dual_qp(
            np.zeros(3), features, kernel, np.r_[np.ones(4), -np.ones(6)], 1.0, 0.5, 10, 4, -0.2
        )
    finally:
        sys.setprofile(previous)
    planned = plan(gram_itemsize=kernel.itemsize)
    for name, block in planned["qp_float64_named_blocks"].items():
        assert captured[name].dtype == np.float64
        assert list(captured[name].shape) == block["shape"]
        assert captured[name].nbytes == block["bytes"]
    assert kernel.nbytes == planned["training_gram_payload_bytes"]
    covariance = _centroid_covariance(features[4:])
    assert covariance.nbytes == planned["covariance_single_matrix_payload_bytes"]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(n_positive=True),
        dict(n_unlabeled=0),
        dict(feature_count=-1),
        dict(gram_itemsize=2),
        dict(gram_itemsize=True),
        dict(prediction_rows=-1),
        dict(max_acs_iter=0),
        dict(max_inner_iter=True),
        dict(memory_cap_bytes=0),
    ],
)
def test_param_invalid_counts_dtype_or_explicit_budgets_rejected(overrides):
    with pytest.raises(ValueError):
        plan(**overrides)


def test_edge_cap_only_rejects_known_payload_never_approves_safety():
    payload = plan()["known_simultaneous_gram_and_qp_payload_bytes"]
    assert plan(memory_cap_bytes=payload - 1)["known_payload_exceeds_cap"]
    at_cap = plan(memory_cap_bytes=payload)
    assert not at_cap["known_payload_exceeds_cap"] and not at_cap["ready_for_formal_run"]
    assert at_cap["below_cap_is_not_fit_approval"] and not at_cap["gram_dtype_verified"]
    assert "previous_iteration_Q_while_building_next_Q" in at_cap["excluded_memory"]


def test_edge_explicit_budget_and_first_iteration_rollback_bound():
    small = plan(max_acs_iter=1, max_inner_iter=7)
    assert small["qp_solves_upper_bound"] == 1 and small["smo_pair_updates_upper_bound"] == 7
    larger = plan(max_acs_iter=3, max_inner_iter=5)
    assert larger["qp_solves_upper_bound"] == 5 and larger["smo_pair_updates_upper_bound"] == 25


def test_edge_actual_forced_rollback_path_reaches_but_never_exceeds_qp_bound(monkeypatch):
    from pu_toolbox.estimators.risk import kldce

    original = kldce._solve_dual_smo
    calls = []

    def decreasing_objective(*args, **kwargs):
        calls.append(1)
        solution, diagnostics = original(*args, **kwargs)
        # Only force the branch condition; keep each genuine tiny QP solution.
        return solution, dict(diagnostics, dual_obj=-float(len(calls)))

    monkeypatch.setattr(kldce, "_solve_dual_smo", decreasing_objective)
    features = np.random.RandomState(19).normal(size=(10, 3))
    labels = np.r_[np.ones(4, int), np.zeros(6, int)]
    with pytest.warns(UserWarning, match="did not converge"):
        fitted = kldce.KLDCEClassifier(
            0.2, mom_groups=2, max_acs_iter=3, max_inner_iter=5, tol=0, random_state=1
        ).fit(features, labels, class_prior=0.5)
    planned = plan(max_acs_iter=3, max_inner_iter=5)
    assert fitted.n_acs_iter_ == 3 and len(calls) == planned["qp_solves_upper_bound"] == 5


def test_basic_cli_large_counts_need_no_numpy_torch_data_or_model_imports():
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'scipy', 'sklearn', 'pu_toolbox'}:
        raise AssertionError('numerical/model import in planner: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--n-positive', '1000000000000', '--n-unlabeled', '1000000000000',
            '--feature-count', '1000000', '--gram-itemsize', '8', '--memory-cap-bytes', '1024']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(PATH)], capture_output=True, text=True)
    assert result.returncode == 2
    planned = json.loads(result.stdout)
    assert planned["known_payload_exceeds_cap"] and not planned["estimator_constructed"]
    assert not planned["cuda_initialized"] and not planned["formal_admission"]
