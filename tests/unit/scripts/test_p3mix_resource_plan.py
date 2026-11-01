"""Torch-free P3Mix planning distinguishes optimizer work and extra forwards."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit
PATH = Path(__file__).resolve().parents[3] / "scripts/prepare_candidate_image_resource_plan.py"
SPEC = importlib.util.spec_from_file_location("p3mix_resource_plan_test", PATH)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


@pytest.fixture(autouse=True)
def stable_capacity(monkeypatch):
    monkeypatch.setattr(planner.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**15))


@pytest.mark.parametrize("variant", ["base", "e", "c"])
@pytest.mark.parametrize(
    "policy,steps,p_rows,u_rows", [("role_balanced", 3, 6, 257), ("merged_shuffle", 3, 2, 257)]
)
def test_param_basic_determ_variant_and_batching_work(
    tmp_path, variant, policy, steps, p_rows, u_rows
):
    plan = planner.build_plan(
        "p3mix",
        target_dir=tmp_path,
        n_positive=2,
        n_unlabeled=257,
        variant=variant,
        batching_policy=policy,
    )
    assert plan == planner.build_plan(
        "p3mix",
        target_dir=tmp_path,
        n_positive=2,
        n_unlabeled=257,
        variant=variant,
        batching_policy=policy,
    )
    assert plan["default_optimizer_updates_per_fit"] == 100 * steps
    assert plan["rows_seen_per_epoch"] == {"P": p_rows, "U": u_rows}
    assert plan["teacher_forward_batches_per_epoch"] == (steps if variant == "e" else 0)
    assert plan["student_training_forward_batches_per_epoch"] == 2 * steps
    assert plan["positive_pool_refresh_forward_batches_per_epoch"] == 1
    assert plan["training_network_instances"] == (2 if variant == "e" else 1)
    assert not plan["formal_admission"] and not plan["cuda_initialized"]
    assert not list(tmp_path.iterdir())


def test_edge_merged_batch_changes_actual_update_count(tmp_path):
    args = dict(target_dir=tmp_path, n_positive=128, n_unlabeled=128)
    balanced = planner.build_plan("p3mix", **args, batching_policy="role_balanced")
    merged = planner.build_plan("p3mix", **args, batching_policy="merged_shuffle")
    assert balanced["default_optimizer_updates_per_fit"] == 100
    assert merged["default_optimizer_updates_per_fit"] == 200


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("pan", {"variant": "e"}),
        ("vpu", {"batching_policy": "merged_shuffle"}),
        ("p3mix", {"variant": "both"}),
        ("p3mix", {"batching_policy": "unknown"}),
    ],
)
def test_param_edge_invalid_policy_or_cross_method_options_refused(tmp_path, method, kwargs):
    with pytest.raises(ValueError):
        planner.build_plan(method, target_dir=tmp_path, n_positive=2, n_unlabeled=3, **kwargs)


def test_basic_cli_e_variant_and_merged_policy_remain_torch_free(tmp_path):
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("planner imported torch")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], "--method", "p3mix", "--target-dir", sys.argv[2],
            "--n-positive", "128", "--n-unlabeled", "128", "--variant", "e",
            "--batching-policy", "merged_shuffle"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(PATH), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["default_optimizer_updates_per_fit"] == 200
    assert plan["training_network_instances"] == 2 and not plan["cuda_initialized"]


@pytest.mark.parametrize("policy", ["role_balanced", "merged_shuffle"])
@pytest.mark.parametrize("variant", ["base", "e", "c"])
def test_param_actual_short_cpu_model_matches_planned_default_work(tmp_path, policy, variant):
    pytest.importorskip("torch")
    from pu_toolbox.estimators.deep.p3mix import P3MixClassifier

    features = np.random.RandomState(8).normal(size=(259, 2)).astype(np.float32)
    labels = np.r_[np.ones(2, dtype=int), np.zeros(257, dtype=int)]
    plan = planner.build_plan(
        "p3mix",
        target_dir=tmp_path,
        n_positive=2,
        n_unlabeled=257,
        variant=variant,
        batching_policy=policy,
    )
    model = P3MixClassifier(
        variant=variant, batching_policy=policy, hidden_dim=4, device="cpu"
    ).fit(features, labels)
    assert model.optimizer_steps_ == plan["default_optimizer_updates_per_fit"]
    assert model.positive_rows_seen_ == 100 * plan["rows_seen_per_epoch"]["P"]
    assert model.unlabeled_rows_seen_ == 100 * plan["rows_seen_per_epoch"]["U"]
