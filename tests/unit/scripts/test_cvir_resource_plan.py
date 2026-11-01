"""CVIR resource planning binds counts/view without claiming verified inputs."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import prepare_candidate_image_resource_plan as planner

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def stable_capacity(monkeypatch):
    monkeypatch.setattr(planner.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**15))


@pytest.mark.parametrize("view,risk,alpha,warm", [("os", 128, 0.2, 1), ("ts", 256, 0.6, 2)])
def test_param_basic_determ_view_and_stage_counts(tmp_path, view, risk, alpha, warm):
    args = dict(
        target_dir=tmp_path,
        n_positive=128,
        n_unlabeled=128,
        population_prior=0.6,
        training_view=view,
    )
    plan = planner.build_plan("cvir", **args)
    assert plan == planner.build_plan("cvir", **args)
    assert plan["risk_pool_rows"] == risk
    assert plan["risk_pool_positive_prior"] == pytest.approx(alpha)
    assert plan["retained_negative_rows"] == 102
    assert plan["default_snapshot_count"] == 21
    assert plan["default_optimizer_updates_per_fit"] == warm + 20
    assert plan["ranking_forward_batches_per_truncation_epoch"] == risk // 128
    assert plan["student_training_forward_batches_per_fit"] == 2 * (warm + 20)
    assert not plan["prior_source_verified"] and not plan["observed_label_counts_verified"]
    assert not plan["bn_cardinality_verified"] and plan["bn_validation_requires_execution_plan"]
    assert not plan["formal_admission"] and not plan["cuda_initialized"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("view", ["os", "ts"])
@pytest.mark.parametrize("positive,unlabeled", [(128, 128), (2, 257)])
def test_param_actual_native_cpu_updates_match_plan(tmp_path, view, positive, unlabeled):
    pytest.importorskip("torch")
    from pu_toolbox.estimators.risk.cvir import CVIRClassifier
    from pu_toolbox.experiment.cvir_adapter import cvir_risk_prior

    plan = planner.build_plan(
        "cvir",
        target_dir=tmp_path,
        n_positive=positive,
        n_unlabeled=unlabeled,
        population_prior=0.6,
        training_view=view,
    )
    alpha = cvir_risk_prior(0.6, n_samples=positive + unlabeled, n_labeled=positive, os_or_ts=view)
    assert plan["risk_pool_positive_prior"] == alpha
    features = np.random.RandomState(9).normal(size=(positive + unlabeled, 2)).astype(np.float32)
    labels = np.r_[np.ones(positive, dtype=int), np.zeros(unlabeled, dtype=int)]
    model = CVIRClassifier(
        unlabeled_positive_prior=alpha,
        prior_source="synthetic test metadata",
        hidden_dim=4,
        device="cpu",
        random_state=9,
    ).fit(features, labels, os_or_ts=view)
    assert model.optimizer_steps_ == plan["default_optimizer_updates_per_fit"]
    assert len(model.epoch_provenance_) == plan["default_snapshot_count"]
    assert len(model.selected_negative_indices_) == plan["retained_negative_rows"]


@pytest.mark.parametrize(
    "prior,view",
    [
        (None, "os"),
        (True, "os"),
        (float("nan"), "os"),
        (0.5, None),
        (0.5, "os"),
        (0.4, "os"),
        (1.0, "ts"),
        (0.6, "unknown"),
    ],
)
def test_param_edge_invalid_or_impossible_prior_is_not_clipped(tmp_path, prior, view):
    with pytest.raises(ValueError):
        planner.build_plan(
            "cvir",
            target_dir=tmp_path,
            n_positive=128,
            n_unlabeled=128,
            population_prior=prior,
            training_view=view,
        )


def test_edge_cvir_options_cannot_silently_affect_other_methods(tmp_path):
    with pytest.raises(ValueError, match="only to cvir"):
        planner.build_plan(
            "pan", target_dir=tmp_path, n_positive=1, n_unlabeled=3, population_prior=0.5
        )


def test_basic_cli_no_numerical_library_imports_or_data_reads(tmp_path):
    code = """
import importlib.abc, runpy, sys
class DenyNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {'torch', 'torchvision', 'numpy', 'scipy', 'sklearn', 'pu_toolbox'}
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('planner imported numerical dependencies')
sys.meta_path.insert(0, DenyNumerics())
sys.argv = ['planner', '--method', 'cvir', '--target-dir', sys.argv[1],
            '--n-positive', '128', '--n-unlabeled', '128',
            '--population-prior', '.6', '--training-view', 'ts']
runpy.run_module('scripts.prepare_candidate_image_resource_plan', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=planner.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["default_optimizer_updates_per_fit"] == 22
    assert not plan["checkpoint_or_dataset_loaded"]
    assert not list(tmp_path.iterdir())
