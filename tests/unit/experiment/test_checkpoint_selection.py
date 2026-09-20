# ruff: noqa: N803, N806
"""Controlled opposite PA/OA optima prove independent selection and restoration."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.base import BaseEstimator
from torch import nn

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer, load_selected_checkpoint
from pu_toolbox.experiment.runner import ExperimentRunner, _checkpoint_context
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA

pytestmark = pytest.mark.unit


class OppositeOptima(BaseEstimator):
    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier

    def fit(self, X, y, *, epoch_callback=None):
        self.model_ = nn.Linear(2, 1)
        self.fit_labels_ = np.array(y)
        for epoch, weights in enumerate(([10.0 * self.multiplier, 0.0], [0.0, 1.0])):
            with torch.no_grad():
                self.model_.weight.copy_(torch.tensor([weights]))
                self.model_.bias.fill_(-5 * self.multiplier if epoch == 0 else -0.5)
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        return self

    def decision_function(self, X):
        with torch.no_grad():
            return self.model_(torch.as_tensor(X, dtype=torch.float32)).reshape(-1).numpy()

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def _bundle():
    return DatasetBundle(
        **{
            role: DatasetPart(
                X=np.array([[1, 1], [0, 0]], dtype=np.float32)
                if role in ("train", "pu_val")
                else np.array([[0, 1], [1, 0]], dtype=np.float32),
                labels=np.array([1, 0]),
                indices=np.arange(index * 2, index * 2 + 2),
                view="clean",
                for_selection=role != "test",
            )
            for index, role in enumerate(("train", "pu_val", "clean_val", "test"))
        }
    )


def _run(tmp_path, bundle=None, *, candidates=None):
    bundle = bundle or _bundle()
    config = {"c": 1.0}
    if candidates is not None:
        config["candidates"] = candidates
    return ExperimentRunner(
        config=config, manifest_path=str(tmp_path / "manifest.json"), class_prior=0.5
    ).fit(OppositeOptima(), bundle.train, bundle.pu_val, bundle.clean_val, bundle.test)


def test_basic_runner_restores_opposite_pa_oa_epoch_optima(tmp_path):
    result = _run(tmp_path)
    assert result.selections["PA"].epoch == 1
    assert result.selections["OA"].epoch == 2
    assert result.selections["PA"].checkpoint_index == 0
    assert result.selections["OA"].checkpoint_index == 1
    np.testing.assert_array_equal(
        result.selected_models["PA"].decision_function(_bundle().test.X), [-5, 5]
    )
    np.testing.assert_array_equal(
        result.selected_models["OA"].decision_function(_bundle().test.X), [0.5, -0.5]
    )
    assert result.test_metrics["OA"]["accuracy"] == 1.0
    assert result.test_metrics["PA"]["accuracy"] == 0.0


def test_basic_manifest_has_separate_selection_refs_and_all_epoch_metrics(tmp_path):
    result = _run(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    left, right = (manifest["selection"][name]["checkpoint"] for name in ("PA", "OA"))
    assert left["path"] != right["path"] and left["sha256"] != right["sha256"]
    assert all(Path(reference["path"]).is_file() for reference in (left, right))
    assert all(
        set(checkpoint["validation_metrics"]) == {"PA", "OA"}
        for checkpoint in manifest["candidate_runs"][0]["epoch_checkpoints"]
    )
    assert result.resources["offline_checkpoint_validation_elapsed_seconds"] > 0
    assert (
        result.resources["offline_selection_elapsed_seconds"]
        >= result.resources["offline_checkpoint_validation_elapsed_seconds"]
    )
    restored = load_selected_checkpoint(manifest["selection"]["OA"], nn.Linear(2, 1))
    np.testing.assert_array_equal(restored.predict(_bundle().test.X), [1, 0])


def test_determ_clean_validation_changes_only_oa_not_pa_choice(tmp_path):
    bundle = _bundle()
    first = _run(tmp_path / "first", bundle)
    changed = replace(
        bundle, clean_val=replace(bundle.clean_val, labels=1 - bundle.clean_val.labels)
    )
    second = _run(tmp_path / "second", changed)
    assert first.selections["PA"] == second.selections["PA"]
    assert first.selections["OA"].epoch != second.selections["OA"].epoch


def test_determ_test_values_and_labels_cannot_change_selection(tmp_path):
    bundle = _bundle()
    first = _run(tmp_path / "first", bundle)
    changed = replace(
        bundle, test=replace(bundle.test, X=bundle.test.X * 17, labels=1 - bundle.test.labels)
    )
    second = _run(tmp_path / "second", changed)
    assert first.selections == second.selections


def test_basic_candidate_pool_and_epoch_are_selected_jointly(tmp_path):
    """Both dimensions are searched, and the two protocols land on different epochs.

    ``multiplier`` only scales the epoch-0 scores: m * (10*x0 - 5) has the same
    sign for every m > 0, so the two candidates predict identically.  Both
    protocols min-max normalise each checkpoint in its own val-side space (the
    runner reuses those affine constants on the test set), which erases the
    scale difference and leaves PA tied -- it takes the earliest candidate,
    exactly as OA already did.  The epochs still differ, which is the joint
    search this test is about.
    """
    result = _run(tmp_path, candidates=[{"multiplier": 1.0}, {"multiplier": 2.0}])
    assert result.manifest["selection"]["PA"]["candidate_index"] == 0
    assert result.manifest["selection"]["OA"]["candidate_index"] == 0
    assert result.selections["PA"].epoch == 1 and result.selections["OA"].epoch == 2


def test_edge_tied_snapshot_scores_choose_earliest_epoch(tmp_path):
    bundle = _bundle()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        OppositeOptima(), bundle.train.X, bundle.train.labels
    )
    val = replace(bundle.clean_val, X=np.array([[1, 1], [0, 0]], dtype=np.float32))
    assert ProtocolOA().select([trajectory], val).epoch == 1


def test_param_corrupt_snapshot_cannot_silently_select_final_weights(tmp_path):
    bundle = _bundle()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        OppositeOptima(), bundle.train.X, bundle.train.labels
    )
    Path(trajectory.checkpoints[0].path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        ProtocolOA().select([trajectory], bundle.clean_val)


def test_edge_pa_missing_unlabeled_validation_fails_loudly(tmp_path):
    bundle = _bundle()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        OppositeOptima(), bundle.train.X, bundle.train.labels
    )
    val = replace(bundle.pu_val, view="pu", labels=np.ones(2, dtype=int))
    with pytest.raises(ValueError, match="both labeled positive and unlabeled"):
        ProtocolPA().select([trajectory], val)


def test_param_truncated_budget_cannot_clear_formal_checkpoint_blocker(tmp_path):
    bundle = _bundle()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        OppositeOptima(), bundle.train.X, bundle.train.labels
    )
    context = {
        "execution_mode": "versioned_pilot",
        "budget": {"epochs": 3},
        "formal_blockers": [
            "collaborator_review",
            "per_epoch_independent_PA_OA_checkpoint_selection",
        ],
    }
    incomplete = _checkpoint_context(context, [trajectory])
    assert "per_epoch_independent_PA_OA_checkpoint_selection" in incomplete["formal_blockers"]
    context["budget"]["epochs"] = 2
    complete = _checkpoint_context(context, [trajectory])
    assert complete["formal_blockers"] == ["collaborator_review"]
    assert complete["selection_checkpoint_scope"] == "independent_per_epoch"
    assert complete["formal_eligible"] is False


def test_basic_in_memory_run_retains_models_but_no_invalid_persistent_paths():
    bundle = _bundle()
    result = ExperimentRunner(config={"c": 1.0}, class_prior=0.5).fit(
        OppositeOptima(), bundle.train, bundle.pu_val, bundle.clean_val, bundle.test
    )
    assert all(
        reference["checkpoint"]["path"] is None
        for reference in result.manifest["selection"].values()
    )
    assert set(result.selected_models) == {"PA", "OA"}
    assert np.isfinite(result.selected_models["PA"].decision_function(bundle.test.X)).all()


def test_param_oa_rejects_pu_view_and_nonselection_test_partition(tmp_path):
    bundle = _bundle()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        OppositeOptima(), bundle.train.X, bundle.train.labels
    )
    for val in (replace(bundle.clean_val, view="pu"), bundle.test):
        with pytest.raises(ValueError, match="selection-enabled clean"):
            ProtocolOA().select([trajectory], val)


def test_edge_loaded_threshold_preserves_float32_val_space_boundary(tmp_path):
    result = _run(tmp_path)
    selection = dict(result.manifest["selection"]["OA"])
    selection["threshold"] = 1.0
    selection["metrics"] = {"val_score_min": -10.0, "val_score_scale": float(np.float32(10.1))}
    model = load_selected_checkpoint(selection, nn.Linear(2, 1))
    X = np.array([[0, 0.6], [0, -9.5]], dtype=np.float32)
    np.testing.assert_array_equal(model.predict(X), [1, 0])
