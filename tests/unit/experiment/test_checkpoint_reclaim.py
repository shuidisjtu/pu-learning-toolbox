# ruff: noqa: N803, N806
"""Non-selected epoch weights are reclaimed without disturbing selection.

D25 makes the reclaim opt-in.  With the switch off (the default) nothing is
deleted and every reference reports ``reclaimed=False``; with it on, only the
weights the two protocols selected survive, and the manifest keeps the original
path and digest of what was removed so an auditor can still tell "existed, then
reclaimed" apart from "never persisted".
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.base import BaseEstimator
from torch import nn

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner

pytestmark = pytest.mark.unit


class ThreeEpochs(BaseEstimator):
    """PA and OA take the first two epochs; the third is never selected.

    Epoch 0 ranks the two val rows in opposite directions, making it PA's
    optimum and OA's worst; epoch 1 is the mirror image and is OA's optimum.
    Epoch 2 scores both rows the same way and only ties with epoch 1, which the
    tie-break resolves in favour of the earlier epoch.
    """

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier

    def fit(self, X, y, *, epoch_callback=None):
        self.model_ = nn.Linear(2, 1)
        plan = (
            ([10.0 * self.multiplier, 0.0], -5.0 * self.multiplier),
            ([0.0, 1.0], -0.5),
            ([1.0, 1.0], -0.5),
        )
        for epoch, (weights, bias) in enumerate(plan):
            with torch.no_grad():
                self.model_.weight.copy_(torch.tensor([weights]))
                self.model_.bias.fill_(bias)
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


def _run(tmp_path, *, reclaim):
    bundle = _bundle()
    config = {"c": 1.0}
    if reclaim is not None:
        config["reclaim_unselected_checkpoints"] = reclaim
    return ExperimentRunner(
        config=config, manifest_path=str(tmp_path / "manifest.json"), class_prior=0.5
    ).fit(ThreeEpochs(), bundle.train, bundle.pu_val, bundle.clean_val, bundle.test)


def _weights_on_disk(tmp_path):
    # str, not Path: manifest references are strings, and the two must compare.
    return sorted(str(path) for path in (tmp_path / "checkpoints").glob("attempt-*/*.pt"))


def _selected_paths(result):
    return {artifact["checkpoint"]["path"] for artifact in result.manifest["selection"].values()}


def _survivors(result):
    """Names of the epoch weights the run kept, in epoch order."""
    return sorted(
        Path(reference["path"]).name
        for reference in result.manifest["candidate_runs"][0]["epoch_checkpoints"]
        if not reference["reclaimed"]
    )


def test_basic_reclaim_keeps_exactly_the_selected_weights(tmp_path):
    result = _run(tmp_path, reclaim=True)
    selected = _selected_paths(result)
    assert len(selected) == 2, "this fixture must put PA and OA on different epochs"
    assert set(_weights_on_disk(tmp_path)) == selected


def test_basic_reclaim_leaves_selection_and_test_metrics_untouched(tmp_path):
    kept = _run(tmp_path / "kept", reclaim=False)
    reclaimed = _run(tmp_path / "reclaimed", reclaim=True)
    assert kept.selections == reclaimed.selections
    assert kept.test_metrics == reclaimed.test_metrics


def test_basic_reclaimed_reference_keeps_the_original_path_and_digest(tmp_path):
    result = _run(tmp_path, reclaim=True)
    references = result.manifest["candidate_runs"][0]["epoch_checkpoints"]
    assert len(references) == 3
    selected = _selected_paths(result)

    for reference in references:
        assert reference["path"] and reference["sha256"]
        if reference["path"] in selected:
            assert reference["reclaimed"] is False
            assert Path(reference["path"]).is_file()
        else:
            assert reference["reclaimed"] is True
            assert not Path(reference["path"]).exists()


def test_edge_switch_off_deletes_nothing_and_marks_nothing(tmp_path):
    result = _run(tmp_path, reclaim=False)
    references = result.manifest["candidate_runs"][0]["epoch_checkpoints"]
    assert len(_weights_on_disk(tmp_path)) == 3
    assert all(reference["reclaimed"] is False for reference in references)
    assert all(Path(reference["path"]).is_file() for reference in references)


def test_edge_absent_switch_defaults_to_keeping_every_checkpoint(tmp_path):
    result = _run(tmp_path, reclaim=None)
    assert len(_weights_on_disk(tmp_path)) == 3
    assert all(
        reference["reclaimed"] is False
        for reference in result.manifest["candidate_runs"][0]["epoch_checkpoints"]
    )


def test_param_failed_unlink_is_not_swallowed_into_a_half_reclaimed_run(tmp_path, monkeypatch):
    """Reclaim failures must surface, not leave the manifest describing a lie."""
    original = Path.unlink

    def refuse(self, *args, **kwargs):
        if self.suffix == ".pt":
            raise PermissionError("checkpoint is locked")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse)
    with pytest.raises(PermissionError, match="checkpoint is locked"):
        _run(tmp_path, reclaim=True)


def test_determ_reclaim_set_is_reproducible_across_runs(tmp_path):
    """What survives is a function of the selection, not of delete order or time."""
    first = _run(tmp_path / "first", reclaim=True)
    second = _run(tmp_path / "second", reclaim=True)

    assert _survivors(first) == _survivors(second) == ["epoch_0001_model.pt", "epoch_0002_model.pt"]
    assert first.selections == second.selections
    assert first.test_metrics == second.test_metrics
