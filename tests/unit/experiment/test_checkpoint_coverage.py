# ruff: noqa: N803, N806, F811
"""Checkpoint coverage follows a declaration, not the set that was written.

``_validate_trajectory`` used to derive the expected component set from
``trajectory.checkpoints`` itself.  An estimator that should have written two
per-epoch networks but wrote one therefore produced a *smaller* expectation
that matched exactly what it saw: the check passed on the very drift it exists
to catch.

These tests pin the declaration-driven form, and the runner-level consequence
that gives it its point.  A dropped component fails the candidate, the run
ends with no trajectories, and the checkpoint blocker is therefore never
discharged -- an incomplete run must not be able to unlock independent
per-epoch PA/OA selection just by omitting data.
"""

import json
import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch
from _survey_script_helpers import make_splits, survey_script  # noqa: F401
from sklearn.base import BaseEstimator
from torch import nn

from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer
from pu_toolbox.experiment.runner import (
    ExperimentRunner,
    _declared_epoch_components,
    _validate_trajectory,
)
from pu_toolbox.experiment.strategies import DeepFitTrainer
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATH, load_protocol, resolve_unit

pytestmark = pytest.mark.unit

_TWO_TEACHERS = ("teacher_1", "teacher_2")


def _data(n=12):
    X = np.random.RandomState(4).normal(size=(n, 3)).astype(np.float32)
    return X, np.array([1, 1, 0] * (n // 3))


def _pu_val(n=12):
    X, y = _data(n)
    return DatasetPart(X=X, labels=y, view="clean", indices=np.arange(n), for_selection=True)


def _selfpu(tmp_path):
    X, y = _data()
    model = SelfPUClassifier(
        0.3,
        hidden_dim=4,
        max_epochs=2,
        batch_size=6,
        threshold=0.7,
        random_state=0,
        device="cpu",
        warmup_epochs=0,
        self_paced_start=0,
        self_paced_end=1,
        distill_start=1,
    )
    with pytest.warns(UserWarning, match="explicit Self-PU ablation"):
        return EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)


def _nnpu(tmp_path):
    X, y = _data()
    model = NonNegativePUClassifier(class_prior=0.3, max_epochs=3, random_state=0, device="cpu")
    return EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)


class _OneNetwork(BaseEstimator):
    """Fires one callback per epoch and exposes exactly one network."""

    epoch_components: tuple = ("model",)

    def __init__(self, epochs=2):
        self.epochs = epochs

    def _expose(self, X):
        self.model_ = nn.Linear(np.shape(X)[1], 1)

    def fit(self, X, y, *, epoch_callback=None):
        self._expose(X)
        for epoch in range(self.epochs):
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        return self

    def decision_function(self, X):
        with torch.no_grad():
            return self.model_(torch.as_tensor(X, dtype=torch.float32)).reshape(-1).numpy()

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


class _DeclaresTwoWritesOne(_OneNetwork):
    """Declares both teachers but exposes only ``model_`` -- the R8(b) drift."""

    epoch_components = _TWO_TEACHERS


class _DeclaresOneWritesTwo(_OneNetwork):
    """Declares the default single network but exposes both teachers."""

    threshold = 0.5

    def _expose(self, X):
        self.model_ = nn.Linear(np.shape(X)[1], 1)
        self.teacher_1_ = self.model_
        self.teacher_2_ = self.model_


class _DroppingTrainer(DeepFitTrainer):
    """Checkpoints normally, then hands back a trajectory missing one teacher.

    A subclass rather than ``DeepFitTrainer`` itself on purpose: the runner
    swaps in its own checkpoint trainer only for the exact base types, so this
    one is used verbatim -- which is how a real drift between an estimator's
    declaration and what got written would reach the validator.
    """

    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trajectory = EpochCheckpointTrainer(checkpoint_dir=self.checkpoint_dir).fit(
                estimator, X, y, class_prior=class_prior, val_pu=val_pu
            )
        return replace(
            trajectory,
            checkpoints=[c for c in trajectory.checkpoints if c.component == "teacher_1"],
        )


class _RecordingTrainer(DeepFitTrainer):
    """A PU trainer that counts fits, for asserting training never started."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        self.calls += 1
        return super().fit(estimator, X, y, class_prior=class_prior, val_pu=val_pu)


def _parts(survey_script, tmp_path):
    make_splits(tmp_path)
    return survey_script.load_split_parts(tmp_path)


def _self_pu_bound(survey_script, tmp_path):
    """A versioned spambase/self_pu unit with its protocol-exact model.

    ``runner_protocol_context`` requires ``type(model) is SelfPUClassifier``,
    so the model has to come from ``assemble_model`` rather than a stub.  That
    is also what makes this the honest two-teacher case: the declaration and
    the writer agree unless something drops a checkpoint in between.
    """
    parts = _parts(survey_script, tmp_path)
    protocol = load_protocol()
    row = resolve_unit(protocol, "spambase", "self_pu")
    model = assemble_model(protocol, row, 3, seed=0, params={}, class_prior=0.3, device="cpu")
    config = {
        "architecture": "mlp",
        "c": 0.5,
        "c_requested_token": "0.5",
        "split_ref": {"dataset": "spambase", "seed": 0},
        "survey_protocol": {
            "path": str(PROTOCOL_PATH),
            "dataset": "spambase",
            "method": "self_pu",
            "mechanism": "scar",
            "seeds": [0],
            "c_tokens": ["0.5"],
        },
    }
    return model, parts, config


# --- the declaration is the expectation ------------------------------------


def test_edge_declared_two_teachers_reject_single_teacher_trajectory(tmp_path):
    """One teacher saved is not a smaller run -- it is a broken one."""
    trajectory = _selfpu(tmp_path)
    assert {checkpoint.component for checkpoint in trajectory.checkpoints} == set(_TWO_TEACHERS)
    single = replace(
        trajectory,
        checkpoints=[c for c in trajectory.checkpoints if c.component == "teacher_1"],
    )
    with pytest.raises(ValueError, match="coverage") as excinfo:
        _validate_trajectory(single, _pu_val(), _TWO_TEACHERS)
    assert "teacher_2" in str(excinfo.value)


def test_basic_complete_trajectories_pass_for_single_and_two_component_estimators(tmp_path):
    """The default declaration stays compatible with every single-network writer."""
    two = _selfpu(tmp_path / "selfpu")
    _validate_trajectory(two, _pu_val(), _declared_epoch_components(two.model))
    assert _declared_epoch_components(two.model) == _TWO_TEACHERS

    one = _nnpu(tmp_path / "nnpu")
    _validate_trajectory(one, _pu_val(), _declared_epoch_components(one.model))
    assert _declared_epoch_components(one.model) == ("model",)


@pytest.mark.parametrize("bad", [(), ("a", "a"), ("a", 1), "model", None])
def test_param_malformed_component_declaration_rejected_before_training(
    tmp_path, survey_script, bad
):  # noqa: F811
    trainer = _RecordingTrainer()
    runner = ExperimentRunner(
        config={"c": 0.5, "trainer": trainer},
        manifest_path=str(tmp_path / "manifest.json"),
    )
    model = type("BadDeclaration", (_OneNetwork,), {"epoch_components": bad})()
    with pytest.raises(ValueError, match="epoch_components"):
        runner.fit(model, *_parts(survey_script, tmp_path))
    assert trainer.calls == 0


def test_param_writer_and_declaration_disagreement_is_named(tmp_path):
    """Drift between the hardcoded writer names and the declaration is named."""
    model = _DeclaresOneWritesTwo()
    with pytest.raises(ValueError, match="epoch_components"):
        EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, *_data())


def test_determ_coverage_verdict_ignores_checkpoint_order(tmp_path):
    trajectory = _selfpu(tmp_path)
    _validate_trajectory(trajectory, _pu_val(), _TWO_TEACHERS)
    shuffled = replace(trajectory, checkpoints=list(reversed(trajectory.checkpoints)))
    _validate_trajectory(shuffled, _pu_val(), _TWO_TEACHERS)


# --- the runner-level consequence ------------------------------------------


def test_edge_dropped_component_cannot_discharge_r9_blocker(tmp_path, survey_script):  # noqa: F811
    """A run that omitted a network must not unlock independent PA/OA selection."""
    model, parts, config = _self_pu_bound(survey_script, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    runner = ExperimentRunner(
        config={**config, "trainer": _DroppingTrainer(tmp_path / "ckpt")},
        manifest_path=str(manifest_path),
    )
    with pytest.raises(RuntimeError, match="all candidate runs failed"):
        runner.fit(model, *parts)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["execution_mode"] == "versioned_pilot"
    messages = [error["message"] for failure in manifest["failures"] for error in failure["errors"]]
    assert any("coverage" in message for message in messages), messages
    assert [run["status"] for run in manifest["candidate_runs"]] == ["excluded"] * len(
        manifest["candidate_runs"]
    )
    assert "per_epoch_independent_PA_OA_checkpoint_selection" in manifest["formal_blockers"]
    assert manifest["selection_checkpoint_scope"] == "pending_actual_trajectory_verification"


def test_basic_complete_versioned_run_does_discharge_the_blocker(tmp_path, survey_script):  # noqa: F811
    """Control: the same unit, undamaged, does clear the checkpoint blocker.

    Without this, the previous test would pass just as well if the guard
    rejected every versioned run.
    """
    model, parts, config = _self_pu_bound(survey_script, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    runner = ExperimentRunner(config=config, manifest_path=str(manifest_path))
    runner.fit(model, *parts)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["failures"] == []
    assert "per_epoch_independent_PA_OA_checkpoint_selection" not in manifest["formal_blockers"]
    assert manifest["selection_checkpoint_scope"] == "independent_per_epoch"
