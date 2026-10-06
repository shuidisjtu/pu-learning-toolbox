"""Stage provenance supports review without approving any stage for formal selection."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import (  # noqa: E402
    GenPUClassifier,
    GradPUClassifier,
    HolisticPUClassifier,
    PULDAClassifier,
    RobustPUClassifier,
    SplitPUClassifier,
)
from pu_toolbox.experiment.checkpoints import (  # noqa: E402
    EpochCheckpointTrainer,
    load_epoch_checkpoint,
)

pytestmark = pytest.mark.unit


def _data():
    return np.random.RandomState(8).normal(size=(18, 3)).astype("float32"), np.r_[
        np.ones(6, int), np.zeros(12, int)
    ]


def _model(method, **overrides):
    common = {"hidden_dim": 4, "random_state": 2, "device": "cpu", **overrides}
    if method == "pulda":
        return PULDAClassifier(
            0.4,
            warmup_epochs=1,
            pu_epochs=2,
            positive_batch_size=4,
            unlabeled_batch_size=4,
            **common,
        )
    common["batch_size"] = 6
    if method == "gradpu":
        return GradPUClassifier(max_epochs=2, **common)
    if method == "robust_pu":
        return RobustPUClassifier(0.4, pretrain_epochs=1, episodes=2, inner_epochs=2, **common)
    if method == "split_pu":
        return SplitPUClassifier(
            0.4, teacher_epochs=1, split_epochs=1, student_epochs=2, rounds=2, **common
        )
    if method == "holistic_pu":
        return HolisticPUClassifier(warmup_epochs=3, max_epochs=2, **common)
    return GenPUClassifier(
        class_prior=0.4, latent_dim=2, max_epochs=1, classifier_epochs=2, **common
    )


@pytest.mark.parametrize(
    "method,expected",
    [
        ("pulda", [("warmup", 1, None), ("pu_mixup", 1, None), ("pu_mixup", 2, None)]),
        ("gradpu", [("pu", 1, None), ("pu", 2, None)]),
        ("robust_pu", [("pretrain", 1, None), ("self_paced", 1, None), ("self_paced", 2, None)]),
        (
            "split_pu",
            [
                ("teacher", 1, None),
                ("splitter", 1, None),
                ("student", 1, 1),
                ("student", 2, 1),
                ("student", 1, 2),
                ("student", 2, 2),
            ],
        ),
        (
            "holistic_pu",
            [
                ("warmup", 1, None),
                ("warmup", 2, None),
                ("warmup", 3, None),
                ("pseudo_pn", 1, None),
                ("pseudo_pn", 2, None),
            ],
        ),
        ("genpu", [("synthetic_pn", 1, None), ("synthetic_pn", 2, None)]),
    ],
)
def test_param_determ_staged_methods_persist_and_restore_exact_provenance(
    tmp_path, method, expected
):
    model, (features, labels) = _model(method), _data()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, features, labels)
    contexts = [checkpoint.reference()["training_context"] for checkpoint in trajectory.checkpoints]
    assert [(c["stage"], c["stage_epoch"], c["round_index"]) for c in contexts] == expected
    steps = [c["optimizer_steps"] for c in contexts]
    assert all(a < b for a, b in zip(steps[:-1], steps[1:], strict=True))
    assert steps[-1] == model.optimizer_steps_
    for checkpoint in trajectory.checkpoints:
        reference = checkpoint.reference()
        assert reference["schema_version"] == "1.1"
        assert reference["training_resume_supported"] is False
        restored = load_epoch_checkpoint(reference, checkpoint.template, device="cpu")
        assert restored.training_context == reference["training_context"]
        np.testing.assert_array_equal(
            restored.decision_function(features), checkpoint.restore().decision_function(features)
        )
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(features),
        model.decision_function(features),
    )
    # A subsequent fit resets counters; old references cannot follow mutable estimator state.
    before = copy.deepcopy(contexts)
    predictions = model.decision_function(features).copy()
    model.fit(features, labels)
    assert contexts == before
    assert model.optimizer_steps_ == steps[-1]
    np.testing.assert_array_equal(model.decision_function(features), predictions)


def test_basic_updates_are_real_steps_not_stage_epochs(tmp_path, monkeypatch):
    counts = []
    original = torch.optim.Adam.step

    def observed(optimizer, *args, **kwargs):
        counts.append(1)
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", observed)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(_model("robust_pu"), *_data())
    assert len(counts) == 14  # 2 pretrain steps + two episodes * 2 inner epochs * 3 batches.
    assert [c.training_context["optimizer_steps"] for c in trajectory.checkpoints] == [2, 8, 14]


def test_edge_split_early_stop_keeps_actual_phase_lengths(tmp_path, monkeypatch):
    from pu_toolbox.estimators.deep import split_pu

    model = _model("split_pu")
    model.set_params(split_epochs=4)
    original = split_pu._batched_score

    def perfect_splitter(network, data, batch_size, device):
        if network is getattr(model, "splitter_", None):
            return original(model.teacher_, data, batch_size, device)
        return original(network, data, batch_size, device)

    monkeypatch.setattr(split_pu, "_batched_score", perfect_splitter)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, *_data())
    assert len(trajectory.checkpoints) == 6 < model.checkpoint_epoch_count == 9
    assert [c.training_context["stage_epoch"] for c in trajectory.checkpoints] == [1, 1, 1, 2, 1, 2]
    assert [c.epoch_position for c in trajectory.checkpoints] == list(range(1, 7))


class _DeclaredStage:
    checkpoint_stages = ("warmup",)

    def __init__(self, bad=None):
        self.bad = bad

    def fit(self, features, labels, *, epoch_callback=None):
        self.model_ = torch.nn.Linear(3, 1)
        for epoch in range(2):
            self.checkpoint_stage_ = "warmup"
            self.checkpoint_stage_epoch_ = epoch + 1
            self.checkpoint_round_ = None
            self.optimizer_steps_ = 2 * (epoch + 1)
            if self.bad:
                name, value = self.bad
                setattr(self, name, value)
            epoch_callback(epoch, self)
        return self


@pytest.mark.parametrize(
    "bad",
    [
        ("checkpoint_stage_", "undeclared"),
        ("checkpoint_stage_", None),
        ("checkpoint_stage_epoch_", 0),
        ("checkpoint_stage_epoch_", True),
        ("checkpoint_round_", -1),
        ("checkpoint_round_", False),
        ("optimizer_steps_", float("nan")),
        ("optimizer_steps_", -1),
        ("checkpoint_stages", []),
        ("checkpoint_stages", ("warmup", "warmup")),
    ],
)
def test_param_writer_refuses_invalid_stage_metadata(tmp_path, bad):
    with pytest.raises(ValueError, match="checkpoint"):
        EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(_DeclaredStage(bad), *_data())
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage", ""),
        ("stage_epoch", False),
        ("stage_epoch", 0),
        ("round_index", -1),
        ("optimizer_steps", -1),
        ("optimizer_steps", 1.5),
    ],
)
def test_param_restore_refuses_malformed_context(tmp_path, field, value):
    checkpoint = (
        EpochCheckpointTrainer(checkpoint_dir=tmp_path)
        .fit(_DeclaredStage(), *_data())
        .checkpoints[0]
    )
    reference = checkpoint.reference()
    reference["training_context"][field] = value
    with pytest.raises(ValueError, match="checkpoint"):
        load_epoch_checkpoint(reference, checkpoint.template)


def test_edge_context_is_detached_from_reference_and_legacy_has_no_stage(tmp_path):
    checkpoint = (
        EpochCheckpointTrainer(checkpoint_dir=tmp_path)
        .fit(_DeclaredStage(), *_data())
        .checkpoints[0]
    )
    reference = checkpoint.reference()
    reference["training_context"]["stage"] = "changed"
    assert checkpoint.training_context["stage"] == "warmup"
    legacy = replace(checkpoint, training_context=None)
    reference = legacy.reference()
    assert reference["schema_version"] == "1.0" and "training_context" not in reference
    assert load_epoch_checkpoint(reference, legacy.template).training_context is None
    for reference in (checkpoint.reference(), legacy.reference()):
        reference["schema_version"] = "1.0" if "training_context" in reference else "1.1"
        with pytest.raises(ValueError, match="schema 1.1"):
            load_epoch_checkpoint(reference, legacy.template)


def test_edge_cumulative_steps_cannot_go_backwards(tmp_path, monkeypatch):
    model = _DeclaredStage()
    original = model.fit

    def decreasing_fit(features, labels, *, epoch_callback=None):
        def record(epoch, fitted):
            fitted.optimizer_steps_ = 2 - epoch
            epoch_callback(epoch, fitted)

        return original(features, labels, epoch_callback=record)

    # Trainer inspects the class signature, while the bound fit carries this controlled drift.
    monkeypatch.setattr(model, "fit", decreasing_fit)
    with pytest.raises(ValueError, match="must not decrease"):
        EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, *_data())
    assert len(list(tmp_path.rglob("*.pt"))) == 1


@pytest.mark.parametrize("method", ["pulda", "robust_pu"])
def test_edge_zero_first_stage_starts_at_local_epoch_one(tmp_path, method):
    model = _model(method)
    model.set_params(**({"warmup_epochs": 0} if method == "pulda" else {"pretrain_epochs": 0}))
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, *_data())
    assert [c.training_context["stage_epoch"] for c in trajectory.checkpoints] == [1, 2]
    assert all(
        c.training_context["stage"] == model.checkpoint_stages[-1] for c in trajectory.checkpoints
    )


@pytest.mark.parametrize("method", ["pulda", "gradpu", "robust_pu", "split_pu"])
@pytest.mark.parametrize("reclaim", [False, True])
def test_param_runner_selection_and_reclaim_keep_stage_provenance(tmp_path, method, reclaim):
    from pu_toolbox.experiment.bundle import DatasetPart
    from pu_toolbox.experiment.runner import ExperimentRunner

    model = _model(method)
    rng = np.random.RandomState(4)
    parts = [
        DatasetPart(
            X=rng.normal(size=(24, 3)).astype("float32"),
            labels=np.tile([0, 1], 12),
            indices=np.arange(role * 24, (role + 1) * 24),
            view="clean",
            for_selection=role != 3,
        )
        for role in range(4)
    ]
    path = tmp_path / "manifest.json"
    result = ExperimentRunner(
        seed=2,
        class_prior=0.4,
        manifest_path=str(path),
        config={
            "method": method,
            "c": 0.5,
            "os_or_ts": "ts",
            "checkpoint_bytes_per_component": 4096,
            "reclaim_unselected_checkpoints": reclaim,
        },
    ).fit(model, *parts)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    references = manifest["candidate_runs"][0]["epoch_checkpoints"]
    assert all(reference["schema_version"] == "1.1" for reference in references)
    assert manifest.get("execution_mode") != "versioned_pilot"
    selected = {s["checkpoint"]["path"] for s in manifest["selection"].values()}
    for reference in references:
        assert reference["training_context"]["stage"] in model.checkpoint_stages
        should_reclaim = reclaim and reference["path"] not in selected
        assert reference["reclaimed"] is should_reclaim
        assert Path(reference["path"]).is_file() is not should_reclaim
    for protocol in ("PA", "OA"):
        selection = manifest["selection"][protocol]["checkpoint"]
        assert selection["training_context"] == result.selected_models[protocol].training_context
        assert selection == references[result.selections[protocol].checkpoint_index]
