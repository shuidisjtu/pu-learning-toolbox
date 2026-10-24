# ruff: noqa: N803, N806
"""Real image trajectories: input isolation, unchanged training and disk bounds."""

import copy
import functools
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart  # noqa: E402
from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier  # noqa: E402
from pu_toolbox.experiment.candidate_image_runner import (  # noqa: E402
    CandidateImageCheckpointTrainer,
)
from pu_toolbox.experiment.image import fit_survey_image_preprocessing  # noqa: E402
from pu_toolbox.experiment.runner import ExperimentRunner  # noqa: E402
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA  # noqa: E402

pytestmark = pytest.mark.unit
BOUND = 64 * 2**20


def setup(method="pulda", seed=3):
    raw = np.random.RandomState(2).randint(0, 256, (12, 3, 8, 8)).astype(np.uint8)
    X, image = fit_survey_image_preprocessing(raw, dataset="cifar10", train_augmentation="none")
    model, record = build_candidate_image_classifier(
        method, image, seed=seed, class_prior=0.4 if method == "pulda" else None, device="cpu"
    )
    if method == "pulda":
        model.set_params(
            warmup_epochs=1, pu_epochs=2, positive_batch_size=2, unlabeled_batch_size=4
        )
    else:
        model.set_params(max_epochs=3, batch_size=4)
    parts = tuple(
        DatasetPart(X.copy(), np.arange(12) % 2, "clean", np.arange(12) + 20 * i, i != 3)
        for i in range(4)
    )
    return model, record, DatasetBundle(*parts)


def config(record, **kwargs):
    return (
        dict(
            c=0.5,
            os_or_ts="ts",
            trainer=CandidateImageCheckpointTrainer(record, checkpoint_bytes_per_component=BOUND),
            checkpoint_bytes_per_component=BOUND,
            reclaim_unselected_checkpoints=True,
        )
        | kwargs
    )


@pytest.mark.parametrize("method", ["pulda", "gradpu"])
def test_basic_real_four_role_fit_selection_cost_and_reclamation(method, tmp_path, monkeypatch):
    model, record, bundle = setup(method)
    original, calls = type(model).fit, []

    @functools.wraps(original)
    def audit(self, X, y, **kwargs):
        calls.append((X.copy(), y.copy(), dict(kwargs)))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(type(model), "fit", audit)
    result = ExperimentRunner(
        seed=3,
        class_prior=0.4,
        protocols=[ProtocolPA(), ProtocolOA()],
        config=config(record),
        manifest_path=str(tmp_path / "manifest.json"),
    ).fit(model, bundle.train, bundle.pu_val, bundle.clean_val, bundle.test)
    assert len(calls) == 1 and not result.failures
    assert set(calls[0][2]) == {"epoch_callback", "os_or_ts"} | (
        {"class_prior"} if method == "pulda" else set()
    )
    np.testing.assert_array_equal(calls[0][0], bundle.train.X)
    assert calls[0][1].sum() == 3 < bundle.train.labels.sum()
    row = result.manifest["candidate_runs"][0]
    facts = row["training_provenance"]
    receipt = facts["bound_fit_receipt"]
    assert receipt["image_setup"] == record and receipt["calibration_applied"]
    assert not facts["external_validation_forwarded_to_fit"] and not facts["formal_admission"]
    assert len(row["epoch_checkpoints"]) == 3 and receipt["optimizer_steps"] == 9
    contexts = [r["training_context"] for r in row["epoch_checkpoints"]]
    assert [r["optimizer_steps"] for r in contexts] == [3, 6, 9]
    assert [r["stage"] for r in contexts] == (
        ["warmup", "pu_mixup", "pu_mixup"] if method == "pulda" else ["pu"] * 3
    )
    if method == "gradpu":
        assert facts["selection_prior_not_forwarded_to_GradPU"]
        assert receipt["algorithm_training"]["batch"]["optimizer_steps"] == 9
    assert set(result.selected_models) == {"PA", "OA"}
    assert sum(not r["reclaimed"] for r in row["epoch_checkpoints"]) <= 2
    for checkpoint in row["epoch_checkpoints"]:
        assert Path(checkpoint["path"]).is_file() is (not checkpoint["reclaimed"])
    assert not result.manifest["formal_eligible"]


@pytest.mark.parametrize("method,seed", [("pulda", 3), ("gradpu", 3), ("gradpu", 7)])
def test_determ_checkpoint_bridge_preserves_plain_fit_weights_history_scores(
    method, seed, tmp_path
):
    model, record, bundle = setup(method, seed)
    plain = copy.deepcopy(model)
    labels = np.r_[np.ones(3, int), np.zeros(9, int)]
    kwargs = {"class_prior": 0.4} if method == "pulda" else {}
    plain.fit(bundle.train.X, labels, os_or_ts="ts", **kwargs)
    trajectory = CandidateImageCheckpointTrainer(
        record, checkpoint_bytes_per_component=BOUND, checkpoint_dir=tmp_path
    ).fit(model, bundle.train.X, labels, class_prior=0.4, os_or_ts="ts")
    assert model.history_ == plain.history_ and model.optimizer_steps_ == plain.optimizer_steps_
    np.testing.assert_array_equal(
        model.decision_function(bundle.test.X), plain.decision_function(bundle.test.X)
    )
    for name, value in model.model_.state_dict().items():
        torch.testing.assert_close(value, plain.model_.state_dict()[name], rtol=0, atol=0)
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(bundle.test.X),
        plain.decision_function(bundle.test.X),
    )


@pytest.mark.parametrize("method", ["pulda", "gradpu"])
def test_basic_preflight_uses_largest_candidate_not_base_budget(method):
    model, record, bundle = setup(method)
    candidates = [
        {},
        {"warmup_epochs": 3, "pu_epochs": 2} if method == "pulda" else {"max_epochs": 5},
    ]
    settings = config(record, candidates=candidates)
    trainer = settings["trainer"]
    trainer.validate_experiment(model, bundle, settings, [ProtocolPA(), ProtocolOA()])
    assert trainer.checkpoint_epoch_bound() == 5


@pytest.mark.parametrize(
    "options",
    [
        {"os_or_ts": "os"},
        {"survey_protocol": {}},
        {"recipe_registry": {}},
        {"capture_epoch_checkpoints": False},
        {"checkpoint_bytes_per_component": True},
        {"candidates": []},
        {"candidates": [{"device": "cuda"}]},
        {"candidates": [{"class_prior": 0.3}]},
    ],
)
def test_param_invalid_contract_refused_before_fit(options, monkeypatch):
    model, record, bundle = setup()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid preflight reached fit")

    monkeypatch.setattr(type(model), "fit", forbidden)
    settings = config(record, **options)
    with pytest.raises(ValueError):
        settings["trainer"].validate_experiment(
            model, bundle, settings, [ProtocolPA(), ProtocolOA()]
        )


def test_edge_no_single_oa_or_sar_approval_or_missing_cuda(monkeypatch):
    model, record, bundle = setup()
    settings = config(record)
    trainer = settings["trainer"]
    with pytest.raises(ValueError, match="independent PA then OA"):
        trainer.validate_experiment(model, bundle, settings, [ProtocolOA()])
    with pytest.raises(ValueError, match="SCAR only"):
        trainer.validate_generation({"mechanism": "scar"}, {"mechanism": "sar_lbe_a"})
    model.device = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no CPU fallback"):
        trainer.validate_experiment(model, bundle, settings, [ProtocolPA(), ProtocolOA()])


def test_edge_oversize_payload_invalidates_fit_and_retains_all_evidence(tmp_path):
    model, record, bundle = setup()
    trainer = CandidateImageCheckpointTrainer(
        record, checkpoint_bytes_per_component=1, checkpoint_dir=tmp_path
    )
    with pytest.raises(ValueError, match="exceeded declared byte bound"):
        trainer.fit(model, bundle.train.X, bundle.train.labels, class_prior=0.4, os_or_ts="ts")
    assert not model._is_fitted and not hasattr(model, "training_provenance_")
    assert len(list(tmp_path.rglob("*.pt"))) == 3
