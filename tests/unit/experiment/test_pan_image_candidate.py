"""Native-OS PAN candidate trajectories keep independent adversaries and paid cost."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier  # noqa: E402
from pu_toolbox.experiment.candidate_image_runner import (  # noqa: E402
    CandidateImageCheckpointTrainer,
)
from pu_toolbox.experiment.feature_adapter import encoder_state_sha256  # noqa: E402
from pu_toolbox.experiment.image import fit_survey_image_preprocessing  # noqa: E402

pytestmark = pytest.mark.unit


def setup():
    raw = np.random.RandomState(9).randint(0, 256, (8, 1, 8, 8)).astype(np.uint8)
    pixels, preprocessing = fit_survey_image_preprocessing(
        raw, dataset="mnist", train_augmentation="none"
    )
    model, record = build_candidate_image_classifier("pan", preprocessing, seed=3, device="cpu")
    model.set_params(max_epochs=3, batch_size=4)
    return model, record, pixels, np.r_[np.ones(2, int), np.zeros(6, int)]


def trainer(record, tmp_path, bound=64 * 2**20):
    return CandidateImageCheckpointTrainer(
        record, checkpoint_bytes_per_component=bound, checkpoint_dir=tmp_path
    )


def test_basic_actual_c_d_updates_and_classifier_only_epoch_snapshots(tmp_path):
    model, record, pixels, labels = setup()
    trajectory = trainer(record, tmp_path).fit(
        model, pixels, labels, class_prior=0.4, os_or_ts="os"
    )
    facts = model.training_provenance_
    receipt = facts["bound_fit_receipt"]
    assert receipt["training_view"] == "os" and not receipt["calibration_applied"]
    assert (
        receipt["independent_C_D_storage_verified"] and receipt["discriminator_encoder_trainable"]
    )
    assert receipt["classifier_only_inference"] and model._class_prior is None
    assert facts["selection_prior_not_forwarded_to_PAN"]
    assert model.optimizer_steps_ == 12
    assert [c.epoch_label for c in trajectory.checkpoints] == [0, 1, 2]
    assert [c.training_context["optimizer_steps"] for c in trajectory.checkpoints] == [4, 8, 12]
    assert all(
        c.component == "model" and c.training_context["stage"] == "adversarial"
        for c in trajectory.checkpoints
    )
    assert encoder_state_sha256(model.encoder) == record["encoder_initial_state_sha256"]
    assert model.encoder_ is not model.discriminator_encoder_
    assert encoder_state_sha256(model.encoder_) != record["encoder_initial_state_sha256"]
    assert (
        encoder_state_sha256(model.discriminator_encoder_) != record["encoder_initial_state_sha256"]
    )
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(pixels),
        model.decision_function(pixels),
    )


def test_determ_checkpoint_adapter_preserves_plain_native_fit_exactly(tmp_path):
    model, record, pixels, labels = setup()
    plain = copy.deepcopy(model).fit(pixels, labels, os_or_ts="os")
    trainer(record, tmp_path).fit(model, pixels, labels, os_or_ts="os")
    assert model.history_ == plain.history_ and model.optimizer_steps_ == plain.optimizer_steps_
    for name in ("model_", "discriminator_"):
        for key, value in getattr(model, name).state_dict().items():
            torch.testing.assert_close(
                value, getattr(plain, name).state_dict()[key], atol=0, rtol=0
            )
    np.testing.assert_array_equal(model.decision_function(pixels), plain.decision_function(pixels))


@pytest.mark.parametrize(
    "fault",
    [
        "ts",
        "shared_record",
        "custom_D",
        "pixels",
        "prior",
        "augmentation",
        "learning_rate",
        "weight",
    ],
)
def test_param_invalid_bound_inputs_refused_before_fit(tmp_path, monkeypatch, fault):
    model, record, pixels, labels = setup()
    if fault == "shared_record":
        record["network_instances"] = 1
    elif fault == "custom_D":
        model.discriminator = torch.nn.Linear(512, 1)
    elif fault == "pixels":
        pixels = pixels.copy() + 0.01
    elif fault in {"learning_rate", "weight"}:
        setattr(model, "learning_rate" if fault == "learning_rate" else "adversarial_weight", True)
    elif fault in {"prior", "augmentation"}:
        _, prep = fit_survey_image_preprocessing(
            np.random.RandomState(2).rand(8, 1, 8, 8).astype(np.float32),
            dataset="mnist",
            train_augmentation="none" if fault == "prior" else "simaugment",
        )
        with pytest.raises(ValueError):
            build_candidate_image_classifier(
                "pan", prep, seed=3, class_prior=0.4 if fault == "prior" else None
            )
        return
    monkeypatch.setattr(
        type(model), "fit", lambda *a, **k: pytest.fail("invalid contract reached training")
    )
    with pytest.raises(ValueError):
        trainer(record, tmp_path).fit(
            model, pixels, labels, os_or_ts="ts" if fault == "ts" else "os"
        )


def test_edge_wrong_actual_epoch_update_count_fails_before_fanout(tmp_path, monkeypatch):
    model, record, pixels, labels = setup()
    original = model.fit

    def bad_fit(features, y, *, epoch_callback, **kwargs):
        def callback(epoch, fitted):
            fitted.optimizer_steps_ -= 1
            epoch_callback(epoch, fitted)

        return original(features, y, epoch_callback=callback, **kwargs)

    monkeypatch.setattr(model, "fit", bad_fit)
    with pytest.raises(ValueError, match="C/D update cost"):
        trainer(record, tmp_path).fit(model, pixels, labels, os_or_ts="os")
    assert not model._is_fitted and not list(tmp_path.rglob("*.pt"))


def test_edge_oversize_snapshot_rejects_fit_without_discarding_evidence(tmp_path):
    model, record, pixels, labels = setup()
    with pytest.raises(ValueError, match="byte bound"):
        trainer(record, tmp_path, bound=1).fit(model, pixels, labels, os_or_ts="os")
    assert not model._is_fitted and not hasattr(model, "training_provenance_")
    assert len(list(tmp_path.rglob("*.pt"))) == 3
