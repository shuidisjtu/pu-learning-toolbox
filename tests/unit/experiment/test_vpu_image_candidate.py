"""VPU native TS marginal, epoch normalization and actual optimizer cost."""

import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier  # noqa: E402
from pu_toolbox.experiment.candidate_image_runner import (  # noqa: E402
    CandidateImageCheckpointTrainer,
)
from pu_toolbox.experiment.image import fit_survey_image_preprocessing  # noqa: E402

pytestmark = pytest.mark.unit


def setup(**parameters):
    raw = np.random.RandomState(13).randint(0, 256, (8, 1, 8, 8)).astype(np.uint8)
    pixels, preprocessing = fit_survey_image_preprocessing(
        raw, dataset="mnist", train_augmentation="none"
    )
    model, record = build_candidate_image_classifier("vpu", preprocessing, seed=3, device="cpu")
    model.set_params(**(dict(max_epochs=3, batch_size=4) | parameters))
    # U-only would give one update; the native P-union-U pool requires two.
    return model, record, pixels, np.r_[np.ones(6, int), np.zeros(2, int)]


def trainer(record, root, bound=64 * 2**20):
    return CandidateImageCheckpointTrainer(
        record, checkpoint_bytes_per_component=bound, checkpoint_dir=root
    )


def test_basic_actual_ts_pool_and_per_epoch_normalization_survive_snapshot(tmp_path):
    model, record, pixels, labels = setup()
    trajectory = trainer(record, tmp_path).fit(
        model, pixels, labels, class_prior=0.4, os_or_ts="ts"
    )
    facts = model.training_provenance_
    receipt = facts["bound_fit_receipt"]
    assert receipt["training_view"] == "ts" and receipt["calibration_applied"]
    assert (
        receipt["training_marginal_rows"] == 8
        and receipt["normalization_training_rows_per_epoch"] == 8
    )
    assert receipt["normalization_forward_batches_per_epoch"] == 2
    assert facts["selection_prior_not_forwarded_to_VPU"] and model._class_prior is None
    assert not facts["source_validation_diagnostic_forwarded"]
    assert (
        not facts["source_validation_risk_used_for_selection"] and model.history_["val_risk"] == []
    )
    assert [c.training_context["optimizer_steps"] for c in trajectory.checkpoints] == [2, 4, 6]
    assert [c.epoch_label for c in trajectory.checkpoints] == [0, 1, 2]
    for checkpoint, normalization in zip(
        trajectory.checkpoints, facts["normalization_history"], strict=True
    ):
        restored = checkpoint.restore()
        assert checkpoint.training_context["stage"] == "variational"
        assert float(restored.model_[-1].bias.detach()) == normalization["normalization_bias"]
        assert math.isclose(
            float(restored.decision_function(pixels).max()), math.log(2), abs_tol=1e-6
        )
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(pixels),
        model.decision_function(pixels),
    )


@pytest.mark.parametrize(
    "schedule,epochs,iterations", [("constant", 3, None), ("source_reset_halving", 21, 1)]
)
def test_determ_plain_fit_and_source_adam_reset_are_unchanged(
    tmp_path, schedule, epochs, iterations
):
    model, record, pixels, labels = setup(
        optimizer_schedule=schedule, max_epochs=epochs, iterations_per_epoch=iterations
    )
    plain = copy.deepcopy(model).fit(pixels, labels, os_or_ts="ts")
    trainer(record, tmp_path).fit(model, pixels, labels, os_or_ts="ts")
    assert model.history_ == plain.history_ and model.optimizer_steps_ == plain.optimizer_steps_
    for key, value in model.model_.state_dict().items():
        torch.testing.assert_close(value, plain.model_.state_dict()[key], atol=0, rtol=0)
    np.testing.assert_array_equal(model.decision_function(pixels), plain.decision_function(pixels))
    if schedule == "source_reset_halving":
        assert model.optimizer_resets_ == 1
        assert model.history_["learning_rate"][19] == model.learning_rate / 2
        assert model.training_provenance_["optimizer_generation_history"] == [0] * 19 + [1] * 2
        assert model.optimizer_steps_ == 21


@pytest.mark.parametrize(
    "fault",
    [
        "os",
        "pool_record",
        "normalization_scope",
        "diagnostic_selection",
        "resident",
        "bad_schedule",
    ],
)
def test_param_invalid_native_contract_never_reaches_fit(tmp_path, monkeypatch, fault):
    model, record, pixels, labels = setup()
    if fault == "pool_record":
        record["marginal_pool"] = "U_only"
    elif fault == "normalization_scope":
        record["normalization_scope"] = "test"
    elif fault == "diagnostic_selection":
        record["validation_diagnostic_selection_eligible"] = True
    elif fault == "resident":
        model.data_device_policy = "resident"
    elif fault == "bad_schedule":
        model.optimizer_schedule = "step_lr"
    monkeypatch.setattr(
        type(model), "fit", lambda *a, **k: pytest.fail("invalid VPU contract reached training")
    )
    with pytest.raises(ValueError):
        trainer(record, tmp_path).fit(
            model, pixels, labels, os_or_ts="os" if fault == "os" else "ts"
        )


@pytest.mark.parametrize("fault", ["bias", "pool", "updates", "maximum"])
def test_edge_corrupt_epoch_state_rejected_before_writing_snapshot(tmp_path, monkeypatch, fault):
    model, record, pixels, labels = setup()
    original = model.fit

    def bad_fit(features, observed, *, epoch_callback, **kwargs):
        def callback(epoch, fitted):
            if fault == "bias":
                with torch.no_grad():
                    fitted.model_[-1].bias.add_(1)
            elif fault == "pool":
                fitted.n_loss_unlabeled_ = fitted.n_unlabeled_
            elif fault == "updates":
                fitted.optimizer_steps_ -= 1
            else:
                fitted.max_log_phi_ = float("nan")
            epoch_callback(epoch, fitted)

        return original(features, observed, epoch_callback=callback, **kwargs)

    monkeypatch.setattr(model, "fit", bad_fit)
    with pytest.raises(ValueError, match="VPU"):
        trainer(record, tmp_path).fit(model, pixels, labels, os_or_ts="ts")
    assert not model._is_fitted and not list(tmp_path.rglob("*.pt"))


def test_edge_oversized_vpu_snapshot_retains_evidence_and_rejects_fit(tmp_path):
    model, record, pixels, labels = setup()
    with pytest.raises(ValueError, match="byte bound"):
        trainer(record, tmp_path, bound=1).fit(model, pixels, labels, os_or_ts="ts")
    assert not model._is_fitted and not hasattr(model, "training_provenance_")
    assert len(list(tmp_path.rglob("*.pt"))) == 3
