"""Bound P3Mix image epochs preserve native OS, pool identities and teacher cost."""

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


def setup(**parameters):
    raw = np.random.RandomState(17).randint(0, 256, (9, 1, 8, 8)).astype(np.uint8)
    pixels, preprocessing = fit_survey_image_preprocessing(
        raw, dataset="mnist", train_augmentation="none"
    )
    model, record = build_candidate_image_classifier("p3mix", preprocessing, seed=3, device="cpu")
    model.set_params(**(dict(max_epochs=2, batch_size=4, hidden_dim=4, pool_size=2) | parameters))
    return model, record, pixels, np.r_[np.ones(2, int), np.zeros(7, int)]


def trainer(record, directory, bound=64 * 2**20):
    return CandidateImageCheckpointTrainer(
        record, checkpoint_bytes_per_component=bound, checkpoint_dir=directory
    )


@pytest.mark.parametrize("variant", ["base", "e", "c"])
@pytest.mark.parametrize("policy", ["role_balanced", "merged_shuffle"])
def test_basic_determ_native_fit_parity_student_snapshots_and_actual_cost(
    tmp_path, variant, policy
):
    model, record, pixels, labels = setup(variant=variant, batching_policy=policy)
    original_pixels, original_labels = pixels.copy(), labels.copy()
    plain = copy.deepcopy(model).fit(pixels, labels, os_or_ts="os")
    trajectory = trainer(record, tmp_path).fit(
        model, pixels, labels, class_prior=0.4, os_or_ts="os"
    )
    facts = model.training_provenance_
    receipt = facts["bound_fit_receipt"]
    assert receipt["training_view"] == "os" and not receipt["calibration_applied"]
    assert receipt["student_only_inference"] and not receipt["backbone_is_paper_architecture"]
    assert facts["selection_prior_not_forwarded_to_P3Mix"] and model._class_prior is None
    assert not facts["external_validation_forwarded_to_fit"]
    assert not facts["teacher_selection_eligible"]
    assert [c.epoch_label for c in trajectory.checkpoints] == [1, 2]
    assert all(c.training_context["stage"] == "train" for c in trajectory.checkpoints)
    rows = 2 if policy == "merged_shuffle" else 4
    assert [c.training_context["optimizer_steps"] for c in trajectory.checkpoints] == (
        [2, 4] if policy == "role_balanced" else [3, 6]
    )
    for epoch, cost in enumerate(facts["epoch_cost_history"], 1):
        steps = 2 if policy == "role_balanced" else 3
        assert cost["positive_rows_seen"] == rows and cost["unlabeled_rows_seen"] == 7
        assert cost["student_training_forward_batches"] == 2 * steps
        assert cost["teacher_forward_batches"] == (steps if variant == "e" else 0)
        assert cost["teacher_ema_updates_completed"] == (epoch if variant == "e" else 0)
        assert cost["positive_pool_refresh_forward_batches"] == 1
    assert model.history_ == plain.history_ and model.epoch_provenance_ == plain.epoch_provenance_
    assert encoder_state_sha256(model.encoder) == record["encoder_initial_state_sha256"]
    assert encoder_state_sha256(model.encoder_) != record["encoder_initial_state_sha256"]
    for key, value in model.model_.state_dict().items():
        torch.testing.assert_close(value, plain.model_.state_dict()[key], rtol=0, atol=0)
    if variant == "e":
        for key, value in model.teacher_.state_dict().items():
            torch.testing.assert_close(value, plain.teacher_.state_dict()[key], rtol=0, atol=0)
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(pixels),
        model.decision_function(pixels),
    )
    np.testing.assert_array_equal(pixels, original_pixels)
    np.testing.assert_array_equal(labels, original_labels)


@pytest.mark.parametrize(
    "fault",
    [
        "ts",
        "pool",
        "teacher_selection",
        "backbone",
        "custom_head",
        "bad_beta",
        "bad_variant",
        "bad_policy",
        "zero_eta",
    ],
)
def test_param_invalid_contract_rejected_before_training(tmp_path, monkeypatch, fault):
    model, record, pixels, labels = setup()
    if fault == "pool":
        record["candidate_pool_scope"] = "clean_validation"
    elif fault == "teacher_selection":
        record["teacher_selection_eligible"] = True
    elif fault == "backbone":
        record["backbone_scope"] = "paper_reproduction"
    elif fault == "custom_head":
        model.model = torch.nn.Linear(512, 1)
    elif fault == "bad_beta":
        model.beta = True
    elif fault == "bad_variant":
        model.variant = "combined"
    elif fault == "bad_policy":
        model.batching_policy = "drop_tail"
    elif fault == "zero_eta":
        model.variant, model.early_learning_weight = "e", 0
    monkeypatch.setattr(type(model), "fit", lambda *a, **k: pytest.fail("invalid contract fit"))
    with pytest.raises(ValueError):
        trainer(record, tmp_path).fit(
            model, pixels, labels, os_or_ts="ts" if fault == "ts" else "os"
        )
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("fault", ["pool", "roles", "updates", "teacher", "shared_teacher"])
def test_edge_corrupt_epoch_is_rejected_before_snapshot(tmp_path, monkeypatch, fault):
    model, record, pixels, labels = setup(variant="e")
    original = model.fit

    def bad_fit(features, observed, *, epoch_callback, **kwargs):
        def callback(epoch, fitted):
            if fault == "pool":
                fitted.candidate_training_positions_[0] = 8
            elif fault == "roles":
                fitted.epoch_provenance_[-1]["positive_rows_seen"] += 1
            elif fault == "updates":
                fitted.optimizer_steps_ -= 1
            elif fault == "shared_teacher":
                fitted.teacher_ = fitted.model_
            else:
                with torch.no_grad():
                    next(fitted.teacher_.parameters()).fill_(float("nan"))
            epoch_callback(epoch, fitted)

        return original(features, observed, epoch_callback=callback, **kwargs)

    monkeypatch.setattr(model, "fit", bad_fit)
    with pytest.raises(ValueError, match="P3Mix"):
        trainer(record, tmp_path).fit(model, pixels, labels, os_or_ts="os")
    assert not model._is_fitted and not hasattr(model, "training_provenance_")
    assert not list(tmp_path.rglob("*.pt"))


def test_edge_byte_bound_rejection_preserves_attempt_evidence(tmp_path):
    model, record, pixels, labels = setup()
    with pytest.raises(ValueError, match="byte bound"):
        trainer(record, tmp_path, bound=1).fit(model, pixels, labels, os_or_ts="os")
    assert not model._is_fitted and not hasattr(model, "training_provenance_")
    assert len(list(tmp_path.rglob("*.pt"))) == 2
