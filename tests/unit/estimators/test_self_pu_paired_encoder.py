# ruff: noqa: E402, N803, N806

"""Regression checks for matched Self-PU freeze and fine-tune arms."""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn

from pu_toolbox.estimators.deep._encoder_policy import set_encoder_training_policy
from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier, _eval_logits

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings("ignore:validation_data was not supplied"),
]


@pytest.mark.parametrize("full_meta", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
def test_basic_paired_encoder_preserves_or_updates_every_student_and_teacher(full_meta, trainable):
    torch.manual_seed(41)
    encoder = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    # Nonzero, nontrivial BN state detects both .train() drift and EMA
    # multiply/add roundoff on an otherwise fixed buffer.
    with torch.no_grad():
        encoder[1].running_mean.copy_(torch.tensor([0.17, -0.37, 0.92, -0.13]))
    eval_forward_sizes = []

    def record_eval_forward(_module, inputs):
        if not torch.is_grad_enabled():
            eval_forward_sizes.append(len(inputs[0]))

    encoder.register_forward_pre_hook(record_eval_forward)
    set_encoder_training_policy(encoder, trainable=trainable)
    template = nn.Sequential(encoder, nn.Linear(4, 1))
    initial = copy.deepcopy(template.state_dict())
    rng = np.random.default_rng(47)
    X = rng.normal(size=(24, 1, 8, 8)).astype(np.float32)
    y = np.r_[np.ones(8, dtype=int), np.zeros(16, dtype=int)]
    support = (
        rng.normal(size=(8, 1, 8, 8)).astype(np.float32),
        np.tile([0, 1], 4),
    )
    observed_epochs: list[int] = []

    def check_epoch(epoch, estimator):
        observed_epochs.append(epoch)
        for model in (
            estimator.student_1_,
            estimator.student_2_,
            estimator.teacher_1_,
            estimator.teacher_2_,
        ):
            state = model.state_dict()
            if trainable:
                assert not torch.equal(state["0.0.weight"], initial["0.0.weight"])
            else:
                assert all(torch.equal(state[k], initial[k]) for k in initial if k.startswith("0."))
                assert not model[0].training
                assert not any(p.requires_grad for p in model[0].parameters())
            assert not torch.equal(state["1.weight"], initial["1.weight"])

    model = SelfPUClassifier(
        0.4,
        backbone=template,
        max_epochs=3,
        batch_size=8,
        warmup_epochs=1,
        self_paced_start=1,
        self_paced_end=2,
        distill_start=2,
        require_validation=full_meta,
        distillation_scope="paper_population" if full_meta else "untrusted_u",
        device="cpu",
        random_state=13,
    )
    model.fit(X, y, validation_data=support if full_meta else None, epoch_callback=check_epoch)

    assert observed_epochs == [1, 2, 3]
    active = [row for row in model.reweight_history_ if row["calibration_active"]]
    assert len(active) == (4 if full_meta else 0)
    assert np.isfinite(model.decision_function(X)).all()
    assert model.checkpoint_prediction_batch_size == 8
    assert eval_forward_sizes and max(eval_forward_sizes) == 8
    assert all(torch.equal(initial[k], template.state_dict()[k]) for k in initial)


@pytest.mark.parametrize("batch_size", [1, 2, 5, None])
def test_param_chunked_predictions_keep_requested_rows_in_order(batch_size):
    module = nn.Sequential(nn.Flatten(), nn.Linear(4, 1)).eval()
    images = torch.arange(40, dtype=torch.float32).reshape(10, 1, 2, 2)
    indices = np.array([5, 4, 7, 3, 1])
    with torch.no_grad():
        expected = module(images[indices])[:, 0]
        actual = _eval_logits(module, images, batch_size, indices=indices)
    torch.testing.assert_close(actual, expected)


def _fixed_full_meta_case():
    torch.manual_seed(7)
    encoder = set_encoder_training_policy(
        nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten()),
        trainable=False,
    )
    model = SelfPUClassifier(
        0.4,
        backbone=nn.Sequential(encoder, nn.Linear(2, 1)),
        max_epochs=2,
        warmup_epochs=0,
        self_paced_start=0,
        self_paced_end=1,
        distill_start=1,
        require_validation=True,
        batch_size=4,
        random_state=7,
        device="cpu",
    )
    rng = np.random.default_rng(3)
    X = rng.normal(size=(16, 1, 4, 4)).astype(np.float32)
    y = np.r_[np.ones(4, dtype=int), np.zeros(12, dtype=int)]
    support = rng.normal(size=(6, 1, 4, 4)).astype(np.float32), np.tile([0, 1], 3)
    return model, X, y, support


def test_edge_frozen_full_meta_refuses_support_in_a_different_representation():
    model, X, y, support = _fixed_full_meta_case()
    with pytest.raises(ValueError, match="sample shape"):
        model.fit(X, y, validation_data=(support[0].reshape(len(support[0]), -1), support[1]))
    assert not hasattr(model, "student_1_")


def test_determ_frozen_full_meta_fit_reproduces_teacher_scores_and_meta_weights():
    model, X, y, support = _fixed_full_meta_case()
    other = copy.deepcopy(model)
    model.fit(X, y, validation_data=support)
    other.fit(X, y, validation_data=support)
    np.testing.assert_array_equal(model.decision_function(X), other.decision_function(X))
    assert model.reweight_history_ == other.reweight_history_
    assert model.teacher_selection_metrics_ == other.teacher_selection_metrics_
