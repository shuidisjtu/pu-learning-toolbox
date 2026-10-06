"""Positive-only LZO equations, selected-prefix state, cost and isolation."""

import copy
import os
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.holistic_pu import (  # noqa: E402
    HolisticPUClassifier,
    holistic_trend_scores,
)
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def problem(architecture="mlp"):
    shape = (27, 3, 4, 4) if architecture == "cnn" else (27, 3)
    features = np.random.RandomState(7).normal(size=shape).astype("float32")
    features[:10] += 1
    labels = np.r_[np.ones(10, int), np.zeros(17, int)]
    torch.manual_seed(17)
    encoder = None
    if architecture == "cnn":
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.2),
        )
    return features, labels, encoder


def model(encoder=None, **overrides):
    params = dict(
        encoder=encoder,
        hidden_dim=4,
        warmup_epochs=4,
        max_epochs=1,
        batch_size=8,
        random_state=3,
        device="cpu",
        warmup_selection="lzo_positive_loss",
        lzo_validation_size=11,
    )
    params.update(overrides)
    return HolisticPUClassifier(**params)


@pytest.mark.math
@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_basic_mixup_matches_scalar_label_invariant_equation_and_pooled_ce(
    monkeypatch, architecture
):
    features, labels, encoder = problem(architecture)
    fitted = model(encoder).fit(features, labels)
    assert np.isin(fitted.lzo_mixup_indices_, np.flatnonzero(labels == 1)).all()
    assert fitted.lzo_candidate_epochs_ == (2, 3, 4)
    assert len(fitted.lzo_losses_) == 4 and np.isfinite(fitted.lzo_losses_).all()
    assert fitted.lzo_evaluated_rows_ == 44 and fitted.lzo_forward_batches_ == 8
    pairs, weights = fitted.lzo_mixup_indices_, fitted.lzo_mixup_weights_.astype("float32")
    independent = np.stack(
        [
            weights[i] * features[left] + (np.float32(1) - weights[i]) * features[right]
            for i, (left, right) in enumerate(pairs)
        ]
    )
    seen = []

    def scalar_scores(rows):
        seen.append(rows.copy())
        return rows.reshape(len(rows), -1).mean(axis=1)

    monkeypatch.setattr(fitted, "_scores", scalar_scores)
    actual = fitted._lzo_loss(features)
    np.testing.assert_array_equal(np.concatenate(seen), independent)
    logits = independent.reshape(len(independent), -1).mean(axis=1).astype(float)
    expected = sum(np.log(1 + np.exp(-float(logit))) for logit in logits) / len(logits)
    assert actual == pytest.approx(expected, rel=1e-12)
    assert [len(rows) for rows in seen] == [8, 3]  # no unweighted average of batch means


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
@pytest.mark.parametrize("initialization", ["continue", "reinitialize"])
def test_determ_selected_prefix_restores_model_adam_rng_and_matches_literal_short_run(
    monkeypatch, architecture, initialization
):
    features, labels, encoder = problem(architecture)

    # Epoch 1 has the lowest score but cannot define a >=2-time trend; 2 and 3
    # tie exactly, so the method must choose 2 and restore its state, not 4.
    def controlled_loss(self, rows):
        return [0.01, 0.2, 0.2, 0.9][self.executed_warmup_epochs_ - 1]

    monkeypatch.setattr(HolisticPUClassifier, "_lzo_loss", controlled_loss)
    selected = model(encoder, pseudo_pn_initialization=initialization).fit(features, labels)
    literal = model(
        encoder,
        pseudo_pn_initialization=initialization,
        warmup_selection="fixed",
        warmup_epochs=2,
    ).fit(features, labels)
    assert selected.selected_warmup_epoch_ == 2 and selected.executed_warmup_epochs_ == 4
    assert selected.prediction_trajectory_.shape == (17, 2)
    assert selected.observed_prediction_trajectory_.shape == (17, 4)
    np.testing.assert_array_equal(selected.prediction_trajectory_, literal.prediction_trajectory_)
    np.testing.assert_array_equal(selected.pseudo_labels_, literal.pseudo_labels_)
    np.testing.assert_array_equal(
        selected.decision_function(features), literal.decision_function(features)
    )
    np.testing.assert_array_equal(
        selected.trend_scores_, holistic_trend_scores(selected.prediction_trajectory_)
    )
    assert selected.stage_optimizer_steps_ == {"warmup": 12, "pseudo_pn": 4}
    assert selected.optimizer_steps_ == 16 and literal.optimizer_steps_ == 10
    assert selected.discarded_warmup_optimizer_steps_ == 6


@pytest.mark.parametrize("initialization", ["continue", "reinitialize"])
def test_determ_clone_pickle_refit_and_full_horizon_stage_snapshots(tmp_path, initialization):
    features, labels, encoder = problem("cnn")
    source = copy.deepcopy(encoder.state_dict())
    fitted = clone(model(encoder, pseudo_pn_initialization=initialization))
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    repeated = clone(model(encoder, pseudo_pn_initialization=initialization)).fit(features, labels)
    np.testing.assert_array_equal(
        fitted.decision_function(features), repeated.decision_function(features)
    )
    np.testing.assert_array_equal(fitted.lzo_mixup_indices_, repeated.lzo_mixup_indices_)
    np.testing.assert_array_equal(fitted.lzo_losses_, repeated.lzo_losses_)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(
        restored.decision_function(features), fitted.decision_function(features)
    )
    assert len(trajectory.checkpoints) == fitted.checkpoint_epoch_count == 5
    assert [
        ref.reference()["training_context"]["optimizer_steps"] for ref in trajectory.checkpoints
    ] == [3, 6, 9, 12, 16]
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(features),
        fitted.decision_function(features),
    )
    fitted.fit(features, labels)
    np.testing.assert_array_equal(
        fitted.decision_function(features), repeated.decision_function(features)
    )
    assert fitted.lzo_evaluated_rows_ == 44 and fitted.optimizer_steps_ == 16
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, source[key], rtol=0, atol=0)


def test_basic_real_argmin_default_size_and_fixed_legacy_path():
    features, labels, _ = problem()
    fitted = model(lzo_validation_size=None).fit(features, labels)
    assert fitted.lzo_validation_size_ == 10
    assert fitted.selected_warmup_epoch_ == int(np.argmin(fitted.lzo_losses_[1:])) + 2
    assert fitted.stopping_rule_ == "lzo_positive_loss_full_horizon_argmin"
    assert fitted.lzo_selection_spec_["source"] == "labeled_training_positive_only"
    assert fitted.lzo_selection_spec_["metric"] == "mean_positive_binary_cross_entropy"
    fixed = model(warmup_selection="fixed").fit(features, labels)
    assert fixed.stopping_rule_ == "fixed_warmup_budget_not_LZO"
    assert fixed.selected_warmup_epoch_ == fixed.executed_warmup_epochs_ == 4
    assert fixed.prediction_trajectory_ is fixed.observed_prediction_trajectory_
    assert fixed.lzo_mixup_indices_.shape == (0, 2) and fixed.lzo_losses_.size == 0
    assert fixed.lzo_evaluated_rows_ == 0 and fixed.discarded_warmup_optimizer_steps_ == 0


def test_edge_validation_eval_preserves_rng_bn_and_mixed_layer_modes(monkeypatch):
    features, labels, encoder = problem("cnn")
    fitted = model(encoder).fit(features, labels)
    fitted.model_.train()
    fitted.encoder_[1].eval()
    modes = [layer.training for layer in fitted.model_.modules()]
    before = copy.deepcopy(fitted.encoder_.state_dict())
    torch_before = torch.get_rng_state().clone()
    original = fitted._scores

    def consume_rng(rows):
        torch.rand(2)
        return original(rows)

    monkeypatch.setattr(fitted, "_scores", consume_rng)
    assert np.isfinite(fitted._lzo_loss(features))
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    assert [layer.training for layer in fitted.model_.modules()] == modes
    for key, value in fitted.encoder_.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"warmup_selection": "oracle"},
        {"warmup_selection": []},
        {"lzo_alpha": 0},
        {"lzo_alpha": float("inf")},
        {"lzo_alpha": True},
        {"lzo_alpha": "bad"},
        {"lzo_validation_size": 0},
        {"lzo_validation_size": True},
        {"lzo_validation_size": 1.2},
    ],
)
def test_param_invalid_recipe_refused_before_fit(overrides):
    features, labels, _ = problem()
    fitted = model(**overrides)
    with pytest.raises(ValueError, match="warmup_selection|lzo_alpha|lzo_validation_size"):
        fitted.fit(features, labels)
    assert not fitted._is_fitted


@pytest.mark.parametrize("bad_loss", [np.nan, np.inf])
def test_edge_nonfinite_validation_cannot_silently_fall_back_to_fixed(monkeypatch, bad_loss):
    features, labels, _ = problem()
    monkeypatch.setattr(HolisticPUClassifier, "_lzo_loss", lambda self, rows: bad_loss)
    fitted = model()
    with pytest.raises(ValueError, match="LZO validation loss"):
        fitted.fit(features, labels)
    assert not fitted._is_fitted and fitted.optimizer_steps_ == 3


def test_basic_best_continue_state_is_one_cpu_copy_with_state_dict_metadata():
    features, labels, _ = problem()
    fitted = model().fit(features, labels)
    original = fitted.model_.state_dict()
    copied = fitted._cpu_state_copy(original)
    assert copied._metadata == original._metadata
    for key, value in copied.items():
        assert value.device.type == "cpu" and value.data_ptr() != original[key].data_ptr()
    with torch.no_grad():
        next(fitted.model_.parameters()).add_(1)
    assert not torch.equal(
        next(iter(copied.values())), next(iter(fitted.model_.state_dict().values()))
    )


def test_edge_one_positive_plan_and_failed_eval_restore_rng_without_fallback(monkeypatch):
    fitted = model()
    with pytest.raises(ValueError, match=">=2 labeled"):
        fitted._prepare_lzo(np.array([0]), np.random.RandomState(3))
    features, labels, encoder = problem("cnn")
    fitted = model(encoder).fit(features, labels)
    state = torch.get_rng_state().clone()

    def fail(rows):
        torch.rand(2)
        raise ValueError("synthetic auxiliary forward failure")

    monkeypatch.setattr(fitted, "_scores", fail)
    with pytest.raises(ValueError, match="auxiliary forward failure"):
        fitted._lzo_loss(features)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)


@pytest.mark.gpu
def test_cuda_best_state_restoration_and_final_checkpoint(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    features, labels, encoder = problem("cnn")
    monkeypatch.setattr(
        HolisticPUClassifier,
        "_lzo_loss",
        lambda self, rows: [0.1, 0.2, 0.3, 0.4][self.executed_warmup_epochs_ - 1],
    )
    fitted = model(encoder, device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    assert fitted.selected_warmup_epoch_ == 2 and next(fitted.model_.parameters()).is_cuda
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
        fitted.decision_function(features),
        rtol=1e-5,
        atol=1e-5,
    )
