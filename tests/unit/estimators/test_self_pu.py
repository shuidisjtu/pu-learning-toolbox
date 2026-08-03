# ruff: noqa: E402, N803, N806

"""Unit tests for Self-PU training components and estimator integration."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.estimators.deep.self_pu import (
    SelfPUClassifier,
    TrustedSetManager,
    calibrate_meta_weights,
    dynamic_trust_target,
    ema_update,
    hard_distillation_loss,
)


@pytest.fixture
def self_pu_data():
    rng = np.random.RandomState(17)
    X = np.vstack([rng.normal(1.0, 0.4, (12, 4)), rng.normal(-1.0, 0.4, (24, 4))]).astype(
        np.float32
    )
    y_pu = np.r_[np.ones(6, dtype=int), np.zeros(30, dtype=int)]
    X_val = np.vstack([rng.normal(1.0, 0.4, (6, 4)), rng.normal(-1.0, 0.4, (6, 4))]).astype(
        np.float32
    )
    y_val = np.r_[np.ones(6, dtype=int), np.zeros(6, dtype=int)]
    return X, y_pu, X_val, y_val


def _small_classifier(**kwargs):
    parameters = {
        "class_prior": 1 / 3,
        "hidden_dim": 8,
        "warmup_epochs": 0,
        "self_paced_start": 0,
        "self_paced_end": 1,
        "distill_start": 2,
        "max_epochs": 3,
        "max_trust_ratio": 0.2,
        "pace_1": 0.1,
        "pace_2": 0.2,
        "batch_size": 16,
        "random_state": 5,
    }
    parameters.update(kwargs)
    return SelfPUClassifier(**parameters)


@pytest.mark.math
class TestSelfPUComponents:
    def test_basic_dynamic_trust_target_matches_linear_schedule(self):
        kwargs = {"start_epoch": 2, "end_epoch": 6, "final_size": 10}
        assert dynamic_trust_target(1, **kwargs) == 0
        assert dynamic_trust_target(2, **kwargs) == 0
        assert dynamic_trust_target(4, **kwargs) == 5
        assert dynamic_trust_target(6, **kwargs) == 10

    def test_basic_trusted_set_is_balanced_and_preserves_soft_labels(self):
        probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        manager = TrustedSetManager(len(probabilities))

        update = manager.update(probabilities, target_size=4)

        assert update.actual_size == 4
        assert update.positive_count == update.negative_count == 2
        assert set(manager.indices) == {0, 1, 2, 3}
        np.testing.assert_allclose(manager.soft_labels, probabilities[manager.indices])
        assert not set(manager.indices) - set(range(len(probabilities)))

    def test_edge_trusted_set_supports_in_and_out_refresh(self):
        manager = TrustedSetManager(6)
        manager.update(np.array([0.0, 0.1, 0.4, 0.5, 0.8, 0.9]), 2)
        update = manager.update(np.array([0.4, 0.1, 0.0, 0.9, 0.8, 0.5]), 2)

        assert set(manager.indices) == {2, 3}
        assert update.entered_count == 2
        assert update.exited_count == 2

    def test_edge_meta_weights_have_stable_zero_gradient_fallback(self):
        weights, statistics = calibrate_meta_weights(np.zeros((4, 2)), gamma=0.25)

        assert np.all(weights >= 0.0)
        assert statistics["ce_fallback"] is True
        assert statistics["pu_fallback"] is True
        assert statistics["ce_active_fraction"] == pytest.approx(0.25)
        assert weights[:, 0].sum() == pytest.approx(1.0)
        assert weights[:, 1].sum() == pytest.approx(1.0)

    def test_param_meta_gamma_limits_ce_support_and_mass(self):
        influences = np.array([[4.0, 1.0], [3.0, 2.0]])
        weights, statistics = calibrate_meta_weights(influences, gamma=0.2)

        assert np.count_nonzero(weights[:, 0]) == 1
        assert weights[:, 0].sum() == pytest.approx(0.4)
        assert statistics["ce_weight_sum"] <= 0.2 * len(influences)
        assert weights[:, 1].sum() == pytest.approx(1.0)

    def test_basic_ema_update_matches_parameter_recurrence(self):
        student = torch.nn.Linear(2, 1, bias=False)
        teacher = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            student.weight.fill_(2.0)
            teacher.weight.zero_()

        ema_update(teacher, student, decay=0.75)

        np.testing.assert_allclose(teacher.weight.detach().numpy(), 0.5)
        assert teacher.weight.grad is None

    def test_basic_hard_distillation_uses_paper_mask(self):
        student = torch.tensor([0.0, 0.5, 1.0])
        peer = torch.tensor([0.0, 0.0, 0.5])
        pu_losses = torch.tensor([0.1, 0.1, 1.0])

        loss, active_fraction = hard_distillation_loss(student, peer, pu_losses, alpha=1.0)

        assert active_fraction.item() == pytest.approx(2 / 3)
        assert loss.item() == pytest.approx(0.125)


@pytest.mark.unit
class TestSelfPUClassifier:
    def test_basic_fit_with_clean_validation_records_all_stages(self, self_pu_data):
        X, y_pu, X_val, y_val = self_pu_data
        classifier = _small_classifier().fit(X, y_pu, validation_data=(X_val, y_val))

        assert classifier.calibration_mode_ == "clean_validation_meta"
        assert classifier.best_teacher_index_ in {1, 2}
        assert classifier.teacher_selection_basis_ == "clean_validation_accuracy"
        assert len(classifier.training_history_) == 6
        assert len(classifier.trusted_history_) == 6
        assert {item["stage"] for item in classifier.training_history_} == {
            "self_paced",
            "distillation",
        }
        assert any(item["calibration_active"] for item in classifier.reweight_history_)
        assert any(item["active"] for item in classifier.distillation_history_)
        assert classifier.get_pu_metadata()["implementation_status"] == "native"
        checkpoint = classifier.get_training_checkpoint()
        assert len(checkpoint["student_states"]) == 2
        assert len(checkpoint["teacher_states"]) == 2
        assert len(checkpoint["optimizer_states"]) == 2
        assert set(checkpoint["histories"]) == {
            "trusted",
            "reweight",
            "distillation",
            "training",
        }

    def test_edge_missing_validation_is_explicit_ablation(self, self_pu_data):
        X, y_pu, _, _ = self_pu_data
        classifier = _small_classifier(max_epochs=2, distill_start=1)

        with pytest.warns(UserWarning, match="explicit Self-PU ablation"):
            classifier.fit(X, y_pu)

        assert classifier.calibration_mode_ == "ablation"
        assert classifier.teacher_selection_basis_ == "training_nnpu_risk_ablation"
        assert not any(item["calibration_active"] for item in classifier.reweight_history_)

    def test_basic_prediction_shapes_probabilities_and_image_input(self):
        rng = np.random.RandomState(3)
        X = rng.normal(size=(24, 1, 2, 2)).astype(np.float32)
        y_pu = np.r_[np.ones(6, dtype=int), np.zeros(18, dtype=int)]
        classifier = _small_classifier(max_epochs=1, distill_start=1).fit(X, y_pu)

        probabilities = classifier.predict_proba(X[:3])
        assert probabilities.shape == (3, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
        assert classifier.predict(X[:3]).dtype == int
        assert classifier.decision_function(X[:3]).shape == (3,)

    def test_param_invalid_configuration_and_validation_protocol(self, self_pu_data):
        X, y_pu, X_val, y_val = self_pu_data
        with pytest.raises(ValueError, match="class_prior"):
            _small_classifier(class_prior=1.0).fit(X, y_pu)
        with pytest.raises(ValueError, match="Require 0 <="):
            _small_classifier(warmup_epochs=2, self_paced_start=1).fit(X, y_pu)
        with pytest.raises(ValueError, match="validation_data is required"):
            _small_classifier(require_validation=True).fit(X, y_pu)
        with pytest.raises(ValueError, match="both classes"):
            _small_classifier().fit(
                X,
                y_pu,
                validation_data=(X_val, np.ones_like(y_val)),
            )
        with pytest.raises(NotImplementedError, match="sample_weight"):
            _small_classifier().fit(X, y_pu, sample_weight=np.ones(len(X)))

    def test_edge_predict_before_fit_and_bad_shape_raise(self, self_pu_data):
        X, y_pu, _, _ = self_pu_data
        with pytest.raises(NotFittedError):
            _small_classifier().predict(X)
        classifier = _small_classifier(max_epochs=1, distill_start=1).fit(X, y_pu)
        with pytest.raises(ValueError, match="sample shape"):
            classifier.predict(np.ones((2, 3), dtype=np.float32))

    def test_deterministic_training_with_fixed_seed(self, self_pu_data):
        X, y_pu, X_val, y_val = self_pu_data
        first = _small_classifier(max_epochs=2, distill_start=1).fit(
            X, y_pu, validation_data=(X_val, y_val)
        )
        second = _small_classifier(max_epochs=2, distill_start=1).fit(
            X, y_pu, validation_data=(X_val, y_val)
        )

        np.testing.assert_allclose(first.decision_function(X), second.decision_function(X))
        assert first.trusted_history_ == second.trusted_history_
        assert first.best_teacher_index_ == second.best_teacher_index_
