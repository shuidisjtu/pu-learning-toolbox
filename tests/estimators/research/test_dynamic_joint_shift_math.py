# ruff: noqa: N803, N806

"""Golden formula and training tests for dynamic joint-shift PU learning."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.research import (  # noqa: E402
    DynamicJointShiftPUClassifier,
    paper_classifier_objective,
    paper_importance_weight_objective,
    paper_pu_risk,
)

pytestmark = pytest.mark.math


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def _data(seed=10):
    rng = np.random.default_rng(seed)
    source_y_true = (rng.random(50) < 0.4).astype(int)
    target_y_true = (rng.random(40) < 0.6).astype(int)
    source = rng.normal(size=(50, 3)) + source_y_true[:, None]
    target = rng.normal(0.3, size=(40, 3)) + target_y_true[:, None]
    source_pu = ((source_y_true == 1) & (rng.random(50) < 0.5)).astype(int)
    target_pu = ((target_y_true == 1) & (rng.random(40) < 0.5)).astype(int)
    return source, source_pu, target, target_pu


def test_basic_equation_19_matches_hand_computation_without_correction():
    values = {
        "target_positive_positive": _tensor([1.0, 2.0]),
        "target_positive_negative": _tensor([0.5, 1.5]),
        "target_unlabeled_negative": _tensor([1.0, 1.0]),
        "source_positive_positive": _tensor([2.0, 2.0]),
        "source_positive_negative": _tensor([1.0, 1.0]),
        "source_unlabeled_negative": _tensor([1.5, 0.5]),
    }
    result = paper_importance_weight_objective(
        **values,
        source_class_prior=0.4,
        target_class_prior=0.6,
        alpha=0.5,
        correction=False,
    )

    def transformed(x):
        return 0.5 * x**2 - 2 * x

    expected = (
        0.6 * np.mean(transformed(np.array([1.0, 2.0])))
        + np.mean(transformed(np.array([1.0, 1.0])))
        - 0.6 * np.mean(transformed(np.array([0.5, 1.5])))
        + 0.4 * 0.5 * np.mean(np.array([2.0, 2.0]) ** 2)
        + 0.5 * (np.mean(np.array([1.5, 0.5]) ** 2) - 0.4 * np.mean(np.ones(2) ** 2))
    )
    assert float(result) == pytest.approx(expected)


def test_param_equation_19_corrections_apply_exact_lower_bounds():
    common = dict(
        target_positive_positive=_tensor([0.0]),
        target_positive_negative=_tensor([2.0]),
        target_unlabeled_negative=_tensor([2.0]),
        source_positive_positive=_tensor([0.0]),
        source_positive_negative=_tensor([2.0]),
        source_unlabeled_negative=_tensor([0.0]),
        source_class_prior=0.8,
        target_class_prior=0.8,
        alpha=0.5,
    )
    corrected = paper_importance_weight_objective(**common, correction=True)
    raw = paper_importance_weight_objective(**common, correction=False)
    assert float(corrected) >= float(raw)


def test_basic_equations_13_and_20_have_expected_beta_endpoints():
    logits_p = _tensor([0.2, 0.4])
    logits_u = _tensor([-0.1, 0.3])
    target = paper_pu_risk(logits_p, logits_u, class_prior=0.5)
    weights = torch.ones(2, dtype=torch.float64)
    source = paper_pu_risk(
        logits_p,
        logits_u,
        class_prior=0.4,
        positive_positive_weight=weights,
        positive_negative_weight=weights,
        unlabeled_negative_weight=weights,
    )
    inputs = dict(
        source_positive_logits=logits_p,
        source_unlabeled_logits=logits_u,
        target_positive_logits=logits_p,
        target_unlabeled_logits=logits_u,
        source_positive_positive_weight=weights,
        source_positive_negative_weight=weights,
        source_unlabeled_negative_weight=weights,
        source_class_prior=0.4,
        target_class_prior=0.5,
    )
    assert float(paper_classifier_objective(**inputs, beta=1.0)) == pytest.approx(float(target))
    assert float(paper_classifier_objective(**inputs, beta=0.0)) == pytest.approx(float(source))


def test_edge_missing_target_and_external_weights_are_rejected():
    source, source_pu, target, target_pu = _data()
    model = DynamicJointShiftPUClassifier(max_epochs=1, hidden_dim=4, feature_dim=3)
    with pytest.raises(ValueError, match="X_target"):
        model.fit(source, source_pu, class_prior=0.4, target_class_prior=0.6)
    with pytest.raises(NotImplementedError, match="paper objective"):
        model.fit(
            source,
            source_pu,
            X_target=target,
            y_target_pu=target_pu,
            class_prior=0.4,
            target_class_prior=0.6,
            sample_weight=np.ones(len(source)),
        )


def test_determ_dynamic_training_reproduces_trace_and_predictions():
    source, source_pu, target, target_pu = _data()
    outputs = []
    for _ in range(2):
        model = DynamicJointShiftPUClassifier(
            max_epochs=3,
            hidden_dim=8,
            feature_dim=4,
            random_state=7,
            device="cpu",
        ).fit(
            source,
            source_pu,
            X_target=target,
            y_target_pu=target_pu,
            class_prior=0.4,
            target_class_prior=0.6,
        )
        outputs.append((model.training_trace_, model.predict(target), model.source_joint_weights_))
        assert model.get_pu_metadata()["objective_equations"] == [13, 19, 20, 21, 22, 23]
        assert np.all(model.source_joint_weights_ >= 0)
        assert np.all(model.source_joint_weights_ <= 10.0)
    assert outputs[0][0] == outputs[1][0]
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_allclose(outputs[0][2], outputs[1][2])
