# ruff: noqa: N803, N806

"""Integration tests for paper comparison methods and ablation factory."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from pu_toolbox.estimators.research import (  # noqa: E402
    JOINT_SHIFT_METHODS,
    DynamicJointShiftPUClassifier,
    JointShiftPUBaseline,
    build_joint_shift_estimator,
)

pytestmark = pytest.mark.integration


def _data(seed=21):
    rng = np.random.default_rng(seed)
    source_true = (rng.random(50) < 0.4).astype(int)
    target_true = (rng.random(40) < 0.6).astype(int)
    source = rng.normal(size=(50, 3)) + source_true[:, None]
    target = rng.normal(0.2, size=(40, 3)) + target_true[:, None]
    source_pu = ((source_true == 1) & (rng.random(50) < 0.5)).astype(int)
    target_pu = ((target_true == 1) & (rng.random(40) < 0.5)).astype(int)
    return source, source_pu, target, target_pu


def _fit(model, data):
    Xs, ys, Xt, yt = data
    return model.fit(Xs, ys, X_target=Xt, y_target_pu=yt, class_prior=0.4, target_class_prior=0.6)


@pytest.mark.parametrize("strategy", ["trpu", "tepu", "fine_tune", "mmd"])
def test_basic_all_paper_baselines_fit_and_predict(strategy):
    data = _data()
    model = _fit(
        JointShiftPUBaseline(
            strategy=strategy,
            max_epochs=1,
            hidden_dim=6,
            feature_dim=4,
            random_state=2,
            device="cpu",
        ),
        data,
    )
    assert model.predict(data[2]).shape == (len(data[2]),)
    assert np.isfinite([row["loss"] for row in model.training_trace_]).all()
    expected_epochs = 2 if strategy == "fine_tune" else 1
    assert len(model.training_trace_) == expected_epochs


def test_param_factory_maps_all_named_ablations_to_exact_flags():
    expected = {
        "dynamic": ("dynamic", True, True),
        "two_step": ("two_step", True, True),
        "without_weight_correction": ("dynamic", False, True),
        "without_classifier_correction": ("dynamic", True, False),
        "without_both_corrections": ("dynamic", False, False),
    }
    for method, flags in expected.items():
        model = build_joint_shift_estimator(method, max_epochs=1)
        assert isinstance(model, DynamicJointShiftPUClassifier)
        assert (model.training_mode, model.weight_correction, model.classifier_correction) == flags
    assert set(JOINT_SHIFT_METHODS) >= set(expected)


def test_edge_unknown_strategy_and_factory_name_fail_early():
    with pytest.raises(ValueError, match="strategy"):
        JointShiftPUBaseline(strategy="unknown")._validate_hyperparameters()
    with pytest.raises(ValueError, match="Unknown joint-shift method"):
        build_joint_shift_estimator("unknown")


def test_determ_mmd_baseline_reproduces_trace_and_predictions():
    data = _data()
    outputs = []
    for _ in range(2):
        model = _fit(
            JointShiftPUBaseline(
                strategy="mmd",
                max_epochs=2,
                hidden_dim=6,
                feature_dim=4,
                random_state=8,
                device="cpu",
            ),
            data,
        )
        outputs.append((model.training_trace_, model.predict(data[2])))
    assert outputs[0][0] == outputs[1][0]
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
