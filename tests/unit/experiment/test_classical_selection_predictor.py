"""Exported non-epoch selection carries VAL constants, never a test-fitted scale."""

import copy
import pickle

import numpy as np
import pytest

from pu_toolbox.experiment.classical_candidate import ClassicalSelectionPredictor

pytestmark = pytest.mark.unit


class RawModel:
    def decision_function(self, features):
        return np.asarray(features)[:, 0].copy()

    def predict(self, features):
        return (self.decision_function(features) >= 0).astype(int)


def selection(threshold=0.6, minimum=-10.0, scale=20.0):
    return dict(
        epoch=None,
        checkpoint_index=None,
        threshold=threshold,
        metrics=dict(val_score_min=minimum, val_score_scale=scale),
    )


def test_basic_determ_pickle_replays_val_rule_not_native_or_test_normalization():
    model = RawModel()
    chosen = selection()
    original = copy.deepcopy(chosen)
    features = np.array([[-100.0], [1.0], [2.0], [200.0]])
    before = features.copy()
    predictor = ClassicalSelectionPredictor(model, chosen)
    expected = ((features[:, 0] + 10) / 20 >= 0.6).astype(int)
    np.testing.assert_array_equal(predictor.predict(features), expected)
    assert not np.array_equal(expected, model.predict(features))
    assert not np.array_equal(expected, ((features[:, 0] + 100) / 300 >= 0.6).astype(int))
    restored = pickle.loads(pickle.dumps(predictor))
    np.testing.assert_array_equal(restored.predict(features), expected)
    np.testing.assert_array_equal(restored.decision_function(features), features[:, 0])
    np.testing.assert_array_equal(features, before)
    assert chosen == original


def test_basic_independent_pa_oa_rules_do_not_mutate_shared_estimator():
    model = RawModel()
    chosen = selection(threshold=0.3, minimum=0, scale=1)
    pa = ClassicalSelectionPredictor(model, chosen)
    oa = ClassicalSelectionPredictor(model, selection(threshold=0.7, minimum=0, scale=1))
    chosen["threshold"] = 0
    features = np.array([[0.4], [0.8]])
    np.testing.assert_array_equal(pa.predict(features), [1, 1])
    np.testing.assert_array_equal(oa.predict(features), [0, 1])
    assert pa.estimator is oa.estimator is model and not hasattr(model, "selection")


@pytest.mark.parametrize(
    "chosen", [selection(threshold=None), selection(minimum=None), selection(scale=0)]
)
def test_edge_none_threshold_or_missing_affine_preserves_runner_fallback(chosen):
    model = RawModel()
    features = np.array([[-1.0], [0.0], [0.8]])
    predictor = ClassicalSelectionPredictor(model, chosen)
    expected = (
        model.predict(features)
        if chosen["threshold"] is None
        else (model.decision_function(features) >= chosen["threshold"]).astype(int)
    )
    np.testing.assert_array_equal(predictor.predict(features), expected)


@pytest.mark.parametrize(
    "change",
    [
        dict(epoch=1),
        dict(checkpoint_index=0),
        dict(threshold=float("nan")),
        dict(threshold=True),
        dict(metrics=[]),
        dict(metrics={"val_score_scale": -1}),
    ],
)
def test_param_invalid_or_checkpoint_selection_refused(change):
    with pytest.raises(ValueError):
        ClassicalSelectionPredictor(RawModel(), selection() | change)
