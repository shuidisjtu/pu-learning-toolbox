# ruff: noqa: N803, N806

"""Integration tests for workflow-level sample-weight propagation."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.core.tags import SampleWeightSupport
from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.workflows import PipelineError, PUPipeline
from tests.helpers import make_scar_data

pytestmark = pytest.mark.integration


class _RecordingWeightedClassifier(BasePUClassifier):
    requires_class_prior = False
    sample_weight_support = SampleWeightSupport.SUPPORTED
    observed_weights: list[np.ndarray] = []

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        type(self).observed_weights.append(np.asarray(sample_weight, dtype=float).copy())
        self._is_fitted = True
        return self

    def _predict(self, X):
        return np.zeros(len(X), dtype=int)

    def _decision_function(self, X):
        return np.zeros(len(X), dtype=float)


def test_basic_weights_reach_each_training_fold_and_refit(rng):
    X, y_pu, _ = make_scar_data(rng, n=90, separation=2.0)
    weights = np.linspace(0.5, 1.5, len(X))
    _RecordingWeightedClassifier.observed_weights = []
    report = PUPipeline(
        classifier=_RecordingWeightedClassifier(),
        cv=3,
    ).fit_evaluate(X, y_pu, sample_weight=weights)

    observed = _RecordingWeightedClassifier.observed_weights
    assert [len(item) for item in observed] == [120, 120, 120, 180]
    assert np.array_equal(observed[-1], weights)
    assert report.provenance["sample_weight"]["supplied"] is True


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("shape", "shape"),
        ("negative", "non-negative"),
        ("nonfinite", "finite"),
        ("zero", "positive value"),
    ],
)
def test_param_invalid_weights_fail_before_training(rng, case, message):
    X, y_pu, _ = make_scar_data(rng, n=90, separation=2.0)
    weights = np.ones(len(X))
    if case == "shape":
        weights = weights[:-1]
    elif case == "negative":
        weights[-1] = -1.0
    elif case == "nonfinite":
        weights[-1] = np.nan
    else:
        weights[:] = 0.0
    with pytest.raises(ValueError, match=message):
        PUPipeline(classifier=_RecordingWeightedClassifier(), cv=3).fit_evaluate(
            X, y_pu, sample_weight=weights
        )


def test_edge_classifier_that_ignores_weights_is_rejected(rng):
    X, y_pu, _ = make_scar_data(rng, n=90, separation=2.0)
    with pytest.raises(PipelineError, match="declared support is 'ignored'"):
        PUPipeline(classifier=UPUClassifier(class_prior=0.4), cv=3).fit_evaluate(
            X, y_pu, sample_weight=np.ones(len(X))
        )


def test_determ_unweighted_provenance_remains_explicit(rng):
    X, y_pu, _ = make_scar_data(rng, n=90, separation=2.0)
    report = PUPipeline(classifier=UPUClassifier(class_prior=0.4), cv=3).fit_evaluate(X, y_pu)
    assert report.provenance["sample_weight"] == {"supplied": False}
