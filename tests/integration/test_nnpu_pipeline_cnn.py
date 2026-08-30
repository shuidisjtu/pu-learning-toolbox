# ruff: noqa: E402, N802, N803, N806
"""nnPU CNN end-to-end through PUPipeline: provenance mapping + save/load
round-trip (dual_architecture_plan.md §5 阶段 2)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import build_encoder  # noqa: E402
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier  # noqa: E402
from pu_toolbox.workflows import PUPipeline  # noqa: E402


def _image_data(n=24, channels=3, size=8, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.5, 0.3, size=(n, channels, size, size)).astype(np.float32)
    y_pu = np.concatenate(
        [
            np.ones(4, dtype=int),
            np.zeros(8, dtype=int),
            np.ones(4, dtype=int),
            np.zeros(8, dtype=int),
        ]
    )
    return X, y_pu


def _table_data(n=40, d=5, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
    y_pu = np.concatenate([np.ones(10, dtype=int), np.zeros(n - 10, dtype=int)])
    return X, y_pu


@pytest.mark.integration
def test_nnpu_cnn_pipeline_reports_native_cnn_provenance():
    X, y_pu = _image_data()
    report = PUPipeline(
        classifier="nnpu",
        architecture="cnn",
        backbone="cnn13",
        cv=2,
        max_epochs=1,
        random_state=42,
        device="cpu",
    ).fit_evaluate(X, y_pu, class_prior=0.3, refit=False)
    p = report.provenance
    assert p["architecture"] == "native_cnn"
    assert p["backbone"] == "cnn13"
    assert p["encoder"] == {"backbone": "cnn13", "in_channels": 3}
    assert p["device"] == {"requested": "cpu", "resolved": "cpu"}


@pytest.mark.integration
def test_nnpu_mlp_pipeline_reports_native_mlp_provenance():
    X, y_pu = _table_data()
    report = PUPipeline(
        classifier="nnpu",
        architecture="mlp",
        cv=2,
        max_epochs=1,
        random_state=42,
    ).fit_evaluate(X, y_pu, class_prior=0.3, refit=False)
    p = report.provenance
    assert p["architecture"] == "native_mlp"
    assert p["backbone"] is None
    assert p["encoder"] is None


@pytest.mark.integration
def test_nnpu_encoder_model_survives_pickle_roundtrip():
    X, y_pu = _image_data()
    clf = NonNegativePUClassifier(
        encoder=build_encoder("cnn", backbone="cnn13", in_channels=3),
        class_prior=0.3,
        max_epochs=1,
        random_state=42,
        device="cpu",
    )
    clf.fit(X, y_pu)
    expected = clf.decision_function(X[:8])
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(restored.decision_function(X[:8]), expected)
