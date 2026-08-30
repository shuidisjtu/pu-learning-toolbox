# ruff: noqa: E402, N802, N803, N806
"""CV fold training isolation: the shared encoder template must not leak
weights across folds (dual_architecture_plan.md §5 阶段 1)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import build_encoder  # noqa: E402
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


def _snapshot(model):
    """Detached weight snapshot of a torch module."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.integration
def test_cv_folds_do_not_leak_encoder_weights():
    X, y_pu = _image_data()
    pipe = PUPipeline(
        classifier="wconpu",
        architecture="cnn",
        backbone="cnn13",
        cv=2,
        max_epochs=1,
        random_state=42,
    )
    pipe._encoder = build_encoder("cnn", backbone="cnn13", in_channels=3)
    template_initial = _snapshot(pipe._encoder)

    # Fold 1 trains a deep copy; the shared template must stay untouched
    # (it is fold 2's starting point).
    clf1 = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
    clf1.fit(X[:12], y_pu[:12], class_prior=0.3)
    assert not _same(_snapshot(clf1.encoder_), template_initial)  # training took effect
    assert _same(_snapshot(pipe._encoder), template_initial)  # template untainted
    fold1_after = _snapshot(clf1.encoder_)

    # Fold 2 trains its own copy; fold 1's weights must not move.
    clf2 = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
    clf2.fit(X[12:], y_pu[12:], class_prior=0.3)
    assert _same(_snapshot(clf1.encoder_), fold1_after)  # fold 2 did not touch fold 1
    assert not _same(_snapshot(clf2.encoder_), template_initial)  # fold 2 trained
    assert _same(_snapshot(pipe._encoder), template_initial)  # template never trained

    # Object isolation: three distinct module objects.
    assert clf1.encoder_ is not clf2.encoder_
    assert clf1.encoder_ is not pipe._encoder
    assert clf2.encoder_ is not pipe._encoder
