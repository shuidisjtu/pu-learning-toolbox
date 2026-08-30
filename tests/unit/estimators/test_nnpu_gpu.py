# ruff: noqa: E402, N802, N803, N806
"""GPU execution-level tests for nnPU with a CNN encoder
(dual_architecture_plan.md §5 阶段 2). Auto-skip without CUDA; run on a
CUDA machine with ``pytest -m gpu``."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import build_encoder  # noqa: E402
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _image_data(n=32, channels=3, size=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.5, 0.3, size=(n, channels, size, size)).astype(np.float32)
    y_pu = np.concatenate([np.ones(8, dtype=int), np.zeros(n - 8, dtype=int)])
    return X, y_pu


def test_nnpu_cnn_trains_and_predicts_on_gpu():
    X, y_pu = _image_data()
    clf = NonNegativePUClassifier(
        encoder=build_encoder("cnn", backbone="cnn13", in_channels=3),
        class_prior=0.4,
        max_epochs=1,
        batch_size=16,
        device="cuda",
        random_state=7,
    )
    clf.fit(X, y_pu)
    assert isinstance(clf.model_, torch.nn.Sequential)
    assert all(p.device.type == "cuda" for p in clf.model_.parameters())
    scores = clf.decision_function(X[:8])
    assert scores.shape == (8,)
