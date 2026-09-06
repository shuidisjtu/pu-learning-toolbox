# tests/unit/experiment/test_strategies_labeling.py

# ruff: noqa: N806

import numpy as np
import pytest

from pu_toolbox.experiment.strategies import (
    SARLBEAGenerator,
    SARLBEBGenerator,
    SCARGenerator,
)

pytestmark = pytest.mark.unit


def _y(n):
    return np.array([1] * (n // 2) + [0] * (n % 2 + n // 2))


def test_scar_fixed_count_and_both_classes():
    y_true = _y(100)  # 50 pos
    y_pu, meta = SCARGenerator().generate(np.zeros((100, 4)), y_true, 0.1, seed=0)
    assert int(meta["c_realized"] * 50) == 5  # round(0.1*50)=5
    assert (y_pu == 1).sum() == 5
    assert set(y_pu).issubset({0, 1})


def test_scar_deterministic_seed():
    X = np.zeros((50, 2))
    y = _y(50)
    a, _ = SCARGenerator().generate(X, y, 0.5, seed=7)
    b, _ = SCARGenerator().generate(X, y, 0.5, seed=7)
    assert np.array_equal(a, b)


def test_lbe_a_uses_posterior_scores():
    # 9 negatives at x=0.0..0.8, positives at x=0.9 and x=10.0. The posterior
    # helper is fitted on REAL labels, so P(y=1|x) peaks at x=10.0, and the
    # k=10 + smoothing posterior weight concentrates on it (fixed count = 1).
    X = np.array([[v] for v in np.arange(0, 0.9, 0.1)] + [[0.9], [10.0]], dtype=float)
    y_true = np.array([0] * 9 + [1, 1])
    y_pu, meta = SARLBEAGenerator().generate(X, y_true, 0.2, seed=0)  # -> 1 labeled
    assert y_pu.sum() == 1  # fixed count: exactly one labeled
    assert np.all(y_true[y_pu == 1] == 1)  # pool = positives only (S=1 ⟹ Y=1)
    assert y_pu[10] == 1  # top-score positive (x=10.0) must be picked
    assert meta["mechanism"] == "sar_lbe_a"
    assert meta["posterior_fit_on"] == "real_labels"
    assert meta["k"] == 10
    assert meta["smoothing"] == (0.9, 0.1)


def test_lbe_b_shrink_coef():
    X = np.array([[i] for i in range(10, 20)], dtype=float)
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pu, meta = SARLBEBGenerator().generate(X, y, 0.1, seed=1)
    assert (y_pu == 1).sum() == 1
    assert np.all(y[y_pu == 1] == 1)  # pool = positives only (S=1 ⟹ Y=1)
    assert meta["shrink_coef"] == 1.0
    assert meta["mechanism"] == "sar_lbe_b"


def test_lbe_b_never_labels_high_weight_negatives():
    # Negatives at low x (posterior ≈ 0) hold the largest LBE-B weights
    # ((2.5 - score)^10 ≈ 9.5e3 vs ≈ 57 for positives), but the sampling
    # pool is true positives only, so no seed may ever label a negative.
    X = np.arange(10, dtype=float).reshape(-1, 1) * 5.0  # x = 0..45
    y_true = np.array([0] * 8 + [1, 1])  # positives at x=40, 45
    for seed in range(20):
        y_pu, _ = SARLBEBGenerator().generate(X, y_true, 0.1, seed=seed)
        assert (y_pu == 1).sum() == 1
        assert np.all(y_true[y_pu == 1] == 1)


def test_lbe_no_positives_returns_all_zero():
    y_true = np.zeros(10, dtype=int)
    for gen in (SARLBEAGenerator(), SARLBEBGenerator()):
        y_pu, meta = gen.generate(np.zeros((10, 2)), y_true, 0.5, seed=0)
        assert np.all(y_pu == 0)
        assert meta["c_realized"] == 0.0
