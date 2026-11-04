# ruff: noqa: N806

"""Single-source RBF kernel formula consistency tests.

Guards the architecture-decay fix: every RBF kernel formula in the
codebase must agree with the canonical ``utils.basis.build_rbf_basis``.
"""

import numpy as np
import pytest

from pu_toolbox.estimators.risk.kldce import _rbf_kernel
from pu_toolbox.utils.basis import build_rbf_basis, rbf_weights


@pytest.mark.unit
def test_basic_rbf_formula_single_source():
    """kldce/pen_l1/kernel_mean 的 RBF 公式必须与 utils.basis 一致."""
    rng = np.random.RandomState(0)
    X = rng.randn(20, 3)
    Z = rng.randn(5, 3)
    sigma = 0.7
    expected = build_rbf_basis(X, Z, sigma)
    assert np.allclose(_rbf_kernel(X, Z, sigma), expected, atol=1e-12)


@pytest.mark.unit
def test_param_rbf_consistency_across_widths():
    """不同 kernel_width 与不同形状下公式仍与 utils.basis 一致."""
    rng = np.random.RandomState(42)
    X = rng.randn(7, 4)
    Z = rng.randn(3, 4)
    for width in (0.3, 1.0, 2.5):
        expected = build_rbf_basis(X, Z, width)
        assert np.allclose(_rbf_kernel(X, Z, width), expected, atol=1e-12)


@pytest.mark.unit
def test_edge_rbf_single_sample_and_identity():
    """退化形状: 单样本、单中心、零距离(X==Z)时仍一致."""
    single = np.array([[0.5]])
    assert np.allclose(
        _rbf_kernel(single, single, 1.0),
        build_rbf_basis(single, single, 1.0),
        atol=1e-12,
    )

    X = np.array([[0.5], [1.5], [2.5]])
    K = _rbf_kernel(X, X, 1.0)
    # X == Z 时零距离对角项必须精确为 1.0 (exp(0))
    assert np.all(np.diag(K) == 1.0)
    assert np.allclose(K, build_rbf_basis(X, X, 1.0), atol=1e-12)


@pytest.mark.unit
def test_determ_rbf_reproducible():
    """同一输入重复构建必须逐位可复现."""
    rng = np.random.RandomState(7)
    X = rng.randn(10, 2)
    Z = rng.randn(4, 2)
    first = build_rbf_basis(X, Z, 0.9)
    second = build_rbf_basis(X, Z, 0.9)
    assert np.array_equal(first, second)


@pytest.mark.unit
def test_param_rbf_weights_matches_basis_matrix_column():
    """rbf_weights(X, w) == build_rbf_basis(X, zeros((1, d)), w)[:, 0]."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((7, 3))
    w = 1.7
    expected = build_rbf_basis(X, np.zeros((1, 3)), w)[:, 0]
    np.testing.assert_allclose(rbf_weights(X, w), expected, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rows,cols", [(0, 5), (3, 0), (3, 3), (7, 5), (8, 8)])
def test_param_kldce_tiles_preserve_full_kernel_dtype_and_inputs(monkeypatch, dtype, rows, cols):
    from pu_toolbox.estimators.risk import kldce
    from pu_toolbox.utils import basis

    rng = np.random.RandomState(91)
    x = rng.normal(size=(rows, 8)).astype(dtype)[:, ::2]
    z = rng.normal(size=(cols, 8)).astype(dtype)[:, ::2]
    before_x, before_z = x.copy(), z.copy()
    expected = build_rbf_basis(x, z, 1.7)
    calls = []

    def record(left, right, width):
        calls.append(len(left))
        assert right is z  # every tile retains every center
        return build_rbf_basis(left, right, width)

    monkeypatch.setattr(kldce, "_RBF_BLOCK_ROWS", 3)
    monkeypatch.setattr(basis, "build_rbf_basis", record)
    actual = kldce._rbf_kernel(x, z, 1.7)
    assert max(calls) <= 3 and sum(calls) == rows
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(z, before_z)
    np.testing.assert_array_equal(actual, kldce._rbf_kernel(x, z, 1.7))


@pytest.mark.unit
def test_basic_kldce_tiling_preserves_small_fit_and_predictions(monkeypatch):
    from pu_toolbox.estimators.risk import kldce

    rng = np.random.RandomState(19)
    x = np.vstack([rng.normal(1.5, 1.0, (15, 3)), rng.normal(-1.5, 1.0, (30, 3))])
    y = np.r_[np.ones(11, int), np.zeros(34, int)]
    options = dict(
        flip_probability=0.3, sigma=2.0, max_acs_iter=3, max_inner_iter=100, random_state=19
    )
    direct = kldce.KLDCEClassifier(**options).fit(x, y)
    monkeypatch.setattr(kldce, "_RBF_BLOCK_ROWS", 7)
    tiled = kldce.KLDCEClassifier(**options).fit(x, y)
    np.testing.assert_allclose(tiled.alpha_full_, direct.alpha_full_, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        tiled.decision_function(x), direct.decision_function(x), rtol=1e-7, atol=1e-8
    )
