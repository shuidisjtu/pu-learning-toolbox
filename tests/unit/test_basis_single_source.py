# ruff: noqa: N806

"""Single-source RBF kernel formula consistency tests.

Guards the architecture-decay fix: every RBF kernel formula in the
codebase must agree with the canonical ``utils.basis.build_rbf_basis``.
"""

import numpy as np
import pytest

from pu_toolbox.estimators.risk.kldce import _rbf_kernel
from pu_toolbox.utils.basis import build_rbf_basis


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
