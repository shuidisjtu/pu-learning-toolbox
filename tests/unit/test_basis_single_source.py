# ruff: noqa: N806

"""Single-source RBF kernel formula consistency tests.

Guards the architecture-decay fix: every RBF kernel formula in the
codebase must agree with the canonical ``utils.basis.build_rbf_basis``.
"""

import numpy as np
import pytest

from pu_toolbox.utils.basis import build_rbf_basis


@pytest.mark.unit
def test_rbf_formula_single_source():
    """kldce/pen_l1/kernel_mean 的 RBF 公式必须与 utils.basis 一致."""
    rng = np.random.RandomState(0)
    X = rng.randn(20, 3)
    Z = rng.randn(5, 3)
    sigma = 0.7
    expected = build_rbf_basis(X, Z, sigma)
    from pu_toolbox.estimators.risk.kldce import _rbf_kernel

    assert np.allclose(_rbf_kernel(X, Z, sigma), expected, atol=1e-12)
