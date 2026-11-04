"""Single-equality box feasibility against an independent LP oracle."""

import numpy as np
import pytest
from scipy.optimize import linprog

from pu_toolbox.estimators.risk.kldce import _find_feasible_init

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("seed", range(8))
def test_basic_random_box_solution_agrees_with_lp_feasibility(seed):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=(1, 31))
    a[0, ::7] = 0
    lower = rng.uniform(-2, 0, 31)
    upper = lower + rng.uniform(0, 4, 31)
    target = float((a @ (lower + rng.uniform(size=31) * (upper - lower)))[0])
    expected = linprog(
        np.zeros(31),
        A_eq=a,
        b_eq=[target],
        bounds=list(zip(lower, upper, strict=True)),
        method="highs",
    )
    assert expected.success
    actual = _find_feasible_init(a, target, lower, upper)
    assert abs(float((a @ actual)[0]) - target) <= 1e-10
    assert np.all(actual >= lower) and np.all(actual <= upper)


@pytest.mark.parametrize("target", [-2.1, 3.1])
def test_param_infeasible_target_rejected(target):
    with pytest.raises(RuntimeError, match="infeasible"):
        _find_feasible_init(np.array([[-1.0, 1.0]]), target, np.zeros(2), np.array([2.0, 3.0]))


@pytest.mark.parametrize("target", [-2.0, 0.0, 3.0])
def test_edge_extreme_vertices_zero_coefficients_and_fixed_variables(target):
    a, lower, upper = np.array([[-1.0, 0.0, 1.0, 3.0]]), np.zeros(4), np.array([2.0, 7.0, 3.0, 0.0])
    result = _find_feasible_init(a, target, lower, upper)
    np.testing.assert_allclose(a @ result, [target], atol=1e-12)
    assert result[3] == 0
    np.testing.assert_array_equal(
        _find_feasible_init(np.zeros((1, 2)), 0.0, np.zeros(2), np.ones(2)), np.zeros(2)
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        _find_feasible_init(np.zeros((1, 2)), 1.0, np.zeros(2), np.ones(2))


@pytest.mark.parametrize(
    "a,lower,upper,target",
    [
        ([[1.0, 2.0], [3.0, 4.0]], [0.0, 0.0], [1.0, 1.0], 0.0),
        ([[float("nan"), 1.0]], [0.0, 0.0], [1.0, 1.0], 0.0),
        ([[1.0, 1.0]], [0.0, 2.0], [1.0, 1.0], 0.0),
        ([[1.0, 1.0]], [0.0, 0.0], [1.0, 1.0], float("inf")),
    ],
)
def test_param_malformed_phase1_contract_rejected(a, lower, upper, target):
    with pytest.raises(ValueError, match="one finite equality"):
        _find_feasible_init(np.asarray(a), target, np.asarray(lower), np.asarray(upper))


def test_determ_large_real_shape_requires_no_lp_and_preserves_inputs(monkeypatch):
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **k: pytest.fail("no LP binding"))
    a = np.r_[np.ones(9000), -np.ones(36000), np.ones(36000)].reshape(1, -1)
    lower, upper = np.zeros(81000), np.r_[np.full(45000, 1 / 45000), np.full(36000, 1 / 90000)]
    first = _find_feasible_init(a, 0.2, lower, upper)
    np.testing.assert_allclose(a @ first, [0.2], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(first, _find_feasible_init(a, 0.2, lower, upper))
    assert not lower.any() and np.all(upper > 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("label_dtype", [np.float32, np.float64, np.int64])
def test_basic_tiled_qp_matches_all_four_dense_blocks_without_full_outer(
    monkeypatch, dtype, label_dtype
):
    from pu_toolbox.estimators.risk import kldce

    rng = np.random.RandomState(12)
    x, n, k = rng.normal(size=(11, 3)).astype(dtype), 11, 4
    labels = np.r_[np.ones(k), -np.ones(n - k)].astype(label_dtype)
    kernel = kldce._rbf_kernel(x, x, 1.0)
    signed = np.outer(labels, labels) * kernel
    expected = (
        np.block([[signed, -signed[:, k:]], [-signed[:, k:].T, signed[k:, k:]]]).astype(float) / 0.6
    )
    monkeypatch.setattr(kldce, "_QP_BLOCK_ROWS", 3)
    monkeypatch.setattr(kldce.np, "outer", lambda *a, **kw: pytest.fail("no dense sign matrix"))
    q, _, _, _, _, _ = kldce._build_dual_qp(np.zeros(3), x, kernel, labels, 0.3, 1.0, n, k, 0.1)
    np.testing.assert_allclose(q, expected, rtol=2e-15, atol=2e-15)
    assert q.dtype == np.float64


def test_edge_overflow_cannot_return_a_nan_feasible_point():
    with pytest.raises(ValueError, match="remain finite"):
        _find_feasible_init(np.array([[1e308]]), 0.0, np.array([-1e308]), np.array([1e308]))
