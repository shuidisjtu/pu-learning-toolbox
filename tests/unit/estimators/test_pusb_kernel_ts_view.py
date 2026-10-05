# ruff: noqa: N803, N806, S101, E501

"""PUSBKernelClassifier 的 TS-OS 训练视图行为（协议 §2.3）。

校准后的视图把承担"无标签风险角色"的集合由 ``X_U`` 换成 ``X_U ∪ X_P``：无标签
经验项的平均在全部训练行上进行。正例项、class prior、RBF 中心候选池、CV fold 与
先验分位数阈值池都不变；**内部 CV 的验证折固定保持 OS**。

数值正确性由独立 golden 负责——期望值是测试内的字面量，由独立脚本用原始数组
重推（不调用生产 helper）；其余测试只观察结构（角色集合、分母、调用形态），
不重复验证公式。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pu_toolbox.estimators.bias_aware import PUSBKernelClassifier
from pu_toolbox.estimators.bias_aware import pusb_kernel as pusb_kernel_module
from pu_toolbox.estimators.bias_aware.pusb_kernel import _pu_objective_and_gradient

pytestmark = pytest.mark.unit


# ── Frozen OS golden ──────────────────────────────────────────────────
# Captured from HEAD 7c716f9 (pre-refactor) on 2026-09-27 with numpy 2.4.6 /
# scipy 1.17.1.  BFGS output can differ in the last digits across BLAS and
# SciPy versions, so coefficients and CV scores are compared with a tolerance
# that is far tighter than any semantic change (a wrong role set moves them by
# O(1)); the discrete selections are compared exactly, since they are the
# drift-resistant anchors.
_OS_GOLDEN_COEF = np.array(
    [
        10.8253853640781,
        -5.9669002818182149,
        -3.5989254633134622,
        -6.8272398694777934,
        6.6846054991945083,
        3.9674756277945504,
        -4.2673254654701216,
        -6.1351444857780359,
        3.4340170052818308,
    ]
)
_OS_GOLDEN_CV_SCORES = np.array(
    [
        [-0.41037665173133642, 1.7702037074866412],
        [-4.5561474182087096, 0.82367556688691401],
    ]
)
_OS_GOLDEN_THRESHOLD = 6.893410268835854
_OS_GOLDEN_SIGMA = 1.5
_OS_GOLDEN_REG_LAMBDA = 0.01
_OS_GOLDEN_CENTERS = np.array([2, 36, 35, 22, 8, 14, 46, 41])
_OS_GOLDEN_FOLDS = np.array(
    [
        2,
        1,
        0,
        2,
        2,
        1,
        2,
        0,
        2,
        1,
        2,
        1,
        1,
        2,
        1,
        0,
        1,
        0,
        2,
        2,
        2,
        1,
        2,
        0,
        0,
        2,
        0,
        0,
        1,
        1,
        1,
        0,
        2,
        1,
        1,
        1,
        0,
        2,
        2,
        0,
        1,
        1,
        0,
        0,
        0,
        2,
        0,
        0,
    ]
)


# ── Independent risk golden ───────────────────────────────────────────
# Four rows, two columns (kernel column + intercept).  Rows 0-1 are positives,
# rows 2-3 unlabeled, so the union view averages over all four rows.  The
# expected numbers below are literals from an independent re-derivation.
_GOLDEN_DESIGN = np.array([[0.5, 1.0], [1.0, 1.0], [-0.5, 1.0], [0.0, 1.0]])
_GOLDEN_Y = np.array([1, 1, 0, 0])
_GOLDEN_COEF = np.array([2.0, -0.5])
_GOLDEN_PI = 0.3
_GOLDEN_LAMBDA = 0.4
_OS_OBJECTIVE = 0.88774513108142972
_OS_GRADIENT = np.array([0.52939361904841098, -0.22001690369774912])
_TS_OBJECTIVE = 1.3877451310814297
_TS_GRADIENT = np.array([0.83439784497284819, 0.0])


def _data(seed=7):
    rng = np.random.RandomState(seed)
    positive = rng.normal(1.0, 0.5, size=(18, 3))
    unlabeled = rng.normal(-0.2, 0.9, size=(30, 3))
    X = np.vstack((positive, unlabeled))
    y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(unlabeled), dtype=int)]
    return X, y


def _small_model(random_state=11):
    return PUSBKernelClassifier(
        n_basis=8,
        cv=3,
        sigma_grid=(0.5, 1.5),
        reg_grid=(0.01, 0.1),
        random_state=random_state,
        max_iter=80,
    )


def _role_spy(monkeypatch):
    """Record every call to the role helper and hand back the real result."""
    original = getattr(pusb_kernel_module, "_risk_role_designs", None)
    if original is None:
        raise AssertionError("_risk_role_designs is not defined: fit cannot build the TS role set")
    calls = []

    def spy(design, y_pu, *, include_positive_in_unlabeled):
        positive, unlabeled = original(
            design, y_pu, include_positive_in_unlabeled=include_positive_in_unlabeled
        )
        calls.append(
            {
                "include_positive": include_positive_in_unlabeled,
                "design_rows": design.shape[0],
                "positive_rows": positive.shape[0],
                "unlabeled_rows": unlabeled.shape[0],
            }
        )
        return positive, unlabeled

    monkeypatch.setattr(pusb_kernel_module, "_risk_role_designs", spy)
    return calls


class TestRiskGolden:
    def test_basic_os_frozen_golden_is_unchanged(self):
        X, y = _data()
        model = _small_model().fit(X, y, class_prior=0.4)

        assert model.sigma_ == _OS_GOLDEN_SIGMA
        assert model.reg_lambda_ == _OS_GOLDEN_REG_LAMBDA
        assert np.array_equal(model.center_indices_, _OS_GOLDEN_CENTERS)
        assert np.array_equal(model.fold_ids_, _OS_GOLDEN_FOLDS)
        assert_allclose(model.coef_, _OS_GOLDEN_COEF, rtol=1e-6, atol=1e-8)
        assert_allclose(model.cv_scores_, _OS_GOLDEN_CV_SCORES, rtol=1e-6, atol=1e-8)
        assert model.threshold_ == pytest.approx(_OS_GOLDEN_THRESHOLD, rel=1e-6)

    @pytest.mark.math
    def test_math_os_risk_matches_independent_golden(self):
        positive = _GOLDEN_DESIGN[_GOLDEN_Y == 1]
        unlabeled = _GOLDEN_DESIGN[_GOLDEN_Y == 0]

        objective, gradient = _pu_objective_and_gradient(
            _GOLDEN_COEF, positive, unlabeled, _GOLDEN_PI, _GOLDEN_LAMBDA
        )

        assert objective == pytest.approx(_OS_OBJECTIVE, rel=1e-10)
        assert_allclose(gradient, _OS_GRADIENT, rtol=1e-10, atol=1e-12)

    @pytest.mark.math
    def test_math_ts_risk_puts_the_positives_in_both_roles(self):
        positive = _GOLDEN_DESIGN[_GOLDEN_Y == 1]
        unlabeled = _GOLDEN_DESIGN  # the union view hands over every row

        objective, gradient = _pu_objective_and_gradient(
            _GOLDEN_COEF, positive, unlabeled, _GOLDEN_PI, _GOLDEN_LAMBDA
        )

        assert objective == pytest.approx(_TS_OBJECTIVE, rel=1e-10)
        assert_allclose(gradient, _TS_GRADIENT, rtol=1e-10, atol=1e-12)
        # The two views are genuinely different objects on the same input.
        assert objective != pytest.approx(_OS_OBJECTIVE, rel=1e-3)

    @pytest.mark.property
    def test_edge_unlabeled_mean_is_over_the_rows_actually_handed_over(self):
        """A frozen denominator would break mean-invariance on duplicated rows."""
        positive = _GOLDEN_DESIGN[_GOLDEN_Y == 1]
        unlabeled = _GOLDEN_DESIGN[_GOLDEN_Y == 0]

        base, _ = _pu_objective_and_gradient(_GOLDEN_COEF, positive, unlabeled, _GOLDEN_PI, 0.0)
        doubled, _ = _pu_objective_and_gradient(
            _GOLDEN_COEF, positive, np.repeat(unlabeled, 2, axis=0), _GOLDEN_PI, 0.0
        )

        assert doubled == pytest.approx(base, rel=1e-12)

    @pytest.mark.property
    def test_determ_gradients_match_finite_difference_in_both_views(self):
        positive = _GOLDEN_DESIGN[_GOLDEN_Y == 1]
        epsilon = 1e-6

        for unlabeled in (_GOLDEN_DESIGN[_GOLDEN_Y == 0], _GOLDEN_DESIGN):
            _, gradient = _pu_objective_and_gradient(
                _GOLDEN_COEF, positive, unlabeled, _GOLDEN_PI, _GOLDEN_LAMBDA
            )
            numerical = np.empty_like(_GOLDEN_COEF)
            for index in range(len(_GOLDEN_COEF)):
                offset = np.zeros_like(_GOLDEN_COEF)
                offset[index] = epsilon
                upper = _pu_objective_and_gradient(
                    _GOLDEN_COEF + offset, positive, unlabeled, _GOLDEN_PI, _GOLDEN_LAMBDA
                )[0]
                lower = _pu_objective_and_gradient(
                    _GOLDEN_COEF - offset, positive, unlabeled, _GOLDEN_PI, _GOLDEN_LAMBDA
                )[0]
                numerical[index] = (upper - lower) / (2.0 * epsilon)

            assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-9)


class TestViewRoles:
    def test_basic_ts_training_fold_loss_uses_every_training_row(self, monkeypatch):
        X, y = _data()
        calls = _role_spy(monkeypatch)
        model = _small_model().fit(X, y, class_prior=0.4, os_or_ts="ts")

        n_total = len(X)
        fold_sizes = set(np.bincount(model.fold_ids_).tolist())
        training = [call for call in calls if call["design_rows"] != n_total]

        assert fold_sizes == {n_total // model.cv}
        # cv folds x (2 sigmas x 2 regs) fits, each built from its own fold.
        assert len(training) == model.cv * 2 * 2
        for call in training:
            assert call["include_positive"] is True
            assert call["design_rows"] in {n_total - size for size in fold_sizes}
            assert call["unlabeled_rows"] == call["design_rows"]
            assert call["positive_rows"] < call["unlabeled_rows"]

    def test_edge_ts_validation_fold_keeps_only_its_own_unlabeled_rows(self, monkeypatch):
        X, y = _data()
        seen = []
        original = pusb_kernel_module._pu_objective_and_gradient

        def spy(coef, positive, unlabeled, class_prior, reg_lambda):
            seen.append((positive.shape[0], unlabeled.shape[0]))
            return original(coef, positive, unlabeled, class_prior, reg_lambda)

        monkeypatch.setattr(pusb_kernel_module, "_pu_objective_and_gradient", spy)
        model = _small_model().fit(X, y, class_prior=0.4, os_or_ts="ts")

        fold_sizes = set(np.bincount(model.fold_ids_).tolist())
        # A validation call is the only one whose two roles together are
        # exactly one fold; training calls carry a whole training fold.
        validation = [pair for pair in seen if sum(pair) in fold_sizes]
        by_fold = {}
        for fold in np.unique(model.fold_ids_):
            mask = model.fold_ids_ == fold
            by_fold[fold] = (int((y[mask] == 1).sum()), int((y[mask] == 0).sum()))
        expected = sorted(list(by_fold.values()) * 4)

        assert len(validation) == model.cv * 4, "every fold x (sigma, reg) is scored"
        assert sorted(validation) == expected
        # Implied by the equality above, but stated as the boundary itself so a
        # failure names the invariant rather than a mismatched tuple.
        for _positive_rows, unlabeled_rows in validation:
            assert unlabeled_rows < (len(X) // model.cv)

    def test_basic_ts_final_refit_loss_uses_every_row(self, monkeypatch):
        X, y = _data()
        calls = _role_spy(monkeypatch)
        _small_model().fit(X, y, class_prior=0.4, os_or_ts="ts")

        refits = [call for call in calls if call["design_rows"] == len(X)]

        assert len(refits) == 1
        assert refits[0]["include_positive"] is True
        assert refits[0]["unlabeled_rows"] == len(X)
        assert refits[0]["positive_rows"] == int((y == 1).sum())

    @pytest.mark.parametrize(("view", "expected"), [("os", False), ("ts", True)])
    def test_determ_role_helper_receives_an_explicit_boolean(self, monkeypatch, view, expected):
        """The helper never takes the run-level view string (plan §4.2)."""
        X, y = _data()
        calls = _role_spy(monkeypatch)
        _small_model().fit(X, y, class_prior=0.4, os_or_ts=view)

        assert len(calls) == 1 + 3 * 2 * 2  # one refit + cv folds x grid
        assert {call["include_positive"] for call in calls} == {expected}
        assert all(isinstance(call["include_positive"], bool | np.bool_) for call in calls)

    def test_determ_role_helper_never_sees_a_validation_fold_design(self, monkeypatch):
        """The helper is called with training-fold and full designs only.

        That is what "the validation fold does not go through the helper" means
        operationally: a validation fold's own design never reaches it.
        """
        X, y = _data()
        calls = _role_spy(monkeypatch)
        model = _small_model().fit(X, y, class_prior=0.4, os_or_ts="ts")

        fold_size = len(X) // model.cv
        expected_rows = {len(X)} | {len(X) - fold_size}

        assert calls
        assert {call["design_rows"] for call in calls} == expected_rows
