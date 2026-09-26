# ruff: noqa: N803, N806, S101, E501

"""UPUClassifier 的 TS-OS 训练视图行为（协议 §2.3）。

校准后的视图把承担"无标签风险角色"的集合由 ``X_U`` 换成 ``X_U ∪ X_P``：
无标签经验项的平均在全部训练行上进行。正例项、class prior 与 RBF 中心
**数量**都不变；RBF 中心**候选池**跟随视图。

数值正确性由独立重推的 squared 闭式 golden 负责；其余测试只观察结构
（solver 收到的集合与分母），不重复验证公式。
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.estimators.risk.upu import UPUClassifier

pytestmark = pytest.mark.unit

_SOLVERS = {
    "squared": "_fit_squared",
    "logistic": "_fit_logistic",
    "double_hinge": "_fit_double_hinge",
}


def _golden_data():
    """d=1、2 个标注正例、3 个无标签样本，取值可手算。"""
    X = np.array([[1.0], [2.0], [-1.0], [0.0], [-2.0]])
    y_pu = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    return X, y_pu


def _separable_data(seed: int = 42, n_p: int = 60, n_u: int = 120):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.uniform(0.1, 1.0, (n_p, 1)), rng.uniform(-1.1, -0.1, (n_u, 1))])
    y_pu = np.concatenate([np.ones(n_p), np.zeros(n_u)])
    return X, y_pu


def _squared_closed_form(X_P, X_loss_unlabeled, n_P, pi, reg_lambda):
    """按论文公式独立重推 squared 闭式解。

    只用原始数组与 ``numpy``：不得调用生产 ``_phi`` / ``_fit_squared``，
    否则就是把实现和它自己比对。``basis="linear"`` 时 φ(x) = x。
    """
    n_U = X_loss_unlabeled.shape[0]
    n_basis = X_P.shape[1]
    H = (X_loss_unlabeled.T @ X_loss_unlabeled) / (2.0 * n_U) + reg_lambda * np.eye(n_basis)
    h = (pi / n_P) * X_P.sum(axis=0) - (1.0 / (2.0 * n_U)) * X_loss_unlabeled.sum(axis=0)
    return np.linalg.solve(H, h)


def _squared_kwargs(**overrides):
    kwargs = dict(
        class_prior=0.4, loss="squared", reg_lambda=0.1, fit_intercept=False, random_state=0
    )
    kwargs.update(overrides)
    return kwargs


def _rbf_data():
    """两个标注正例远离三个无标签行，便于确定性判断中心来源。"""
    X = np.array([[10.0], [11.0], [-1.0], [0.0], [1.0]])
    y_pu = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    return X, y_pu


class TestViewResolution:
    def test_basic_default_view_matches_explicit_os(self):
        """不传 os_or_ts 与显式 os 必须逐位相同。"""
        X, y_pu = _separable_data()
        default = UPUClassifier(**_squared_kwargs()).fit(X, y_pu)
        explicit = UPUClassifier(**_squared_kwargs()).fit(X, y_pu, os_or_ts="os")
        np.testing.assert_array_equal(default.coef_, explicit.coef_)

    @pytest.mark.parametrize("bad", ["TS", "case_control", "", "ts "])
    def test_param_invalid_view_value_rejected(self, bad):
        X, y_pu = _separable_data()
        with pytest.raises(ValueError, match="os_or_ts"):
            UPUClassifier(**_squared_kwargs()).fit(X, y_pu, os_or_ts=bad)


class TestSquaredGolden:
    def test_basic_os_squared_matches_independent_golden(self):
        X, y_pu = _golden_data()
        clf = UPUClassifier(**_squared_kwargs()).fit(X, y_pu, os_or_ts="os")
        expected = _squared_closed_form(X[y_pu == 1], X[y_pu == 0], 2, 0.4, 0.1)
        np.testing.assert_allclose(clf.coef_, expected, rtol=1e-12)
        # 手算值：H = (1+0+4)/6 + 0.1 = 14/15，h = 0.6 + 0.5 = 11/10 ⇒ 33/28
        np.testing.assert_allclose(clf.coef_, np.array([33.0 / 28.0]), rtol=1e-12)

    def test_basic_ts_squared_matches_independent_golden(self):
        X, y_pu = _golden_data()
        clf = UPUClassifier(**_squared_kwargs()).fit(X, y_pu, os_or_ts="ts")
        expected = _squared_closed_form(X[y_pu == 1], X, 2, 0.4, 0.1)
        np.testing.assert_allclose(clf.coef_, expected, rtol=1e-12)
        # 手算值：H = 10/10 + 0.1 = 11/10，h = 0.6 - 0 = 3/5 ⇒ 6/11
        np.testing.assert_allclose(clf.coef_, np.array([6.0 / 11.0]), rtol=1e-12)

    def test_basic_ts_changes_the_fitted_coefficients(self):
        """路径证据：ts 与 os 不能得出同一组系数。"""
        X, y_pu = _golden_data()
        os_coef = UPUClassifier(**_squared_kwargs()).fit(X, y_pu, os_or_ts="os").coef_
        ts_coef = UPUClassifier(**_squared_kwargs()).fit(X, y_pu, os_or_ts="ts").coef_
        assert not np.allclose(os_coef, ts_coef)

    def test_determ_repeated_ts_fits_are_identical(self):
        X, y_pu = _separable_data()
        first = UPUClassifier(**_squared_kwargs()).fit(X, y_pu, os_or_ts="ts").coef_
        second = UPUClassifier(**_squared_kwargs()).fit(X, y_pu, os_or_ts="ts").coef_
        np.testing.assert_array_equal(first, second)


class TestSolverPoolAndDenominator:
    @pytest.mark.parametrize("loss", ["squared", "logistic", "double_hinge"])
    @pytest.mark.parametrize("view", ["os", "ts"])
    def test_param_every_loss_receives_the_view_pool(self, monkeypatch, loss, view):
        """OS 用 X_U 的基数，TS 用全训练集的基数；三条 solver 路径一致。"""
        X, y_pu = _separable_data()
        n_u = int((y_pu == 0).sum())
        expected = X.shape[0] if view == "ts" else n_u
        seen = {}
        original = getattr(UPUClassifier, _SOLVERS[loss])

        def spy(self, Phi_P, Phi_U, n_P, n_unlabeled_loss, pi):
            seen["rows"] = Phi_U.shape[0]
            seen["denominator"] = n_unlabeled_loss
            return original(self, Phi_P, Phi_U, n_P, n_unlabeled_loss, pi)

        monkeypatch.setattr(UPUClassifier, _SOLVERS[loss], spy)
        kwargs = _squared_kwargs(loss=loss, reg_lambda=10.0, max_iter=500)
        UPUClassifier(**kwargs).fit(X, y_pu, os_or_ts=view)

        assert seen["rows"] == expected
        assert seen["denominator"] == expected

    @pytest.mark.parametrize("loss", ["squared", "logistic", "double_hinge"])
    def test_param_every_loss_fits_finite_parameters_under_ts(self, loss):
        X, y_pu = _separable_data()
        clf = UPUClassifier(**_squared_kwargs(loss=loss, reg_lambda=10.0, max_iter=500))
        clf.fit(X, y_pu, os_or_ts="ts")
        assert np.all(np.isfinite(clf.coef_))

    def test_edge_a_single_unlabeled_row_still_uses_the_union_denominator(self, monkeypatch):
        """n_U = 1 的极端情形：分母是 n_P + 1，不是 1。"""
        X = np.array([[1.0], [2.0], [-3.0]])
        y_pu = np.array([1.0, 1.0, 0.0])
        seen = {}
        original = UPUClassifier._fit_squared

        def spy(self, Phi_P, Phi_U, n_P, n_unlabeled_loss, pi):
            seen["denominator"] = n_unlabeled_loss
            return original(self, Phi_P, Phi_U, n_P, n_unlabeled_loss, pi)

        monkeypatch.setattr(UPUClassifier, "_fit_squared", spy)
        UPUClassifier(**_squared_kwargs()).fit(X, y_pu, os_or_ts="ts")
        assert seen["denominator"] == 3


class TestRBFCentrePool:
    def test_basic_ts_rbf_centers_come_from_the_union_pool(self):
        X, y_pu = _rbf_data()
        clf = UPUClassifier(
            **_squared_kwargs(basis="rbf", kernel_width=1.0, n_centers=X.shape[0])
        ).fit(X, y_pu, os_or_ts="ts")
        np.testing.assert_allclose(np.sort(clf._centers_.ravel()), np.sort(X.ravel()))

    def test_basic_os_rbf_centers_come_from_the_unlabeled_pool_only(self):
        X, y_pu = _rbf_data()
        clf = UPUClassifier(
            **_squared_kwargs(basis="rbf", kernel_width=1.0, n_centers=X.shape[0])
        ).fit(X, y_pu, os_or_ts="os")
        np.testing.assert_allclose(np.sort(clf._centers_.ravel()), np.sort(X[y_pu == 0].ravel()))

    def test_basic_default_rbf_center_count_is_view_independent(self):
        """中心数量由校准前的 n_U 推出：两个视图的中心数（与特征维度）一致。"""
        X, y_pu = _rbf_data()
        kwargs = _squared_kwargs(basis="rbf", kernel_width=1.0)
        os_clf = UPUClassifier(**kwargs).fit(X, y_pu, os_or_ts="os")
        ts_clf = UPUClassifier(**kwargs).fit(X, y_pu, os_or_ts="ts")
        assert os_clf._centers_.shape == ts_clf._centers_.shape
        assert os_clf._n_basis_ == ts_clf._n_basis_
        assert os_clf._n_basis_ == int((y_pu == 0).sum())
