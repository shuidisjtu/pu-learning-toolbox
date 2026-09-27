# ruff: noqa: N803, N806, S101, E501

"""PUSBKernelClassifier 视图不变量的结构与接口测试（协议 §2.3）。

校准只允许改变"承担无标签风险角色"的集合。本文件锁定**不被允许变化**的那些量：
RBF 中心的候选池与数量、CV fold、距离矩阵行数、先验分位数阈值池，以及公共接口
（默认视图、非法值、sample_weight 行为）。同时锁定"角色重合但物理样本不复制"
——并集是角色指派，不是把训练矩阵复制一份。

公式本身的正确性由 test_pusb_kernel_ts_view.py 的独立 golden 负责，本文件只
观察结构与接口。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pu_toolbox.estimators.bias_aware import PUSBKernelClassifier
from pu_toolbox.estimators.bias_aware import pusb_kernel as pusb_kernel_module

pytestmark = pytest.mark.unit

_CLASS_PRIOR = 0.4
_CV = 3
_GRID_POINTS = 4  # 2 sigmas x 2 regs


def _data(seed=7):
    rng = np.random.RandomState(seed)
    positive = rng.normal(1.0, 0.5, size=(18, 3))
    unlabeled = rng.normal(-0.2, 0.9, size=(30, 3))
    X = np.vstack((positive, unlabeled))
    y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(unlabeled), dtype=int)]
    return X, y


def _small_model(random_state=11, **overrides):
    params = {
        "n_basis": 8,
        "cv": _CV,
        "sigma_grid": (0.5, 1.5),
        "reg_grid": (0.01, 0.1),
        "random_state": random_state,
        "max_iter": 80,
    }
    params.update(overrides)
    return PUSBKernelClassifier(**params)


def _fit(path):
    X, y = _data()
    model = _small_model().fit(X, y, class_prior=_CLASS_PRIOR, os_or_ts=path)
    return model, X, y


class TestViewInvariants:
    def test_basic_centres_folds_and_grid_are_view_independent(self):
        os_model, _, _ = _fit("os")
        ts_model, _, _ = _fit("ts")

        assert np.array_equal(os_model.center_indices_, ts_model.center_indices_)
        assert_allclose(os_model.centers_, ts_model.centers_)
        assert os_model.centers_.shape == ts_model.centers_.shape
        assert os_model.centers_.shape[0] == os_model.n_basis
        assert np.array_equal(os_model.fold_ids_, ts_model.fold_ids_)
        assert os_model.cv_scores_.shape == ts_model.cv_scores_.shape

    def test_basic_oversized_n_basis_is_capped_by_the_row_count(self):
        X, y = _data()

        os_model = _small_model(n_basis=1000).fit(X, y, class_prior=_CLASS_PRIOR)
        ts_model = _small_model(n_basis=1000).fit(X, y, class_prior=_CLASS_PRIOR, os_or_ts="ts")

        assert os_model.centers_.shape[0] == len(X)
        assert ts_model.centers_.shape[0] == len(X)

    def test_basic_design_row_count_never_grows_with_the_view(self):
        """The distance matrix is built once, from X -- no row is appended."""
        seen = []
        original = pusb_kernel_module._rbf_design

        def spy(distances, sigma):
            seen.append(distances.shape[0])
            return original(distances, sigma)

        for path in ("os", "ts"):
            seen.clear()
            X, y = _data()
            with pytest.MonkeyPatch.context() as patcher:
                patcher.setattr(pusb_kernel_module, "_rbf_design", spy)
                _small_model().fit(X, y, class_prior=_CLASS_PRIOR, os_or_ts=path)
            assert seen, f"no design was built for {path}"
            assert set(seen) == {len(X)}

    def test_edge_ts_roles_overlap_without_duplicating_training_rows(self):
        roles = []
        fits = []
        original_roles = pusb_kernel_module._risk_role_designs
        original_fit = pusb_kernel_module._fit_coefficients

        def roles_spy(design, y_pu, *, include_positive_in_unlabeled):
            positive, unlabeled = original_roles(
                design, y_pu, include_positive_in_unlabeled=include_positive_in_unlabeled
            )
            roles.append(
                {
                    "design": design,
                    "y": y_pu,
                    "positive": positive,
                    "unlabeled": unlabeled,
                    "include_positive": include_positive_in_unlabeled,
                }
            )
            return positive, unlabeled

        def fit_spy(positive, unlabeled, class_prior, reg_lambda, max_iter, tol):
            fits.append({"positive": positive, "unlabeled": unlabeled})
            return original_fit(positive, unlabeled, class_prior, reg_lambda, max_iter, tol)

        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(pusb_kernel_module, "_risk_role_designs", roles_spy)
            patcher.setattr(pusb_kernel_module, "_fit_coefficients", fit_spy)
            X, y = _data()
            _small_model().fit(X, y, class_prior=_CLASS_PRIOR, os_or_ts="ts")

        assert roles and len(roles) == len(fits) == 1 + _CV * _GRID_POINTS
        for role, fit in zip(roles, fits, strict=True):
            # The role arrays reach the solver unchanged.
            assert fit["positive"] is role["positive"]
            assert fit["unlabeled"] is role["unlabeled"]
            assert role["include_positive"] is True
            # The unlabeled role is the training design itself: same rows, same
            # length, nothing concatenated on.
            assert fit["unlabeled"].shape == role["design"].shape
            assert np.array_equal(fit["unlabeled"], role["design"])
            assert fit["positive"].shape[0] == int((role["y"] == 1).sum())
            # Every positive row also carries the unlabeled-risk role.
            for row in fit["positive"]:
                assert any(np.array_equal(row, other) for other in fit["unlabeled"])

    @pytest.mark.parametrize("path", ["os", "ts"])
    def test_basic_threshold_pool_keeps_the_full_training_size(self, path):
        calls = []
        original = pusb_kernel_module.prior_quantile_predict

        def spy(scores, class_prior):
            calls.append((scores.size, class_prior))
            return original(scores, class_prior)

        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(pusb_kernel_module, "prior_quantile_predict", spy)
            X, y = _data()
            _small_model().fit(X, y, class_prior=_CLASS_PRIOR, os_or_ts=path)

        assert calls, f"no threshold was derived for {path}"
        assert [size for size, _ in calls] == [len(X)]
        assert [pi for _, pi in calls] == [_CLASS_PRIOR]

    def test_basic_default_view_matches_explicit_os(self):
        X, y = _data()
        default = _small_model().fit(X, y, class_prior=_CLASS_PRIOR)
        explicit = _small_model().fit(X, y, class_prior=_CLASS_PRIOR, os_or_ts="os")

        assert np.array_equal(default.coef_, explicit.coef_)
        assert default.threshold_ == explicit.threshold_
        assert np.array_equal(default.cv_scores_, explicit.cv_scores_)

    def test_basic_ts_changes_the_fitted_solution(self):
        os_model, X, _ = _fit("os")
        ts_model, _, _ = _fit("ts")

        # Not a no-op: the calibrated risk fits different coefficients, while
        # the grid it was chosen from is the same one.
        assert os_model.cv_scores_.shape == ts_model.cv_scores_.shape
        assert not np.array_equal(os_model.coef_, ts_model.coef_)
        assert not np.array_equal(os_model.decision_function(X), ts_model.decision_function(X))

    @pytest.mark.parametrize("bad", ["TS", ""])
    def test_param_invalid_view_value_is_rejected(self, bad):
        X, y = _data()
        with pytest.raises(ValueError, match="os_or_ts must be"):
            _small_model().fit(X, y, class_prior=_CLASS_PRIOR, os_or_ts=bad)

    def test_param_sample_weight_is_still_rejected_under_both_views(self):
        X, y = _data()
        for path in ("os", "ts"):
            with pytest.raises(NotImplementedError):
                _small_model().fit(
                    X,
                    y,
                    class_prior=_CLASS_PRIOR,
                    sample_weight=np.ones(len(X)),
                    os_or_ts=path,
                )


class TestSelectionConsistency:
    @pytest.mark.parametrize("path", ["os", "ts"])
    def test_determ_selection_matches_the_stored_cv_scores(self, path):
        """The chosen (sigma, lambda) comes from this view's own scores."""
        model, _, _ = _fit(path)

        best = np.unravel_index(np.argmin(model.cv_scores_), model.cv_scores_.shape)

        assert model.sigma_ == model.sigma_grid[best[0]]
        assert model.reg_lambda_ == model.reg_grid[best[1]]

    def test_determ_validation_roles_are_identical_across_views(self):
        """The validation objective has the same definition in both views."""
        assert _validation_roles("os") == _validation_roles("ts")
        assert len(_validation_roles("ts")) == _CV * _GRID_POINTS


def _validation_roles(path):
    """Role shapes of every validation-objective call, sorted."""
    seen = []
    original = pusb_kernel_module._pu_objective_and_gradient

    def spy(coef, positive, unlabeled, class_prior, reg_lambda):
        seen.append((positive.shape[0], unlabeled.shape[0]))
        return original(coef, positive, unlabeled, class_prior, reg_lambda)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(pusb_kernel_module, "_pu_objective_and_gradient", spy)
        X, y = _data()
        model = _small_model().fit(X, y, class_prior=_CLASS_PRIOR, os_or_ts=path)

    fold_sizes = set(np.bincount(model.fold_ids_).tolist())
    return sorted(pair for pair in seen if sum(pair) in fold_sizes)
