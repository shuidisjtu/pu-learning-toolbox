# ruff: noqa: N802, N803, N806, E501

"""Tests for UPUClassifier and UPULoss — convex PU learning.

Covers method card §9: separable boundary, class-prior sensitivity,
API contract, error handling, convergence sanity, and solver metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.exceptions import NotFittedError, ValidationError
from pu_toolbox.estimators.risk.upu import UPUClassifier, _pu_validation_risk
from pu_toolbox.losses.upu import UPULoss

# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _make_separable_data(rng: np.random.RandomState, n_p: int = 200, n_u: int = 400):
    """§9.1: p(x|y=+1)=Uniform(0.1,1.0), p(x|y=-1)=Uniform(-1.1,-0.1)."""
    X_p = rng.uniform(0.1, 1.0, (n_p, 1))
    X_n = rng.uniform(-1.1, -0.1, (n_u, 1))
    X = np.vstack([X_p, X_n])
    y_pu = np.concatenate([np.ones(n_p), np.zeros(n_u)])
    return X, y_pu


# ═════════════════════════════════════════════════════════════════════
# UPULoss
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestUPULoss:
    """Method card §8.4 — risk computation and gradients."""

    def test_basic_risk_and_prior_effect(self):
        """Well-separated scores → lower risk; higher π → lower risk."""
        loss = UPULoss("logistic")
        r_good = loss(np.array([2.0, 3.0]), np.array([-5.0, -4.0]), class_prior=0.5)
        r_bad = loss(np.array([0.0, 0.0]), np.array([0.0, 0.0]), class_prior=0.5)
        assert r_good < r_bad

        r_lo = loss(np.array([2.0, 2.0]), np.array([-1.0, -1.0]), class_prior=0.3)
        r_hi = loss(np.array([2.0, 2.0]), np.array([-1.0, -1.0]), class_prior=0.7)
        assert r_lo > r_hi

    def test_param_invalid_args_and_gradient(self):
        """Invalid loss/prior/empty input → ValueError; gradient matches finite diff."""
        with pytest.raises(ValueError):
            UPULoss("double_hinge")
        loss = UPULoss("logistic")
        for bad_prior in (0.0, 1.0, -0.5):
            with pytest.raises(ValueError, match="class_prior"):
                loss(np.array([1.0]), np.array([0.0]), class_prior=bad_prior)
        with pytest.raises(ValueError, match="positive_scores"):
            loss(np.array([]), np.array([0.0]), class_prior=0.5)
        with pytest.raises(ValueError, match="unlabeled_scores"):
            loss(np.array([0.0]), np.array([]), class_prior=0.5)

        # Squared-loss analytical gradient vs finite differences
        loss_sq = UPULoss("squared")
        p, u, pi = np.array([1.0, -1.0]), np.array([0.5, -0.5, 0.0]), 0.5
        eps = 1e-6
        dP, dU = loss_sq.gradient(p, u, class_prior=pi)
        assert dP.shape == (2,) and dU.shape == (3,)
        f_plus = loss_sq(np.array([p[0] + eps, p[1]]), u, class_prior=pi)
        f_minus = loss_sq(np.array([p[0] - eps, p[1]]), u, class_prior=pi)
        assert abs(dP[0] - (f_plus - f_minus) / (2 * eps)) < 1e-6


# ═════════════════════════════════════════════════════════════════════
# UPUClassifier — fit / predict / metadata
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestUPUClassifier:
    """Core API contract and metadata."""

    def test_basic_metadata_and_api(self):
        clf = UPUClassifier(class_prior=0.5)
        assert clf.loss == "double_hinge"
        assert clf.family.value == "risk_estimation"
        assert clf.requires_class_prior is True
        assert clf.get_pu_metadata()["is_fitted"] is False
        assert clf.get_params()["loss"] == "double_hinge"
        clf.set_params(reg_lambda=0.1)
        assert clf.reg_lambda == 0.1

    def test_basic_fit_and_predict(self, rng):
        """fit() sets all expected attributes with correct shapes."""
        X, y_pu = _make_separable_data(rng)
        kw = dict(class_prior=0.5, loss="logistic", reg_lambda=1.0, max_iter=500, random_state=42)
        clf = UPUClassifier(**kw)
        clf.fit(X, y_pu)
        assert clf._is_fitted
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))
        pred = clf.predict(X)
        assert pred.shape == (X.shape[0],) and set(np.unique(pred)).issubset({0, 1})
        scores = clf.decision_function(X)
        assert scores.shape == (X.shape[0],)
        np.testing.assert_array_equal(clf.score_samples(X), scores)
        # Class prior overridable in fit
        clf2 = UPUClassifier(**kw)
        clf2.fit(X, y_pu, class_prior=0.6)
        assert clf2._class_prior == 0.6
        # Metadata
        meta = clf.get_pu_metadata()
        assert meta["is_fitted"] is True and meta["loss"] == "logistic" and meta["n_basis"] == 1

    @pytest.mark.parametrize("loss_name", ["double_hinge", "logistic", "squared"])
    def test_param_all_loss_variants_run(self, rng, loss_name):
        """Every loss variant fits and predicts on synthetic data."""
        X, y_pu = _make_separable_data(rng)
        kw = dict(class_prior=0.5, loss=loss_name, reg_lambda=10.0, max_iter=500, random_state=42)
        if loss_name == "squared":
            kw["fit_intercept"] = False
        clf = UPUClassifier(**kw)
        clf.fit(X, y_pu)
        assert clf.coef_ is not None
        risk = clf.pu_validation_risk(X, y_pu)
        assert np.isfinite(risk)


# ═════════════════════════════════════════════════════════════════════
# Separable boundary (§9.1)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSeparableBoundary:
    """Method card §9.1: recovering the true decision boundary."""

    def test_deterministic_squared_recovers_boundary(self, rng):
        """Squared loss closed-form: boundary ∈ [-0.15, 0.15]."""
        X, y_pu = _make_separable_data(rng, n_p=200, n_u=400)
        clf = UPUClassifier(
            class_prior=0.5, loss="squared", fit_intercept=False,
            reg_lambda=1.0, random_state=42,
        )
        clf.fit(X, y_pu)
        boundary = -clf.intercept_ / clf.coef_[0]
        assert abs(boundary) <= 0.15, f"boundary={boundary:.4f}"

    @pytest.mark.parametrize("loss_name", ["logistic", "double_hinge"])
    def test_param_pu_cv_finds_working_lambda(self, rng, loss_name):
        """PU-CV selects λ that yields >80 % P-accuracy."""
        X, y_pu = _make_separable_data(rng, n_p=300, n_u=600)
        lambda_grid = np.logspace(-2, 2, 10) if loss_name == "logistic" else np.logspace(-1, 2, 7)

        # 3-fold PU-CV
        mP = y_pu == 1
        iP, iU = np.where(mP)[0], np.where(~mP)[0]
        rng.shuffle(iP)
        rng.shuffle(iU)
        fP = np.array_split(iP, 3)
        fU = np.array_split(iU, 3)
        pi = 0.5
        best_lam, best_risk = lambda_grid[0], np.inf
        for lam in lambda_grid:
            fold_risks = []
            for k in range(3):
                tr_P = np.concatenate([fP[i] for i in range(3) if i != k])
                tr_U = np.concatenate([fU[i] for i in range(3) if i != k])
                X_tr = np.vstack([X[tr_P], X[tr_U]])
                y_tr = np.concatenate([np.ones(len(tr_P)), np.zeros(len(tr_U))])
                clf = UPUClassifier(
                    class_prior=pi, loss=loss_name, reg_lambda=lam,
                    max_iter=500, random_state=42,
                )
                clf.fit(X_tr, y_tr)
                r = _pu_validation_risk(
                    clf.decision_function(X[fP[k]]),
                    clf.decision_function(X[fU[k]]), pi,
                )
                fold_risks.append(r)
            if (m := float(np.mean(fold_risks))) < best_risk:
                best_risk, best_lam = m, lam

        clf = UPUClassifier(
            class_prior=pi, loss=loss_name, reg_lambda=best_lam,
            max_iter=1000, random_state=42,
        )
        clf.fit(X, y_pu)
        acc_P = np.mean(clf.predict(X)[y_pu == 1] == 1)
        assert acc_P >= 0.8, f"{loss_name}: P-acc={acc_P:.3f} with λ={best_lam}"


# ═════════════════════════════════════════════════════════════════════
# Error handling & boundary conditions (§9.6)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestErrorHandling:
    """Method card §9.6 — API contract violations."""

    def test_param_not_fitted_raises(self):
        clf = UPUClassifier(class_prior=0.5)
        with pytest.raises(NotFittedError):
            clf.predict(np.array([[0.5]]))
        with pytest.raises(NotFittedError):
            clf.decision_function(np.array([[0.5]]))

    @pytest.mark.parametrize(
        "clf, match",
        [
            (UPUClassifier(class_prior=0.0, reg_lambda=1.0), "class_prior"),
            (UPUClassifier(class_prior=1.0, reg_lambda=1.0), "class_prior"),
            (UPUClassifier(class_prior=0.5, reg_lambda=0.0), "reg_lambda"),
            (UPUClassifier(class_prior=0.5, loss="hinge", reg_lambda=1.0), "Unknown loss"),
        ],
    )
    def test_param_invalid_params_raise(self, rng, clf, match):
        X, y_pu = _make_separable_data(rng)
        with pytest.raises(ValueError, match=match):
            clf.fit(X, y_pu)

    def test_edge_rbf_and_nan_raises(self, rng):
        """RBF without kernel_width, NaN in X, no positives, predict_proba."""
        X, y_pu = _make_separable_data(rng)
        # RBF without kernel_width
        clf_rbf = UPUClassifier(class_prior=0.5, basis="rbf")
        with pytest.raises(ValueError, match="kernel_width"):
            clf_rbf.fit(X, y_pu)
        # NaN in X
        X_nan, y_nan = _make_separable_data(rng, n_p=10, n_u=20)
        X_nan[0, 0] = np.nan
        clf_nan = UPUClassifier(class_prior=0.5, reg_lambda=1.0)
        with pytest.raises(ValueError, match="NaN"):
            clf_nan.fit(X_nan, y_nan)
        # No positives
        with pytest.raises(ValidationError):
            UPUClassifier(class_prior=0.5).fit(np.array([[0.5], [0.6]]), np.array([0, 0]))
        # predict_proba not implemented
        clf = UPUClassifier(class_prior=0.5, loss="logistic", reg_lambda=1.0, max_iter=500, random_state=42)
        clf.fit(X, y_pu)
        with pytest.raises(NotImplementedError):
            clf.predict_proba(X)


# ═════════════════════════════════════════════════════════════════════
# RBF basis
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestRBFBasis:
    def test_basic_rbf_fit_and_coef_shape(self, rng):
        X, y_pu = _make_separable_data(rng, n_p=100, n_u=200)
        clf = UPUClassifier(
            class_prior=0.5, loss="squared", fit_intercept=False, basis="rbf",
            kernel_width=0.5, n_centers=50, random_state=42,
        )
        clf.fit(X, y_pu)
        assert clf._is_fitted and clf.coef_.shape == (50,)
        assert len(clf.predict(X)) == X.shape[0]


# ═════════════════════════════════════════════════════════════════════
# Convergence & sensitivity (§9.4 / §9.5)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestConvergenceAndSensitivity:
    """Larger dataset → better risk; accuracy peaks near true class prior."""

    def test_basic_more_data_improves_risk(self, rng):
        """Larger dataset → not-worse PU validation risk."""
        risks = []
        for n in (100, 200):
            X, y_pu = _make_separable_data(rng, n_p=n // 2, n_u=n // 2)
            clf = UPUClassifier(
                class_prior=0.5, loss="squared", fit_intercept=False,
                reg_lambda=1.0, random_state=42,
            )
            clf.fit(X, y_pu)
            risks.append(clf.pu_validation_risk(X, y_pu))
        assert risks[0] >= risks[1] - 0.1

    def test_deterministic_class_prior_sensitivity(self, rng):
        """Accuracy peaks near true π=0.5; degrades with perturbation."""
        X, y_pu = _make_separable_data(rng, n_p=200, n_u=400)
        y_true = (y_pu == 1).astype(int)
        accs = {}
        for delta in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            pi = 0.5 + delta
            if not (0.0 < pi < 1.0):
                continue
            clf = UPUClassifier(
                class_prior=pi, loss="squared", fit_intercept=False,
                reg_lambda=1.0, random_state=42,
            )
            clf.fit(X, y_pu)
            accs[delta] = float(np.mean(clf.predict(X) == y_true))
        assert accs[0.0] >= max(accs.values()) - 0.05


# ═════════════════════════════════════════════════════════════════════
# _pu_validation_risk unit tests
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestValidationRisk:
    """Eq. (2): R = 2π·f_n + f_pu − π."""

    def test_basic_validation_risk_formula(self):
        """Hand-computed expected values."""
        # Perfect: all P>0, all U≤0 → f_n=0, f_pu=0 → R = −π
        assert _pu_validation_risk(np.array([1.0, 2.0]), np.array([-1.0, -2.0]), 0.5) == -0.5
        # Random: f_n=0.5, f_pu=0.5 → R = 2·0.5·0.5 + 0.5 − 0.5 = 0.5
        assert abs(_pu_validation_risk(np.array([1.0, -1.0]), np.array([1.0, -1.0]), 0.5) - 0.5) < 1e-9
        # Empty → inf
        assert _pu_validation_risk(np.array([]), np.array([0.0]), 0.5) == np.inf
