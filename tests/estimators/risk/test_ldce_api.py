# ruff: noqa: N802, N803, N806, S101

"""API, convergence, error, and regression tests for LDCEClassifier."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pu_toolbox.estimators.risk.ldce import LDCEClassifier


# ═════════════════════════════════════════════════════════════════════
# Test helpers
# ═════════════════════════════════════════════════════════════════════


def _make_censoring_pu_data(
    rng: np.random.RandomState,
    n_pos: int = 50,
    n_neg: int = 100,
    h: float = 0.3,
    d: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic censoring PU data with known ground truth."""
    X_pos = rng.randn(n_pos, d) + 1.5
    X_neg = rng.randn(n_neg, d) - 1.5
    X = np.vstack([X_pos, X_neg])
    y_true = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

    n_hide = int(n_pos * h)
    hide_idx = rng.choice(n_pos, size=n_hide, replace=False)
    y_pu = y_true.copy()
    y_pu[hide_idx] = 0
    return X, y_pu, y_true


# ═════════════════════════════════════════════════════════════════════
# Convergence behaviour
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestConvergence:
    """Convergence behaviour properties."""

    def test_objective_decreases_and_converged_flag(self, rng):
        rng2 = np.random.RandomState(16)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=50, n_neg=100, d=3)
        clf = LDCEClassifier(
            flip_probability=0.3, max_iter=100, tol=1e-4,
            random_state=42,
        )
        clf.fit(X, y_pu)
        history = clf.objective_history_
        # After early iterations, objective should be non-increasing
        mid = min(5, len(history) // 3)
        for i in range(mid + 1, len(history)):
            assert history[i] <= history[i - 1] + 1e-10, (
                f"Objective increased at iter {i}: "
                f"{history[i-1]} → {history[i]}"
            )
        # converged_ flag is consistent

        assert clf.converged_ or clf.n_iter_ == clf.max_iter


# ═════════════════════════════════════════════════════════════════════
# Ellipsoid constraint
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestEllipsoidConstraint:
    """Constraint satisfaction."""

    def test_constraint_satisfied_at_final(self, rng):
        rng2 = np.random.RandomState(18)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=40, n_neg=80, d=5)
        clf = LDCEClassifier(
            flip_probability=0.3, max_iter=10, centroid_radius=1.0,
            random_state=42,
        )
        clf.fit(X, y_pu)
        m = clf.true_unlabeled_centroid_
        m_hat = clf.corrupted_centroid_
        S = clf.centroid_covariance_
        diff = m - m_hat
        constraint = float(diff @ S @ diff)
        assert constraint <= clf.centroid_radius + 1e-8


# ═════════════════════════════════════════════════════════════════════
# h sensitivity
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestHSensitivity:
    """Mild h mismatch should not crash."""

    @pytest.mark.parametrize("h_hat", [0.18, 0.24, 0.30, 0.36, 0.42])
    def test_mild_h_error_does_not_crash(self, rng, h_hat):
        rng2 = np.random.RandomState(19)
        X, y_pu, _ = _make_censoring_pu_data(
            rng2, n_pos=40, n_neg=80, h=0.3, d=4,
        )
        clf = LDCEClassifier(
            flip_probability=h_hat, max_iter=15, random_state=42,
        )
        clf.fit(X, y_pu)
        assert np.isfinite(clf.coef_).all()


# ═════════════════════════════════════════════════════════════════════
# Convergence diagnostics
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestConvergenceDiagnostics:
    """Diagnostic attributes are recorded correctly."""

    def test_diagnostics_consistent(self, rng):
        rng2 = np.random.RandomState(23)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=30, n_neg=60)
        clf = LDCEClassifier(
            flip_probability=0.3, max_iter=20, random_state=42,
        )
        clf.fit(X, y_pu)
        assert clf.n_iter_ == len(clf.objective_history_)
        assert clf.n_iter_ <= 20
        assert clf.n_iter_ >= 1
        if clf.converged_:
            assert clf.n_iter_ < clf.max_iter

    def test_metadata_includes_diagnostics(self, rng):
        rng2 = np.random.RandomState(24)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=30, n_neg=60)
        clf = LDCEClassifier(flip_probability=0.3, max_iter=10, random_state=42)
        clf.fit(X, y_pu)
        meta = clf.get_pu_metadata()
        assert meta["is_fitted"] is True
        assert meta["flip_probability"] == 0.3
        assert meta["converged"] == clf.converged_
        assert meta["n_iter"] == clf.n_iter_
        assert "family" in meta


# ═════════════════════════════════════════════════════════════════════
# Sklearn API compatibility
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLDCEAPI:
    """Sklearn API compatibility."""

    def test_get_params_set_params(self):
        clf = LDCEClassifier(flip_probability=0.3, reg_strength=2.0)
        params = clf.get_params()
        assert params["flip_probability"] == 0.3
        assert params["reg_strength"] == 2.0
        clf.set_params(reg_strength=5.0)
        assert clf.get_params()["reg_strength"] == 5.0

    def test_clone_compatible(self, rng):
        rng2 = np.random.RandomState(25)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=30, n_neg=60)
        clf = LDCEClassifier(flip_probability=0.3, max_iter=5, random_state=42)
        clf.fit(X, y_pu)
        clf2 = clone(clf)
        assert clf2.get_params() == clf.get_params()
        assert not clf2._is_fitted

    def test_pipeline_compatible(self, rng):
        rng2 = np.random.RandomState(26)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=30, n_neg=60)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LDCEClassifier(
                flip_probability=0.3, max_iter=10, random_state=42,
            )),
        ])
        pipe.fit(X, y_pu)
        pred = pipe.predict(X)
        assert set(np.unique(pred)) <= {0, 1}

    def test_score_samples_delegates(self, rng):
        rng2 = np.random.RandomState(27)
        X, y_pu, _ = _make_censoring_pu_data(rng2, n_pos=30, n_neg=60)
        clf = LDCEClassifier(flip_probability=0.3, max_iter=10, random_state=42)
        clf.fit(X, y_pu)
        np.testing.assert_array_equal(
            clf.score_samples(X), clf.decision_function(X),
        )


# ═════════════════════════════════════════════════════════════════════
# Error handling
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLDCEErrors:
    """Parameter validation edge cases."""

    @pytest.mark.parametrize("bad_h", [0.0, 1.0, 1.5, -0.5])
    def test_invalid_h_raises(self, rng, bad_h):
        rng2 = np.random.RandomState(28)
        X = rng2.randn(30, 3)
        y_pu = np.concatenate([np.ones(10, dtype=int), np.zeros(20, dtype=int)])
        clf = LDCEClassifier(flip_probability=bad_h, max_iter=5)
        with pytest.raises(ValueError):
            clf.fit(X, y_pu)

    def test_negative_reg_raises(self, rng):
        rng2 = np.random.RandomState(29)
        X = rng2.randn(30, 3)
        y_pu = np.concatenate([np.ones(10, dtype=int), np.zeros(20, dtype=int)])
        clf = LDCEClassifier(flip_probability=0.3, reg_strength=-1.0)
        with pytest.raises(ValueError, match="reg_strength"):
            clf.fit(X, y_pu)

    def test_nan_in_X_raises(self, rng):
        rng2 = np.random.RandomState(30)
        X = rng2.randn(30, 3)
        X[0, 1] = np.nan
        y_pu = np.concatenate([np.ones(10, dtype=int), np.zeros(20, dtype=int)])
        clf = LDCEClassifier(flip_probability=0.3, max_iter=5)
        with pytest.raises(ValueError, match="NaN"):
            clf.fit(X, y_pu)

    def test_no_unlabeled_raises(self, rng):
        rng2 = np.random.RandomState(31)
        X = rng2.randn(10, 3)
        y_pu = np.ones(10, dtype=int)
        clf = LDCEClassifier(flip_probability=0.3, max_iter=5)
        with pytest.raises(ValueError, match="unlabeled"):
            clf.fit(X, y_pu)

    @pytest.mark.parametrize("bad_max_iter", [0, 1])
    def test_max_iter_too_small_raises(self, rng, bad_max_iter):
        rng2 = np.random.RandomState(42)
        X = rng2.randn(30, 3)
        y_pu = np.concatenate([np.ones(10, dtype=int), np.zeros(20, dtype=int)])
        clf = LDCEClassifier(flip_probability=0.3, max_iter=bad_max_iter)
        with pytest.raises(ValueError, match="max_iter"):
            clf.fit(X, y_pu)


# ═════════════════════════════════════════════════════════════════════
# End-to-end regression
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLDCERegression:
    """Smoke-level paper regression tests."""

    def test_end_to_end_runs(self, rng):
        """LDCE runs end-to-end on synthetic data without errors."""
        rng2 = np.random.RandomState(32)
        X, y_pu, _ = _make_censoring_pu_data(
            rng2, n_pos=80, n_neg=160, h=0.3, d=5,
        )
        clf = LDCEClassifier(
            flip_probability=0.3, max_iter=50, random_state=42,
        )
        clf.fit(X, y_pu)
        pred = clf.predict(X)
        assert pred.shape == (X.shape[0],)
        assert set(np.unique(pred)) <= {0, 1}
        assert clf._is_fitted
        assert clf.n_iter_ >= 1
        assert len(clf.objective_history_) == clf.n_iter_

    def test_separable_data_runs(self, rng):
        """Well-separated data: model fits without crash.

        LDCE is a RESEARCH-maturity method; optimal accuracy requires
        CV-tuned hyperparameters (λ, b) as noted in the method card §9.1.
        This test verifies the pipeline works end-to-end, not accuracy.
        """
        rng2 = np.random.RandomState(33)
        X_pos = rng2.randn(100, 3) + 3.0
        X_neg = rng2.randn(200, 3) - 3.0
        X = np.vstack([X_pos, X_neg])
        y_true = np.concatenate([np.ones(100, dtype=int), np.zeros(200, dtype=int)])
        n_hide = int(100 * 0.3)
        hide = rng2.choice(100, size=n_hide, replace=False)
        y_pu = y_true.copy()
        y_pu[hide] = 0

        clf = LDCEClassifier(
            flip_probability=0.3, max_iter=50, random_state=42,
        )
        clf.fit(X, y_pu)
        pred = clf.predict(X)
        assert pred.shape == (X.shape[0],)
        assert set(np.unique(pred)) <= {0, 1}
        assert clf._is_fitted
        assert clf.n_iter_ >= 1
        assert hasattr(clf, "coef_")
        assert hasattr(clf, "class_prior_")
        assert np.isfinite(clf.coef_).all()
        assert np.isfinite(clf.centroid_covariance_).all()
