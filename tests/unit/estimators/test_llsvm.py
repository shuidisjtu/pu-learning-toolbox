# ruff: noqa: N802, N803, N806, E501

"""Tests for LLSVMClassifier.

Covers API contract, input validation, end-to-end training on synthetic
data, class prior paths, and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.exceptions import NotFittedError, ValidationError
from pu_toolbox.estimators.classic.llsvm import LLSVMClassifier


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _make_two_gaussian_pu(rng, n_p=50, n_u=150, d=2, separation=3.0):
    """Two-Gaussian PU data: P ~ N(+sep/2, I), hidden N ~ N(-sep/2, I)."""
    mu_p = np.full(d, separation / 2)
    mu_n = np.full(d, -separation / 2)
    X_p = rng.randn(n_p, d) + mu_p
    n_pos_in_u = n_u // 3
    n_neg_in_u = n_u - n_pos_in_u
    X_u = np.vstack([
        rng.randn(n_pos_in_u, d) + mu_p,
        rng.randn(n_neg_in_u, d) + mu_n,
    ])
    X = np.vstack([X_p, X_u])
    y_pu = np.concatenate([np.ones(n_p), np.zeros(n_u)])
    class_prior = n_pos_in_u / n_u
    return X, y_pu, class_prior


# ═════════════════════════════════════════════════════════════════════
# API contract
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLLSVMAPI:
    """API contract: shapes, types, metadata, get/set_params."""

    def test_basic_metadata(self):
        clf = LLSVMClassifier()
        assert clf.family.value == "risk_estimation"
        assert clf.requires_class_prior is True
        assert clf.backend.value == "numpy"
        meta = clf.get_pu_metadata()
        assert meta["is_fitted"] is False

    def test_basic_get_set_params(self):
        clf = LLSVMClassifier(alpha=3.0, gamma=20.0)
        params = clf.get_params()
        assert params["alpha"] == 3.0
        assert params["gamma"] == 20.0
        clf.set_params(alpha=5.0)
        assert clf.alpha == 5.0

    def test_basic_fit_predict_shapes(self, rng):
        X, y_pu, pi = _make_two_gaussian_pu(rng)
        clf = LLSVMClassifier(max_epochs=50, random_state=42)
        clf.fit(X, y_pu, class_prior=pi)
        assert clf._is_fitted
        assert clf.coef_.shape == (X.shape[1],)
        assert isinstance(clf.intercept_, float)
        pred = clf.predict(X)
        assert pred.shape == (X.shape[0],)
        assert set(np.unique(pred)).issubset({-1, 1})
        scores = clf.decision_function(X)
        assert scores.shape == (X.shape[0],)
        np.testing.assert_array_equal(clf.score_samples(X), scores)
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

    def test_basic_not_fitted_raises(self):
        clf = LLSVMClassifier()
        with pytest.raises(NotFittedError):
            clf.predict(np.array([[1.0, 2.0]]))
        with pytest.raises(NotFittedError):
            clf.decision_function(np.array([[1.0, 2.0]]))

    def test_basic_predict_proba_not_implemented(self, rng):
        X, y_pu, pi = _make_two_gaussian_pu(rng)
        clf = LLSVMClassifier(max_epochs=10, random_state=42)
        clf.fit(X, y_pu, class_prior=pi)
        with pytest.raises(NotImplementedError):
            clf.predict_proba(X)


# ═════════════════════════════════════════════════════════════════════
# Input validation
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLLSVMValidation:
    """Input validation: bad labels, empty sets, bad hyperparams."""

    def test_param_no_positives_raises(self, rng):
        X = rng.randn(10, 2)
        y_all_zero = np.zeros(10)
        with pytest.raises(ValidationError):
            LLSVMClassifier(max_epochs=10).fit(X, y_all_zero, class_prior=0.5)

    def test_param_invalid_class_prior_raises(self, rng):
        X, y_pu, _ = _make_two_gaussian_pu(rng)
        for bad_pi in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="class_prior"):
                LLSVMClassifier(max_epochs=10).fit(X, y_pu, class_prior=bad_pi)

    def test_param_invalid_hyperparams_raise(self, rng):
        X, y_pu, pi = _make_two_gaussian_pu(rng)
        with pytest.raises(ValueError, match="alpha"):
            LLSVMClassifier(alpha=-1.0, max_epochs=10).fit(X, y_pu, class_prior=pi)
        with pytest.raises(ValueError, match="learning_rate"):
            LLSVMClassifier(learning_rate=0.0, max_epochs=10).fit(X, y_pu, class_prior=pi)
        with pytest.raises(ValueError, match="max_epochs"):
            LLSVMClassifier(max_epochs=0).fit(X, y_pu, class_prior=pi)


# ═════════════════════════════════════════════════════════════════════
# End-to-end training
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLLSVMTraining:
    """End-to-end training on synthetic data."""

    def test_basic_two_gaussian_separable(self, rng):
        """Well-separated data → positive samples get positive scores."""
        X, y_pu, pi = _make_two_gaussian_pu(rng, n_p=80, n_u=200, separation=4.0)
        clf = LLSVMClassifier(
            max_epochs=500, learning_rate=1e-4, random_state=42,
        )
        clf.fit(X, y_pu, class_prior=pi)
        scores = clf.decision_function(X)
        # Positive samples should mostly get positive scores
        pos_scores = scores[y_pu == 1]
        pos_accuracy = np.mean(pos_scores > 0)
        assert pos_accuracy >= 0.6, f"P-accuracy={pos_accuracy:.3f}"

    def test_basic_loss_decreases(self, rng):
        """Loss history should generally decrease over training."""
        X, y_pu, pi = _make_two_gaussian_pu(rng)
        clf = LLSVMClassifier(max_epochs=200, random_state=42)
        clf.fit(X, y_pu, class_prior=pi)
        history = clf.loss_history_
        assert len(history) == 200
        # First loss > last loss (allowing noise, compare first 10% vs last 10%)
        early = np.mean(history[:20])
        late = np.mean(history[-20:])
        assert late < early, f"Loss did not decrease: early={early:.4f}, late={late:.4f}"

    def test_basic_calibration_prevents_all_positive(self, rng):
        """With calibration (gamma>0), not all predictions should be +1."""
        X, y_pu, pi = _make_two_gaussian_pu(rng, n_p=50, n_u=200)
        clf = LLSVMClassifier(
            gamma=10.0, max_epochs=300, random_state=42,
        )
        clf.fit(X, y_pu, class_prior=pi)
        preds = clf.predict(X)
        # Should have both +1 and -1 predictions
        assert len(np.unique(preds)) == 2, "All predictions are the same class"

    def test_basic_fitted_attributes(self, rng):
        """All expected fitted attributes are set."""
        X, y_pu, pi = _make_two_gaussian_pu(rng)
        clf = LLSVMClassifier(max_epochs=10, random_state=42)
        clf.fit(X, y_pu, class_prior=pi)
        assert clf.class_prior_ == pytest.approx(pi)
        assert clf.calibration_target_ == pytest.approx(2 * pi - 1)
        assert clf.n_positive_ == int(np.sum(y_pu == 1))
        assert clf.n_unlabeled_ == int(np.sum(y_pu == 0))
        assert hasattr(clf, "loss_history_")


# ═════════════════════════════════════════════════════════════════════
# Class prior paths
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLLSVMClassPrior:
    """Class prior: explicit vs auto-estimated."""

    def test_basic_explicit_prior(self, rng):
        """Explicit class_prior is stored correctly."""
        X, y_pu, _ = _make_two_gaussian_pu(rng)
        clf = LLSVMClassifier(max_epochs=10, random_state=42)
        clf.fit(X, y_pu, class_prior=0.4)
        assert clf.class_prior_ == pytest.approx(0.4)
        assert clf.calibration_target_ == pytest.approx(2 * 0.4 - 1)

    def test_basic_auto_estimated_prior(self, rng):
        """Without explicit prior, penL1 estimates it."""
        X, y_pu, _ = _make_two_gaussian_pu(rng, n_p=80, n_u=200)
        clf = LLSVMClassifier(max_epochs=10, random_state=42)
        clf.fit(X, y_pu)
        assert 0.0 < clf.class_prior_ < 1.0
        assert clf.calibration_target_ == pytest.approx(2 * clf.class_prior_ - 1)


# ═════════════════════════════════════════════════════════════════════
# Reproducibility
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLLSVMReproducibility:
    """Same random_state → identical results."""

    def test_deterministic_same_seed(self, rng):
        X, y_pu, pi = _make_two_gaussian_pu(rng)
        clf1 = LLSVMClassifier(max_epochs=50, random_state=123)
        clf1.fit(X, y_pu, class_prior=pi)
        clf2 = LLSVMClassifier(max_epochs=50, random_state=123)
        clf2.fit(X, y_pu, class_prior=pi)
        np.testing.assert_array_equal(clf1.coef_, clf2.coef_)
        assert clf1.intercept_ == clf2.intercept_
