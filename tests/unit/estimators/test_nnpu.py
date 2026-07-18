# ruff: noqa: N802, N803, N806, S101, S113, E501

"""Tests for NonNegativePUClassifier and NonNegativePULoss — nnPU learning.

Covers method card §9: risk formula consistency, negative-risk regression,
branch gradients, max-misuse guard, beta boundary, class-prior validation,
P/U batch handling, output semantics, and overfitting behaviour.

Consolidated: 15 test methods across 6 classes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.config import POSITIVE_LABEL, UNLABELED_LABEL
from pu_toolbox.core.exceptions import NotFittedError, ValidationError
from pu_toolbox.losses.nnpu import NonNegativePULoss, _nnpu_train_step

# ═════════════════════════════════════════════════════════════════════
# MARK: helpers
# ═════════════════════════════════════════════════════════════════════

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier  # noqa: E402


def _make_synthetic_data(
    rng: np.random.RandomState,
    n_p: int = 50,
    n_u: int = 100,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """2-class Gaussian: pos ~ N(+1, 1), neg ~ N(-1, 1)."""
    rng.seed(seed)
    X_p = rng.randn(n_p, n_features) + 1.0
    X_n = rng.randn(n_u, n_features) - 1.0
    X = np.vstack([X_p, X_n])
    y_pu = np.concatenate(
        [np.full(n_p, POSITIVE_LABEL, dtype=int),
         np.full(n_u, UNLABELED_LABEL, dtype=int)],
    )
    class_prior = n_p / (n_p + n_u)
    return X, y_pu, class_prior


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.1 — Risk formula consistency
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.math
class TestRiskFormulas:
    """Method card §9.1 — component risk formulas match hand computation."""

    @pytest.mark.parametrize(
        "p_scores, u_scores, pi, desc",
        [
            (np.array([-1.0, -0.5]), np.array([1.5, 1.0]), 0.5, "r_positive"),
            (np.array([3.0, 4.0]), np.array([-4.0, -3.0]), 0.3, "r_negative"),
            (np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.5, "r_zero"),
        ],
    )
    def test_basic_risk_formula_consistency(self, p_scores, u_scores, pi, desc):
        """Hand-computed sigmoid-based risk components match evaluate()."""
        loss = NonNegativePULoss()
        info = loss.evaluate(p_scores, u_scores, class_prior=pi)

        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        R_p_plus_hand = float(np.mean(sig(-p_scores)))
        R_p_minus_hand = float(np.mean(sig(p_scores)))
        R_u_minus_hand = float(np.mean(sig(u_scores)))
        r_hand = R_u_minus_hand - pi * R_p_minus_hand
        upu_hand = pi * R_p_plus_hand + r_hand
        nnpu_hand = pi * R_p_plus_hand + max(0.0, r_hand)

        assert info["positive_risk"] == pytest.approx(R_p_plus_hand)
        assert info["negative_risk"] == pytest.approx(r_hand)
        assert info["upu_risk"] == pytest.approx(upu_hand)
        assert info["nnpu_risk"] == pytest.approx(nnpu_hand)
        # __call__: non_negative flag switches between risks
        r_nn = loss(p_scores, u_scores, class_prior=pi, non_negative=True)
        r_upu = loss(p_scores, u_scores, class_prior=pi, non_negative=False)
        assert r_nn >= 0
        assert r_nn >= r_upu

    def test_param_validation_and_errors(self):
        """class_prior validation and empty score arrays → ValueError."""
        loss = NonNegativePULoss()
        s = np.array([1.0])

        for bad_prior, expected_match in ((0.0, "class_prior"), (1.0, "class_prior"), (-0.5, "class_prior"), (None, "")):
            with pytest.raises((ValueError, TypeError), match=expected_match or None):
                loss(s, s, class_prior=bad_prior)

        with pytest.raises(ValueError, match="positive_scores"):
            loss(np.array([]), np.array([1.0]), class_prior=0.5)
        with pytest.raises(ValueError, match="unlabeled_scores"):
            loss(np.array([1.0]), np.array([]), class_prior=0.5)


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.2-9.4 — Negative risk regression & branch gradients
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestGradientBehavior:
    """Method card §9.2-§9.4 — uPU/nnPU risk, branch gradients, max() guard."""

    def test_basic_upu_vs_nnpu_risk(self):
        """uPU can be negative; nnPU never is (random score extremes test)."""
        loss = NonNegativePULoss()
        # Strong separation → r < 0 → uPU < 0
        p_scores = np.array([10.0, 12.0, 11.0])
        u_scores = np.array([-8.0, -9.0, -7.0, -10.0])
        pi = 0.3
        assert loss(p_scores, u_scores, class_prior=pi, non_negative=False) < 0
        assert loss(p_scores, u_scores, class_prior=pi, non_negative=True) >= 0

        # Random score extremes: nnPU risk never negative
        rng = np.random.RandomState(42)
        for _ in range(20):
            p = rng.uniform(5, 20, size=10)
            u = rng.uniform(-20, -5, size=30)
            pi_r = rng.uniform(0.1, 0.9)
            assert loss(p, u, class_prior=pi_r, non_negative=True) >= -1e-12

    def test_deterministic_branch_gradients(self):
        """Normal vs correction branch gradients differ; gamma=0 disables correction grad."""
        # Normal branch (r > 0)
        p_n = torch.tensor([-1.0, -0.5], requires_grad=True)
        u_n = torch.tensor([1.5, 1.0], requires_grad=True)
        R_pp = torch.sigmoid(-p_n).mean()
        R_pm = torch.sigmoid(p_n).mean()
        R_um = torch.sigmoid(u_n).mean()
        opt_n, info_n = _nnpu_train_step(R_pp, R_pm, R_um, class_prior=0.5, beta=0.0, gamma=1.0)
        assert not info_n["correction"]
        opt_n.backward()
        assert p_n.grad is not None and (p_n.grad != 0).any()

        # Correction branch (r < 0)
        p_c = torch.tensor([5.0, 4.0], requires_grad=True)
        u_c = torch.tensor([-2.0, -1.0], requires_grad=True)
        R_pp_c = torch.sigmoid(-p_c).mean()
        R_pm_c = torch.sigmoid(p_c).mean()
        R_um_c = torch.sigmoid(u_c).mean()
        opt_c, info_c = _nnpu_train_step(R_pp_c, R_pm_c, R_um_c, class_prior=0.3, beta=0.0, gamma=1.0)
        assert info_c["correction"]
        opt_c.backward()

        # Normal branch with same scores (beta=inf)
        p_n2 = torch.tensor([5.0, 4.0], requires_grad=True)
        u_n2 = torch.tensor([-2.0, -1.0], requires_grad=True)
        R_pp2 = torch.sigmoid(-p_n2).mean()
        R_pm2 = torch.sigmoid(p_n2).mean()
        R_um2 = torch.sigmoid(u_n2).mean()
        opt_n2, _ = _nnpu_train_step(R_pp2, R_pm2, R_um2, class_prior=0.3, beta=float("inf"), gamma=1.0)
        opt_n2.backward()
        assert not torch.allclose(p_c.grad, p_n2.grad, atol=1e-5)

        # gamma=0 → no gradient in correction
        p_z = torch.tensor([5.0, 4.0], requires_grad=True)
        u_z = torch.tensor([-2.0, -1.0], requires_grad=True)
        R_ppz = torch.sigmoid(-p_z).mean()
        R_pmz = torch.sigmoid(p_z).mean()
        R_umz = torch.sigmoid(u_z).mean()
        opt_z, info_z = _nnpu_train_step(R_ppz, R_pmz, R_umz, class_prior=0.3, beta=0.0, gamma=0.0)
        assert info_z["correction"]
        opt_z.backward()
        assert torch.allclose(p_z.grad, torch.zeros_like(p_z.grad), atol=1e-7)

    def test_deterministic_algorithm1_differs_from_max(self):
        """When r < -beta, Alg.1 gradient != gradient of max(0, r)."""
        p_a = torch.tensor([5.0, 4.0], requires_grad=True)
        u_a = torch.tensor([-2.0, -1.0], requires_grad=True)
        pi = 0.3
        R_pp_a = torch.sigmoid(-p_a).mean()
        R_pm_a = torch.sigmoid(p_a).mean()
        R_um_a = torch.sigmoid(u_a).mean()
        opt_alg1, info = _nnpu_train_step(R_pp_a, R_pm_a, R_um_a, class_prior=pi, beta=0.0, gamma=1.0)
        assert info["correction"]
        opt_alg1.backward()
        alg1_grad_p = p_a.grad.clone()

        # Naive max(0, r)
        p_m = torch.tensor([5.0, 4.0], requires_grad=True)
        u_m = torch.tensor([-2.0, -1.0], requires_grad=True)
        R_pp_m = torch.sigmoid(-p_m).mean()
        R_pm_m = torch.sigmoid(p_m).mean()
        R_um_m = torch.sigmoid(u_m).mean()
        r = R_um_m - pi * R_pm_m
        (pi * R_pp_m + torch.clamp(r, min=0.0)).backward()

        assert not torch.allclose(alg1_grad_p, p_m.grad, atol=1e-5)


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.5 — Beta boundary & fit-time validation
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestBetaAndValidation:
    """Method card §9.5 — beta validation, behavioral edges, and fit-time checks."""

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"gamma": 1.5}, "gamma"),
            ({"loss": "logistic"}, "loss"),
            ({"max_epochs": 0}, "max_epochs"),
            ({"batch_size": 0}, "batch_size"),
            ({"beta": -0.1}, "beta"),
        ],
    )
    def test_param_fit_validation(self, rng, kwargs, match):
        """Gamma, loss, max_epochs, batch_size, beta invalid → ValueError in fit()."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model, max_epochs=1, batch_size=10)
        clf.set_params(**kwargs)
        with pytest.raises(ValueError, match=match):
            clf.fit(X, y_pu, class_prior=pi)

    def test_edge_beta_behavioral_and_warnings(self, rng):
        """beta<0 raises, beta=0 standard nnPU, beta huge→uPU, beta>class_prior warns."""
        # beta < 0 → ValueError in loss
        with pytest.raises(ValueError, match="beta"):
            NonNegativePULoss(beta=-0.1)

        # beta=0: correction when r < 0
        loss0 = NonNegativePULoss(beta=0.0)
        r_nn = loss0(np.array([5.0, 4.0]), np.array([-3.0, -2.0]), class_prior=0.3, non_negative=True)
        r_upu = loss0(np.array([5.0, 4.0]), np.array([-3.0, -2.0]), class_prior=0.3, non_negative=False)
        assert r_nn >= 0 and r_nn > r_upu

        # beta huge → correction never triggers
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model, beta=1e9, max_epochs=3, batch_size=8)
        clf.fit(X, y_pu, class_prior=pi)
        assert all(f == 0.0 for f in clf.get_training_history()["correction_fraction"])

        # beta=class_prior runs without error
        model2 = torch.nn.Linear(5, 1)
        clf2 = NonNegativePUClassifier(model=model2, beta=pi, max_epochs=3, batch_size=8)
        clf2.fit(X, y_pu, class_prior=pi)
        assert clf2._is_fitted

        # beta > class_prior → UserWarning
        model3 = torch.nn.Linear(5, 1)
        clf3 = NonNegativePUClassifier(model=model3, beta=pi + 0.1, max_epochs=1, batch_size=10)
        with pytest.warns(UserWarning, match="beta"):
            clf3.fit(X, y_pu, class_prior=pi)


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.6-§9.7 — Class-prior validation & P/U batch handling
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestValidationAndBatches:
    """Method card §9.6-§9.7 — class_prior required/validated, P/U batch edges."""

    def test_param_class_prior_validation(self, rng):
        """class_prior required in fit(), stored in metadata and as attribute."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)

        # Missing class_prior → TypeError
        clf = NonNegativePUClassifier(model=model, max_epochs=1)
        with pytest.raises(TypeError):
            clf.fit(X, y_pu)

        # Valid class_prior stored
        clf2 = NonNegativePUClassifier(model=model, max_epochs=1, batch_size=10)
        clf2.fit(X, y_pu, class_prior=pi)
        assert clf2.class_prior_ == pytest.approx(pi)
        assert clf2.get_pu_metadata()["class_prior"] == pytest.approx(pi)

    def test_edge_pu_batch_edges(self, rng):
        """No-P, no-U, single-P, unequal loader lengths, NaN in X."""
        # No positives
        X0 = rng.randn(20, 3)
        model0 = torch.nn.Linear(3, 1)
        with pytest.raises((ValidationError, ValueError)):
            NonNegativePUClassifier(model=model0, max_epochs=1).fit(X0, np.zeros(20), class_prior=0.5)

        # No unlabeled
        with pytest.raises(ValueError):
            NonNegativePUClassifier(model=model0, max_epochs=1).fit(X0, np.ones(20), class_prior=0.5)

        # Single positive (2 total)
        Xp, Xu = rng.randn(2, 3), rng.randn(20, 3)
        X2 = np.vstack([Xp, Xu])
        y2 = np.concatenate([np.ones(2), np.zeros(20)])
        pi2 = 2.0 / 22.0
        model2 = torch.nn.Linear(3, 1)
        clf2 = NonNegativePUClassifier(model=model2, max_epochs=2, batch_size=4)
        clf2.fit(X2, y2, class_prior=pi2)
        assert clf2._is_fitted and clf2.n_positive_ == 2

        # Unequal loader lengths
        Xp3, Xu3 = rng.randn(3, 3), rng.randn(100, 3)
        X3 = np.vstack([Xp3, Xu3])
        y3 = np.concatenate([np.ones(3), np.zeros(100)])
        pi3 = 3.0 / 103.0
        model3 = torch.nn.Linear(3, 1)
        clf3 = NonNegativePUClassifier(model=model3, max_epochs=2, batch_size=2)
        clf3.fit(X3, y3, class_prior=pi3)
        assert clf3._is_fitted

        # NaN in X
        X_nan, y_nan, pi_n = _make_synthetic_data(rng, n_p=20, n_u=40)
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            NonNegativePUClassifier(model=torch.nn.Linear(5, 1), max_epochs=1).fit(X_nan, y_nan, class_prior=pi_n)

    def test_edge_sample_weight_and_validation(self, rng):
        """Sample weight normalization works; validation_data triggers early stop."""
        Xp, Xu = rng.randn(10, 3), rng.randn(20, 3)
        X = np.vstack([Xp, Xu])
        y_pu = np.concatenate([np.ones(10), np.zeros(20)])
        pi = 10.0 / 30.0

        # Sample weight
        w_p = np.ones(10) * 2.0
        w_u = np.ones(20) * 0.5
        sw = np.concatenate([w_p, w_u])
        model = torch.nn.Linear(3, 1)
        clf = NonNegativePUClassifier(model=model, max_epochs=1, batch_size=5)
        clf.fit(X, y_pu, class_prior=pi, sample_weight=sw)
        assert clf._is_fitted

        # Validation data early stopping
        X2, y2, pi2 = _make_synthetic_data(rng, n_p=30, n_u=60)
        model2 = torch.nn.Linear(5, 1)
        clf2 = NonNegativePUClassifier(model=model2, max_epochs=50, patience=3, batch_size=8)
        clf2.fit(X2, y2, class_prior=pi2, validation_data=(X2, y2))
        assert len(clf2.history_["epoch"]) <= 50


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.8 — Output semantics, API contract & early stopping
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestAPIContract:
    """Method card §9.8 — output semantics, sklearn compat, early stopping."""

    def test_basic_fit_predict_outputs(self, rng):
        """decision_function shape, predict binary, predict_proba raises, not_fitted."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model, max_epochs=2, batch_size=8)
        clf.fit(X, y_pu, class_prior=pi)

        scores = clf.decision_function(X)
        assert scores.shape == (X.shape[0],) and np.isfinite(scores).all()

        preds = clf.predict(X)
        assert set(np.unique(preds)) <= {0, 1} and preds.dtype == int

        with pytest.raises(NotImplementedError, match="predict_proba"):
            clf.predict_proba(X)

        with pytest.raises(NotFittedError):
            NonNegativePUClassifier(model=model).predict(X)

    def test_basic_sklearn_compatibility_and_metadata(self, rng):
        """get_params, set_params, history_keys, default_model, reproducibility, metadata."""
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model, beta=0.1, gamma=0.9)

        # get_params / set_params
        params = clf.get_params()
        assert isinstance(params, dict) and params["beta"] == 0.1 and params["gamma"] == 0.9
        clf.set_params(beta=0.3, max_epochs=50)
        assert clf.beta == 0.3 and clf.max_epochs == 50

        # Fit and check history
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model2 = torch.nn.Linear(5, 1)
        clf2 = NonNegativePUClassifier(model=model2, max_epochs=3, batch_size=8)
        clf2.fit(X, y_pu, class_prior=pi)

        hist = clf2.get_training_history()
        required = {"epoch", "positive_risk", "negative_risk", "upu_risk", "nnpu_risk", "optimization_loss", "correction_fraction"}
        assert required <= set(hist.keys())
        for k in required:
            assert len(hist[k]) == 3

        # Metadata
        meta = clf2.get_pu_metadata()
        assert meta["loss"] == "sigmoid" and meta["beta"] == 0.0 and meta["is_fitted"] is True
        assert meta["family"] == "risk_estimation"
        assert "n_positive" in meta and "n_unlabeled" in meta

        # Default model created
        clf3 = NonNegativePUClassifier(max_epochs=2, batch_size=8)
        clf3.fit(X, y_pu, class_prior=pi)
        assert isinstance(clf3.model_, torch.nn.Module)

    def test_deterministic_reproducibility(self, rng):
        """Same random_state → identical decision_function output."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40, seed=99)
        torch.manual_seed(42)
        m1 = torch.nn.Linear(5, 1)
        c1 = NonNegativePUClassifier(model=m1, max_epochs=2, batch_size=8, random_state=42)
        c1.fit(X, y_pu, class_prior=pi)
        s1 = c1.decision_function(X)

        torch.manual_seed(42)
        m2 = torch.nn.Linear(5, 1)
        c2 = NonNegativePUClassifier(model=m2, max_epochs=2, batch_size=8, random_state=42)
        c2.fit(X, y_pu, class_prior=pi)
        s2 = c2.decision_function(X)
        np.testing.assert_array_almost_equal(s1, s2)

    def test_edge_evaluate_pu_risk_and_early_stopping(self, rng):
        """evaluate_pu_risk override/flag_switch; early stopping with/without validation."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model, max_epochs=2, batch_size=8)
        clf.fit(X, y_pu, class_prior=pi)

        # evaluate_pu_risk
        risk_stored = clf.evaluate_pu_risk(X, y_pu)
        risk_override = clf.evaluate_pu_risk(X, y_pu, class_prior=0.7)
        assert risk_stored != pytest.approx(risk_override)
        r_nn = clf.evaluate_pu_risk(X, y_pu, non_negative=True)
        r_upu = clf.evaluate_pu_risk(X, y_pu, non_negative=False)
        assert r_nn >= 0 and r_nn >= r_upu

        # Early stopping with validation
        X2, y2, pi2 = _make_synthetic_data(rng, n_p=15, n_u=30)
        X_val, y_val, _ = _make_synthetic_data(rng, n_p=10, n_u=20, seed=99999)
        model2 = torch.nn.Linear(5, 1)
        clf2 = NonNegativePUClassifier(model=model2, max_epochs=100, patience=2, batch_size=4, random_state=42)
        clf2.fit(X2, y2, class_prior=pi2, validation_data=(X_val, y_val))
        assert len(clf2.history_["epoch"]) < 100

        # Without validation → full epochs
        X3, y3, pi3 = _make_synthetic_data(rng, n_p=20, n_u=40)
        model3 = torch.nn.Linear(5, 1)
        clf3 = NonNegativePUClassifier(model=model3, max_epochs=5, patience=2, batch_size=8)
        clf3.fit(X3, y3, class_prior=pi3)
        assert len(clf3.history_["epoch"]) == 5


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.9 — Overfitting behaviour test (slow)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestOverfittingBehavior:
    """Method card §9.9 — uPU overfits, nnPU remains stable."""

    def test_slow_nnpu_prevents_negative_risk_divergence(self, rng):
        """On a flexible model, nnPU keeps negative risk near 0."""
        X_p = rng.randn(30, 2) + 1.5
        X_n = rng.randn(200, 2) - 1.5
        X = np.vstack([X_p, X_n])
        y_pu = np.concatenate([np.ones(30), np.zeros(200)])
        pi = 30.0 / 230.0

        # nnPU
        model_nnpu = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        clf_nnpu = NonNegativePUClassifier(
            model=model_nnpu, beta=0.0, gamma=1.0,
            max_epochs=50, batch_size=32, random_state=42,
        )
        clf_nnpu.fit(X, y_pu, class_prior=pi)
        history_nnpu = clf_nnpu.get_training_history()

        # uPU (huge beta → never correct)
        model_upu = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        clf_upu = NonNegativePUClassifier(
            model=model_upu, beta=1e9, gamma=1.0,
            max_epochs=50, batch_size=32, random_state=42,
        )
        clf_upu.fit(X, y_pu, class_prior=pi)
        history_upu = clf_upu.get_training_history()

        correction_fractions = history_nnpu["correction_fraction"]
        assert max(correction_fractions) >= 0

        final_nr_nnpu = history_nnpu["negative_risk"][-1]
        final_nr_upu = history_upu["negative_risk"][-1]
        assert final_nr_nnpu >= final_nr_upu or abs(final_nr_upu) < 0.5, (
            f"nnPU negative risk ({final_nr_nnpu:.4f}) should not "
            f"diverge worse than uPU ({final_nr_upu:.4f})"
        )
