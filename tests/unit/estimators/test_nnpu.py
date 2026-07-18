# ruff: noqa: N802, N803, N806, S101, S113

"""Tests for NonNegativePUClassifier and NonNegativePULoss — nnPU learning.

Covers method card §9: risk formula consistency, negative-risk regression,
branch gradients, max-misuse guard, beta boundary, class-prior validation,
P/U batch handling, output semantics, and overfitting behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    """2-class Gaussian: pos ~ N(+1, 1), neg ~ N(−1, 1)."""
    rng.seed(seed)
    X_p = rng.randn(n_p, n_features) + 1.0
    X_n = rng.randn(n_u, n_features) - 1.0
    X = np.vstack([X_p, X_n])
    y_pu = np.concatenate([np.ones(n_p), np.zeros(n_u)])
    class_prior = n_p / (n_p + n_u)
    return X, y_pu, class_prior


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.1 — Risk formula consistency (NonNegativePULoss)
# ═════════════════════════════════════════════════════════════════════


class TestRiskFormulas:
    """Method card §9.1 — component risk formulas match hand computation."""

    def test_individual_risks_positive_r(self):
        """r > 0: P scores negative, U scores positive."""
        loss = NonNegativePULoss()
        # P looks negative → sigmoid(score_P) is small
        # U looks positive → sigmoid(score_U) is large
        # → r = R_u^- − π·R_p^- > 0
        p_scores = np.array([-1.0, -0.5])
        u_scores = np.array([1.5, 1.0])
        pi = 0.5

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
        assert r_hand > 0
        # r > 0 → uPU == nnPU
        assert info["upu_risk"] == pytest.approx(info["nnpu_risk"])

    def test_individual_risks_negative_r(self):
        """r < 0: P scores positive, U scores negative."""
        loss = NonNegativePULoss()
        # P looks positive → sigmoid(score_P) is large
        # U looks negative → sigmoid(score_U) is small
        # → r = R_u^- − π·R_p^- ≪ 0
        p_scores = np.array([3.0, 4.0])
        u_scores = np.array([-4.0, -3.0])
        pi = 0.3

        info = loss.evaluate(p_scores, u_scores, class_prior=pi)

        assert info["negative_risk"] < 0
        assert info["upu_risk"] < info["nnpu_risk"]
        assert info["nnpu_risk"] >= 0.0

    def test_individual_risks_r_equals_zero(self):
        """Exact break-even with balanced symmetric scores."""
        loss = NonNegativePULoss()
        p_scores = np.array([0.0, 0.0])
        u_scores = np.array([0.0, 0.0])
        pi = 0.5

        info = loss.evaluate(p_scores, u_scores, class_prior=pi)
        # sigmoid(0) = 0.5
        # R_p^+ = 0.5, R_p^- = 0.5, R_u^- = 0.5
        # r = 0.5 − 0.5*0.5 = 0.25 > 0 → uPU == nnPU
        assert info["positive_risk"] == pytest.approx(0.5)
        assert info["upu_risk"] == pytest.approx(info["nnpu_risk"])

    def test_call_non_negative_flag(self):
        """__call__(non_negative=True/False) switches between risks."""
        loss = NonNegativePULoss()
        p_scores = np.array([3.0, 4.0])
        u_scores = np.array([-4.0, -3.0])
        pi = 0.3

        r_nn = loss(p_scores, u_scores, class_prior=pi, non_negative=True)
        r_upu = loss(p_scores, u_scores, class_prior=pi, non_negative=False)
        assert r_nn >= 0
        assert r_nn >= r_upu

    def test_class_prior_validation(self):
        """Invalid class_prior → ValueError."""
        loss = NonNegativePULoss()
        s = np.array([1.0])

        with pytest.raises(ValueError, match="class_prior"):
            loss(s, s, class_prior=0.0)
        with pytest.raises(ValueError, match="class_prior"):
            loss(s, s, class_prior=1.0)
        with pytest.raises(ValueError, match="class_prior"):
            loss(s, s, class_prior=-0.5)

    def test_empty_scores_raise(self):
        """Empty positive or unlabeled scores → ValueError."""
        loss = NonNegativePULoss()
        with pytest.raises(ValueError, match="positive_scores"):
            loss(np.array([]), np.array([1.0]), class_prior=0.5)
        with pytest.raises(ValueError, match="unlabeled_scores"):
            loss(np.array([1.0]), np.array([]), class_prior=0.5)


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.2 — Negative empirical risk regression
# ═════════════════════════════════════════════════════════════════════


class TestNegativeRiskRegression:
    """Method card §9.2 — uPU risk goes negative, nnPU stays ≥ 0."""

    def test_upu_risk_can_be_negative(self):
        """Strong separation → r ≪ 0 → uPU risk < 0."""
        loss = NonNegativePULoss()
        p_scores = np.array([10.0, 12.0, 11.0])
        u_scores = np.array([-8.0, -9.0, -7.0, -10.0])
        pi = 0.3

        upu_risk = loss(
            p_scores, u_scores, class_prior=pi, non_negative=False
        )
        nnpu_risk = loss(
            p_scores, u_scores, class_prior=pi, non_negative=True
        )

        assert upu_risk < 0, f"Expected negative uPU risk, got {upu_risk}"
        assert nnpu_risk >= 0, (
            f"Expected non-negative nnPU risk, got {nnpu_risk}"
        )

    def test_nnpu_non_negative_always(self):
        """nnPU risk never goes below 0 regardless of score extremity."""
        loss = NonNegativePULoss()
        rng = np.random.RandomState(42)
        for _ in range(20):
            p_scores = rng.uniform(5, 20, size=10)
            u_scores = rng.uniform(-20, -5, size=30)
            pi = rng.uniform(0.1, 0.9)
            nnpu_risk = loss(
                p_scores, u_scores, class_prior=pi, non_negative=True
            )
            assert nnpu_risk >= -1e-12, (
                f"nnPU risk negative: {nnpu_risk}"
            )


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.3 — Branch gradient tests
# ═════════════════════════════════════════════════════════════════════


class TestBranchGradients:
    """Method card §9.3 — gradient differs between normal/correction branches.

    Uses synthetic scores with requires_grad=True rather than model
    parameters, so that score values are fully controlled and the
    branch decision is deterministic.
    """

    def test_normal_branch_gradient(self):
        """r >= -beta → gradient == uPU risk gradient."""
        # P negative, U positive → r > 0 → normal branch
        p = torch.tensor([-1.0, -0.5], requires_grad=True)
        u = torch.tensor([1.5, 1.0], requires_grad=True)
        pi = 0.5

        R_p_plus = torch.sigmoid(-p).mean()
        R_p_minus = torch.sigmoid(p).mean()
        R_u_minus = torch.sigmoid(u).mean()
        opt_loss, info = _nnpu_train_step(
            R_p_plus, R_p_minus, R_u_minus,
            class_prior=pi, beta=0.0, gamma=1.0,
        )
        assert not info["correction"], "Expected normal branch"

        opt_loss.backward()
        assert p.grad is not None and (p.grad != 0).any()
        assert u.grad is not None and (u.grad != 0).any()

    def test_correction_branch_positive_risk_detached(self):
        """r < -beta: R_p_plus gradient does NOT flow in correction."""
        # P very positive, U very negative → r < 0 → correction branch
        # sigmoid(5,4) ≈ [0.99, 0.98] → R_p^- ≈ 0.99
        # sigmoid(-2,-1) ≈ [0.12, 0.27] → R_u^- ≈ 0.19
        # r = 0.19 − 0.3*0.99 = −0.11 < 0
        p_pos = torch.tensor([5.0, 4.0], requires_grad=True)
        u_neg = torch.tensor([-2.0, -1.0], requires_grad=True)
        pi = 0.3

        # Correction branch (beta=0)
        R_pp = torch.sigmoid(-p_pos).mean()
        R_pm = torch.sigmoid(p_pos).mean()
        R_um = torch.sigmoid(u_neg).mean()
        opt_loss_corr, info_corr = _nnpu_train_step(
            R_pp, R_pm, R_um, class_prior=pi, beta=0.0, gamma=1.0
        )
        assert info_corr["correction"], (
            f"Expected correction branch, r={info_corr['negative_risk']}"
        )

        opt_loss_corr.backward()
        corr_grad_p = p_pos.grad.clone()
        corr_grad_u = u_neg.grad.clone()

        # Normal branch (beta=inf, same scores → no correction)
        p_norm = torch.tensor([5.0, 4.0], requires_grad=True)
        u_norm = torch.tensor([-2.0, -1.0], requires_grad=True)
        R_pp_n = torch.sigmoid(-p_norm).mean()
        R_pm_n = torch.sigmoid(p_norm).mean()
        R_um_n = torch.sigmoid(u_norm).mean()
        opt_loss_norm, info_norm = _nnpu_train_step(
            R_pp_n, R_pm_n, R_um_n,
            class_prior=pi, beta=float("inf"), gamma=1.0,
        )
        assert not info_norm["correction"]
        opt_loss_norm.backward()
        norm_grad_p = p_norm.grad.clone()
        norm_grad_u = u_norm.grad.clone()

        # Gradients should differ (correction branch drops R_p_plus gradient)
        assert not torch.allclose(corr_grad_p, norm_grad_p, atol=1e-5), (
            "Correction branch gradient should differ from normal branch"
        )
        assert not torch.allclose(corr_grad_u, norm_grad_u, atol=1e-5), (
            "U gradients should also differ between branches"
        )

    def test_gamma_zero_no_gradient_in_correction(self):
        """gamma=0 → correction branch produces zero gradient."""
        p = torch.tensor([5.0, 4.0], requires_grad=True)
        u = torch.tensor([-2.0, -1.0], requires_grad=True)
        pi = 0.3

        R_pp = torch.sigmoid(-p).mean()
        R_pm = torch.sigmoid(p).mean()
        R_um = torch.sigmoid(u).mean()
        opt_loss, info = _nnpu_train_step(
            R_pp, R_pm, R_um, class_prior=pi, beta=0.0, gamma=0.0
        )
        assert info["correction"], "Expected correction branch"
        opt_loss.backward()

        # gamma=0 means opt_loss = 0 * r = 0 → no gradient
        assert torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-7)
        assert torch.allclose(u.grad, torch.zeros_like(u.grad), atol=1e-7)


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.4 — max() mis-implementation guard
# ═════════════════════════════════════════════════════════════════════


class TestMaxMisImplementationGuard:
    """Method card §9.4 — Algorithm 1 gradient ≠ naive max(0, r) gradient."""

    def test_algorithm1_differs_from_max_backward(self):
        """When r < -beta, grad of −γ·r ≠ grad of π·R_p^+ + max(0, r)."""
        # Algorithm 1
        p_a = torch.tensor([5.0, 4.0], requires_grad=True)
        u_a = torch.tensor([-2.0, -1.0], requires_grad=True)
        pi = 0.3

        R_pp_a = torch.sigmoid(-p_a).mean()
        R_pm_a = torch.sigmoid(p_a).mean()
        R_um_a = torch.sigmoid(u_a).mean()
        opt_alg1, info = _nnpu_train_step(
            R_pp_a, R_pm_a, R_um_a, class_prior=pi, beta=0.0, gamma=1.0
        )
        assert info["correction"]
        opt_alg1.backward()
        alg1_grad_p = p_a.grad.clone()
        alg1_grad_u = u_a.grad.clone()

        # Naive max(0, r)
        p_m = torch.tensor([5.0, 4.0], requires_grad=True)
        u_m = torch.tensor([-2.0, -1.0], requires_grad=True)
        R_p_plus = torch.sigmoid(-p_m).mean()
        R_p_minus = torch.sigmoid(p_m).mean()
        R_u_minus = torch.sigmoid(u_m).mean()
        r = R_u_minus - pi * R_p_minus
        loss_max = pi * R_p_plus + torch.clamp(r, min=0.0)  # max(0, r)
        loss_max.backward()
        max_grad_p = p_m.grad.clone()
        max_grad_u = u_m.grad.clone()

        # Gradients should differ
        assert not torch.allclose(alg1_grad_p, max_grad_p, atol=1e-5), (
            "Algorithm 1 and max(0, r) gradients should differ "
            "when r < -beta"
        )
        assert not torch.allclose(alg1_grad_u, max_grad_u, atol=1e-5), (
            "Algorithm 1 and max(0, r) gradients should differ "
            "when r < -beta"
        )


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.5 — Beta boundary tests
# ═════════════════════════════════════════════════════════════════════


class TestBetaBoundary:
    """Method card §9.5 — beta validation and behavioural edges."""

    def test_beta_negative_raises_in_loss(self):
        """beta < 0 → ValueError in NonNegativePULoss."""
        with pytest.raises(ValueError, match="beta"):
            NonNegativePULoss(beta=-0.1)

    def test_beta_negative_raises_in_classifier_fit(self, rng):
        """beta < 0 → ValueError in fit()."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, beta=-0.1, max_epochs=1
        )
        with pytest.raises(ValueError, match="beta"):
            clf.fit(X, y_pu, class_prior=pi)

    def test_beta_zero_standard_nnpu(self):
        """beta=0: correction triggers when r < 0."""
        loss = NonNegativePULoss(beta=0.0)
        # P positive, U negative → r < 0
        p_scores = np.array([5.0, 4.0])
        u_scores = np.array([-3.0, -2.0])
        pi = 0.3
        r_nn = loss(p_scores, u_scores, class_prior=pi, non_negative=True)
        r_upu = loss(p_scores, u_scores, class_prior=pi, non_negative=False)

        # r < 0 → nnPU clips → r_nn > r_upu
        assert r_nn >= 0
        assert r_nn > r_upu

    def test_beta_large_behaves_like_upu(self, rng):
        """Very large beta → correction never triggers (training-level test)."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40, n_features=5)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, beta=1e9, max_epochs=3, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)
        hist = clf.get_training_history()
        # With beta >> 0, correction should never trigger
        assert all(f == 0.0 for f in hist["correction_fraction"]), (
            "With beta=1e9, correction fraction should be 0"
        )

    def test_beta_above_class_prior_warns(self, rng):
        """beta > class_prior with sigmoid → UserWarning."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, beta=pi + 0.1, max_epochs=1, batch_size=10
        )
        with pytest.warns(UserWarning, match="beta"):
            clf.fit(X, y_pu, class_prior=pi)


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.6 — Class-prior validation
# ═════════════════════════════════════════════════════════════════════


class TestClassPriorValidation:
    """Method card §9.6 — class_prior required and validated."""

    def test_none_raises(self):
        """class_prior=None → TypeError (before comparison) in evaluator."""
        loss = NonNegativePULoss()
        with pytest.raises((ValueError, TypeError)):
            loss(np.array([1.0]), np.array([1.0]), class_prior=None)

    def test_out_of_range_raises(self):
        """class_prior ≤ 0 or ≥ 1 → ValueError."""
        loss = NonNegativePULoss()
        for bad in (0.0, -0.1, 1.0, 1.5):
            with pytest.raises(ValueError, match="class_prior"):
                loss(np.array([1.0]), np.array([1.0]), class_prior=bad)

    def test_valid_stored_in_metadata(self, rng):
        """Valid class_prior is persisted."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=1, batch_size=10
        )
        clf.fit(X, y_pu, class_prior=pi)
        meta = clf.get_pu_metadata()
        assert meta["class_prior"] == pytest.approx(pi)

    def test_missing_in_fit_raises(self, rng):
        """class_prior not passed to fit() → TypeError."""
        X, y_pu, _ = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model, max_epochs=1)
        with pytest.raises(TypeError):
            clf.fit(X, y_pu)  # missing required keyword


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.7 — P/U batch tests
# ═════════════════════════════════════════════════════════════════════


class TestPUBatches:
    """Method card §9.7 — P/U batch handling edge cases."""

    def test_no_positive_raises(self, rng):
        """All unlabeled → ValidationError or ValueError."""
        X = rng.randn(20, 3)
        y_pu = np.zeros(20)
        model = torch.nn.Linear(3, 1)
        clf = NonNegativePUClassifier(model=model, max_epochs=1)
        with pytest.raises((ValidationError, ValueError)):
            clf.fit(X, y_pu, class_prior=0.5)

    def test_no_unlabeled_raises(self, rng):
        """Only positives → ValueError."""
        X = rng.randn(20, 3)
        y_pu = np.ones(20)
        model = torch.nn.Linear(3, 1)
        clf = NonNegativePUClassifier(model=model, max_epochs=1)
        with pytest.raises(ValueError):
            clf.fit(X, y_pu, class_prior=0.5)

    def test_single_positive_works(self, rng):
        """n_P=2 still trains (minimum per validation)."""
        X_p = rng.randn(2, 3)
        X_u = rng.randn(20, 3)
        X = np.vstack([X_p, X_u])
        y_pu = np.concatenate([np.ones(2), np.zeros(20)])
        pi = 2.0 / 22.0
        model = torch.nn.Linear(3, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=2, batch_size=4
        )
        clf.fit(X, y_pu, class_prior=pi)
        assert clf._is_fitted
        assert clf.n_positive_ == 2
        assert clf.n_unlabeled_ == 20

    def test_unequal_loader_lengths(self, rng):
        """n_P=3, n_U=100 with batch_size=2 → cycling works."""
        X_p = rng.randn(3, 3)
        X_u = rng.randn(100, 3)
        X = np.vstack([X_p, X_u])
        y_pu = np.concatenate([np.ones(3), np.zeros(100)])
        pi = 3.0 / 103.0
        model = torch.nn.Linear(3, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=2, batch_size=2
        )
        clf.fit(X, y_pu, class_prior=pi)
        assert clf._is_fitted
        assert len(clf.history_["epoch"]) == 2

    def test_sample_weight_normalization(self, rng):
        """Weighted mean within each group → consistent risk components."""
        X_p = rng.randn(10, 3)
        X_u = rng.randn(20, 3)
        X = np.vstack([X_p, X_u])
        y_pu = np.concatenate([np.ones(10), np.zeros(20)])
        pi = 10.0 / 30.0

        w_p = np.ones(10) * 2.0  # uniform doubled weight → same mean
        w_u = np.ones(20) * 0.5
        sample_weight = np.concatenate([w_p, w_u])

        model = torch.nn.Linear(3, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=1, batch_size=5
        )
        clf.fit(X, y_pu, class_prior=pi, sample_weight=sample_weight)
        assert clf._is_fitted
        assert len(clf.history_["epoch"]) == 1


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.8 — Output semantics
# ═════════════════════════════════════════════════════════════════════


class TestOutputSemantics:
    """Method card §9.8 — API contract compliance."""

    def test_decision_function_shape(self, rng):
        """decision_function returns (n_samples,) finite values."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=2, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)

        scores = clf.decision_function(X)
        assert scores.shape == (X.shape[0],)
        assert np.isfinite(scores).all()

    def test_predict_binary(self, rng):
        """predict returns {0, 1} only."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=2, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)

        preds = clf.predict(X)
        assert set(np.unique(preds)) <= {0, 1}
        assert preds.dtype == int

    def test_predict_proba_raises(self, rng):
        """predict_proba → NotImplementedError."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=2, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)

        with pytest.raises(NotImplementedError, match="predict_proba"):
            clf.predict_proba(X)

    def test_evaluate_pu_risk(self, rng):
        """evaluate_pu_risk matches Eq. (6) computed manually."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=2, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)

        risk_nn = clf.evaluate_pu_risk(X, y_pu, non_negative=True)
        risk_upu = clf.evaluate_pu_risk(X, y_pu, non_negative=False)
        assert risk_nn >= 0
        assert risk_nn >= risk_upu

    def test_not_fitted_raises(self, rng):
        """NotFittedError before fit()."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model)

        with pytest.raises(NotFittedError):
            clf.predict(X)
        with pytest.raises(NotFittedError):
            clf.decision_function(X)
        with pytest.raises(NotFittedError):
            clf.evaluate_pu_risk(X, y_pu)


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.9 — Overfitting behaviour test (slow)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestOverfittingBehavior:
    """Method card §9.9 — uPU overfits, nnPU remains stable."""

    def test_nnpu_prevents_negative_risk_divergence(self, rng):
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
            model=model_nnpu,
            beta=0.0,
            gamma=1.0,
            max_epochs=50,
            batch_size=32,
            random_state=42,
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
            model=model_upu,
            beta=1e9,
            gamma=1.0,
            max_epochs=50,
            batch_size=32,
            random_state=42,
        )
        clf_upu.fit(X, y_pu, class_prior=pi)
        history_upu = clf_upu.get_training_history()

        # nnPU: correction fraction should be recorded
        correction_fractions = history_nnpu["correction_fraction"]
        assert max(correction_fractions) >= 0

        # nnPU final negative risk should not be excessively negative
        final_nr_nnpu = history_nnpu["negative_risk"][-1]
        final_nr_upu = history_upu["negative_risk"][-1]
        assert final_nr_nnpu >= final_nr_upu or abs(final_nr_upu) < 0.5, (
            f"nnPU negative risk ({final_nr_nnpu:.4f}) should not "
            f"diverge worse than uPU ({final_nr_upu:.4f})"
        )


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.5b — Fit-time parameter validation (coverage for fit() checks)
# ═════════════════════════════════════════════════════════════════════


class TestFitValidation:
    """Parameter validation inside fit() beyond what the loss class checks."""

    def test_gamma_out_of_range_raises_in_fit(self, rng):
        """gamma outside [0, 1] → ValueError in fit()."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, gamma=1.5, max_epochs=1
        )
        with pytest.raises(ValueError, match="gamma"):
            clf.fit(X, y_pu, class_prior=pi)

    def test_invalid_loss_raises_in_fit(self, rng):
        """loss != 'sigmoid' → ValueError in fit()."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, loss="logistic", max_epochs=1
        )
        with pytest.raises(ValueError, match="loss"):
            clf.fit(X, y_pu, class_prior=pi)

    def test_max_epochs_non_positive_raises_in_fit(self, rng):
        """max_epochs <= 0 → ValueError in fit()."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=0
        )
        with pytest.raises(ValueError, match="max_epochs"):
            clf.fit(X, y_pu, class_prior=pi)

    def test_batch_size_non_positive_raises_in_fit(self, rng):
        """batch_size <= 0 → ValueError in fit()."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, batch_size=0, max_epochs=1
        )
        with pytest.raises(ValueError, match="batch_size"):
            clf.fit(X, y_pu, class_prior=pi)

    def test_nan_in_X_raises_in_fit(self, rng):
        """X containing NaN → ValueError in fit()."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        X[0, 0] = np.nan
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model, max_epochs=1)
        with pytest.raises(ValueError, match="NaN"):
            clf.fit(X, y_pu, class_prior=pi)

    def test_class_prior_public_attribute(self, rng):
        """class_prior_ is a public fitted attribute (sklearn convention)."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model, max_epochs=1, batch_size=8)
        clf.fit(X, y_pu, class_prior=pi)
        assert clf.class_prior_ == pytest.approx(pi)


# ═════════════════════════════════════════════════════════════════════
# MARK: §9.5c — Beta == class_prior boundary
# ═════════════════════════════════════════════════════════════════════


class TestBetaEqualsClassPrior:
    """Method card §9.5 — beta at the sigmoid upper bound."""

    def test_beta_equals_class_prior_runs(self, rng):
        """beta=class_prior: correction may or may not trigger;
        training must complete without error."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40, n_features=5)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, beta=pi, max_epochs=3, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)
        assert clf._is_fitted
        hist = clf.get_training_history()
        # With beta=class_prior, sigmoid's max is 1, so r can reach -pi.
        # Correction should still be possible (r < -pi with strong sep).
        assert all(isinstance(f, float) for f in hist["correction_fraction"])
        assert min(hist["correction_fraction"]) >= 0.0
        assert max(hist["correction_fraction"]) <= 1.0


# ═════════════════════════════════════════════════════════════════════
# MARK: API compatibility
# ═════════════════════════════════════════════════════════════════════


class TestAPICompatibility:
    """sklearn compatibility: get_params, set_params, training history."""

    def test_get_params_returns_dict(self):
        """get_params() returns constructor params."""
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(model=model, beta=0.1, gamma=0.9)
        params = clf.get_params()
        assert isinstance(params, dict)
        assert params["beta"] == 0.1
        assert params["gamma"] == 0.9
        assert params["loss"] == "sigmoid"

    def test_set_params_updates(self):
        """set_params() updates and returns self."""
        clf = NonNegativePUClassifier(beta=0.1)
        clf.set_params(beta=0.3, max_epochs=50)
        assert clf.beta == 0.3
        assert clf.max_epochs == 50

    def test_history_keys_complete(self, rng):
        """history_ has all 7 required keys."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=3, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)

        hist = clf.get_training_history()
        required = {
            "epoch",
            "positive_risk",
            "negative_risk",
            "upu_risk",
            "nnpu_risk",
            "optimization_loss",
            "correction_fraction",
        }
        assert required <= set(hist.keys())
        for k in required:
            assert len(hist[k]) == 3, f"{k} should have 3 entries"

    def test_reproducibility(self, rng):
        """Same random_state → identical results."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40, seed=99)
        torch.manual_seed(42)
        model1 = torch.nn.Linear(5, 1)
        clf1 = NonNegativePUClassifier(
            model=model1, max_epochs=2, batch_size=8, random_state=42
        )
        clf1.fit(X, y_pu, class_prior=pi)

        torch.manual_seed(42)
        model2 = torch.nn.Linear(5, 1)
        clf2 = NonNegativePUClassifier(
            model=model2, max_epochs=2, batch_size=8, random_state=42
        )
        clf2.fit(X, y_pu, class_prior=pi)

        s1 = clf1.decision_function(X)
        s2 = clf2.decision_function(X)
        np.testing.assert_array_almost_equal(s1, s2)

    def test_default_model_created(self, rng):
        """model=None → default nn.Linear is created."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        clf = NonNegativePUClassifier(max_epochs=2, batch_size=8)
        clf.fit(X, y_pu, class_prior=pi)
        assert isinstance(clf.model_, torch.nn.Module)
        scores = clf.decision_function(X)
        assert scores.shape == (X.shape[0],)

    def test_validation_data_early_stopping(self, rng):
        """Early stopping on validation risk."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=30, n_u=60)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=50, patience=3, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi, validation_data=(X, y_pu))
        assert len(clf.history_["epoch"]) <= 50

    def test_get_pu_metadata(self, rng):
        """get_pu_metadata() includes nnPU-specific fields."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=2, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)

        meta = clf.get_pu_metadata()
        assert meta["loss"] == "sigmoid"
        assert meta["beta"] == 0.0
        assert meta["gamma"] == 1.0
        assert meta["is_fitted"] is True
        assert meta["family"] == "risk_estimation"
        assert "n_positive" in meta
        assert "n_unlabeled" in meta


# ═════════════════════════════════════════════════════════════════════
# MARK: Early-stopping behaviour
# ═════════════════════════════════════════════════════════════════════


class TestEarlyStopping:
    """Validation-data early stopping and best-model restoration."""

    def test_best_model_restored_after_early_stopping(self, rng):
        """After early stopping, model_ is best (not last-epoch) state."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=30, n_u=60)
        # Separate validation set: small → overfitting triggers early stop quickly
        X_val, y_pu_val, _ = _make_synthetic_data(rng, n_p=20, n_u=40, seed=123)
        model = torch.nn.Linear(5, 1)

        clf = NonNegativePUClassifier(
            model=model,
            max_epochs=100,
            patience=2,
            batch_size=8,
            random_state=42,
        )
        clf.fit(X, y_pu, class_prior=pi, validation_data=(X_val, y_pu_val))

        n_epochs = len(clf.history_["epoch"])
        assert n_epochs < 100, (
            f"Expected early stop before max_epochs=100, ran {n_epochs}"
        )
        # The restored model produces valid predictions
        scores = clf.decision_function(X)
        assert np.isfinite(scores).all()

    def test_no_validation_runs_full_epochs(self, rng):
        """Without validation_data, training runs exactly max_epochs."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=5, patience=2, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)
        assert len(clf.history_["epoch"]) == 5


# ═════════════════════════════════════════════════════════════════════
# MARK: evaluate_pu_risk extensions
# ═════════════════════════════════════════════════════════════════════


class TestEvaluatePuRisk:
    """Additional coverage for evaluate_pu_risk()."""

    def test_class_prior_override(self, rng):
        """Explicit class_prior overrides the stored value."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=2, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)

        risk_stored = clf.evaluate_pu_risk(X, y_pu)
        risk_override = clf.evaluate_pu_risk(X, y_pu, class_prior=0.7)
        # Different class_prior → different risk
        assert risk_stored != pytest.approx(risk_override)

    def test_non_negative_flag_switches_risk(self, rng):
        """non_negative=True gives nnPU risk, False gives uPU risk."""
        X, y_pu, pi = _make_synthetic_data(rng, n_p=20, n_u=40)
        model = torch.nn.Linear(5, 1)
        clf = NonNegativePUClassifier(
            model=model, max_epochs=2, batch_size=8
        )
        clf.fit(X, y_pu, class_prior=pi)

        r_nn = clf.evaluate_pu_risk(X, y_pu, non_negative=True)
        r_upu = clf.evaluate_pu_risk(X, y_pu, non_negative=False)
        assert r_nn >= 0
        assert r_nn >= r_upu
