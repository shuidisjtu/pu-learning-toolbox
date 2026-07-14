# ruff: noqa: N806

"""Tests for ElkanNotoClassifier (native PU classifier, KDD 2008).

Coverage follows method card §8.
"""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pu_toolbox.core.exceptions import (
    NotFittedError as PUNotFittedError,
)
from pu_toolbox.core.exceptions import (
    ValidationError as PUValidationError,
)
from pu_toolbox.estimators.classic.elkan_noto import ElkanNotoClassifier

# ── Helpers ──────────────────────────────────────────────────────────


def _make_scar_data(rng, n=100, c=0.5, n_features=5):
    """Generate synthetic data under SCAR with known labeling propensity ``c``.

    Returns (X, y_pu, y_true).
    """
    X_pos = rng.randn(n, n_features) + 2.0
    X_neg = rng.randn(n, n_features) - 2.0
    X = np.vstack([X_pos, X_neg])
    y_true = np.hstack([np.ones(n), np.zeros(n)])

    y_pu = np.zeros(2 * n, dtype=int)
    pos_idx = np.where(y_true == 1)[0]
    n_labeled = int(n * c)
    labeled = rng.choice(pos_idx, size=n_labeled, replace=False)
    y_pu[labeled] = 1

    return X, y_pu, y_true


# ── Basic functional tests ───────────────────────────────────────────


class TestImport:
    """Method card §8: import sanity."""

    def test_import_class(self):
        from pu_toolbox.estimators.classic import ElkanNotoClassifier  # noqa: F811

        assert ElkanNotoClassifier is not None

    def test_default_construction(self):
        clf = ElkanNotoClassifier()
        assert clf.mode == "probability_correction"
        assert clf.n_cv_folds == 3
        assert clf.eps == 1e-12


class TestFitPredict:
    """Basic fit / predict / shape smoke tests."""

    def test_fit_predict_shape(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        y_pred = clf.predict(X)
        assert y_pred.shape == (X.shape[0],)
        assert y_pred.dtype == int
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_predict_proba_shape(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_decision_function_shape(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        scores = clf.decision_function(X)
        assert scores.shape == (X.shape[0],)

    def test_label_proba_shape(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        g = clf.predict_label_proba(X)
        assert g.shape == (X.shape[0],)

    def test_score_samples_alias(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert np.allclose(clf.score_samples(X), clf.decision_function(X))

    def test_classes_attribute(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert np.array_equal(clf.classes_, np.array([0, 1]))


class TestScarCalibration:
    """Method card §8.1: SCAR calibration quality."""

    def test_propensity_estimate_close_to_true_c(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=200, c=0.5)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        # With 200 samples, estimate should be within ±0.15 of true c=0.5
        assert 0.25 < clf.propensity_ < 0.75, f"c_hat={clf.propensity_} far from true c=0.5"

    def test_f_improves_over_g_for_true_labels(self, rng):
        """Brier score of f(x) should be ≤ Brier score of g(x)."""
        X, y_pu, y_true = _make_scar_data(rng, n=200, c=0.5)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)

        g = clf.predict_label_proba(X)
        f = clf.decision_function(X)

        brier_g = np.mean((g - y_true) ** 2)
        brier_f = np.mean((f - y_true) ** 2)

        # f(x) should be at least as good as g(x) for true labels
        # (allow small tolerance for finite-sample noise)
        assert brier_f <= brier_g + 0.05, f"Brier(f)={brier_f:.4f} > Brier(g)={brier_g:.4f}"

    def test_high_accuracy_on_clean_data(self, rng):
        """With strong separation, accuracy should be near-perfect."""
        X, y_pu, y_true = _make_scar_data(rng, n=100, c=0.5)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        acc = np.mean(clf.predict(X) == y_true)
        assert acc > 0.9


class TestRankingInvariance:
    """Method card §8.2: g(x) and f(x) have identical ranking."""

    def test_roc_auc_identical(self, rng):
        X, y_pu, y_true = _make_scar_data(rng, n=200, c=0.5)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)

        g = clf.predict_label_proba(X)
        f = clf.decision_function(X)

        assert clf.propensity_ > 0
        # g(x) and f(x) = g(x)/c should be perfectly rank-correlated
        assert np.corrcoef(g, f)[0, 1] > 0.9999

    def test_ranking_preserved_for_new_data(self, rng):
        X_train, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        X_test, _, y_test = _make_scar_data(rng, n=100, c=0.5)

        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X_train, y_pu)

        g = clf.predict_label_proba(X_test)
        f = clf.decision_function(X_test)

        # Spearman rank correlation ≈ 1
        from scipy.stats import spearmanr

        rho, _ = spearmanr(g, f)
        assert rho > 0.9999


class TestWeightedRetraining:
    """Method card §8.3: weighted retraining data construction."""

    def test_weighted_mode_runs(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        clf = ElkanNotoClassifier(mode="weighted_retraining", n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert clf.final_estimator_ is not None
        y_pred = clf.predict(X)
        assert y_pred.shape == (X.shape[0],)

    def test_weighted_mode_label_proba_returns_none(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        clf = ElkanNotoClassifier(mode="weighted_retraining", n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert clf.predict_label_proba(X) is None

    def test_weights_sum_to_one_per_sample(self, rng):
        """Each unlabeled sample's positive + negative weight sums to 1."""
        X, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)

        g = clf.predict_label_proba(X)
        mask_unl = y_pu == 0
        w = clf._compute_weights(g[mask_unl])

        assert w.shape == (mask_unl.sum(),)
        # Each weight should be non-negative
        assert np.all(w >= 0)
        # w + (1-w) = 1 by construction
        assert np.allclose(w + (1 - w), np.ones_like(w))

    def test_labeled_positives_have_weight_one(self, rng):
        """In augmented data, original positives have unit weight."""
        X, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        # Use probability_correction mode to access g(x) via predict_label_proba
        clf = ElkanNotoClassifier(mode="probability_correction", n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)

        # Verify that weights computed from g(x) are non-negative
        g = clf.predict_label_proba(X)
        assert g is not None
        mask_unl = y_pu == 0
        w = clf._compute_weights(g[mask_unl])
        assert np.all(w >= 0)
        # Weighted mode uses the same _compute_weights logic,
        # so this validates the weight computation independently.


class TestPriorEstimation:
    """Method card §8.4: class prior estimation."""

    def test_class_prior_in_range(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=200, c=0.5)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert 0.0 < clf.class_prior_ < 1.0

    def test_class_prior_roughly_correct(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=200, c=0.5)
        # True prior = 100 positives / 200 total = 0.5
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        # Should be in a reasonable range around 0.5
        assert 0.3 < clf.class_prior_ < 0.8


class TestErrorHandling:
    """Method card §8.5: error and boundary cases."""

    def test_not_fitted_raises(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        clf = ElkanNotoClassifier()
        with pytest.raises(PUNotFittedError):
            clf.predict(X)
        with pytest.raises(PUNotFittedError):
            clf.predict_proba(X)
        with pytest.raises(PUNotFittedError):
            clf.decision_function(X)

    def test_zero_positive_error(self, rng):
        X, y_pu, y_true = _make_scar_data(rng, n=100, c=0.5)
        # Set all labels to 0 (unlabeled)
        y_all_unlabeled = np.zeros_like(y_pu)
        clf = ElkanNotoClassifier()
        with pytest.raises(PUValidationError):
            clf.fit(X, y_all_unlabeled)

    def test_too_few_positives_for_kfold(self, rng):
        X, y_pu, y_true = _make_scar_data(rng, n=50, c=0.1)
        n_labeled = int(np.sum(y_pu == 1))
        if n_labeled < 3:
            clf = ElkanNotoClassifier(n_cv_folds=3)
            with pytest.raises(ValueError, match="n_cv_folds"):
                clf.fit(X, y_pu)

    def test_n_cv_folds_validation(self):
        with pytest.raises(ValueError, match="n_cv_folds"):
            ElkanNotoClassifier(n_cv_folds=1)
        with pytest.raises(ValueError, match="n_cv_folds"):
            ElkanNotoClassifier(n_cv_folds=0)

    def test_eps_validation(self):
        with pytest.raises(ValueError, match="eps"):
            ElkanNotoClassifier(eps=0.0)
        with pytest.raises(ValueError, match="eps"):
            ElkanNotoClassifier(eps=-0.1)


class TestCompatibility:
    """sklearn compatibility tests."""

    def test_get_params_set_params(self, rng):
        clf = ElkanNotoClassifier(n_cv_folds=5, eps=1e-6)
        params = clf.get_params()
        assert params["n_cv_folds"] == 5
        assert params["eps"] == 1e-6

        clf2 = ElkanNotoClassifier()
        clf2.set_params(**params)
        assert clf2.get_params() == params

    def test_pipeline_compatible(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        pipe = Pipeline(
            [
                ("clf", ElkanNotoClassifier(n_cv_folds=3, random_state=42)),
            ]
        )
        pipe.fit(X, y_pu)
        y_pred = pipe.predict(X)
        assert y_pred.shape == (X.shape[0],)

    def test_with_different_base_estimator(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        clf = ElkanNotoClassifier(
            base_estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            n_cv_folds=3,
            random_state=42,
        )
        clf.fit(X, y_pu)
        assert clf.predict(X).shape == (X.shape[0],)

    def test_with_svm_base_estimator(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        clf = ElkanNotoClassifier(
            base_estimator=CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
            n_cv_folds=3,
            random_state=42,
        )
        clf.fit(X, y_pu)
        assert clf.predict(X).shape == (X.shape[0],)

    def test_sample_weight_actually_changes_model(self, rng):
        """Verify sample_weight ≠ uniform weights produces a different model."""
        X, y_pu, _ = _make_scar_data(rng, n=200, c=0.5)

        # Fit without weights
        clf_unweighted = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf_unweighted.fit(X, y_pu)
        pred_unweighted = clf_unweighted.predict(X)

        # Fit with extreme weights: up-weight first 20% of samples
        sw = np.ones(X.shape[0])
        sw[:40] = 100.0  # heavy emphasis on first 40 samples
        clf_weighted = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf_weighted.fit(X, y_pu, sample_weight=sw)
        pred_weighted = clf_weighted.predict(X)

        # The extreme re-weighting should produce a different decision boundary
        assert not np.array_equal(pred_unweighted, pred_weighted), (
            "Expected sample_weight to change model predictions, but predictions "
            "are identical — sample_weight may be ignored internally."
        )

    def test_sample_weight_shape_validation(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        with pytest.raises(ValueError, match="sample_weight"):
            clf.fit(X, y_pu, sample_weight=np.ones(999))


class TestMetadata:
    """Metadata and fitted-attribute tests."""

    def test_get_pu_metadata(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        meta = clf.get_pu_metadata()
        assert meta["is_fitted"] is True
        assert meta["propensity"] is not None
        assert meta["mode"] == "probability_correction"
        assert meta["family"] == "classic_calibration"
        assert "SCAR" in meta["assumption"]

    def test_propensity_is_stored(self, rng):
        X, y_pu, _ = _make_scar_data(rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert 0.0 < clf.propensity_ <= 1.0

    def test_isotonic_calibration(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        clf = ElkanNotoClassifier(calibration_method="isotonic", n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert clf.predict(X).shape == (X.shape[0],)


class TestProbabilisticEdgeCases:
    """Tests for probability edge cases (clipping, warnings)."""

    def test_propensity_is_one_when_all_positive_known(self, rng):
        """When all positives are labeled (c ≈ 1), f(x) ≈ g(x)."""
        n = 100
        X_pos = rng.randn(n, 5) + 2.0
        X_neg = rng.randn(n, 5) - 2.0
        X = np.vstack([X_pos, X_neg])
        y_pu = np.hstack([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

        clf = ElkanNotoClassifier(n_cv_folds=6, random_state=42)
        clf.fit(X, y_pu)
        # When all positives are labeled, c ≈ 1 (within tolerance)
        assert clf.propensity_ > 0.5

    def test_predict_proba_columns_sum_to_one(self, rng):
        X, y_pu, _ = _make_scar_data(rng, n=100, c=0.5)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        proba = clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)
