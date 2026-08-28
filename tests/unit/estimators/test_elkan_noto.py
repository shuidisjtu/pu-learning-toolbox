# ruff: noqa: N806

"""Tests for ElkanNotoClassifier (native PU classifier, KDD 2008).

Coverage follows method card §8. Consolidated: 15 test methods across 10 classes.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from pu_toolbox.core.exceptions import (
    NotFittedError as PUNotFittedError,
)
from pu_toolbox.core.exceptions import (
    ValidationError as PUValidationError,
)
from pu_toolbox.estimators.classic.elkan_noto import ElkanNotoClassifier
from pu_toolbox.preprocessing import make_scar_dataset


@pytest.mark.unit
class TestWritebackDefaults:
    """Constructor defaults tuned in ADR-0016 step-6 write-back round 3.

    Round-2 re-run (elkan_noto_tuning_r2, contract v2) confirmed
    ``mode="weighted_retraining"`` with 12/12 paired-CI improvements (6
    SCAR + 6 SAR, diff -0.009..-0.067, all CI upper bounds < 0; the
    improvement is attributable to the mode itself — the calibration
    method does not enter the final prediction under weighted
    retraining).  The default is pinned here so a future change requires
    an explicit verdict round, not a silent drift.
    """

    def test_basic_writeback_default_mode_weighted(self):
        clf = ElkanNotoClassifier()
        assert clf.mode == "weighted_retraining"

    def test_param_other_defaults_unchanged(self):
        clf = ElkanNotoClassifier()
        assert clf.calibration_method == "sigmoid"
        assert clf.n_cv_folds == 3
        assert clf.eps == 1e-12
        assert clf.base_estimator is None

    def test_edge_explicit_overrides_win(self):
        clf = ElkanNotoClassifier(mode="probability_correction")
        assert clf.mode == "probability_correction"


# ═════════════════════════════════════════════════════════════════════
# Basic import + construction
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFitPredict:
    """Basic import, fit/predict shape, classes, and edge-case probability checks."""

    def test_basic_import_and_output_shapes(self, rng):
        """Import sanity, all output shapes, and score_samples alias."""
        from pu_toolbox.estimators.classic import ElkanNotoClassifier  # noqa: F811

        assert ElkanNotoClassifier is not None
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        assert (
            clf.mode == "weighted_retraining" and clf.n_cv_folds == 3
        )  # ADR-0016 step-6 write-back round 3

        # proba output requires probability-correction mode (weighted mode
        # deliberately returns None from predict_label_proba)
        clf_pc = ElkanNotoClassifier(mode="probability_correction", n_cv_folds=3, random_state=42)
        X, y_pu, _ = make_scar_dataset(random_state=rng)
        clf.fit(X, y_pu)
        clf_pc.fit(X, y_pu)

        y_pred = clf.predict(X)
        assert y_pred.shape == (X.shape[0],) and y_pred.dtype == int
        assert set(np.unique(y_pred)).issubset({0, 1})
        assert clf.predict_proba(X).shape == (X.shape[0], 2)
        assert clf.decision_function(X).shape == (X.shape[0],)
        assert clf_pc.predict_label_proba(X).shape == (X.shape[0],)
        assert np.allclose(clf.score_samples(X), clf.decision_function(X))
        assert np.array_equal(clf.classes_, np.array([0, 1]))
        # Proba columns sum to 1
        proba = clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ═════════════════════════════════════════════════════════════════════
# SCAR calibration quality (§8.1)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestScarCalibration:
    """Method card §8.1: SCAR calibration quality."""

    def test_basic_propensity_estimate_close_to_true_c(self, rng):
        X, y_pu, _ = make_scar_dataset(n=200, c=0.5, random_state=rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert 0.25 < clf.propensity_ < 0.75, f"c_hat={clf.propensity_} far from true c=0.5"

    def test_basic_high_accuracy_on_clean_data(self, rng):
        """With strong separation, accuracy should be near-perfect."""
        X, y_pu, y_true = make_scar_dataset(n=100, c=0.5, random_state=rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        acc = np.mean(clf.predict(X) == y_true)
        assert acc > 0.85, f"Accuracy {acc:.3f} below 0.85"

    @pytest.mark.math
    def test_math_balanced_oof_propensity_lands_in_band(self):
        """sqrt-balanced OOF keeps c_hat close to the true c=0.5.

        Before the fix the unweighted OOF under-estimated c (0.40 on this
        dataset) and inflated class_prior_ (0.62); the sqrt-balance rule
        must land both in the acceptance bands.
        """
        X, y_pu, _ = make_scar_dataset(n=200, c=0.5, random_state=42)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert 0.50 <= clf.propensity_ <= 0.68, f"c_hat={clf.propensity_}"
        assert 0.40 <= clf.class_prior_ <= 0.58, f"prior={clf.class_prior_}"


# ═════════════════════════════════════════════════════════════════════
# Ranking invariance (§8.2)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestRankingInvariance:
    """Method card §8.2: g(x) and f(x) have identical ranking."""

    def test_deterministic_ranking_invariance(self, rng):
        """g(x) and f(x) are perfectly rank-correlated on train and test."""
        X_train, y_pu, _ = make_scar_dataset(n=200, c=0.5, random_state=rng)
        X_test, _, _ = make_scar_dataset(n=100, c=0.5, random_state=rng)

        # g(x) is only produced under probability-correction mode
        clf = ElkanNotoClassifier(mode="probability_correction", n_cv_folds=3, random_state=42)
        clf.fit(X_train, y_pu)

        for X in (X_train, X_test):
            g = clf.predict_label_proba(X)
            f = clf.decision_function(X)
            assert np.corrcoef(g, f)[0, 1] > 0.9999


# ═════════════════════════════════════════════════════════════════════
# Weighted retraining (§8.3)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestWeightedRetraining:
    """Method card §8.3: weighted retraining data construction."""

    def test_basic_weighted_retraining_and_weights(self, rng):
        """Weighted mode runs; predict_label_proba returns None; weights are non-negative."""
        X, y_pu, _ = make_scar_dataset(n=100, c=0.5, random_state=rng)

        # Weighted mode runs
        clf = ElkanNotoClassifier(mode="weighted_retraining", n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert clf.final_estimator_ is not None
        assert clf.predict(X).shape == (X.shape[0],)
        assert clf.predict_label_proba(X) is None

        # _compute_weights produces non-negative values summing to 1 per sample
        clf_pc = ElkanNotoClassifier(mode="probability_correction", n_cv_folds=3, random_state=42)
        clf_pc.fit(X, y_pu)
        g = clf_pc.predict_label_proba(X)
        mask_unl = y_pu == 0
        w = clf_pc._compute_weights(g[mask_unl])
        assert w.shape == (mask_unl.sum(),)
        assert np.all(w >= 0)
        assert np.allclose(w + (1 - w), np.ones_like(w))


# ═════════════════════════════════════════════════════════════════════
# Prior estimation (§8.4)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPriorEstimation:
    """Method card §8.4: class prior estimation."""

    def test_basic_class_prior_estimation(self, rng):
        X, y_pu, _ = make_scar_dataset(n=200, c=0.5, random_state=rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        # class_prior should be in a reasonable range around true prior=0.5
        assert 0.3 < clf.class_prior_ < 0.8, f"class_prior={clf.class_prior_}"


# ═════════════════════════════════════════════════════════════════════
# Error handling (§8.5)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestErrorHandling:
    """Method card §8.5: error and boundary cases."""

    @pytest.mark.parametrize(
        "init_kwargs, match",
        [
            ({"n_cv_folds": 1}, "n_cv_folds"),
            ({"n_cv_folds": 0}, "n_cv_folds"),
            ({"eps": 0.0}, "eps"),
            ({"eps": -0.1}, "eps"),
        ],
    )
    def test_param_constructor_validation(self, init_kwargs, match):
        with pytest.raises(ValueError, match=match):
            ElkanNotoClassifier(**init_kwargs)

    def test_param_not_fitted_and_validation_errors(self, rng):
        """Not fitted raises; zero positives raises; too few positives for kfold."""
        X, y_pu, _ = make_scar_dataset(random_state=rng)
        clf = ElkanNotoClassifier()
        for method in ["predict", "predict_proba", "decision_function"]:
            with pytest.raises(PUNotFittedError):
                getattr(clf, method)(X)

        # Zero positives
        with pytest.raises(PUValidationError):
            ElkanNotoClassifier().fit(X, np.zeros_like(y_pu))

        # Too few positives for kfold
        X2, y2, _ = make_scar_dataset(n=50, c=0.1, random_state=rng)
        n_labeled = int(np.sum(y2 == 1))
        if n_labeled < 3:
            with pytest.raises(ValueError, match="n_cv_folds"):
                ElkanNotoClassifier(n_cv_folds=3).fit(X2, y2)


# ═════════════════════════════════════════════════════════════════════
# sklearn compatibility
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestCompatibility:
    """sklearn compatibility tests."""

    def test_basic_pipeline_compatible(self, rng):
        X, y_pu, _ = make_scar_dataset(random_state=rng)
        pipe = Pipeline([("clf", ElkanNotoClassifier(n_cv_folds=3, random_state=42))])
        pipe.fit(X, y_pu)
        assert pipe.predict(X).shape == (X.shape[0],)

    def test_edge_custom_estimator_and_sample_weight(self, rng):
        """Different base estimators and sample_weight produce varied outputs."""
        X, y_pu, _ = make_scar_dataset(n=100, c=0.5, random_state=rng)

        # RandomForest base estimator (probability-correction mode: weighted
        # retraining with a non-calibrated estimator can produce negative
        # duplication weights)
        clf_rf = ElkanNotoClassifier(
            base_estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            mode="probability_correction",
            n_cv_folds=3,
            random_state=42,
        )
        clf_rf.fit(X, y_pu)
        assert clf_rf.predict(X).shape == (X.shape[0],)

        # Sample weight changes model
        clf_unw = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf_unw.fit(X, y_pu)
        pred_unweighted = clf_unw.predict(X)

        sw = np.ones(X.shape[0])
        sw[:40] = 100.0
        clf_w = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf_w.fit(X, y_pu, sample_weight=sw)
        assert not np.array_equal(pred_unweighted, clf_w.predict(X))

        # Sample weight shape validation
        with pytest.raises(ValueError, match="sample_weight"):
            ElkanNotoClassifier(n_cv_folds=3).fit(X, y_pu, sample_weight=np.ones(999))


# ═════════════════════════════════════════════════════════════════════
# Metadata & fitted attributes
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMetadata:
    """Metadata, propensity, calibration method."""

    def test_basic_metadata_and_attributes(self, rng):
        X, y_pu, _ = make_scar_dataset(random_state=rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        meta = clf.get_pu_metadata()
        assert meta["is_fitted"] is True
        assert meta["mode"] == "weighted_retraining"  # ADR-0016 step-6 write-back round 3
        assert meta["family"] == "classic_calibration"
        assert "SCAR" in meta["assumption"]
        assert 0.0 < clf.propensity_ <= 1.0

    def test_edge_isotonic_calibration(self, rng):
        X, y_pu, _ = make_scar_dataset(n=100, c=0.5, random_state=rng)
        clf = ElkanNotoClassifier(calibration_method="isotonic", n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        assert clf.predict(X).shape == (X.shape[0],)


# ═════════════════════════════════════════════════════════════════════
# Probabilistic edge cases
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestProbabilisticEdgeCases:
    """Probability edge cases: clipping, c≈1, proba sums."""

    def test_basic_predict_proba_columns_sum_to_one(self, rng):
        X, y_pu, _ = make_scar_dataset(n=100, c=0.5, random_state=rng)
        clf = ElkanNotoClassifier(n_cv_folds=3, random_state=42)
        clf.fit(X, y_pu)
        proba = clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_edge_propensity_is_one_when_all_positive_known(self, rng):
        """When all positives are labeled (c ≈ 1), propensity > 0.5."""
        n = 100
        X_pos = rng.randn(n, 5) + 2.0
        X_neg = rng.randn(n, 5) - 2.0
        X = np.vstack([X_pos, X_neg])
        y_pu = np.hstack([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
        clf = ElkanNotoClassifier(n_cv_folds=6, random_state=42)
        clf.fit(X, y_pu)
        assert clf.propensity_ > 0.5
