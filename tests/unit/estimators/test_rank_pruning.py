# ruff: noqa: N806
"""RP confident counts, author strict rank ties, OOF isolation and weighting."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.estimators.classic.rank_pruning import (
    RankPruningClassifier,
    confident_positive_noise_rate,
    rank_pruning_pu_mask,
)
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

pytestmark = pytest.mark.unit


def data():
    rng = np.random.RandomState(8)
    X = np.r_[rng.normal(2, 0.5, (12, 3)), rng.normal(2, 0.5, (6, 3)), rng.normal(-2, 0.5, (18, 3))]
    return X, np.r_[np.ones(12, int), np.zeros(24, int)]


@pytest.mark.math
def test_confident_counts_and_pruning_match_pu_equations():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.86, 0.2, 0.1, 0.05, 0.02, 0.01])
    assert confident_positive_noise_rate(y, probabilities) == 0.5
    mask, prior, pi0 = rank_pruning_pu_mask(y, probabilities, 0.5, min_retained=1)
    assert prior == pytest.approx(0.5) and pi0 == pytest.approx(1 / 3)
    np.testing.assert_array_equal(mask, [False, False, True, True, False, False, False, False])


def test_edge_strict_threshold_ties_and_minimum_retention_match_author_code():
    y = np.array([1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.85, 0.2, 0.2, 0.05])
    mask, _, _ = rank_pruning_pu_mask(y, probabilities, 0.5, min_retained=1)
    np.testing.assert_array_equal(mask, [False, False, True, False, False, False])
    assert not rank_pruning_pu_mask(y, probabilities, 0.5)[0].any()


def test_fit_crossfit_isolation_reweighting_pickle_and_registry():
    X, y = data()
    fitted = clone(RankPruningClassifier(random_state=3, min_retained=2))
    with pytest.raises(NotFittedError):
        fitted.predict(X)
    fitted.fit(X, y)
    heldout_all = []
    for fold in fitted.cv_fold_indices_:
        assert not set(fold["train"]) & set(fold["heldout"])
        heldout_all.extend(fold["heldout"])
    assert sorted(heldout_all) == list(range(len(y)))
    assert fitted.rho0_ == 0 and fitted.rho1_ == np.mean(fitted.fold_noise_rates_)
    assert not fitted.pruned_mask_[y == 1].any()
    kept_y = y[fitted.retained_indices_]
    np.testing.assert_array_equal(fitted.final_sample_weights_[kept_y == 0], 1)
    np.testing.assert_allclose(fitted.final_sample_weights_[kept_y == 1], 1 / (1 - fitted.rho1_))
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.predict_proba(X), fitted.predict_proba(X))
    np.testing.assert_array_equal(fitted.predict(X), fitted.predict_proba(X)[:, 1] >= 0.5)
    register_all_builtin_methods()
    assert get_algorithm("rp") is RankPruningClassifier
    assert get_algorithm("rank_pruning") is RankPruningClassifier


def test_seed_known_noise_and_fail_closed_views_and_base_estimator():
    X, y = data()
    first = RankPruningClassifier(frac_pos2neg=0.2, random_state=9).fit(X, y)
    second = RankPruningClassifier(frac_pos2neg=0.2, random_state=9).fit(X, y, class_prior=0.4)
    assert first.rho1_ == 0.2 and first.rho0_ == 0
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
    with pytest.raises(ValueError, match="native OS"):
        RankPruningClassifier().fit(X, y, os_or_ts="ts")
    with pytest.raises(TypeError, match="sample_weight"):
        RankPruningClassifier(base_estimator=KNeighborsClassifier()).fit(X, y)
    with pytest.raises(NotImplementedError):
        RankPruningClassifier().fit(X, y, sample_weight=np.ones(len(y)))


@pytest.mark.parametrize(
    "kwargs", [{"n_cv_folds": 1}, {"n_cv_folds": 15}, {"frac_pos2neg": 1}, {"min_retained": 0}]
)
def test_invalid_parameters(kwargs):
    X, y = data()
    with pytest.raises(ValueError):
        RankPruningClassifier(**kwargs).fit(X, y)
