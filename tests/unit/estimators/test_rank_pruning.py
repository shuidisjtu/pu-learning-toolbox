# ruff: noqa: N806
"""RP confident counts, author strict rank ties, OOF isolation and weighting."""

import pickle
import subprocess
import sys

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


@pytest.mark.parametrize("keyword", ["y_true", "support_labels", "test_data", "missing_labels"])
def test_param_binding_failure_invalidates_previous_fit(keyword):
    features, labels = data()
    model = RankPruningClassifier(random_state=3, min_retained=2).fit(features, labels)
    with pytest.raises(TypeError):
        if keyword == "missing_labels":
            model.fit(features)
        else:
            model.fit(features, labels, **{keyword: labels})
    for name in ("predict", "predict_proba", "decision_function", "score_samples"):
        with pytest.raises(NotFittedError):
            getattr(model, name)(features)


def test_basic_determ_guard_preserves_rp_folds_weights_and_refit():
    features, labels = data()
    model = RankPruningClassifier(random_state=3, min_retained=2)
    reference = clone(model)
    model.fit(features, labels)
    type(reference).fit.__wrapped__(reference, features, labels)
    expected = reference.predict_proba(features)
    np.testing.assert_array_equal(model.predict_proba(features), expected)
    np.testing.assert_array_equal(model.retained_indices_, reference.retained_indices_)
    np.testing.assert_array_equal(model.final_sample_weights_, reference.final_sample_weights_)
    for actual, other in zip(model.cv_fold_indices_, reference.cv_fold_indices_, strict=True):
        for role in ("train", "heldout"):
            np.testing.assert_array_equal(actual[role], other[role])
    with pytest.raises(TypeError):
        model.fit(features, labels, y_true=labels)
    assert not model._is_fitted
    model.fit(features, labels)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(features), expected)


def test_edge_rp_lifecycle_guard_does_not_require_torch():
    script = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch deliberately unavailable')
sys.meta_path.insert(0, NoTorch())
import numpy as np
from pu_toolbox.estimators.classic.rank_pruning import RankPruningClassifier
from pu_toolbox.core.exceptions import NotFittedError
rng = np.random.RandomState(8)
x = np.r_[rng.normal(2, .5, (12, 3)), rng.normal(2, .5, (6, 3)), rng.normal(-2, .5, (18, 3))]
y = np.r_[np.ones(12, int), np.zeros(24, int)]
model = RankPruningClassifier(random_state=3, min_retained=2).fit(x, y)
assert model.predict(x).shape == (36,)
try:
    model.fit(x, y, y_true=y)
except TypeError:
    pass
else:
    raise AssertionError('unknown labels accepted')
assert not model._is_fitted
try:
    model.predict(x)
except NotFittedError:
    pass
else:
    raise AssertionError('failed refit left usable predictions')
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
