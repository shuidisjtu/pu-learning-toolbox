"""Intentional PUET adaptations must stay distinguishable from author replay."""

import numpy as np
import pytest

from pu_toolbox.estimators.risk.puet import PUExtraTreesClassifier

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("minimum,leaves", [(1, 2), (2, 1)])
def test_basic_param_child_minimum_rejects_positive_gain_singleton(minimum, leaves):
    features = np.r_[0.0, np.ones(7)][:, None]
    observed = np.r_[np.ones(4, int), np.zeros(4, int)]
    model = PUExtraTreesClassifier(
        0.4,
        n_estimators=1,
        max_depth=1,
        min_samples_leaf=minimum,
        max_features="all",
        random_state=7,
    ).fit(features, observed)
    # Every candidate separates one P from 3P/4U: risk .96 -> 0 + .84.
    # Parent size 8 exceeds either limit; only the *child* guard can reject it.
    assert model.n_leaves_[0] == leaves
    if minimum == 1:
        assert model.feature_importances_[0] == pytest.approx(0.12)
    else:
        assert model.trees_[0].feature == -1
        np.testing.assert_array_equal(model.feature_importances_, [0.0])


class FixedCandidate:
    def __init__(self, fraction):
        self.fraction = fraction
        self.draws = 0

    def choice(self, count, *, size, replace):
        assert count == size == 1 and replace is False
        return np.array([0])

    def random_sample(self):
        self.draws += 1
        return self.fraction


@pytest.mark.parametrize("fraction", [0.01, 0.99])
def test_edge_full_threshold_interval_is_not_author_trimmed_interval(fraction):
    features = np.array([[0.0], [1.0], [9.0], [10.0]])
    observed = np.array([1, 1, 0, 0])
    rng = FixedCandidate(fraction)
    model = PUExtraTreesClassifier(0.4, n_estimators=1, max_depth=1, max_features="all")
    root, importance, leaves, depth = model._build_tree(
        features,
        observed,
        np.arange(4),
        positive_weight=0.2,
        unlabeled_weight=0.5,
        feature_budget=1,
        rng=rng,
    )
    assert rng.draws == 1 and root.feature == 0 and leaves == 2 and depth == 1
    assert root.threshold == pytest.approx(10 * fraction)
    # Locked author tree.py:347-348 draws in [0+.4*(1-0), 9+.6*(10-9)].
    assert root.threshold < 0.4 or root.threshold > 9.6
    assert importance[0] > 0


@pytest.mark.parametrize("prior", [0.95, 0.99])
def test_edge_strictly_negative_nnpu_gain_keeps_parent_leaf(prior):
    features = np.r_[np.zeros(10), np.ones(10)][:, None]
    observed = np.r_[np.ones(9, int), 0, 1, np.zeros(9, int)]
    # Only cut: 9P/1U vs 1P/9U. The left child clamps to zero; the
    # right child's nonnegative risk exceeds the positive parent risk.
    parent = 4 * prior * (1 - prior)
    right_positive = prior / 10
    children = 4 * right_positive * (0.9 - right_positive) / 0.9
    assert children > parent > 0  # Not the already-covered zero-gain case.
    model = PUExtraTreesClassifier(
        prior,
        n_estimators=1,
        max_depth=1,
        max_features="all",
        max_candidates=3,
        random_state=7,
    ).fit(features, observed)
    assert model.trees_[0].feature == -1 and model.n_leaves_[0] == 1
    np.testing.assert_array_equal(model.feature_importances_, [0.0])


@pytest.mark.parametrize("budget", [4, 100])
def test_determ_oversized_feature_budget_clips_without_mutating_parameter(budget):
    rng = np.random.RandomState(4)
    features = rng.normal(size=(40, 3))
    observed = np.r_[np.ones(12, int), np.zeros(28, int)]
    params = dict(n_estimators=5, max_depth=3, max_candidates=3, random_state=9)
    oversized = PUExtraTreesClassifier(0.4, max_features=budget, **params).fit(features, observed)
    all_features = PUExtraTreesClassifier(0.4, max_features="all", **params).fit(features, observed)
    assert oversized.max_features == oversized.get_params()["max_features"] == budget
    np.testing.assert_array_equal(
        oversized.decision_function(features), all_features.decision_function(features)
    )
    np.testing.assert_array_equal(oversized.feature_importances_, all_features.feature_importances_)
    np.testing.assert_array_equal(oversized.n_leaves_, all_features.n_leaves_)
