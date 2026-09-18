"""PU Extra Trees with recursive greedy nnPU quadratic-risk reduction.

Clean-room tabular implementation of the nnPU/quadratic branch in Wilton et
al. (NeurIPS 2022).  The author's repository also includes uPU and logistic
variants; those are deliberately not claimed here.
"""

# ruff: noqa: N803, N806

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.base import BasePUClassifier
from ...core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    SampleWeightSupport,
    Scenario,
    SourceStatus,
)
from ...core.validation import validate_pu_X_y


def _nnpu_quadratic_node_risk(
    n_positive: int, n_unlabeled: int, positive_weight: float, unlabeled_weight: float
) -> float:
    """Paper Proposition 2(a), with global P/U weights retained at each node."""
    wp = n_positive * positive_weight
    wu = n_unlabeled * unlabeled_weight
    if wu <= wp or wp == 0:
        return 0.0
    return float(4.0 * wp * (wu - wp) / wu)


@dataclass(slots=True)
class _Node:
    vote: int
    feature: int = -1
    threshold: float = 0.0
    left: _Node | None = None
    right: _Node | None = None


def _tree_votes(root: _Node, X: np.ndarray) -> np.ndarray:
    """Apply a fitted tree to a batch without recursive Python calls."""
    votes = np.empty(len(X), dtype=np.int8)
    stack = [(root, np.arange(len(X)))]
    while stack:
        node, rows = stack.pop()
        if node.feature < 0:
            votes[rows] = node.vote
            continue
        assert node.left is not None and node.right is not None
        left_mask = X[rows, node.feature] <= node.threshold
        stack.append((node.left, rows[left_mask]))
        stack.append((node.right, rows[~left_mask]))
    return votes


class PUExtraTreesClassifier(BasePUClassifier):
    """Extra Trees trained by nnPU quadratic risk reduction on tabular P/U data.

    Each tree randomly samples nonconstant features and random thresholds,
    then maximizes the decrease in the closed-form nnPU node risk.  Leaves
    vote +1 when their estimated positive mass exceeds negative mass; the
    ensemble returns majority votes.  Scores are vote margins, *not*
    calibrated posterior probabilities.

    ``bootstrap=False`` follows the released author implementation, which
    relies on random split candidates for forest diversity.  Set it to True
    for separate P/U bootstrap samples as described in the paper's Section 4.
    """

    family = AlgorithmFamily.RISK_ESTIMATION
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_RELATED
    backend = Backend.NUMPY
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED

    def __init__(
        self,
        class_prior: float,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: int | str = "sqrt",
        max_candidates: int = 1,
        bootstrap: bool = False,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_candidates = max_candidates
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> PUExtraTreesClassifier:
        """Build independent randomized trees from reliable P and marginal U."""
        if sample_weight is not None:
            raise NotImplementedError("PUET does not implement sample_weight")
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="PUET")
        if not np.issubdtype(X.dtype, np.number) or not np.isrealobj(X):
            raise ValueError("X must contain real numeric values")
        if not np.isfinite(X).all():
            raise ValueError("X must contain finite values")
        X = np.asarray(X, dtype=np.float64)
        n_positive = int(np.count_nonzero(y_pu == 1))
        n_unlabeled = len(X) - n_positive
        if n_unlabeled == 0:
            raise ValueError("PUET needs at least one unlabeled sample")
        prior = self.class_prior if class_prior is None else class_prior
        if (
            not isinstance(prior, int | float | np.number)
            or not np.isfinite(prior)
            or not 0 < prior < 1
        ):
            raise ValueError("class_prior must be finite and in (0, 1)")
        for name in ("n_estimators", "min_samples_leaf", "max_candidates"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_depth is not None and (type(self.max_depth) is not int or self.max_depth < 1):
            raise ValueError("max_depth must be None or a positive integer")
        if self.max_features not in ("sqrt", "all") and (
            type(self.max_features) is not int or self.max_features < 1
        ):
            raise ValueError("max_features must be 'sqrt', 'all', or a positive integer")
        if type(self.bootstrap) is not bool:
            raise ValueError("bootstrap must be a bool")

        rng = np.random.RandomState(self.random_state)
        positive_rows = np.flatnonzero(y_pu == 1)
        unlabeled_rows = np.flatnonzero(y_pu == 0)
        positive_weight = float(prior) / n_positive
        unlabeled_weight = 1.0 / n_unlabeled
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            feature_budget = max(1, int(np.ceil(np.sqrt(n_features))))
        elif self.max_features == "all":
            feature_budget = n_features
        else:
            feature_budget = min(self.max_features, n_features)

        trees = []
        importances = np.zeros(n_features, dtype=np.float64)
        leaf_counts = []
        depths = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                rows = np.concatenate(
                    [
                        rng.choice(positive_rows, n_positive, replace=True),
                        rng.choice(unlabeled_rows, n_unlabeled, replace=True),
                    ]
                )
            else:
                rows = np.arange(len(X))
            root, importance, leaves, depth = self._build_tree(
                X,
                y_pu,
                rows,
                positive_weight=positive_weight,
                unlabeled_weight=unlabeled_weight,
                feature_budget=feature_budget,
                rng=rng,
            )
            trees.append(root)
            importances += importance
            leaf_counts.append(leaves)
            depths.append(depth)

        self.trees_ = trees
        self.feature_importances_ = importances / self.n_estimators
        self.n_leaves_ = np.asarray(leaf_counts, dtype=int)
        self.tree_depths_ = np.asarray(depths, dtype=int)
        self.n_features_in_ = n_features
        self.classes_ = np.array([0, 1])
        self._class_prior = float(prior)
        self._X_shape_ = X.shape
        self._is_fitted = True
        return self

    def _build_tree(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        root_rows: np.ndarray,
        *,
        positive_weight: float,
        unlabeled_weight: float,
        feature_budget: int,
        rng: np.random.RandomState,
    ) -> tuple[_Node, np.ndarray, int, int]:
        root = _Node(vote=-1)
        importances = np.zeros(X.shape[1], dtype=np.float64)
        leaf_count = 0
        deepest = 0
        stack = [(root, root_rows, 0)]
        while stack:
            node, rows, depth = stack.pop()
            deepest = max(deepest, depth)
            n_p = int(np.count_nonzero(labels[rows] == 1))
            n_u = len(rows) - n_p
            wp = n_p * positive_weight
            wu = n_u * unlabeled_weight
            node.vote = 1 if 2.0 * wp > wu else -1
            parent_risk = _nnpu_quadratic_node_risk(n_p, n_u, positive_weight, unlabeled_weight)
            if (
                parent_risk == 0
                or len(rows) < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)
            ):
                leaf_count += 1
                continue

            limits = []
            for feature in range(X.shape[1]):
                column = X[rows, feature]
                low, high = float(column.min()), float(column.max())
                if low < high:
                    limits.append((feature, low, high))
            if not limits:
                leaf_count += 1
                continue

            best_gain = 0.0
            best_split = None
            selected = rng.choice(len(limits), size=min(feature_budget, len(limits)), replace=False)
            for position in selected:
                feature, low, high = limits[int(position)]
                for _ in range(self.max_candidates):
                    fraction = rng.random_sample()
                    threshold = float((1.0 - fraction) * low + fraction * high)
                    mask = X[rows, feature] <= threshold
                    left_size = int(mask.sum())
                    right_size = len(rows) - left_size
                    if min(left_size, right_size) < self.min_samples_leaf:
                        continue
                    left_p = int(np.count_nonzero(labels[rows[mask]] == 1))
                    right_p = n_p - left_p
                    children_risk = _nnpu_quadratic_node_risk(
                        left_p, left_size - left_p, positive_weight, unlabeled_weight
                    ) + _nnpu_quadratic_node_risk(
                        right_p, right_size - right_p, positive_weight, unlabeled_weight
                    )
                    gain = parent_risk - children_risk
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_split = (feature, threshold, mask)
            if best_split is None:
                leaf_count += 1
                continue

            feature, node.threshold, mask = best_split
            node.feature = feature
            node.left = _Node(vote=-1)
            node.right = _Node(vote=-1)
            importances[feature] += best_gain
            stack.append((node.right, rows[~mask], depth + 1))
            stack.append((node.left, rows[mask], depth + 1))
        return root, importances, leaf_count, deepest

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("PUET prediction X must have the fitted 2-D feature shape")
        if not np.issubdtype(X.dtype, np.number) or not np.isrealobj(X) or not np.isfinite(X).all():
            raise ValueError("PUET prediction X must contain finite real numeric values")
        X = np.asarray(X, dtype=np.float64)
        votes = np.zeros(len(X), dtype=np.float64)
        for root in self.trees_:
            votes += _tree_votes(root, X)
        return votes / len(self.trees_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self._decision_function(X) > 0.0).astype(int)
