"""Paper Algorithm 1 BBE, explicitly estimating the positive fraction in U.

This module is deliberately not a BasePriorEstimator or registry entry: its
target is alpha_U, not the population pi consumed by the OS PU pipeline.
The scoring model must not have trained on the samples used to count tails.
"""

# Public feature/role arguments follow the project's X/y convention.
# ruff: noqa: N803, N806

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from ..core.exceptions import NotFittedError


def _unit_parameter(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number strictly between 0 and 1")
    value = float(value)
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return float(value)


def _scores(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty 1-D real score array")
    array = array.astype(float)
    if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must contain finite scores in [0, 1], high = positive")
    return array


@dataclass(frozen=True)
class BBEResult:
    """Unclipped tail ratio and the complete threshold-selection receipt.

    ``selection_upper_bounds`` are Algorithm 1's selection criterion, not a
    certified confidence interval on the true alpha_U for arbitrary data.
    Equality to alpha_U needs the paper's identifiability assumptions.
    """

    alpha_u: float
    threshold: float
    selected_index: int
    thresholds: tuple[float, ...]
    positive_counts: tuple[int, ...]
    unlabeled_counts: tuple[int, ...]
    ratios: tuple[float, ...]
    selection_upper_bounds: tuple[float, ...]
    n_positive: int
    n_unlabeled: int
    delta: float
    gamma: float


def best_bin_estimate(
    positive_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
    *,
    delta: float = 0.1,
    gamma: float = 0.01,
) -> BBEResult:
    """Select an inclusive upper tail by Garg et al. (2021), Algorithm 1.

    Inputs are scores from a fixed model on *held-out*, independently sampled
    P and marginal U. No clean U labels are accepted. All distinct empirical
    tails are enumerated, including threshold zero; zero-P-mass tails are
    undefined and excluded. Ties in the criterion choose the lowest threshold.
    Runtime is O((n_P+n_U) log(n_P+n_U)); no sample-by-threshold matrix is built.
    """
    delta = _unit_parameter(delta, "delta")
    gamma = _unit_parameter(gamma, "gamma")
    p = np.sort(_scores(positive_scores, "positive_scores"))
    u = np.sort(_scores(unlabeled_scores, "unlabeled_scores"))
    thresholds = np.unique(np.r_[0.0, p, u])
    p_counts = len(p) - np.searchsorted(p, thresholds, side="left")
    u_counts = len(u) - np.searchsorted(u, thresholds, side="left")
    valid = p_counts > 0
    thresholds, p_counts, u_counts = thresholds[valid], p_counts[valid], u_counts[valid]
    qp, qu = p_counts / len(p), u_counts / len(u)
    ratios = qu / qp
    # log(4 / delta) written without an overflowing division for tiny delta.
    log_confidence = np.log(4.0) - np.log(delta)
    penalty = (1 + gamma) * (
        np.sqrt(log_confidence / (2 * len(p))) + np.sqrt(log_confidence / (2 * len(u)))
    )
    upper = ratios + penalty / qp
    selected = int(np.argmin(upper))
    return BBEResult(
        alpha_u=float(ratios[selected]),
        threshold=float(thresholds[selected]),
        selected_index=selected,
        thresholds=tuple(float(x) for x in thresholds),
        positive_counts=tuple(int(x) for x in p_counts),
        unlabeled_counts=tuple(int(x) for x in u_counts),
        ratios=tuple(float(x) for x in ratios),
        selection_upper_bounds=tuple(float(x) for x in upper),
        n_positive=len(p),
        n_unlabeled=len(u),
        delta=delta,
        gamma=gamma,
    )


class BBEMixtureEstimator(BaseEstimator):
    """Fit PvU scores on train rows, then run BBE on disjoint holdout rows.

    This standalone, dense 2-D engineering path requires explicit
    ``input_sampling='ts'``: role 1 is an independent positive sample and role
    0 a marginal unlabeled mixture. It is not an automatic OS prior estimator,
    a TEDn classifier, or a reproduction of the author's deep recipes.

    Parameters
    ----------
    input_sampling : str or None, default=None
        Must explicitly be ``'ts'``. Merely setting this flag cannot certify
        the sampling design or independence of actual input data.
    holdout_fraction : float, default=0.2
        Ceil of this fraction of each observed role is held out. Both train
        and holdout must contain at least one row of each role.
    delta, gamma : float, default=0.1, 0.01
        Algorithm 1 selection-penalty parameters, strictly between 0 and 1.
    classifier : sklearn estimator or None, default=None
        Cloned PvU model implementing fit/predict_proba and classes_={0,1}.
        Default: train-only StandardScaler + balanced LogisticRegression.
    random_state : int, RandomState or None, default=None
        Controls within-role holdout split and the default classifier.
    """

    def __init__(
        self,
        *,
        input_sampling: str | None = None,
        holdout_fraction: float = 0.2,
        delta: float = 0.1,
        gamma: float = 0.01,
        classifier: Any = None,
        random_state: Any = None,
    ) -> None:
        self.input_sampling = input_sampling
        self.holdout_fraction = holdout_fraction
        self.delta = delta
        self.gamma = gamma
        self.classifier = classifier
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y_pu: np.ndarray, *, sample_ids: np.ndarray | None = None
    ) -> BBEMixtureEstimator:
        """Fit using observed P/U roles only; no validation/test truth input.

        Optional sample_ids must be unique, nonempty strings identifying
        original observations. Positional disjointness is always enforced;
        callers still own identity/group and external-role isolation.
        """
        # A failed refit must never leave a previous estimate usable.
        for name in ("result_", "classifier_", "train_indices_", "holdout_indices_", "provenance_"):
            self.__dict__.pop(name, None)
        if self.input_sampling != "ts":
            raise ValueError("BBE requires explicit input_sampling='ts'; result is alpha_U, not pi")
        fraction = _unit_parameter(self.holdout_fraction, "holdout_fraction")
        _unit_parameter(self.delta, "delta")
        _unit_parameter(self.gamma, "gamma")
        features = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
        roles = np.asarray(y_pu)
        if (
            roles.ndim != 1
            or len(roles) != len(features)
            or roles.dtype.kind not in "biuf"
            or not np.all(np.isin(roles, [0, 1]))
        ):
            raise ValueError("y_pu must be a 1-D array of observed roles 1=P and 0=U")
        if sample_ids is not None:
            ids = np.asarray(sample_ids)
            if (
                ids.ndim != 1
                or len(ids) != len(features)
                or ids.dtype.kind != "U"
                or any(not item.strip() for item in ids)
                or len(np.unique(ids)) != len(ids)
            ):
                raise ValueError("sample_ids must be unique nonempty strings, one per input row")
        rng = check_random_state(self.random_state)
        train, held = [], []
        for role in (1, 0):
            indices = rng.permutation(np.flatnonzero(roles == role))
            n_held = int(np.ceil(fraction * len(indices)))
            if not 1 <= n_held < len(indices):
                raise ValueError("each P/U role needs nonempty train and holdout rows")
            held.extend(indices[:n_held])
            train.extend(indices[n_held:])
        train_indices = np.array(train, dtype=int)
        holdout_indices = np.array(held, dtype=int)
        if self.classifier is None:
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=self.random_state
                ),
            )
        else:
            classifier = clone(self.classifier)
        classifier.fit(features[train_indices], roles[train_indices])
        classes = np.asarray(classifier.classes_)
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError("classifier classes_ must be exactly {0, 1}, 1=P")
        probabilities = np.asarray(classifier.predict_proba(features[holdout_indices]))
        if (
            probabilities.shape != (len(holdout_indices), 2)
            or probabilities.dtype.kind not in "iuf"
            or not np.all(np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=1), 1)
        ):
            raise ValueError("classifier predict_proba must return finite binary probabilities")
        scores = probabilities[:, int(np.flatnonzero(classes == 1)[0])]
        held_roles = roles[holdout_indices]
        result = best_bin_estimate(
            scores[held_roles == 1], scores[held_roles == 0], delta=self.delta, gamma=self.gamma
        )
        self.classifier_ = classifier
        self.train_indices_ = train_indices
        self.holdout_indices_ = holdout_indices
        self.result_ = result
        self.provenance_ = {
            "target": "unlabeled_positive_fraction_alpha_U",
            "input_sampling": "ts",
            "sampling_design_verified": False,
            "split_policy": "within_observed_role_random_holdout_ceil",
            "holdout_fraction": float(fraction),
            "train_positive": int(np.sum(roles[train_indices] == 1)),
            "train_unlabeled": int(np.sum(roles[train_indices] == 0)),
            "holdout_positive": result.n_positive,
            "holdout_unlabeled": result.n_unlabeled,
            "train_holdout_positionally_disjoint": True,
            "sample_ids_checked": sample_ids is not None,
            "clean_labels_read": False,
            "selection_uses_holdout_only": True,
            "population_prior_conversion": None,
            "formal_admission": False,
        }
        return self

    def estimate_unlabeled_prior(self) -> float:
        """Return alpha_U, deliberately not the OS pipeline's estimate() API."""
        if not hasattr(self, "result_"):
            raise NotFittedError("BBEMixtureEstimator is not fitted. Call fit() first.")
        return self.result_.alpha_u
