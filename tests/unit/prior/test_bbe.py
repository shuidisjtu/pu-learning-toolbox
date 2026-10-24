"""Independent paper-statistic oracle and train/holdout isolation for BBE."""

# Feature naming mirrors the public fit API.
# ruff: noqa: N803, N806

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone

from pu_toolbox.core.base import BasePriorEstimator
from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.prior import BBEMixtureEstimator, best_bin_estimate

pytestmark = pytest.mark.unit


class RecordingScorer(BaseEstimator):
    def __init__(self, reverse=False, bad_probabilities=False):
        self.reverse = reverse
        self.bad_probabilities = bad_probabilities

    def fit(self, X, y):
        self.seen_X_ = X.copy()
        self.seen_y_ = y.copy()
        self.classes_ = np.array([1, 0] if self.reverse else [0, 1])
        return self

    def predict_proba(self, X):
        self.scored_X_ = X.copy()
        p = X[:, 0]
        proba = np.c_[1 - p, p]
        if self.bad_probabilities:
            proba[:] = 0.2
        return proba[:, ::-1] if self.reverse else proba


def _data():
    # First coordinate is a fixed scorer's score, second uniquely identifies rows.
    p = np.linspace(0.65, 0.98, 40)
    u = np.r_[np.linspace(0.67, 0.97, 20), np.linspace(0.01, 0.3, 60)]
    return np.c_[np.r_[p, u], np.arange(120)], np.r_[np.ones(40), np.zeros(80)]


def _scalar_reference(p, u, delta, gamma):
    rows = []
    for c in sorted({0.0, *p, *u}):
        count_p = sum(score >= c for score in p)
        count_u = sum(score >= c for score in u)
        if count_p == 0:
            continue
        qp, qu = count_p / len(p), count_u / len(u)
        ratio = qu / qp
        bound = (
            (1 + gamma)
            / qp
            * (
                math.sqrt(math.log(4 / delta) / (2 * len(p)))
                + math.sqrt(math.log(4 / delta) / (2 * len(u)))
            )
        )
        rows.append((c, count_p, count_u, ratio, ratio + bound))
    return rows, min(range(len(rows)), key=lambda i: rows[i][4])


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("delta,gamma", [(0.1, 0.01), (0.8, 0.7)])
def test_basic_matches_independent_scalar_reference(seed, delta, gamma):
    rng = np.random.RandomState(seed)
    p = rng.randint(0, 11, size=37) / 10
    u = rng.randint(0, 11, size=53) / 10
    result = best_bin_estimate(p, u, delta=delta, gamma=gamma)
    rows, idx = _scalar_reference(p.tolist(), u.tolist(), delta, gamma)
    assert result.thresholds == tuple(row[0] for row in rows)
    assert result.positive_counts == tuple(row[1] for row in rows)
    assert result.unlabeled_counts == tuple(row[2] for row in rows)
    np.testing.assert_allclose(result.ratios, [row[3] for row in rows])
    np.testing.assert_allclose(result.selection_upper_bounds, [row[4] for row in rows])
    assert result.selected_index == idx
    assert result.alpha_u == pytest.approx(rows[idx][3])


def test_basic_selection_penalty_not_minimum_raw_ratio():
    p = np.r_[np.full(99, 0.8), 0.99]
    u = np.r_[np.full(50, 0.8), np.zeros(50)]
    result = best_bin_estimate(p, u)
    assert min(result.ratios) == 0
    assert result.threshold == 0.8
    assert result.alpha_u == 0.5
    # Return the ratio at the selected bin, not the UCB used for selection.
    assert result.selection_upper_bounds[result.selected_index] > result.alpha_u


def test_edge_ties_zero_denominator_and_endpoint_bins():
    result = best_bin_estimate([0, 0.5, 0.5], [0, 0.5, 1])
    assert result.thresholds == (0, 0.5)
    assert result.positive_counts == (3, 2)
    assert result.unlabeled_counts == (3, 2)
    assert best_bin_estimate([1], [1]).alpha_u == 1
    assert best_bin_estimate([0], [0]).threshold == 0
    assert best_bin_estimate([1] * 100, [0] * 100).alpha_u == 0
    with pytest.raises(FrozenInstanceError):
        result.alpha_u = 0


@pytest.mark.parametrize(
    "bad", [[], [[0.2]], [np.nan], [np.inf], [-0.1], [1.01], [1j], [True], ["0.3"]]
)
@pytest.mark.parametrize("side", ["positive_scores", "unlabeled_scores"])
def test_param_invalid_scores(bad, side):
    inputs = {"positive_scores": [0.5], "unlabeled_scores": [0.2], side: bad}
    with pytest.raises(ValueError, match=side):
        best_bin_estimate(**inputs)


@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1, np.nan, np.inf, True, "0.1"])
@pytest.mark.parametrize("name", ["delta", "gamma"])
def test_param_invalid_selection_parameters(value, name):
    with pytest.raises(ValueError, match=name):
        best_bin_estimate([0.5], [0.2], **{name: value})


def test_deterministic_permutation_and_input_immutability():
    p, u = np.array([0.1, 0.8, 0.8, 1]), np.array([0, 0.1, 0.3, 0.8, 1])
    p_before, u_before = p.copy(), u.copy()
    assert best_bin_estimate(p, u) == best_bin_estimate(p[::-1], u[::-1])
    np.testing.assert_array_equal(p, p_before)
    np.testing.assert_array_equal(u, u_before)


@pytest.mark.parametrize("reverse", [False, True])
def test_basic_end_to_end_clone_and_train_holdout_isolation(reverse):
    X, roles = _data()
    supplied = RecordingScorer(reverse=reverse)
    estimator = BBEMixtureEstimator(
        input_sampling="ts", classifier=supplied, random_state=42, holdout_fraction=0.25
    ).fit(X, roles, sample_ids=np.array([f"row-{i}" for i in range(len(X))]))
    train, held = estimator.train_indices_, estimator.holdout_indices_
    assert set(train).isdisjoint(held)
    assert sorted(np.r_[train, held]) == list(range(len(X)))
    np.testing.assert_array_equal(estimator.classifier_.seen_X_, X[train])
    np.testing.assert_array_equal(estimator.classifier_.seen_y_, roles[train])
    np.testing.assert_array_equal(estimator.classifier_.scored_X_, X[held])
    assert not hasattr(supplied, "seen_X_")
    p_scores = X[held][roles[held] == 1, 0]
    u_scores = X[held][roles[held] == 0, 0]
    assert estimator.result_ == best_bin_estimate(p_scores, u_scores)
    assert estimator.estimate_unlabeled_prior() == estimator.result_.alpha_u
    assert estimator.provenance_["holdout_positive"] == 10
    assert estimator.provenance_["holdout_unlabeled"] == 20
    assert estimator.provenance_["sample_ids_checked"] is True
    assert estimator.provenance_["clean_labels_read"] is False
    assert estimator.provenance_["population_prior_conversion"] is None
    assert not hasattr(estimator, "estimate")


def test_basic_default_scaler_fits_train_only_and_seed_reproduces():
    X, roles = _data()
    original = BBEMixtureEstimator(input_sampling="ts", random_state=9)
    first, second = clone(original).fit(X, roles), clone(original).fit(X, roles)
    np.testing.assert_array_equal(first.train_indices_, second.train_indices_)
    np.testing.assert_array_equal(first.holdout_indices_, second.holdout_indices_)
    np.testing.assert_allclose(
        first.classifier_.named_steps["standardscaler"].mean_,
        X[first.train_indices_].mean(axis=0),
    )
    assert first.result_ == second.result_
    assert 0 <= first.estimate_unlabeled_prior() <= 1
    assert not hasattr(original, "result_")


@pytest.mark.parametrize("sampling", [None, "os", "sar", "TS"])
def test_param_requires_explicit_sampling_semantics(sampling):
    X, roles = _data()
    with pytest.raises(ValueError, match="alpha_U, not pi"):
        BBEMixtureEstimator(input_sampling=sampling).fit(X, roles)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, True, np.nan, 0.999])
def test_param_holdout_fraction_has_no_silent_empty_train_fallback(fraction):
    X, roles = _data()
    with pytest.raises(ValueError, match="holdout|train"):
        BBEMixtureEstimator(input_sampling="ts", holdout_fraction=fraction).fit(X, roles)


@pytest.mark.parametrize(
    "case", ["missing_p", "missing_u", "one_p", "bad_role", "column", "nan", "duplicate_id"]
)
def test_edge_invalid_roles_or_identities(case):
    X, roles = _data()
    ids = np.array([f"r{i}" for i in range(len(X))])
    if case == "missing_p":
        roles[:] = 0
    elif case == "missing_u":
        roles[:] = 1
    elif case == "one_p":
        roles[:] = 0
        roles[0] = 1
    elif case == "bad_role":
        roles[0] = -1
    elif case == "column":
        roles = roles[:, None]
    elif case == "nan":
        X[0, 0] = np.nan
    else:
        ids[1] = ids[0]
    with pytest.raises(ValueError):
        BBEMixtureEstimator(input_sampling="ts").fit(X, roles, sample_ids=ids)


def test_edge_failed_refit_clears_previous_estimate_and_bad_probability_rejected():
    X, roles = _data()
    model = BBEMixtureEstimator(input_sampling="ts", random_state=3).fit(X, roles)
    model.set_params(classifier=RecordingScorer(bad_probabilities=True))
    with pytest.raises(ValueError, match="probabilities"):
        model.fit(X, roles)
    with pytest.raises(NotFittedError):
        model.estimate_unlabeled_prior()
    assert not hasattr(model, "classifier_")


def test_param_clean_truth_argument_not_accepted():
    X, roles = _data()
    with pytest.raises(TypeError, match="y_true"):
        BBEMixtureEstimator(input_sampling="ts").fit(X, roles, y_true=roles)


def test_edge_smallest_positive_delta_has_finite_selection_bounds():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = best_bin_estimate([0.2, 0.8], [0.1, 0.7], delta=np.nextafter(0.0, 1.0))
    assert np.isfinite(result.selection_upper_bounds).all()


def test_basic_ledger_preserves_prior_and_admission_boundaries():
    root = Path(__file__).resolve().parents[3]
    ledger = json.loads(
        (root / "pu_toolbox/experiment/method_ledger.json").read_text(encoding="utf-8")
    )
    cvir = ledger["methods"]["cvir"]
    extension = cvir["independent_extensions"]["bbe"]
    review = json.loads((root / extension["review_ref"]).read_text(encoding="utf-8"))
    implementation = review["implementation"]
    assert not issubclass(BBEMixtureEstimator, BasePriorEstimator)
    assert extension["target"] == "alpha_U, not automatic population pi"
    assert cvir["calibration_applied"] is False
    assert cvir["calibration_hooked"] is True
    for key in ("calibration_applied", "calibration_hooked", "formal_admission"):
        assert extension[key] is False
        assert implementation[key] is False
    assert implementation["wired_to_existing_cvir"] is False
    assert implementation["registered_as_population_prior_estimator"] is False
    assert implementation["tedn_implemented"] is False
    assert review["owner_signatures"] == []
