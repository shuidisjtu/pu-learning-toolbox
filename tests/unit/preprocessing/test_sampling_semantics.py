# ruff: noqa: N803, N806

"""Sampling designs, legacy aliases, and independent Gaussian P/U sources."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pu_toolbox import preprocessing

pytestmark = pytest.mark.unit


def test_basic_fixed_count_preserves_os_pool_and_empirical_prior():
    y_true = np.array([1] * 40 + [0] * 60)
    labels = preprocessing.make_fixed_count_labels(y_true, n_labeled=20, random_state=7)
    assert labels.shape == y_true.shape
    assert np.count_nonzero(labels) == 20
    assert np.all(labels[y_true == 0] == 0)
    assert np.mean(y_true[labels == 0]) == pytest.approx(0.25)
    dispatched = preprocessing.make_pu_labels(
        y_true, mechanism="fixed_count", n_labeled=20, random_state=7
    )
    np.testing.assert_array_equal(labels, dispatched)


@pytest.mark.parametrize("use_dispatcher", [False, True])
def test_determ_legacy_case_control_alias_keeps_seed_and_warns(use_dispatcher):
    y_true = np.array([1] * 40 + [0] * 60)
    expected = preprocessing.make_fixed_count_labels(y_true, n_labeled=20, random_state=7)
    with pytest.warns(FutureWarning, match="make_fixed_count_labels.*2.0"):
        if use_dispatcher:
            labels = preprocessing.make_pu_labels(
                y_true, mechanism="case_control", n_labeled=20, random_state=7
            )
        else:
            labels = preprocessing.make_case_control_labels(y_true, 20, random_state=7)
    np.testing.assert_array_equal(labels, expected)


@pytest.mark.parametrize("n_labeled", [True, 1.5, 0, -1, 4])
def test_param_fixed_count_rejects_invalid_counts(n_labeled):
    with pytest.raises((ValueError, TypeError), match="n_labeled"):
        preprocessing.make_fixed_count_labels(np.array([1, 1, 0]), n_labeled=n_labeled)


def test_edge_fixed_count_all_positive_and_missing_dispatch_count():
    labels = preprocessing.make_fixed_count_labels(np.ones(3), n_labeled=3, random_state=0)
    np.testing.assert_array_equal(labels, np.ones(3))
    with pytest.raises(ValueError, match="n_labeled"):
        preprocessing.make_pu_labels(np.array([1, 0]), mechanism="fixed_count")


def test_basic_independent_ts_records_population_and_empirical_priors():
    X, y_pu, y_true, metadata = preprocessing.make_gaussian_case_control_data(
        n_p=50, n_u=10000, class_prior=0.4, n_features=2, random_state=17
    )
    assert X.shape == (10050, 2)
    assert np.all(y_true[y_pu == 1] == 1)
    assert np.count_nonzero(y_pu) == 50
    assert 0.37 < y_true[y_pu == 0].mean() < 0.43
    assert metadata["input_sampling"] == "ts"
    assert metadata["population_class_prior"] == 0.4
    assert metadata["unlabeled_class_prior"] == pytest.approx(y_true[y_pu == 0].mean())
    assert metadata["training_class_prior"] == pytest.approx(y_true.mean())
    assert metadata["sampling_policy"] == "independent_generative"
    sample_ids = metadata["sample_ids"]
    assert len(set(sample_ids)) == len(X)
    assert all(identity.startswith("positive:") for identity in sample_ids[:50])
    assert all(identity.startswith("population:") for identity in sample_ids[50:])


def test_determ_independent_ts_draws_reproduce_both_class_distributions():
    kwargs = dict(n_p=8, n_u=40, class_prior=0.4, n_features=2, random_state=13)
    first = preprocessing.make_gaussian_case_control_data(**kwargs)
    second = preprocessing.make_gaussian_case_control_data(**kwargs)
    for actual, expected in zip(first[:3], second[:3], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert first[3] == second[3]
    X, y_pu, y_true, _ = first
    rng = np.random.RandomState(13)
    positive = rng.normal(size=(8, 2)) + 1.0
    unlabeled_truth = (rng.uniform(size=40) < 0.4).astype(int)
    unlabeled = rng.normal(size=(40, 2)) + (2 * unlabeled_truth[:, None] - 1)
    np.testing.assert_array_equal(X[y_pu == 1], positive)
    np.testing.assert_array_equal(X[y_pu == 0], unlabeled)
    np.testing.assert_array_equal(y_true[y_pu == 0], unlabeled_truth)


@pytest.mark.parametrize("class_prior", [0.0, 1.0])
def test_edge_ts_population_can_have_a_single_class(class_prior):
    _, y_pu, y_true, metadata = preprocessing.make_gaussian_case_control_data(
        n_p=1, n_u=3, class_prior=class_prior, random_state=0
    )
    assert np.all(y_true[y_pu == 0] == class_prior)
    assert metadata["unlabeled_class_prior"] == class_prior


def test_determ_ts_randomstate_reuses_stream_and_metadata_serializes():
    rng = np.random.RandomState(23)
    seeded = preprocessing.make_gaussian_case_control_data(random_state=np.int64(23))
    streamed = preprocessing.make_gaussian_case_control_data(random_state=rng)
    for actual, expected in zip(streamed[:3], seeded[:3], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert seeded[3]["random_state"] == 23
    assert streamed[3]["random_state"] is None
    assert json.loads(json.dumps(streamed[3], allow_nan=False)) == streamed[3]
    following = preprocessing.make_gaussian_case_control_data(random_state=rng)
    assert not np.array_equal(following[0], streamed[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_p": 0},
        {"n_u": -1},
        {"n_u": 1.5},
        {"n_features": True},
        {"class_prior": -0.1},
        {"class_prior": 1.1},
        {"class_prior": float("nan")},
        {"class_prior": True},
        {"class_prior": "0.4"},
        {"separation": float("inf")},
        {"separation": -1},
    ],
)
def test_param_ts_rejects_invalid_distribution_parameters(kwargs):
    with pytest.raises((TypeError, ValueError)):
        preprocessing.make_gaussian_case_control_data(**kwargs)


def test_determ_legacy_gaussian_retains_negative_u_source():
    X, y_pu, prior = preprocessing.make_gaussian_pu_data(
        n_p=3, n_u=7, n_features=2, random_state=11
    )
    rng = np.random.RandomState(11)
    expected_p = rng.randn(3, 2) + 1.0
    expected_n = rng.randn(7, 2) - 1.0
    np.testing.assert_array_equal(X[y_pu == 1], expected_p)
    np.testing.assert_array_equal(X[y_pu == 0], expected_n)
    assert prior == 0.3
