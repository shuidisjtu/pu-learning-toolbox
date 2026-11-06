"""Mass identities and declared targets for positive class priors."""

import numpy as np
import pytest

from pu_toolbox.prior import (
    ClassPriorEstimator,
    KernelMeanPriorEstimator,
    ReCPEEstimator,
    native_unlabeled_to_population,
    population_to_native_unlabeled,
)

pytestmark = pytest.mark.unit


def test_basic_os_mass_identity_and_round_trip():
    population = native_unlabeled_to_population(0.25, 0.2)
    assert population == pytest.approx(0.4)
    assert population_to_native_unlabeled(population, 0.2) == pytest.approx(0.25)


@pytest.mark.parametrize("q", [0.0, 0.2, 0.8, 1.0 - 1e-10])
@pytest.mark.parametrize("prior_u", [0.0, 0.25, 0.7, 1.0])
def test_param_mass_identity_at_boundaries(q, prior_u):
    population = native_unlabeled_to_population(prior_u, q)
    assert population_to_native_unlabeled(population, q) == pytest.approx(prior_u, abs=1e-6)
    assert native_unlabeled_to_population(prior_u, q, input_sampling="ts") == prior_u
    assert population_to_native_unlabeled(prior_u, q, input_sampling="ts") == prior_u


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"class_prior": np.nan}, ValueError),
        ({"class_prior": np.inf}, ValueError),
        ({"class_prior": -0.1}, ValueError),
        ({"class_prior": 1.01}, ValueError),
        ({"class_prior": True}, TypeError),
        ({"class_prior": np.bool_(False)}, TypeError),
        ({"class_prior": [0.2]}, TypeError),
        ({"class_prior": "0.2"}, TypeError),
        ({"labeled_fraction": True}, TypeError),
        ({"labeled_fraction": 1.0}, ValueError),
        ({"labeled_fraction": np.nan}, ValueError),
        ({"labeled_fraction": -0.01}, ValueError),
        ({"input_sampling": "invalid"}, ValueError),
        ({"input_sampling": None}, ValueError),
    ],
)
@pytest.mark.parametrize(
    "convert", [native_unlabeled_to_population, population_to_native_unlabeled]
)
def test_edge_invalid_probability_arguments(kwargs, error, convert):
    parameters = {"class_prior": 0.4, "labeled_fraction": 0.2, **kwargs}
    with pytest.raises(error):
        convert(**parameters)


def test_edge_os_prior_below_labeled_fraction_raises():
    with pytest.raises(ValueError, match="labeled_fraction"):
        population_to_native_unlabeled(0.2, 0.4)
    assert population_to_native_unlabeled(0.2, 0.4, input_sampling="ts") == 0.2


def test_deterministic_numpy_scalars_and_repeated_conversion():
    first = native_unlabeled_to_population(np.float64(0.25), np.float64(0.2))
    second = native_unlabeled_to_population(np.float64(0.25), np.float64(0.2))
    assert isinstance(first, float)
    assert first == second


@pytest.mark.parametrize(
    "estimator", [KernelMeanPriorEstimator(), ClassPriorEstimator(), ReCPEEstimator()]
)
def test_param_builtin_mixture_estimators_declare_native_u(estimator):
    assert estimator.estimate_target == "native_unlabeled"
    assert estimator.estimation_pool == "native_unlabeled"


@pytest.mark.parametrize("prior_u", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("method", ["km1", "km2", "pen_l1"])
def test_param_identifiable_mixture_estimates_native_u(prior_u, method):
    """Disjoint class supports identify U's mixture weight directly."""
    n_positive_u = int(80 * prior_u)
    positive = np.ones((24, 1))
    unlabeled = np.vstack([np.ones((n_positive_u, 1)), -np.ones((80 - n_positive_u, 1))])
    features = np.vstack([positive, unlabeled])
    labels = np.r_[np.ones(24, dtype=int), np.zeros(80, dtype=int)]
    estimator = (
        ClassPriorEstimator(sigma=0.5)
        if method == "pen_l1"
        else KernelMeanPriorEstimator(variant=method, kernel_width=0.5)
    ).fit(features, labels)
    assert estimator.estimate() == pytest.approx(prior_u, abs=0.02)
