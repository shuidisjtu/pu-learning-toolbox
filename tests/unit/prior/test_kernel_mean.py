# ruff: noqa: N806

"""Tests for KM1/KM2 kernel-mean class-prior estimation."""

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.prior import KernelMeanPriorEstimator
from pu_toolbox.prior.kernel_mean import _nearest_simplex_distance
from tests.helpers import make_scar_data


def _mixture(seed=4):
    rng = np.random.RandomState(seed)
    component = rng.normal(2.5, 0.35, size=(35, 2))
    hidden_positive = rng.normal(2.5, 0.35, size=(25, 2))
    hidden_negative = rng.normal(-1.5, 0.5, size=(55, 2))
    X = np.vstack([component, hidden_positive, hidden_negative])
    y_pu = np.r_[np.ones(len(component), dtype=int), np.zeros(80, dtype=int)]
    return X, y_pu


@pytest.mark.math
def test_basic_identity_kernel_qp_matches_simplex_projection():
    u = np.array([1.2, -0.2])
    solution, distance_squared, _, gap = _nearest_simplex_distance(u, np.eye(2), tolerance=1e-10)
    np.testing.assert_allclose(solution, [1.0, 0.0], atol=1e-7)
    assert distance_squared == pytest.approx(0.08, abs=1e-7)
    assert gap <= 1e-7


@pytest.mark.unit
def test_basic_km_variants_are_bounded_and_expose_diagnostics():
    X, y_pu = _mixture()
    estimator = KernelMeanPriorEstimator(max_qp_iter=500, qp_tolerance=1e-6).fit(X, y_pu)
    assert 0.0 <= estimator.estimate() <= 1.0
    assert 0.0 <= estimator.km2_estimate_ <= 1.0
    assert estimator.kernel_width_ > 0
    assert set(estimator.diagnostics_) == {"km1", "km2"}
    assert estimator.n_mixture_ == 80
    assert estimator.n_component_ == 35


@pytest.mark.unit
def test_determ_km_is_reproducible_with_subsampling():
    X, y_pu = _mixture()
    kwargs = {"max_samples_per_group": 20, "random_state": 9, "qp_tolerance": 1e-6}
    first = KernelMeanPriorEstimator(**kwargs).fit(X, y_pu)
    second = KernelMeanPriorEstimator(**kwargs).fit(X, y_pu)
    assert first.estimate() == pytest.approx(second.estimate())
    assert first.kernel_width_ == pytest.approx(second.kernel_width_)


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["km1", "km2"])
def test_param_selected_variant_matches_named_estimate(variant):
    X, y_pu = _mixture()
    estimator = KernelMeanPriorEstimator(
        variant=variant, max_samples_per_group=20, qp_tolerance=1e-6
    ).fit(X, y_pu)
    expected = estimator.km1_estimate_ if variant == "km1" else estimator.km2_estimate_
    assert estimator.estimate() == expected


# ── math (relative width golden) ───────────────────────────────


@pytest.mark.math
def test_math_auto_width_formula(rng):
    """Relative selection: kernel_width_ == scale x median pairwise distance."""
    X, y_pu, _ = make_scar_data(rng, n=200, separation=2.0)
    est = KernelMeanPriorEstimator(variant="km2").fit(X, y_pu)
    # Implementation takes the median over the full squared-distance matrix
    # (diagonal zeros included); replicate exactly.
    d2 = pairwise_distances(X, metric="sqeuclidean")
    expected = 0.1 * float(np.sqrt(np.median(d2)))
    assert est.kernel_width_ == pytest.approx(expected, rel=1e-6)


@pytest.mark.math
@pytest.mark.parametrize("variant", ["km1", "km2"])
def test_math_variants_land_in_band(rng, variant):
    """Default relative width lands both variants in the acceptance band."""
    X, y_pu, _ = make_scar_data(rng, n=200, separation=2.0)
    est = KernelMeanPriorEstimator(variant=variant).fit(X, y_pu).estimate()
    assert 0.35 <= est <= 0.75


# ── param (selection modes) ────────────────────────────────────


@pytest.mark.unit
def test_param_mmd_grid_selection_preserved(rng):
    """mmd_grid keeps the author width search and differs from relative."""
    X, y_pu, _ = make_scar_data(rng, n=200, separation=2.0)
    rel = KernelMeanPriorEstimator(variant="km2").fit(X, y_pu)
    grid = KernelMeanPriorEstimator(variant="km2", width_selection="mmd_grid").fit(X, y_pu)
    assert grid.kernel_width_ > 0
    assert grid.kernel_width_ != pytest.approx(rel.kernel_width_)


@pytest.mark.unit
def test_param_explicit_width_beats_selection(rng):
    """Explicit kernel_width overrides any width_selection mode."""
    X, y_pu, _ = make_scar_data(rng, n=200, separation=2.0)
    est = KernelMeanPriorEstimator(kernel_width=0.5, width_selection="relative").fit(X, y_pu)
    assert est.kernel_width_ == pytest.approx(0.5)


# ── edge ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_edge_invalid_parameters_and_unfitted_estimate_raise():
    X, y_pu = _mixture()
    with pytest.raises(NotFittedError):
        KernelMeanPriorEstimator().estimate()
    with pytest.raises(ValueError, match="variant"):
        KernelMeanPriorEstimator(variant="unknown").fit(X, y_pu)
    with pytest.raises(ValueError, match="identical"):
        KernelMeanPriorEstimator().fit(np.ones_like(X), y_pu)
