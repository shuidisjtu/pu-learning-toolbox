# ruff: noqa: N803, N806

"""Kernel-mean mixture-proportion estimation (KM1/KM2)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import pairwise_distances

from ..core.base import BasePriorEstimator
from ..core.exceptions import NotFittedError
from ..core.validation import validate_pu_X_y


def _nearest_simplex_distance(
    u: np.ndarray,
    kernel: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    max_iter: int = 2000,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, float, int, float]:
    """Minimize ``(u-v)^T K (u-v)`` over the probability simplex."""
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive")
    n = len(u)
    if kernel.shape != (n, n):
        raise ValueError("kernel shape must agree with u")
    if initial is None:
        v = np.full(n, 1.0 / n)
    else:
        v = np.maximum(np.asarray(initial, dtype=float), 0.0)
        total = v.sum()
        v = np.full(n, 1.0 / n) if total <= 0 else v / total

    kernel_u = kernel @ u
    kernel_v = kernel @ v
    gap = float("inf")
    for _iteration in range(1, max_iter + 1):
        gradient = 2.0 * (kernel_v - kernel_u)
        vertex = int(np.argmin(gradient))
        gap = float(np.dot(gradient, v) - gradient[vertex])
        if gap <= tolerance:
            break
        direction_kernel = kernel[:, vertex] - kernel_v
        denominator = 2.0 * float(kernel[vertex, vertex] - 2 * kernel_v[vertex] + v @ kernel_v)
        step = 1.0 if denominator <= 1e-15 else np.clip(gap / denominator, 0.0, 1.0)
        v *= 1.0 - step
        v[vertex] += step
        kernel_v += step * direction_kernel

    residual = u - v
    distance_squared = max(0.0, float(residual @ (kernel @ residual)))
    return v, distance_squared, _iteration, gap


class KernelMeanPriorEstimator(BasePriorEstimator):
    """Estimate the positive class prior with the paper's KM1 or KM2 algorithm.

    The unlabeled observations form mixture ``F`` and labeled positives form
    component ``H``. The returned estimate is ``kappa`` in
    ``F = (1-kappa)G + kappa H``.
    """

    def __init__(
        self,
        *,
        variant: Literal["km1", "km2"] = "km1",
        kernel_width: float | None = None,
        width_factors: tuple[float, ...] = (0.1, 0.316227766, 1.0, 3.16227766, 10.0),
        epsilon: float = 0.04,
        lambda_upper_bound: float = 8.0,
        km2_final_slope_weight: float = 0.2,
        max_qp_iter: int = 2000,
        qp_tolerance: float = 1e-7,
        max_samples_per_group: int | None = None,
        standardize: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.variant = variant
        self.kernel_width = kernel_width
        self.width_factors = width_factors
        self.epsilon = epsilon
        self.lambda_upper_bound = lambda_upper_bound
        self.km2_final_slope_weight = km2_final_slope_weight
        self.max_qp_iter = max_qp_iter
        self.qp_tolerance = qp_tolerance
        self.max_samples_per_group = max_samples_per_group
        self.standardize = standardize
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if self.variant not in {"km1", "km2"}:
            raise ValueError("variant must be 'km1' or 'km2'")
        if self.kernel_width is not None and self.kernel_width <= 0:
            raise ValueError("kernel_width must be positive or None")
        if not self.width_factors or any(value <= 0 for value in self.width_factors):
            raise ValueError("width_factors must contain positive values")
        if self.epsilon <= 0 or self.lambda_upper_bound <= 1 + self.epsilon:
            raise ValueError("epsilon must be positive and lambda_upper_bound sufficiently above 1")
        if not 0 <= self.km2_final_slope_weight <= 1:
            raise ValueError("km2_final_slope_weight must be in [0, 1]")
        if self.max_qp_iter < 1 or self.qp_tolerance <= 0:
            raise ValueError("max_qp_iter and qp_tolerance must be positive")
        if self.max_samples_per_group is not None and self.max_samples_per_group < 2:
            raise ValueError("max_samples_per_group must be >= 2 or None")

    def _select_samples(self, values: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        limit = self.max_samples_per_group
        if limit is None or len(values) <= limit:
            return values
        return values[rng.choice(len(values), size=limit, replace=False)]

    def _select_kernel(
        self, mixture: np.ndarray, component: np.ndarray
    ) -> tuple[float, np.ndarray, float]:
        values = np.vstack([mixture, component])
        distances_squared = pairwise_distances(values, metric="sqeuclidean")
        n_mixture = len(mixture)
        weights = np.r_[
            np.full(n_mixture, 1.0 / n_mixture),
            np.full(len(component), -1.0 / len(component)),
        ]
        if self.kernel_width is None:
            median_squared = float(np.median(distances_squared))
            if median_squared <= 0:
                positive = distances_squared[distances_squared > 0]
                if not len(positive):
                    raise ValueError("kernel width is undefined because all samples are identical")
                median_squared = float(np.median(positive))
            median_width = np.sqrt(median_squared)
            widths = [median_width * factor for factor in self.width_factors]
        else:
            widths = [self.kernel_width]

        best: tuple[float, np.ndarray, float] | None = None
        for width in widths:
            kernel = np.exp(-distances_squared / (2.0 * width**2))
            distance = np.sqrt(max(0.0, float(weights @ (kernel @ weights))))
            if best is None or distance > best[2]:
                best = (float(width), kernel, distance)
        assert best is not None
        return best

    def _distance(
        self,
        lambda_value: float,
        kernel: np.ndarray,
        n_mixture: int,
        n_component: int,
        initial: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, int, float]:
        u = np.r_[
            np.full(n_mixture, lambda_value / n_mixture),
            np.full(n_component, (1.0 - lambda_value) / n_component),
        ]
        solution, distance_squared, iterations, gap = _nearest_simplex_distance(
            u,
            kernel,
            initial=initial,
            max_iter=self.max_qp_iter,
            tolerance=self.qp_tolerance,
        )
        return np.sqrt(distance_squared), solution, iterations, gap

    def _estimate_lambda(
        self,
        threshold: float,
        kernel: np.ndarray,
        n_mixture: int,
        n_component: int,
    ) -> tuple[float, list[dict[str, float]]]:
        left, right = 1.0, self.lambda_upper_bound
        diagnostics = []
        while right - left > self.epsilon:
            midpoint = (left + right) / 2.0
            first, solution, first_iterations, first_gap = self._distance(
                midpoint, kernel, n_mixture, n_component
            )
            second, _, second_iterations, second_gap = self._distance(
                midpoint + self.epsilon / 2.0,
                kernel,
                n_mixture,
                n_component,
                initial=solution,
            )
            slope = 2.0 * (second - first) / self.epsilon
            diagnostics.append(
                {
                    "lambda": midpoint,
                    "slope": slope,
                    "qp_iterations": float(first_iterations + second_iterations),
                    "max_qp_gap": max(first_gap, second_gap),
                }
            )
            if slope > threshold:
                right = midpoint
            else:
                left = midpoint
        return (left + right) / 2.0, diagnostics

    def fit(self, X: np.ndarray, y_pu: np.ndarray) -> KernelMeanPriorEstimator:
        """Fit KM1/KM2 to positive-component and unlabeled-mixture samples."""
        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            estimator_name="KernelMeanPriorEstimator",
        )
        self._validate_parameters()
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values")
        if self.standardize:
            self.mean_ = X.mean(axis=0)
            self.scale_ = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
            X = (X - self.mean_) / self.scale_

        rng = np.random.RandomState(self.random_state)
        mixture = self._select_samples(X[y_pu == 0], rng)
        component = self._select_samples(X[y_pu == 1], rng)
        width, kernel, distribution_distance = self._select_kernel(mixture, component)
        if distribution_distance <= 1e-12:
            raise ValueError("mixture and component kernel means are indistinguishable")

        d_one, solution, _, _ = self._distance(1.0, kernel, len(mixture), len(component))
        d_next, _, _, _ = self._distance(
            1.05, kernel, len(mixture), len(component), initial=solution
        )
        initial_slope = (d_next - d_one) / 0.05
        weight = self.km2_final_slope_weight
        km2_threshold = (1.0 - weight) * initial_slope + weight * distribution_distance
        km1_threshold = 1.0 / np.sqrt(min(len(mixture), len(component)))
        if km1_threshold / distribution_distance > 0.9:
            km1_threshold = km2_threshold

        lambda_km1, diagnostics_km1 = self._estimate_lambda(
            km1_threshold, kernel, len(mixture), len(component)
        )
        lambda_km2, diagnostics_km2 = self._estimate_lambda(
            km2_threshold, kernel, len(mixture), len(component)
        )
        self.km1_estimate_ = float(np.clip((lambda_km1 - 1.0) / lambda_km1, 0.0, 1.0))
        self.km2_estimate_ = float(np.clip((lambda_km2 - 1.0) / lambda_km2, 0.0, 1.0))
        self.class_prior_ = self.km1_estimate_ if self.variant == "km1" else self.km2_estimate_
        self.kernel_width_ = width
        self.distribution_distance_ = distribution_distance
        self.thresholds_ = {"km1": km1_threshold, "km2": km2_threshold}
        self.diagnostics_ = {"km1": diagnostics_km1, "km2": diagnostics_km2}
        self.n_mixture_ = len(mixture)
        self.n_component_ = len(component)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def estimate(self) -> float:
        """Return the selected KM1 or KM2 mixture proportion."""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("KernelMeanPriorEstimator is not fitted. Call fit() first.")
        return self.class_prior_
