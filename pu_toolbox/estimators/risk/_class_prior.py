"""Shared class-prior helpers for risk-based estimators."""

from __future__ import annotations


def solve_prior_from_positive_fraction(k: int, n: int, h: float) -> float:
    """Estimate the class prior ``p = k / (n * (1 - h))``.

    Parameters
    ----------
    k : int
        Number of labeled positives.
    n : int
        Total number of samples.
    h : float
        Class-conditional positive proportion in the denominator,
        ``h`` in ``[0, 1)``.

    Returns
    -------
    float
        The derived class prior, guaranteed to lie in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``n <= 0``, if ``h`` is outside ``[0, 1)``, or if the derived
        prior ``p`` falls outside ``(0, 1]``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if not 0.0 <= h < 1.0:
        raise ValueError(f"h must be in [0, 1), got {h}.")
    p = k / (n * (1.0 - h))
    if not (0.0 < p <= 1.0):
        raise ValueError(
            f"Derived class prior p = {p} is out of (0, 1]. "
            f"Check flip_probability (h={h}) and data: "
            f"k={k}, n={n}. Formula: p = k / [n·(1−h)]."
        )
    return p


def stable_centroid_denominator(
    p: float,
    h: float,
    *,
    tolerance: float = 1e-12,
) -> float:
    """Return the LDCE centroid denominator after a stability check.

    Both LDCE variants divide by ``1 - 2 * p * h`` when constructing the
    centroid term. Values close to zero make that term ill-conditioned and
    must be rejected consistently.

    Parameters
    ----------
    p : float
        Positive-class prior.
    h : float
        Positive-label flip probability.
    tolerance : float, default=1e-12
        Absolute denominator values below this threshold are rejected.

    Returns
    -------
    float
        The stable denominator ``1 - 2 * p * h``.

    Raises
    ------
    ValueError
        If the denominator's absolute value is below ``tolerance``.
    """
    denominator = 1.0 - 2.0 * p * h
    if abs(denominator) < tolerance:
        raise ValueError(
            f"Denominator 1−2ph = {denominator:.2e} is near-zero "
            f"(h={h}, p={p}). The centroid term is ill-conditioned "
            "for this data / flip_probability combination."
        )
    return denominator
