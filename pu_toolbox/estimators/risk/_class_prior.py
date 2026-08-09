"""Shared class-prior estimation helper for risk-based estimators."""

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
