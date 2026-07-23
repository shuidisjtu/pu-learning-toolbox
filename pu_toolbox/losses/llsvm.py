# ruff: noqa: N803, N806

"""LLSVM loss components and gradients.

Implements the objective function from the official MATLAB code of:

    Gong, C., Liu, T., Yang, J., & Tao, D.
    "Large-Margin Label-Calibrated Support Vector Machines
    for Positive and Unlabeled Learning."
    IEEE TNNLS, 30(11), 3471-3482, 2019.

Formulas follow the official code, NOT the paper (see method card §4.3):
- Exponential coefficient: exp(-5 f^2)  (paper uses -3)
- Squash function: (A/pi) * arctan(f)   (paper fixes A=2; code uses A=10)
- P and U hat terms: un-normalised sums  (paper divides by p and u)
"""

from __future__ import annotations

import numpy as np


def positive_hinge_loss(
    X_p: np.ndarray,
    w: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """Squared hinge loss on positive samples.

    Loss:  alpha * sum_P [max(1 - f, 0)]^2
    Grad:  alpha * sum_P 2*min(f - 1, 0) * x
    """
    if len(X_p) == 0:
        return 0.0, np.zeros_like(w)

    scores = X_p @ w  # (n_p,)
    residuals = np.maximum(1.0 - scores, 0.0)  # (n_p,)
    loss = alpha * float(np.sum(residuals ** 2))

    # Gradient: d/dw [max(1-f,0)]^2 = 2*min(f-1, 0) * x  (when f < 1)
    coeffs = 2.0 * alpha * np.minimum(scores - 1.0, 0.0)  # (n_p,)
    grad = X_p.T @ coeffs  # (d,)

    return loss, grad


def unlabeled_hat_loss(
    X_u: np.ndarray,
    w: np.ndarray,
    beta: float,
) -> tuple[float, np.ndarray]:
    """Gaussian-like hat loss on unlabeled samples.

    Loss:  beta * sum_U exp(-5 f^2)
    Grad:  beta * sum_U (-10 f exp(-5 f^2)) * x
    """
    if len(X_u) == 0:
        return 0.0, np.zeros_like(w)

    scores = X_u @ w  # (n_u,)
    exp_terms = np.exp(-5.0 * scores ** 2)  # (n_u,)
    loss = beta * float(np.sum(exp_terms))

    coeffs = beta * (-10.0 * scores * exp_terms)  # (n_u,)
    grad = X_u.T @ coeffs  # (d,)

    return loss, grad


def calibration_loss(
    X_u: np.ndarray,
    w: np.ndarray,
    gamma: float,
    t: float,
    A: float,
    n_unlabeled: int,
) -> tuple[float, np.ndarray]:
    """Label calibration loss on unlabeled samples.

    Loss:  (gamma / u) * sum_U [max(A/pi * arctan(f) - t, 0)]^2
    Grad:  (gamma / u) * sum_U 2*A / (pi*(1+f^2)) * max(phi-t, 0) * x
    where phi = A/pi * arctan(f)
    """
    if len(X_u) == 0:
        return 0.0, np.zeros_like(w)

    scores = X_u @ w  # (n_u,)
    phi = A / np.pi * np.arctan(scores)  # (n_u,)
    violations = np.maximum(phi - t, 0.0)  # (n_u,)

    loss = gamma / n_unlabeled * float(np.sum(violations ** 2))

    # d/dw phi = A / (pi * (1 + f^2)) * x
    # d/dw [max(phi-t,0)]^2 = 2 * max(phi-t,0) * A / (pi*(1+f^2)) * x
    coeffs = (
        2.0 * A * gamma / (np.pi * n_unlabeled)
        * violations / (1.0 + scores ** 2)
    )  # (n_u,)
    grad = X_u.T @ coeffs  # (d,)

    return loss, grad


def llsvm_objective(
    w: np.ndarray,
    X_p: np.ndarray,
    X_u: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    t: float,
    A: float,
    reg_lambda: float,
) -> tuple[float, np.ndarray]:
    """Full LLSVM objective: three loss terms + L2 regularisation.

    Returns (loss, gradient) w.r.t. w.
    """
    l_p, g_p = positive_hinge_loss(X_p, w, alpha)
    l_u, g_u = unlabeled_hat_loss(X_u, w, beta)
    n_u = max(len(X_u), 1)
    l_c, g_c = calibration_loss(X_u, w, gamma, t, A, n_u)

    reg_loss = 0.5 * reg_lambda * float(np.dot(w, w))
    reg_grad = reg_lambda * w

    total_loss = l_p + l_u + l_c + reg_loss
    total_grad = g_p + g_u + g_c + reg_grad

    return total_loss, total_grad
