# ruff: noqa: N803, N806

"""PNU risk / loss functions.

Implements the Positive-Negative-Unlabeled (PNU) risk combination from
Sakai, du Plessis, Niu, Sugiyama (ICML 2017):

    R_PNU^eta(g) = (1-eta)R_PN(g) + eta*R_PU(g)     (eta >= 0)
    R_PNU^eta(g) = (1+eta)R_PN(g) - eta*R_NU(g)     (eta < 0)

v1 supports only the squared-loss convex formulation where the
composite loss simplifies to ell_tilde(m) = -m.

Reference
---------
Sakai, T., du Plessis, M. C., Niu, G., & Sugiyama, M.
"Semi-Supervised Classification Based on Classification from
Positive and Unlabeled Data."  ICML, 2017.
"""

from __future__ import annotations

import numpy as np


def _eta_to_gamma(eta: float) -> tuple[float, str]:
    """Convert PNU eta to non-negative gamma and branch label.

    R_PNU^eta = (1-gamma)*R_PN + gamma*R_PU   (eta >= 0)
                (1-gamma)*R_PN + gamma*R_NU   (eta < 0)

    Returns (gamma, branch) where gamma = |eta| and branch is "pu" or "nu".
    """
    if eta >= 0.0:
        return eta, "pu"
    else:
        return -eta, "nu"


# ═════════════════════════════════════════════════════════════════════
# Component risk helpers (squared loss — convex formulation)
# ═════════════════════════════════════════════════════════════════════


def _compute_pn_risk(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    class_prior: float,
) -> float:
    """PN (supervised) risk with squared composite loss.

    R_PN(g) = theta_P * mean_P[-g] + theta_N * mean_N[g]
    """
    theta_N = 1.0 - class_prior
    return float(class_prior * np.mean(-positive_scores) + theta_N * np.mean(negative_scores))


def _compute_pu_risk_squared(
    positive_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
    class_prior: float,
) -> float:
    """C-PU risk with squared composite loss.

    R_C-PU(g) = theta_P * mean_P[-g] + mean_U[g]
    """
    return float(class_prior * np.mean(-positive_scores) + np.mean(unlabeled_scores))


def _compute_nu_risk_squared(
    negative_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
    class_prior: float,
) -> float:
    """C-NU risk with squared composite loss.

    R_C-NU(g) = theta_N * mean_N[g] + mean_U[-g]
    """
    theta_N = 1.0 - class_prior
    return float(theta_N * np.mean(negative_scores) + np.mean(-unlabeled_scores))


def _compute_pnu_risk(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
    class_prior: float,
    eta: float,
) -> float:
    """Combined PNU risk.

    R_PNU^eta(g) = (1-eta)*R_PN + eta*R_C-PU   (eta >= 0)
                   (1+eta)*R_PN - eta*R_C-NU   (eta < 0)
    """
    r_pn = _compute_pn_risk(positive_scores, negative_scores, class_prior)
    gamma, branch = _eta_to_gamma(eta)

    if branch == "pu":
        r_component = _compute_pu_risk_squared(positive_scores, unlabeled_scores, class_prior)
    else:
        r_component = _compute_nu_risk_squared(negative_scores, unlabeled_scores, class_prior)
    return float((1.0 - gamma) * r_pn + gamma * r_component)
