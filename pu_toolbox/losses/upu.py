# ruff: noqa: N803, N806

"""Unbiased PU (uPU) loss / risk functions.

Implements the convex PU risk family from du Plessis, Niu, Sugiyama
(ICML 2015):

    J(g) = −π E_P[g(X)] + E_U[ℓ(−g(X))]

where ℓ is a margin loss whose composite ℓ̃(z) = ℓ(z) − ℓ(−z) = −z
(linear), guaranteeing convexity of the overall risk.

Reference
---------
du Plessis, M. C., Niu, G., & Sugiyama, M.
"Convex Formulation for Learning from Positive and Unlabeled Data."
ICML, 2015.
"""

from __future__ import annotations

import numpy as np

from ..core.base import BasePULoss

# ═════════════════════════════════════════════════════════════════════
# Numerical helpers
# ═════════════════════════════════════════════════════════════════════


def _softplus_stable(z: np.ndarray) -> np.ndarray:
    """Compute log(1 + exp(z)) stably via logaddexp."""
    return np.logaddexp(0.0, z)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid: 1 / (1 + exp(−z))."""
    # Clip to avoid overflow in exp.
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


# ═════════════════════════════════════════════════════════════════════
# UPULoss
# ═════════════════════════════════════════════════════════════════════


class UPULoss(BasePULoss):
    """Unbiased PU risk for logistic or squared margin loss.

    This is the *risk* (not a trainable loss with parameters).  The
    estimator calls it to evaluate the PU objective for a given set of
    model scores.  It can also be reused by nnPU / PNU implementations
    that share the same risk decomposition framework (§8.7 of the method
    card).

    Parameters
    ----------
    loss : {"logistic", "squared"}, default "logistic"
        Margin loss ℓ.  Both satisfy ℓ̃(z) = −z, making the PU risk convex.

        * ``"logistic"``  — ℓ(z) = log(1 + exp(−z)),
          ℓ(−g(x_U)) = softplus(g(x_U)).
        * ``"squared"``   — ℓ(z) = ¼(z − 1)²,
          ℓ(−g(x_U)) = ¼(g(x_U) + 1)².

    Notes
    -----
    ``"double_hinge"`` is **not** supported here because it requires a
    QP solver; use :class:`~pu_toolbox.estimators.risk.upu.UPUClassifier`
    instead.
    """

    requires_class_prior: bool = True

    def __init__(self, loss: str = "logistic") -> None:
        if loss not in ("logistic", "squared"):
            raise ValueError(
                f"loss must be 'logistic' or 'squared'; got {loss!r}. "
                "Use UPUClassifier for double_hinge."
            )
        self.loss = loss

    def __call__(
        self,
        positive_scores: np.ndarray,
        unlabeled_scores: np.ndarray,
        *,
        class_prior: float,
    ) -> float:
        """Compute the uPU risk.

        Parameters
        ----------
        positive_scores : np.ndarray of shape (n_P,)
            g(x) for labeled-positive samples.
        unlabeled_scores : np.ndarray of shape (n_U,)
            g(x) for unlabeled samples.
        class_prior : float
            Class prior π = P(y=1).  Must satisfy 0 < π < 1.

        Returns
        -------
        float
            Scalar risk value.
        """
        if not (0.0 < class_prior < 1.0):
            raise ValueError(f"class_prior must be in (0, 1); got {class_prior}.")
        if len(positive_scores) == 0:
            raise ValueError("positive_scores must not be empty.")
        if len(unlabeled_scores) == 0:
            raise ValueError("unlabeled_scores must not be empty.")

        pos_term = -class_prior * float(np.mean(positive_scores))

        if self.loss == "logistic":
            unlabeled_term = float(np.mean(_softplus_stable(unlabeled_scores)))
        else:  # squared
            unlabeled_term = float(np.mean((unlabeled_scores + 1.0) ** 2)) / 4.0

        return pos_term + unlabeled_term

    # ── Gradient (for C-LL, used by L-BFGS optimisers) ────────────────

    def gradient(
        self,
        positive_scores: np.ndarray,
        unlabeled_scores: np.ndarray,
        *,
        class_prior: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return dJ/dg_P and dJ/dg_U for the current batch of scores.

        Useful for gradient-based optimisers (e.g. L-BFGS in
        :class:`UPUClassifier` with ``loss="logistic"``).

        Returns
        -------
        dpos : np.ndarray of shape (n_P,)
            ∂J / ∂g(x_i^P).
        dunl : np.ndarray of shape (n_U,)
            ∂J / ∂g(x_j^U).
        """
        n_P = len(positive_scores)
        n_U = len(unlabeled_scores)
        if n_P == 0 or n_U == 0:
            raise ValueError("positive_scores and unlabeled_scores must be non-empty.")

        dpos = np.full(n_P, -class_prior / n_P)

        if self.loss == "logistic":
            dunl = _sigmoid(unlabeled_scores) / n_U
        else:  # squared
            dunl = 0.5 * (unlabeled_scores + 1.0) / n_U

        return dpos, dunl
