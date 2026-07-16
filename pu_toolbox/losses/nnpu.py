# ruff: noqa: N803, N806

"""Non-Negative PU (nnPU) loss / risk functions.

Implements the non-negative PU risk estimator from Kiryo, Niu,
du Plessis, Sugiyama (NIPS 2017):

    R̃_pu(g) = π R_p^+(g) + max{0, R_u^−(g) − π R_p^−(g)}

where each component risk uses the sigmoid surrogate loss:

    R_p^+(g) = (1/n_p) Σ_i σ(−g(x_i^p))       # P as positive
    R_p^−(g) = (1/n_p) Σ_i σ(+g(x_i^p))       # P as negative
    R_u^−(g) = (1/n_u) Σ_i σ(+g(x_i^u))       # U as negative

The training step implements Algorithm 1 with separate optimisation
and reporting quantities — the correction branch gradient comes from
−γ·r, NOT from differentiating max(0, r).

Reference
---------
Kiryo, R., Niu, G., du Plessis, M. C., & Sugiyama, M.
"Positive-Unlabeled Learning with Non-Negative Risk Estimator."
NIPS, 2017.
"""

from __future__ import annotations

import numpy as np

from ..core.base import BasePULoss

# ═════════════════════════════════════════════════════════════════════
# Numerical helpers
# ═════════════════════════════════════════════════════════════════════


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid: 1 / (1 + exp(−z))."""
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


# ═════════════════════════════════════════════════════════════════════
# NonNegativePULoss — NumPy evaluator
# ═════════════════════════════════════════════════════════════════════


class NonNegativePULoss(BasePULoss):
    """Non-negative PU risk evaluator (NumPy).

    Computes the three component risks, the negative-risk term *r*,
    and both the uPU and nnPU risk values from pre-computed model
    scores.  Intended for full-dataset evaluation (after each epoch)
    rather than mini-batch training.

    Parameters
    ----------
    loss : {"sigmoid"}, default "sigmoid"
        Surrogate loss.  MVP only supports sigmoid.
    beta : float, default 0.0
        Non-negativity threshold (paper default).
    gamma : float, default 1.0
        Correction-branch step-size discount (paper default).

    Notes
    -----
    This class is stateless; ``beta`` and ``gamma`` affect only the
    reporting of ``nnpu_risk``, not the internal risk decomposition.
    """

    requires_class_prior: bool = True

    def __init__(
        self,
        loss: str = "sigmoid",
        beta: float = 0.0,
        gamma: float = 1.0,
    ) -> None:
        if loss != "sigmoid":
            raise ValueError(
                f"loss must be 'sigmoid' in MVP; got {loss!r}."
            )
        if beta < 0:
            raise ValueError(f"beta must be >= 0; got {beta}.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1]; got {gamma}.")
        self.loss = loss
        self.beta = beta
        self.gamma = gamma

    def __call__(
        self,
        positive_scores: np.ndarray,
        unlabeled_scores: np.ndarray,
        *,
        class_prior: float,
        non_negative: bool = True,
    ) -> float:
        """Compute the PU risk.

        Parameters
        ----------
        positive_scores : np.ndarray of shape (n_P,)
            Model scores g(x) for labeled-positive samples.
        unlabeled_scores : np.ndarray of shape (n_U,)
            Model scores g(x) for unlabeled samples.
        class_prior : float
            Class prior π = P(y=1).  Must satisfy 0 < π < 1.
        non_negative : bool, default True
            If True, return the nnPU risk R̃_pu (Eq. 6).
            If False, return the unbiased uPU risk R̂_pu.

        Returns
        -------
        float
            Scalar risk value.
        """
        info = self.evaluate(
            positive_scores, unlabeled_scores, class_prior=class_prior
        )
        return info["nnpu_risk"] if non_negative else info["upu_risk"]

    def evaluate(
        self,
        positive_scores: np.ndarray,
        unlabeled_scores: np.ndarray,
        *,
        class_prior: float,
    ) -> dict:
        """Compute all risk components.

        Parameters
        ----------
        positive_scores : np.ndarray of shape (n_P,)
        unlabeled_scores : np.ndarray of shape (n_U,)
        class_prior : float

        Returns
        -------
        dict with keys:
            positive_risk, negative_risk, upu_risk, nnpu_risk
        """
        if not (0.0 < class_prior < 1.0):
            raise ValueError(
                f"class_prior must be in (0, 1); got {class_prior}."
            )
        if len(positive_scores) == 0:
            raise ValueError("positive_scores must not be empty.")
        if len(unlabeled_scores) == 0:
            raise ValueError("unlabeled_scores must not be empty.")

        # Component risks (Eqs. 4.1–4.3)
        R_p_plus = float(np.mean(_sigmoid_stable(-positive_scores)))
        R_p_minus = float(np.mean(_sigmoid_stable(positive_scores)))
        R_u_minus = float(np.mean(_sigmoid_stable(unlabeled_scores)))

        pi = class_prior
        r = R_u_minus - pi * R_p_minus  # negative-risk term
        upu_risk = pi * R_p_plus + r     # unbiased risk

        # nnPU risk (Eq. 6)
        nnpu_risk = pi * R_p_plus + max(0.0, r)

        return {
            "positive_risk": R_p_plus,
            "negative_risk": r,
            "upu_risk": upu_risk,
            "nnpu_risk": nnpu_risk,
        }


# ═════════════════════════════════════════════════════════════════════
# PyTorch training step
# ═════════════════════════════════════════════════════════════════════


def _nnpu_train_step(
    R_p_plus: torch.Tensor,  # noqa: F821
    R_p_minus: torch.Tensor,  # noqa: F821
    R_u_minus: torch.Tensor,  # noqa: F821
    *,
    class_prior: float,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, dict]:  # noqa: F821
    """Single nnPU training step — Algorithm 1 (Kiryo et al. 2017).

    Applies the branch logic to pre-computed component risks from the
    current P/U mini-batches, producing an optimisation loss whose
    ``.backward()`` yields the correct Algorithm-1 gradients.

    Parameters
    ----------
    R_p_plus : torch.Tensor (scalar)
        Empirical risk on P samples treated as positive:
        ``mean(σ(−g(x_i^p)))``.
    R_p_minus : torch.Tensor (scalar)
        Empirical risk on P samples treated as negative:
        ``mean(σ(+g(x_i^p)))``.
    R_u_minus : torch.Tensor (scalar)
        Empirical risk on U samples treated as negative:
        ``mean(σ(+g(x_i^u)))``.
    class_prior : float
        Class prior π = P(y=1).
    beta : float, default 0.0
        Non-negativity threshold.
    gamma : float, default 1.0
        Correction-branch step-size discount.

    Returns
    -------
    opt_loss : torch.Tensor
        Scalar tensor.  Call ``opt_loss.backward()`` to accumulate
        correct Algorithm-1 gradients.
    info : dict
        Scalar Python floats for logging:
        ``positive_risk``, ``negative_risk``, ``upu_risk``,
        ``nnpu_risk``, ``optimization_loss``, ``correction``.

    Notes
    -----
    Callers are responsible for computing the three component risks
    from model scores.  This decoupling allows the classifier to use
    weighted means (sample_weight) while gradient-level tests pass raw
    ``.mean()`` risks.
    """
    import torch

    pi = class_prior
    r = R_u_minus - pi * R_p_minus  # negative-risk term
    upu_risk = pi * R_p_plus + r     # unbiased uPU risk

    # ── Branch decision (per mini-batch, per Algorithm 1) ─────────
    # Using Python if/else on a float (.item()) means autograd builds
    # the correct computational graph for each branch independently.
    # This matches the official Chainer implementation's separation of
    # self.loss (reporting) from self.x_out (backward source).

    if r.item() < -beta:
        # CORRECTION BRANCH: r < −β
        #   Reported risk:  π·R_p^+ − β          (detached, logging only)
        #   Optimisation:   −γ·r                  (pushes r back toward 0)
        #   R_p^+ is detached → its gradient does NOT flow.
        nnpu_report = pi * R_p_plus.detach() - beta
        opt_loss = -gamma * r
        in_correction = True
    else:
        # NORMAL BRANCH: r ≥ −β
        #   Both report and optimisation use the full uPU risk.
        nnpu_report = upu_risk
        opt_loss = upu_risk
        in_correction = False

    info = {
        "positive_risk": R_p_plus.item(),
        "negative_risk": r.item(),
        "upu_risk": upu_risk.item(),
        "nnpu_risk": nnpu_report.item()
        if isinstance(nnpu_report, torch.Tensor)
        else nnpu_report,
        "optimization_loss": opt_loss.item(),
        "correction": in_correction,
    }
    return opt_loss, info
