# ruff: noqa: N803, N806

"""Non-Negative PU (nnPU) classifier — Algorithm 1 (Kiryo et al. 2017).

Implements the non-negative PU risk estimator for flexible (deep) models
trained via mini-batch SGD.  The core training logic separates the
*reported* nnPU risk from the *optimisation* quantity so that the
correction branch produces gradients from −γ·r alone (not from
differentiating max(0, r)).

Reference
---------
Kiryo, R., Niu, G., du Plessis, M. C., & Sugiyama, M.
"Positive-Unlabeled Learning with Non-Negative Risk Estimator."
NIPS, 2017.
"""

from __future__ import annotations

import copy
import warnings
from collections import defaultdict
from itertools import cycle
from typing import Literal

import numpy as np

from ...core.base import BasePUClassifier
from ...core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    Scenario,
    SourceStatus,
)
from ...core.validation import validate_pu_X_y
from ...losses.nnpu import NonNegativePULoss, _nnpu_train_step

# ═════════════════════════════════════════════════════════════════════
# NonNegativePUClassifier
# ═════════════════════════════════════════════════════════════════════


class NonNegativePUClassifier(BasePUClassifier):
    """Non-negative PU classifier (nnPU).

    Fits a flexible model g(x) by minimising the non-negative PU risk
    via mini-batch SGD (Algorithm 1 of Kiryo et al. 2017).

    Parameters
    ----------
    model : torch.nn.Module or None, default None
        PyTorch model that outputs raw scores g(x).
        If None, a default ``nn.Linear(n_features, 1)`` is created.
    loss : {"sigmoid"}, default "sigmoid"
        Surrogate loss.  MVP only supports sigmoid.
    beta : float, default 0.0
        Non-negativity threshold.  Must be >= 0.
    gamma : float, default 1.0
        Correction-branch step-size discount.  Must be in [0, 1].
    optimizer : torch.optim.Optimizer or None, default None
        PyTorch optimiser instance.  If None, ``Adam(lr=1e-3)`` is used.
    batch_size : int, default 256
        Mini-batch size.  The P and U batches are independently sized
        to ``min(batch_size, n_P)`` and ``min(batch_size, n_U)``.
    max_epochs : int, default 200
        Maximum number of training epochs.
    patience : int, default 20
        Early-stopping patience (epochs).  Only used when
        ``validation_data`` is passed to :meth:`fit`.
    random_state : int or None, default None
        Seed for PyTorch and NumPy RNGs (reproducibility).

    Attributes
    ----------
    model_ : torch.nn.Module
        Fitted model.
    class_prior_ : float
        Class prior used during training.
    n_positive_ : int
        Number of labeled-positive samples.
    n_unlabeled_ : int
        Number of unlabeled samples.
    history_ : dict of list
        Per-epoch training metrics.  Keys: ``epoch``,
        ``positive_risk``, ``negative_risk``, ``upu_risk``,
        ``nnpu_risk``, ``optimization_loss``, ``correction_fraction``.
    classes_ : np.ndarray
        ``np.array([0, 1])``.
    """

    # ── Class-level metadata ──────────────────────────────────────────
    family: AlgorithmFamily = AlgorithmFamily.RISK_ESTIMATION
    assumption: tuple[Assumption, ...] = (Assumption.SCAR,)
    scenario: tuple[Scenario, ...] = (Scenario.CASE_CONTROL,)
    requires_class_prior: bool = True
    implementation_status: ImplementationStatus = ImplementationStatus.NATIVE
    source_status: SourceStatus = SourceStatus.OFFICIAL_EXACT
    backend: Backend = Backend.TORCH
    maturity: Maturity = Maturity.STABLE

    def __init__(
        self,
        model: torch.nn.Module | None = None,  # noqa: F821
        *,
        loss: Literal["sigmoid"] = "sigmoid",
        beta: float = 0.0,
        gamma: float = 1.0,
        optimizer: torch.optim.Optimizer | None = None,  # noqa: F821
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.beta = beta
        self.gamma = gamma
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

    # ── fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float,
        sample_weight: np.ndarray | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> NonNegativePUClassifier:
        """Fit the nnPU classifier via mini-batch SGD (Algorithm 1).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y_pu : np.ndarray of shape (n_samples,)
            PU labels.  +1 = labeled positive, 0 = unlabeled.
        class_prior : float
            Class prior π = P(y=1).  **Required.** Must be in (0, 1).
        sample_weight : np.ndarray of shape (n_samples,), optional
            Per-sample weights.  Normalised independently within the
            P and U groups.
        validation_data : tuple (X_val, y_pu_val), optional
            PU validation data for early stopping.  If provided,
            training stops after ``patience`` epochs without
            improvement in nnPU validation risk.

        Returns
        -------
        self : NonNegativePUClassifier
        """
        import torch

        # ── Validate inputs ───────────────────────────────────────
        X, y_pu = validate_pu_X_y(
            X, y_pu, estimator_name="NonNegativePUClassifier"
        )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values.")
        if not (0.0 < class_prior < 1.0):
            raise ValueError(
                f"class_prior must be in (0, 1); got {class_prior}."
            )
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0; got {self.beta}.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(
                f"gamma must be in [0, 1]; got {self.gamma}."
            )
        if self.loss != "sigmoid":
            raise ValueError(
                f"loss must be 'sigmoid' in MVP; got {self.loss!r}."
            )
        if self.max_epochs <= 0:
            raise ValueError(
                f"max_epochs must be > 0; got {self.max_epochs}."
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be > 0; got {self.batch_size}."
            )

        # Warn if beta exceeds sigmoid upper bound
        if self.beta > class_prior:
            warnings.warn(
                f"beta ({self.beta}) > class_prior ({class_prior}). "
                "With sigmoid loss (bounded by 1), the correction "
                "branch may never activate. Consider beta <= class_prior.",
                UserWarning,
                stacklevel=2,
            )

        # ── Reproducibility ───────────────────────────────────────
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(rng.randint(0, 2**31))

        # ── Split P / U ───────────────────────────────────────────
        mask_P = y_pu == 1
        X_P = X[mask_P]
        X_U = X[~mask_P]
        n_P = X_P.shape[0]
        n_U = X_U.shape[0]
        d = X.shape[1]

        if n_P == 0:
            raise ValueError("Need at least 1 labeled positive sample.")
        if n_U == 0:
            raise ValueError("Need at least 1 unlabeled sample.")

        self.n_positive_ = n_P
        self.n_unlabeled_ = n_U
        self._class_prior = class_prior
        self.class_prior_ = class_prior
        self._X_shape_ = X.shape

        # ── Sample weights ────────────────────────────────────────
        w_P = None
        w_U = None
        if sample_weight is not None:
            w_P = sample_weight[mask_P]
            w_U = sample_weight[~mask_P]

        # ── Build model ───────────────────────────────────────────
        if self.model is not None:
            self.model_ = copy.deepcopy(self.model)
        else:
            self.model_ = torch.nn.Linear(d, 1)

        # ── Device ────────────────────────────────────────────────
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_.to(device)

        # ── Build optimiser ───────────────────────────────────────
        if self.optimizer is not None:
            opt = self.optimizer
            # Re-bind to current model parameters if optimiser was
            # created with a (now-stale) param group.
            if hasattr(opt, "param_groups"):
                opt.param_groups[0]["params"] = list(
                    self.model_.parameters()
                )
        else:
            opt = torch.optim.Adam(self.model_.parameters(), lr=1e-3)

        # ── Convert data to tensors ───────────────────────────────
        X_P_t = torch.tensor(X_P, dtype=torch.float32)
        X_U_t = torch.tensor(X_U, dtype=torch.float32)

        # Always wrap in TensorDataset so DataLoader yields tuples
        if w_P is not None:
            w_P_t = torch.tensor(w_P, dtype=torch.float32)
            ds_P = torch.utils.data.TensorDataset(X_P_t, w_P_t)
        else:
            ds_P = torch.utils.data.TensorDataset(X_P_t)
        if w_U is not None:
            w_U_t = torch.tensor(w_U, dtype=torch.float32)
            ds_U = torch.utils.data.TensorDataset(X_U_t, w_U_t)
        else:
            ds_U = torch.utils.data.TensorDataset(X_U_t)

        # ── Batch sizes (cannot exceed class counts) ──────────────
        batch_P = min(self.batch_size, n_P)
        batch_U = min(self.batch_size, n_U)

        # ── Initialise history ────────────────────────────────────
        self.history_ = {
            "epoch": [],
            "positive_risk": [],
            "negative_risk": [],
            "upu_risk": [],
            "nnpu_risk": [],
            "optimization_loss": [],
            "correction_fraction": [],
        }

        # ── Validation data ───────────────────────────────────────
        X_val_t = None
        y_val_pu = None
        if validation_data is not None:
            X_val, y_val_pu = validation_data
            X_val, y_val_pu = validate_pu_X_y(
                X_val,
                y_val_pu,
                estimator_name="NonNegativePUClassifier[val]",
            )
            X_val_t = torch.tensor(X_val, dtype=torch.float32)

        # ── Training loop ─────────────────────────────────────────
        best_val_risk = float("inf")
        best_state = copy.deepcopy(self.model_.state_dict())
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.model_.train()

            # Shuffle: re-create loaders each epoch
            p_loader = torch.utils.data.DataLoader(
                ds_P, batch_size=batch_P, shuffle=True, drop_last=False
            )
            u_loader = torch.utils.data.DataLoader(
                ds_U, batch_size=batch_U, shuffle=True, drop_last=False
            )

            # Cycle the shorter loader
            n_batches = max(len(p_loader), len(u_loader))
            p_iter = (
                iter(cycle(p_loader))
                if len(p_loader) < n_batches
                else iter(p_loader)
            )
            u_iter = (
                iter(cycle(u_loader))
                if len(u_loader) < n_batches
                else iter(u_loader)
            )

            epoch_sum = defaultdict(float)
            n_corrections = 0

            for _step in range(n_batches):
                p_item = next(p_iter)
                u_item = next(u_iter)

                # Handle optional sample-weight channel
                if w_P is not None:
                    batch_P_x, batch_P_w = p_item
                    batch_P_w = batch_P_w.to(device)
                else:
                    (batch_P_x,) = p_item
                    batch_P_w = None
                if w_U is not None:
                    batch_U_x, batch_U_w = u_item
                    batch_U_w = batch_U_w.to(device)
                else:
                    (batch_U_x,) = u_item
                    batch_U_w = None

                batch_P_x = batch_P_x.to(device)
                batch_U_x = batch_U_x.to(device)

                opt.zero_grad()

                scores_P = self.model_(batch_P_x).squeeze(-1)
                scores_U = self.model_(batch_U_x).squeeze(-1)

                # Apply sample weights within each group
                if batch_P_w is not None:
                    # Weighted mean: Σ(w·ℓ) / Σw
                    loss_pos_p = torch.sigmoid(-scores_P)
                    loss_pos_n = torch.sigmoid(scores_P)
                    R_p_plus = (
                        (batch_P_w * loss_pos_p).sum() / batch_P_w.sum()
                    )
                    R_p_minus = (
                        (batch_P_w * loss_pos_n).sum() / batch_P_w.sum()
                    )
                else:
                    R_p_plus = torch.sigmoid(-scores_P).mean()
                    R_p_minus = torch.sigmoid(scores_P).mean()

                if batch_U_w is not None:
                    loss_unl_n = torch.sigmoid(scores_U)
                    R_u_minus = (
                        (batch_U_w * loss_unl_n).sum() / batch_U_w.sum()
                    )
                else:
                    R_u_minus = torch.sigmoid(scores_U).mean()

                opt_loss, step_info = _nnpu_train_step(
                    R_p_plus,
                    R_p_minus,
                    R_u_minus,
                    class_prior=class_prior,
                    beta=self.beta,
                    gamma=self.gamma,
                )

                opt_loss.backward()
                opt.step()

                # Accumulate per-step metrics
                epoch_sum["positive_risk"] += step_info["positive_risk"]
                epoch_sum["negative_risk"] += step_info["negative_risk"]
                epoch_sum["upu_risk"] += step_info["upu_risk"]
                epoch_sum["nnpu_risk"] += step_info["nnpu_risk"]
                epoch_sum["optimization_loss"] += step_info[
                    "optimization_loss"
                ]
                if step_info["correction"]:
                    n_corrections += 1

            # ── End of epoch ──────────────────────────────────────
            n_steps = n_batches
            self.history_["epoch"].append(epoch)
            self.history_["positive_risk"].append(
                epoch_sum["positive_risk"] / n_steps
            )
            self.history_["negative_risk"].append(
                epoch_sum["negative_risk"] / n_steps
            )
            self.history_["upu_risk"].append(
                epoch_sum["upu_risk"] / n_steps
            )
            self.history_["nnpu_risk"].append(
                epoch_sum["nnpu_risk"] / n_steps
            )
            self.history_["optimization_loss"].append(
                epoch_sum["optimization_loss"] / n_steps
            )
            self.history_["correction_fraction"].append(
                n_corrections / n_steps
            )

            # ── Early stopping (on validation nnPU risk) ──────────
            if validation_data is not None and X_val_t is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_scores = (
                        self.model_(X_val_t.to(device))
                        .squeeze(-1)
                        .cpu()
                        .numpy()
                    )
                mask_val_P = y_val_pu == 1
                evaluator = NonNegativePULoss(
                    loss=self.loss, beta=self.beta, gamma=self.gamma
                )
                val_risk = evaluator(
                    val_scores[mask_val_P],
                    val_scores[~mask_val_P],
                    class_prior=class_prior,
                    non_negative=True,
                )

                if val_risk < best_val_risk:
                    best_val_risk = val_risk
                    best_state = copy.deepcopy(self.model_.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    break

        # ── Restore best model (if early-stopped) ─────────────────
        if validation_data is not None:
            self.model_.load_state_dict(best_state)

        # ── Finalise ──────────────────────────────────────────────
        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    # ── Decision function / predict ─────────────────────────────────

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """g(x) raw scores from the fitted model."""
        import torch

        self._check_is_fitted()
        X_t = torch.tensor(X, dtype=torch.float32)
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            scores = self.model_(X_t.to(device)).squeeze(-1).cpu().numpy()
        return scores

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Binary labels: 1 if g(x) >= 0 else 0."""
        return (self._decision_function(X) >= 0.0).astype(int)

    # ── predict_proba ────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Not implemented — g(x) is not a calibrated probability.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "NonNegativePUClassifier does not implement predict_proba. "
            "The decision function g(x) is not a calibrated probability. "
            "Use decision_function() instead."
        )

    # ── evaluate_pu_risk ─────────────────────────────────────────────

    def evaluate_pu_risk(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        non_negative: bool = True,
    ) -> float:
        """Compute uPU or nnPU risk on the full dataset.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y_pu : np.ndarray of shape (n_samples,)
            PU labels.
        class_prior : float, optional
            Override the stored class prior.
        non_negative : bool, default True
            If True, return nnPU risk R̃_pu (Eq. 6).
            If False, return unbiased uPU risk R̂_pu.

        Returns
        -------
        float
        """
        from ...core.labels import normalize_pu_labels

        self._check_is_fitted()
        pi = (
            class_prior
            if class_prior is not None
            else self._class_prior
        )
        if pi is None:
            raise ValueError(
                "class_prior must be provided (was not stored during fit)."
            )

        y_pu = normalize_pu_labels(y_pu)
        scores = self._decision_function(X)
        mask_P = y_pu == 1

        evaluator = NonNegativePULoss(
            loss=self.loss, beta=self.beta, gamma=self.gamma
        )
        return evaluator(
            scores[mask_P],
            scores[~mask_P],
            class_prior=pi,
            non_negative=non_negative,
        )

    # ── Training history ─────────────────────────────────────────────

    def get_training_history(self) -> dict:
        """Return per-epoch training metrics.

        Returns
        -------
        dict
            Keys: ``epoch``, ``positive_risk``, ``negative_risk``,
            ``upu_risk``, ``nnpu_risk``, ``optimization_loss``,
            ``correction_fraction``.
        """
        self._check_is_fitted()
        return dict(self.history_)

    # ── Metadata ─────────────────────────────────────────────────────

    def get_pu_metadata(self) -> dict:
        """Return PU metadata including nnPU-specific diagnostics."""
        meta = super().get_pu_metadata()
        meta.update(
            {
                "loss": self.loss,
                "beta": self.beta,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "n_positive": getattr(self, "n_positive_", None),
                "n_unlabeled": getattr(self, "n_unlabeled_", None),
                "final_correction_fraction": (
                    self.history_["correction_fraction"][-1]
                    if getattr(self, "history_", None)
                    and self.history_["correction_fraction"]
                    else None
                ),
            }
        )
        return meta
