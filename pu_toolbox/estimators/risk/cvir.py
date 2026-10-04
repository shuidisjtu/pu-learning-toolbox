"""Conditional Value Ignoring Risk (CVIR), a tabular PyTorch adaptation.

Garg et al., NeurIPS 2021, Algorithm 2.  The supplied mixture proportion is
the positive share *inside the unlabeled pool*.  BBE/TEDn prior estimation,
image/text backbones and the paper's convergence stopping rule are not here.
"""

# ruff: noqa: N803, N806, N812

from __future__ import annotations

import numpy as np

from ...core.base import BasePUClassifier
from ...core.device import resolve_device
from ...core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    SampleWeightSupport,
    Scenario,
    SourceStatus,
)
from ...core.validation import validate_pu_X_y


def select_cvir_negatives(scores: np.ndarray, unlabeled_positive_prior: float) -> np.ndarray:
    """Keep the lowest-scored ``1-alpha`` share of the original U rows.

    Stable ordering makes tied scores deterministic.  The count is rounded
    down (and clamped to one), an explicit finite-sample adaptation of the
    paper's fractional population definition.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be a non-empty finite one-dimensional array")
    if not np.isfinite(unlabeled_positive_prior) or not 0 < unlabeled_positive_prior < 1:
        raise ValueError("unlabeled_positive_prior must be in (0, 1)")
    count = max(1, int(np.floor((1 - unlabeled_positive_prior) * len(scores))))
    return np.argsort(scores, kind="stable")[:count]


class CVIRClassifier(BasePUClassifier):
    """Fit CVIR given the positive fraction within the original U pool.

    ``class_prior`` is *not* accepted as a substitute for the required
    ``unlabeled_positive_prior`` unless the caller has established that its U
    pool is an independent sample of the population marginal.  The frozen
    survey runner passes population pi, so this adapter is not in its matrix.
    """

    family = AlgorithmFamily.RISK_ESTIMATION
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = False  # population pi is not the required alpha_U
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp"})
    input_ndims = frozenset({2})
    encoder_parameter = None
    trains_encoder = False

    def __init__(
        self,
        *,
        unlabeled_positive_prior: float | None = None,
        hidden_dim: int = 64,
        warm_start_epochs: int = 1,
        max_epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.unlabeled_positive_prior = unlabeled_positive_prior
        self.hidden_dim = hidden_dim
        self.warm_start_epochs = warm_start_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
        epoch_callback=None,
    ) -> CVIRClassifier:
        """Train on P and U; do not infer alpha_U from the observed P/U ratio."""
        import torch
        from torch.nn import functional as F

        if sample_weight is not None:
            raise NotImplementedError("CVIR does not implement sample_weight")
        if class_prior is not None:
            raise ValueError(
                "CVIR needs unlabeled_positive_prior=alpha_U; class_prior is the "
                "population prior and cannot be substituted without a verified marginal U pool"
            )
        alpha = self.unlabeled_positive_prior
        if alpha is None or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("unlabeled_positive_prior must be in (0, 1)")
        for name in ("hidden_dim", "warm_start_epochs", "max_epochs", "batch_size"):
            value = getattr(self, name)
            lower = 0 if name == "warm_start_epochs" else 1
            if type(value) is not int or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="CVIRClassifier")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("X must be finite numeric")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X must remain finite after float32 conversion")
        p_idx = np.flatnonzero(y_pu == 1)
        u_idx = np.flatnonzero(y_pu == 0)
        if not len(u_idx):
            raise ValueError("CVIR needs original unlabeled rows")

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        device = resolve_device(self.device)
        model = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        ).to(device)
        data = torch.as_tensor(X, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.model_ = model
        self.n_features_in_ = X.shape[1]
        self.n_positive_, self.n_unlabeled_ = len(p_idx), len(u_idx)
        self.unlabeled_positive_prior_ = float(alpha)
        self.history_ = {"warm_start_loss": [], "cvir_loss": [], "selected_negative_count": []}
        self.selected_negative_indices_ = np.array([], dtype=int)
        self._is_fitted = False

        def train_epoch(negative_idx: np.ndarray, *, reweight: bool) -> float:
            positive_order = rng.permutation(p_idx)
            negative_order = rng.permutation(negative_idx)
            steps = max(
                int(np.ceil(len(positive_order) / self.batch_size)),
                int(np.ceil(len(negative_order) / self.batch_size)),
            )
            losses = []
            model.train()
            for step in range(steps):
                p = positive_order[(step * self.batch_size) % len(p_idx) :][: self.batch_size]
                n = negative_order[(step * self.batch_size) % len(negative_idx) :][
                    : self.batch_size
                ]
                positive_loss = F.softplus(-model(data[p]).reshape(-1)).mean()
                negative_loss = F.softplus(model(data[n]).reshape(-1)).mean()
                loss = (
                    (alpha * positive_loss + (1 - alpha) * negative_loss)
                    if reweight
                    else (positive_loss + negative_loss)
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("CVIR loss became non-finite")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            return float(np.mean(losses))

        epoch = 0
        for _ in range(self.warm_start_epochs):
            self.history_["warm_start_loss"].append(train_epoch(u_idx, reweight=False))
            self._is_fitted = True
            if epoch_callback is not None:
                epoch_callback(epoch, self)
            epoch += 1
        for _ in range(self.max_epochs):
            model.eval()
            with torch.no_grad():
                scores = model(data[u_idx]).reshape(-1).cpu().numpy()
            local_negative = select_cvir_negatives(scores, float(alpha))
            negative_idx = u_idx[local_negative]
            self.selected_negative_indices_ = negative_idx.copy()
            self.history_["selected_negative_count"].append(len(negative_idx))
            self.history_["cvir_loss"].append(train_epoch(negative_idx, reweight=True))
            self._is_fitted = True
            if epoch_callback is not None:
                epoch_callback(epoch, self)
            epoch += 1
        self.classes_ = np.array([0, 1])
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("CVIR prediction X must have fitted 2-D feature shape")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("CVIR prediction X must be finite numeric")
        self.model_.eval()
        with torch.no_grad():
            device = next(self.model_.parameters()).device
            scores = self.model_(torch.as_tensor(X, dtype=torch.float32, device=device))
        return scores.reshape(-1).cpu().numpy()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self._decision_function(X) >= 0).astype(int)
