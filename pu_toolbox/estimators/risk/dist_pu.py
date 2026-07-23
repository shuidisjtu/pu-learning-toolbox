# ruff: noqa: N803, N806, E501

"""Dist-PU label-distribution alignment classifier.

The implementation follows the paper's three train-time signals: positive
supervision, unlabeled expectation alignment, and entropy minimisation.  A
small Mixup term is included when ``mixup_weight`` is non-zero.  PyTorch is an
optional dependency and is imported only when ``fit`` is called.
"""

from __future__ import annotations

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


class DistPUClassifier(BasePUClassifier):
    """Train a small MLP using Dist-PU's label-distribution objective."""

    family = AlgorithmFamily.RISK_ESTIMATION
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.TORCH
    maturity = Maturity.RESEARCH

    def __init__(self, class_prior: float, *, hidden_dim: int = 64, epochs: int = 100, batch_size: int = 128,
                 learning_rate: float = 1e-3, alignment_weight: float = 1.0, entropy_weight: float = 0.05,
                 mixup_weight: float = 0.1, random_state: int | None = 0, device: str = "cpu") -> None:
        super().__init__()
        self.class_prior = class_prior
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alignment_weight = alignment_weight
        self.entropy_weight = entropy_weight
        self.mixup_weight = mixup_weight
        self.random_state = random_state
        self.device = device

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("DistPUClassifier requires the optional 'torch' dependency") from exc
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="DistPUClassifier")
        pi = self.class_prior if class_prior is None else class_prior
        if not 0.0 < pi < 1.0 or self.epochs < 1 or self.hidden_dim < 1:
            raise ValueError("class_prior must be in (0, 1), epochs >= 1, hidden_dim >= 1")
        X = np.asarray(X, dtype=np.float32)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        device = torch.device(self.device)
        self.model_ = nn.Sequential(nn.Linear(X.shape[1], self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        tx = torch.as_tensor(X, device=device)
        ty = torch.as_tensor(y_pu, device=device)
        p_mask, u_mask = ty == 1, ty == 0
        self.loss_history_ = []
        bce = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model_(tx).squeeze(1).clamp(-10, 10)
            probs = torch.sigmoid(logits)
            positive_loss = bce(logits[p_mask], torch.ones_like(logits[p_mask]))
            alignment = (probs[u_mask].mean() - pi).pow(2)
            entropy = -(probs[u_mask] * torch.log(probs[u_mask] + 1e-6) + (1 - probs[u_mask]) * torch.log(1 - probs[u_mask] + 1e-6)).mean()
            loss = positive_loss + self.alignment_weight * alignment + self.entropy_weight * entropy
            if self.mixup_weight > 0 and len(X) > 1:
                perm = torch.randperm(len(X), device=device)
                lam = torch.rand((), device=device)
                mix_x = lam * tx + (1 - lam) * tx[perm]
                mix_y = lam * probs.detach() + (1 - lam) * probs.detach()[perm]
                loss = loss + self.mixup_weight * bce(self.model_(mix_x).squeeze(1), mix_y)
            loss.backward()
            optimizer.step()
            self.loss_history_.append(float(loss.detach().cpu()))
        self.classes_ = np.array([0, 1])
        self._class_prior, self._X_shape_, self._is_fitted = pi, X.shape, True
        self.device_ = device
        return self

    def _decision_function(self, X):
        import torch
        with torch.no_grad():
            return self.model_(torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device_)).squeeze(1).cpu().numpy()

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        score = 1.0 / (1.0 + np.exp(-np.clip(self._decision_function(X), -40, 40)))
        return np.column_stack([1.0 - score, score])
