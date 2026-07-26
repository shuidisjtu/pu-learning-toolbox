# ruff: noqa: N803, N806

"""Information-maximisation representation learning from PU data."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ...core.base import BasePUClassifier
from ...core.exceptions import NotFittedError
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
from ...prior.pen_l1 import ClassPriorEstimator


def pu_smi_objective(
    positive_ratio: np.ndarray,
    unlabeled_ratio: np.ndarray,
) -> float:
    """Compute the empirical PU-SMI density-ratio objective (paper Eq. 4)."""
    positive_ratio = np.asarray(positive_ratio, dtype=float)
    unlabeled_ratio = np.asarray(unlabeled_ratio, dtype=float)
    if positive_ratio.size == 0 or unlabeled_ratio.size == 0:
        raise ValueError("positive_ratio and unlabeled_ratio must be non-empty")
    if not np.isfinite(positive_ratio).all() or not np.isfinite(unlabeled_ratio).all():
        raise ValueError("density-ratio values must be finite")
    return float(0.5 * np.mean(unlabeled_ratio**2) - np.mean(positive_ratio))


class InfoMaxPURepresentation(BaseEstimator, TransformerMixin):
    """Learn a low-dimensional PURL representation by PU-SMI maximisation."""

    def __init__(
        self,
        *,
        representation_dim: int = 20,
        hidden_dim: int = 60,
        ratio_steps: int = 4,
        encoder_steps: int = 1,
        max_epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        standardize: bool = True,
        random_state: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.representation_dim = representation_dim
        self.hidden_dim = hidden_dim
        self.ratio_steps = ratio_steps
        self.encoder_steps = encoder_steps
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.standardize = standardize
        self.random_state = random_state
        self.device = device

    def fit(self, X: np.ndarray, y_pu: np.ndarray) -> InfoMaxPURepresentation:
        """Fit the encoder and density-ratio head with alternating updates."""
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                "InfoMaxPURepresentation requires the optional 'torch' dependency"
            ) from exc

        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            estimator_name="InfoMaxPURepresentation",
        )
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values")
        for name, value in (
            ("representation_dim", self.representation_dim),
            ("hidden_dim", self.hidden_dim),
            ("ratio_steps", self.ratio_steps),
            ("encoder_steps", self.encoder_steps),
            ("max_epochs", self.max_epochs),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")

        if self.standardize:
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.0)
            X_train = (X - self.mean_) / self.scale_
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=np.float32)
            self.scale_ = np.ones(X.shape[1], dtype=np.float32)
            X_train = X

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        device = torch.device(self.device)
        self.encoder_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.representation_dim),
        ).to(device)
        self.ratio_head_ = nn.Linear(self.representation_dim, 1).to(device)

        encoder_optimizer = torch.optim.SGD(
            self.encoder_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        ratio_optimizer = torch.optim.SGD(
            self.ratio_head_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        tx = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        mask_p = torch.as_tensor(y_pu == 1, device=device)
        mask_u = ~mask_p
        self.history_ = {"objective": []}

        def objective(*, detach_encoder: bool) -> torch.Tensor:
            representation = self.encoder_(tx)
            if detach_encoder:
                representation = representation.detach()
            ratio = self.ratio_head_(representation).squeeze(1)
            return 0.5 * ratio[mask_u].square().mean() - ratio[mask_p].mean()

        for _ in range(self.max_epochs):
            for _ in range(self.ratio_steps):
                ratio_optimizer.zero_grad()
                ratio_loss = objective(detach_encoder=True)
                ratio_loss.backward()
                ratio_optimizer.step()

            for parameter in self.ratio_head_.parameters():
                parameter.requires_grad_(False)
            for _ in range(self.encoder_steps):
                encoder_optimizer.zero_grad()
                encoder_loss = objective(detach_encoder=False)
                encoder_loss.backward()
                encoder_optimizer.step()
            for parameter in self.ratio_head_.parameters():
                parameter.requires_grad_(True)

            with torch.no_grad():
                self.history_["objective"].append(float(objective(detach_encoder=False).cpu()))

        self.n_features_in_ = X.shape[1]
        self.device_ = device
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map inputs to the learned PURL representation."""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("InfoMaxPURepresentation is not fitted. Call fit() first.")
        import torch

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        X = (X - self.mean_) / self.scale_
        self.encoder_.eval()
        with torch.no_grad():
            result = self.encoder_(torch.as_tensor(X, dtype=torch.float32, device=self.device_))
        return result.cpu().numpy()

    def density_ratio(self, X: np.ndarray) -> np.ndarray:
        """Return the learned unconstrained density-ratio-head output."""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("InfoMaxPURepresentation is not fitted. Call fit() first.")
        import torch

        representation = self.transform(X)
        self.ratio_head_.eval()
        with torch.no_grad():
            result = self.ratio_head_(
                torch.as_tensor(
                    representation,
                    dtype=torch.float32,
                    device=self.device_,
                )
            ).squeeze(1)
        return result.cpu().numpy()

    def get_training_history(self) -> dict[str, list[float]]:
        """Return a copy of the per-epoch PU-SMI objective history."""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("InfoMaxPURepresentation is not fitted. Call fit() first.")
        return {key: list(values) for key, values in self.history_.items()}


class InfoMaxPUClassifier(BasePUClassifier):
    """PURL followed by class-prior estimation and an nnPU classifier."""

    family = AlgorithmFamily.DEEP_PU
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = False
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.NOT_FOUND
    backend = Backend.TORCH
    maturity = Maturity.RESEARCH

    def __init__(
        self,
        *,
        class_prior: float | None = None,
        representation_dim: int = 20,
        hidden_dim: int = 60,
        representation_epochs: int = 200,
        classifier_epochs: int = 200,
        learning_rate: float = 1e-3,
        prior_estimator: BaseEstimator | None = None,
        random_state: int | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.representation_dim = representation_dim
        self.hidden_dim = hidden_dim
        self.representation_epochs = representation_epochs
        self.classifier_epochs = classifier_epochs
        self.learning_rate = learning_rate
        self.prior_estimator = prior_estimator
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> InfoMaxPUClassifier:
        """Fit PURL, resolve the class prior, then train nnPU."""
        import torch

        from ..risk.nnpu import NonNegativePUClassifier

        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            estimator_name="InfoMaxPUClassifier",
        )
        X = np.asarray(X, dtype=np.float32)
        self.representation_ = InfoMaxPURepresentation(
            representation_dim=self.representation_dim,
            hidden_dim=self.hidden_dim,
            max_epochs=self.representation_epochs,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            device=self.device,
        ).fit(X, y_pu)
        representation = self.representation_.transform(X)

        resolved_prior = self.class_prior if class_prior is None else class_prior
        if resolved_prior is None:
            estimator = (
                ClassPriorEstimator() if self.prior_estimator is None else self.prior_estimator
            )
            self.prior_estimator_ = estimator.fit(representation, y_pu)
            resolved_prior = float(self.prior_estimator_.estimate())
        if not 0.0 < resolved_prior < 1.0:
            raise ValueError("class_prior must be in (0, 1)")

        model = torch.nn.Linear(self.representation_dim, 1).to(self.device)
        self.classifier_ = NonNegativePUClassifier(
            model=model,
            class_prior=resolved_prior,
            max_epochs=self.classifier_epochs,
            random_state=self.random_state,
        ).fit(representation, y_pu, sample_weight=sample_weight)
        self.class_prior_ = float(resolved_prior)
        self.classes_ = np.array([0, 1])
        self._class_prior = self.class_prior_
        self._X_shape_ = X.shape
        self._is_fitted = True
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.classifier_.decision_function(self.representation_.transform(X))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self._decision_function(X) >= 0.0).astype(int)
