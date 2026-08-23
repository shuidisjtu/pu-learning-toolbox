# ruff: noqa: N803, N806

"""Clean-room implementation of the AISTATS 2025 dynamic joint-shift PU objective.

The objective functions map directly to Kumagai et al. equations (13),
(19)--(23). Torch is imported lazily so core installations remain usable.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.core.device import resolve_device
from pu_toolbox.core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    SampleWeightSupport,
    Scenario,
    SourceStatus,
)
from pu_toolbox.core.validation import check_scalar_in_range, validate_pu_X_y

__all__ = [
    "DynamicJointShiftPUClassifier",
    "paper_classifier_objective",
    "paper_importance_weight_objective",
    "paper_pu_risk",
]


def _mean(values: Any, name: str) -> Any:
    if values.numel() == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return values.mean()


def paper_importance_weight_objective(
    *,
    target_positive_positive: Any,
    target_positive_negative: Any,
    target_unlabeled_negative: Any,
    source_positive_positive: Any,
    source_positive_negative: Any,
    source_unlabeled_negative: Any,
    source_class_prior: float,
    target_class_prior: float,
    alpha: float,
    correction: bool = True,
) -> Any:
    """Return equation (19), including its exact lower-bound corrections."""
    import torch

    check_scalar_in_range(source_class_prior, 0.0, 1.0, "source_class_prior", inclusive=False)
    check_scalar_in_range(target_class_prior, 0.0, 1.0, "target_class_prior", inclusive=False)
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1].")

    def transformed(weight: Any) -> Any:
        return alpha * weight.square() - 2.0 * weight

    target_positive = target_class_prior * _mean(
        transformed(target_positive_positive), "target_positive_positive"
    )
    target_negative_raw = _mean(
        transformed(target_unlabeled_negative), "target_unlabeled_negative"
    ) - target_class_prior * _mean(
        transformed(target_positive_negative), "target_positive_negative"
    )
    target_lower_bound = -(1.0 - target_class_prior) / alpha
    if correction:
        target_negative = torch.abs(target_negative_raw - target_lower_bound) + target_lower_bound
    else:
        target_negative = target_negative_raw

    source_positive = (
        source_class_prior
        * (1.0 - alpha)
        * _mean(source_positive_positive.square(), "source_positive_positive")
    )
    source_negative_raw = (1.0 - alpha) * (
        _mean(source_unlabeled_negative.square(), "source_unlabeled_negative")
        - source_class_prior * _mean(source_positive_negative.square(), "source_positive_negative")
    )
    source_negative = torch.abs(source_negative_raw) if correction else source_negative_raw
    return target_positive + target_negative + source_positive + source_negative


def _sigmoid_loss(logits: Any, label: int) -> Any:
    import torch

    signed = 1.0 if label == 1 else -1.0
    return torch.sigmoid(-signed * logits)


def paper_pu_risk(
    positive_logits: Any,
    unlabeled_logits: Any,
    *,
    class_prior: float,
    positive_positive_weight: Any | None = None,
    positive_negative_weight: Any | None = None,
    unlabeled_negative_weight: Any | None = None,
    correction: bool = True,
) -> Any:
    """Return equation (4)/(13) with sigmoid loss and optional joint weights."""
    import torch

    check_scalar_in_range(class_prior, 0.0, 1.0, "class_prior", inclusive=False)
    positive_loss = _sigmoid_loss(positive_logits, 1)
    positive_negative_loss = _sigmoid_loss(positive_logits, -1)
    unlabeled_negative_loss = _sigmoid_loss(unlabeled_logits, -1)
    if positive_positive_weight is not None:
        positive_loss = positive_loss * positive_positive_weight
    if positive_negative_weight is not None:
        positive_negative_loss = positive_negative_loss * positive_negative_weight
    if unlabeled_negative_weight is not None:
        unlabeled_negative_loss = unlabeled_negative_loss * unlabeled_negative_weight
    positive_term = class_prior * _mean(positive_loss, "positive_logits")
    negative_raw = _mean(unlabeled_negative_loss, "unlabeled_logits") - class_prior * _mean(
        positive_negative_loss, "positive_logits"
    )
    negative_term = torch.abs(negative_raw) if correction else negative_raw
    return positive_term + negative_term


def paper_classifier_objective(
    *,
    source_positive_logits: Any,
    source_unlabeled_logits: Any,
    target_positive_logits: Any,
    target_unlabeled_logits: Any,
    source_positive_positive_weight: Any,
    source_positive_negative_weight: Any,
    source_unlabeled_negative_weight: Any,
    source_class_prior: float,
    target_class_prior: float,
    beta: float,
    correction: bool = True,
) -> Any:
    """Return equation (20): target PU risk plus joint-weighted source PU risk."""
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0, 1].")
    target_risk = paper_pu_risk(
        target_positive_logits,
        target_unlabeled_logits,
        class_prior=target_class_prior,
        correction=correction,
    )
    source_risk = paper_pu_risk(
        source_positive_logits,
        source_unlabeled_logits,
        class_prior=source_class_prior,
        positive_positive_weight=source_positive_positive_weight,
        positive_negative_weight=source_positive_negative_weight,
        unlabeled_negative_weight=source_unlabeled_negative_weight,
        correction=correction,
    )
    return beta * target_risk + (1.0 - beta) * source_risk


class DynamicJointShiftPUClassifier(BasePUClassifier):
    """Dynamic shared-feature joint importance-weighted PU classifier.

    This is a clean-room implementation of equations (19)--(23) and
    Algorithm 1. It requires PU samples and known class priors in both domains.
    """

    family = AlgorithmFamily.BIAS_AWARE
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.SELECTION_BIASED,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.NOT_FOUND
    backend = Backend.TORCH
    maturity = Maturity.RESEARCH
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        beta: float = 0.5,
        hidden_dim: int = 128,
        feature_dim: int = 128,
        max_epochs: int = 200,
        classifier_learning_rate: float = 1e-4,
        weight_learning_rate: float = 1e-3,
        weight_correction: bool = True,
        classifier_correction: bool = True,
        training_mode: Literal["dynamic", "two_step"] = "dynamic",
        random_state: int | None = 42,
        device: str | None = "auto",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.max_epochs = max_epochs
        self.classifier_learning_rate = classifier_learning_rate
        self.weight_learning_rate = weight_learning_rate
        self.weight_correction = weight_correction
        self.classifier_correction = classifier_correction
        self.training_mode = training_mode
        self.random_state = random_state
        self.device = device

    def _validate_hyperparameters(self) -> None:
        if not np.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must lie in (0, 1].")
        if not np.isfinite(self.beta) or not 0 <= self.beta <= 1:
            raise ValueError("beta must lie in [0, 1].")
        for name, value in {
            "hidden_dim": self.hidden_dim,
            "feature_dim": self.feature_dim,
            "max_epochs": self.max_epochs,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1.")
        for name, value in {
            "classifier_learning_rate": self.classifier_learning_rate,
            "weight_learning_rate": self.weight_learning_rate,
        }.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0.")
        if self.training_mode not in {"dynamic", "two_step"}:
            raise ValueError("training_mode must be 'dynamic' or 'two_step'.")
        if not isinstance(self.weight_correction, bool) or not isinstance(
            self.classifier_correction, bool
        ):
            raise TypeError("correction flags must be bool values.")

    def fit(
        self,
        X: Any,
        y_pu: Any,
        *,
        X_target: Any | None = None,
        y_target_pu: Any | None = None,
        class_prior: float | None = None,
        target_class_prior: float | None = None,
        sample_weight: Any | None = None,
    ) -> DynamicJointShiftPUClassifier:
        """Fit with source and target PU datasets using Algorithm 1."""
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                "DynamicJointShiftPUClassifier requires the optional 'torch' dependency."
            ) from exc
        self._validate_hyperparameters()
        if sample_weight is not None:
            raise NotImplementedError("External sample_weight is not part of the paper objective.")
        if X_target is None or y_target_pu is None:
            raise ValueError("X_target and y_target_pu are required.")
        if class_prior is None or target_class_prior is None:
            raise ValueError("class_prior and target_class_prior are both required.")
        check_scalar_in_range(class_prior, 0.0, 1.0, "class_prior", inclusive=False)
        check_scalar_in_range(target_class_prior, 0.0, 1.0, "target_class_prior", inclusive=False)
        X_source, source_y = validate_pu_X_y(
            X, y_pu, accept_sparse=False, estimator_name=type(self).__name__
        )
        X_target_array, target_y = validate_pu_X_y(
            X_target,
            y_target_pu,
            accept_sparse=False,
            estimator_name=f"{type(self).__name__}(target)",
        )
        if X_source.shape[1] != X_target_array.shape[1]:
            raise ValueError("Source and target must have the same number of features.")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
        device = resolve_device(self.device)
        source_x = torch.as_tensor(np.asarray(X_source, dtype=np.float32), device=device)
        target_x = torch.as_tensor(np.asarray(X_target_array, dtype=np.float32), device=device)
        source_labels = torch.as_tensor(source_y, device=device)
        target_labels = torch.as_tensor(target_y, device=device)
        source_positive = source_labels == 1
        source_unlabeled = source_labels == 0
        target_positive = target_labels == 1
        target_unlabeled = target_labels == 0
        self.feature_extractor_ = nn.Sequential(
            nn.Linear(X_source.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
            nn.ReLU(),
        ).to(device)
        self.classifier_head_ = nn.Linear(self.feature_dim, 1).to(device)
        self.weight_head_ = nn.Sequential(
            nn.Linear(self.feature_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        ).to(device)
        weight_optimizer = torch.optim.Adam(
            self.weight_head_.parameters(), lr=self.weight_learning_rate
        )
        classifier_optimizer = torch.optim.Adam(
            list(self.feature_extractor_.parameters()) + list(self.classifier_head_.parameters()),
            lr=self.classifier_learning_rate,
        )
        self.training_trace_: list[dict[str, float]] = []

        def weight(features: Any, label: int) -> Any:
            one_hot = torch.zeros((len(features), 2), device=device)
            one_hot[:, 1 if label == 1 else 0] = 1.0
            return self.weight_head_(torch.cat([features, one_hot], dim=1)).squeeze(1) / self.alpha

        def weight_step() -> Any:
            weight_optimizer.zero_grad()
            source_features = self.feature_extractor_(source_x).detach()
            target_features = self.feature_extractor_(target_x).detach()
            objective = paper_importance_weight_objective(
                target_positive_positive=weight(target_features[target_positive], 1),
                target_positive_negative=weight(target_features[target_positive], -1),
                target_unlabeled_negative=weight(target_features[target_unlabeled], -1),
                source_positive_positive=weight(source_features[source_positive], 1),
                source_positive_negative=weight(source_features[source_positive], -1),
                source_unlabeled_negative=weight(source_features[source_unlabeled], -1),
                source_class_prior=float(class_prior),
                target_class_prior=float(target_class_prior),
                alpha=self.alpha,
                correction=self.weight_correction,
            )
            objective.backward()
            weight_optimizer.step()
            return objective.detach()

        def classifier_step() -> Any:
            classifier_optimizer.zero_grad()
            source_features = self.feature_extractor_(source_x)
            target_features = self.feature_extractor_(target_x)
            source_logits = self.classifier_head_(source_features).squeeze(1)
            target_logits = self.classifier_head_(target_features).squeeze(1)
            with torch.no_grad():
                source_detached = source_features.detach()
                source_pp = weight(source_detached[source_positive], 1)
                source_pn = weight(source_detached[source_positive], -1)
                source_un = weight(source_detached[source_unlabeled], -1)
            objective = paper_classifier_objective(
                source_positive_logits=source_logits[source_positive],
                source_unlabeled_logits=source_logits[source_unlabeled],
                target_positive_logits=target_logits[target_positive],
                target_unlabeled_logits=target_logits[target_unlabeled],
                source_positive_positive_weight=source_pp,
                source_positive_negative_weight=source_pn,
                source_unlabeled_negative_weight=source_un,
                source_class_prior=float(class_prior),
                target_class_prior=float(target_class_prior),
                beta=self.beta,
                correction=self.classifier_correction,
            )
            objective.backward()
            classifier_optimizer.step()
            return objective.detach()

        if self.training_mode == "two_step":
            weight_losses = [weight_step() for _ in range(self.max_epochs)]
            for epoch in range(self.max_epochs):
                classifier_loss = classifier_step()
                self.training_trace_.append(
                    {
                        "epoch": float(epoch + 1),
                        "weight_loss": float(weight_losses[-1].cpu()),
                        "classifier_loss": float(classifier_loss.cpu()),
                    }
                )
        else:
            for epoch in range(self.max_epochs):
                weight_loss = weight_step()
                classifier_loss = classifier_step()
                self.training_trace_.append(
                    {
                        "epoch": float(epoch + 1),
                        "weight_loss": float(weight_loss.cpu()),
                        "classifier_loss": float(classifier_loss.cpu()),
                    }
                )
        with torch.no_grad():
            source_features = self.feature_extractor_(source_x)
            self.source_joint_weights_ = np.column_stack(
                [
                    weight(source_features, -1).cpu().numpy(),
                    weight(source_features, 1).cpu().numpy(),
                ]
            )
        self.device_ = device
        self.source_class_prior_ = float(class_prior)
        self.target_class_prior_ = float(target_class_prior)
        self._class_prior = float(target_class_prior)
        self._X_shape_ = (len(source_y) + len(target_y), X_source.shape[1])
        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    def _decision_function(self, X: Any) -> np.ndarray:
        import torch

        array = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            features = self.feature_extractor_(torch.as_tensor(array, device=self.device_))
            return self.classifier_head_(features).squeeze(1).cpu().numpy()

    def _predict(self, X: Any) -> np.ndarray:
        return (self._decision_function(X) >= 0.0).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = 1.0 / (1.0 + np.exp(-np.clip(self._decision_function(X), -40, 40)))
        return np.column_stack([1.0 - probability, probability])

    def get_pu_metadata(self) -> dict[str, Any]:
        metadata = super().get_pu_metadata()
        metadata.update(
            {
                "paper": "Kumagai et al., AISTATS 2025",
                "objective_equations": [13, 19, 20, 21, 22, 23],
                "training_algorithm": "Algorithm 1",
                "training_mode": self.training_mode,
                "weight_correction": self.weight_correction,
                "classifier_correction": self.classifier_correction,
                "guarantee": "clean_room_paper_objective",
            }
        )
        return metadata
