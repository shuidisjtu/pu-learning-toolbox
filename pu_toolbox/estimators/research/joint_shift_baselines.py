# ruff: noqa: N803, N806

"""Paper-aligned baselines and ablation factory for joint-shift PU experiments."""

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

from .dynamic_joint_shift import DynamicJointShiftPUClassifier, paper_pu_risk

BaselineStrategy = Literal["trpu", "tepu", "fine_tune", "mmd"]
JointShiftMethod = Literal[
    "dynamic",
    "trpu",
    "tepu",
    "fine_tune",
    "mmd",
    "two_step",
    "without_weight_correction",
    "without_classifier_correction",
    "without_both_corrections",
]

JOINT_SHIFT_METHODS: tuple[JointShiftMethod, ...] = (
    "dynamic",
    "trpu",
    "tepu",
    "fine_tune",
    "mmd",
    "two_step",
    "without_weight_correction",
    "without_classifier_correction",
    "without_both_corrections",
)

__all__ = [
    "JOINT_SHIFT_METHODS",
    "JointShiftPUBaseline",
    "build_joint_shift_estimator",
]


def _mmd_rbf_mixture(source: Any, target: Any) -> Any:
    """Five-kernel RBF MMD used by the paper's domain-adaptation baseline."""
    import torch

    combined = torch.cat([source, target], dim=0)
    distances = torch.cdist(combined, combined).square()
    positive = distances.detach()[distances.detach() > 0]
    median = positive.median() if positive.numel() else torch.ones((), device=source.device)
    scales = (0.25, 0.5, 1.0, 2.0, 4.0)

    def kernel(left: Any, right: Any) -> Any:
        squared = torch.cdist(left, right).square()
        return sum(torch.exp(-squared / (2.0 * median * scale + 1e-8)) for scale in scales)

    return (
        kernel(source, source).mean()
        + kernel(target, target).mean()
        - 2 * kernel(source, target).mean()
    )


class JointShiftPUBaseline(BasePUClassifier):
    """Comparable neural trPU, tePU, fine-tuning, and MMD baselines."""

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
        strategy: BaselineStrategy,
        beta: float = 0.5,
        mmd_weight: float = 0.1,
        hidden_dim: int = 128,
        feature_dim: int = 128,
        max_epochs: int = 200,
        learning_rate: float = 1e-4,
        random_state: int | None = 42,
        device: str | None = "auto",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.beta = beta
        self.mmd_weight = mmd_weight
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device

    def _validate_hyperparameters(self) -> None:
        if self.strategy not in {"trpu", "tepu", "fine_tune", "mmd"}:
            raise ValueError("strategy must be trpu, tepu, fine_tune, or mmd.")
        if not np.isfinite(self.beta) or not 0 <= self.beta <= 1:
            raise ValueError("beta must lie in [0, 1].")
        if not np.isfinite(self.mmd_weight) or self.mmd_weight < 0:
            raise ValueError("mmd_weight must be finite and non-negative.")
        for name, value in {
            "hidden_dim": self.hidden_dim,
            "feature_dim": self.feature_dim,
            "max_epochs": self.max_epochs,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and > 0.")

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
    ) -> JointShiftPUBaseline:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                "Joint-shift baselines require the optional 'torch' dependency."
            ) from exc
        self._validate_hyperparameters()
        if sample_weight is not None:
            raise NotImplementedError("External sample_weight is not part of these baselines.")
        if X_target is None or y_target_pu is None:
            raise ValueError("X_target and y_target_pu are required for comparable baselines.")
        if class_prior is None or target_class_prior is None:
            raise ValueError("class_prior and target_class_prior are both required.")
        check_scalar_in_range(class_prior, 0.0, 1.0, "class_prior", inclusive=False)
        check_scalar_in_range(target_class_prior, 0.0, 1.0, "target_class_prior", inclusive=False)
        source_x, source_y = validate_pu_X_y(
            X, y_pu, accept_sparse=False, estimator_name=f"{type(self).__name__}(source)"
        )
        target_x, target_y = validate_pu_X_y(
            X_target,
            y_target_pu,
            accept_sparse=False,
            estimator_name=f"{type(self).__name__}(target)",
        )
        if source_x.shape[1] != target_x.shape[1]:
            raise ValueError("Source and target must have the same number of features.")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
        device = resolve_device(self.device)
        source_tensor = torch.as_tensor(np.asarray(source_x, dtype=np.float32), device=device)
        target_tensor = torch.as_tensor(np.asarray(target_x, dtype=np.float32), device=device)
        source_labels = torch.as_tensor(source_y, device=device)
        target_labels = torch.as_tensor(target_y, device=device)
        self.feature_extractor_ = nn.Sequential(
            nn.Linear(source_x.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
            nn.ReLU(),
        ).to(device)
        self.classifier_head_ = nn.Linear(self.feature_dim, 1).to(device)
        optimizer = torch.optim.Adam(
            list(self.feature_extractor_.parameters()) + list(self.classifier_head_.parameters()),
            lr=self.learning_rate,
        )
        self.training_trace_: list[dict[str, float | str]] = []

        def domain_risk(features: Any, labels: Any, prior: float) -> Any:
            logits = self.classifier_head_(features).squeeze(1)
            return paper_pu_risk(
                logits[labels == 1],
                logits[labels == 0],
                class_prior=prior,
                correction=True,
            )

        def train_epoch(stage: str) -> None:
            optimizer.zero_grad()
            source_features = self.feature_extractor_(source_tensor)
            target_features = self.feature_extractor_(target_tensor)
            source_risk = domain_risk(source_features, source_labels, float(class_prior))
            target_risk = domain_risk(target_features, target_labels, float(target_class_prior))
            mmd = torch.zeros((), device=device)
            if stage == "source":
                loss = source_risk
            elif stage == "target":
                loss = target_risk
            else:
                mmd = _mmd_rbf_mixture(source_features, target_features)
                loss = (
                    (1.0 - self.beta) * source_risk
                    + self.beta * target_risk
                    + self.mmd_weight * mmd
                )
            loss.backward()
            optimizer.step()
            self.training_trace_.append(
                {
                    "stage": stage,
                    "loss": float(loss.detach().cpu()),
                    "source_risk": float(source_risk.detach().cpu()),
                    "target_risk": float(target_risk.detach().cpu()),
                    "mmd": float(mmd.detach().cpu()),
                }
            )

        if self.strategy == "fine_tune":
            for _ in range(self.max_epochs):
                train_epoch("source")
            for _ in range(self.max_epochs):
                train_epoch("target")
        else:
            stage = {"trpu": "source", "tepu": "target", "mmd": "mmd"}[self.strategy]
            for _ in range(self.max_epochs):
                train_epoch(stage)
        self.device_ = device
        self.source_class_prior_ = float(class_prior)
        self.target_class_prior_ = float(target_class_prior)
        self._class_prior = float(target_class_prior)
        self._X_shape_ = (len(source_y) + len(target_y), source_x.shape[1])
        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    def _decision_function(self, X: Any) -> np.ndarray:
        import torch

        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device_)
            return self.classifier_head_(self.feature_extractor_(tensor)).squeeze(1).cpu().numpy()

    def _predict(self, X: Any) -> np.ndarray:
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = 1.0 / (1.0 + np.exp(-np.clip(self._decision_function(X), -40, 40)))
        return np.column_stack([1.0 - probability, probability])

    def get_pu_metadata(self) -> dict[str, Any]:
        metadata = super().get_pu_metadata()
        metadata.update(
            {
                "baseline_strategy": self.strategy,
                "paper_comparison": True,
                "mmd_kernel_count": 5 if self.strategy == "mmd" else 0,
                "guarantee": "research_baseline",
            }
        )
        return metadata


def build_joint_shift_estimator(
    method: JointShiftMethod,
    *,
    alpha: float = 0.1,
    beta: float = 0.5,
    mmd_weight: float = 0.1,
    hidden_dim: int = 128,
    feature_dim: int = 128,
    max_epochs: int = 200,
    classifier_learning_rate: float = 1e-4,
    weight_learning_rate: float = 1e-3,
    random_state: int | None = 42,
    device: str | None = "auto",
) -> BasePUClassifier:
    """Build a paper method, comparison baseline, or named ablation."""
    if method not in JOINT_SHIFT_METHODS:
        raise ValueError(
            f"Unknown joint-shift method {method!r}. Available: {JOINT_SHIFT_METHODS}."
        )
    shared = {
        "beta": beta,
        "hidden_dim": hidden_dim,
        "feature_dim": feature_dim,
        "max_epochs": max_epochs,
        "random_state": random_state,
        "device": device,
    }
    if method in {"trpu", "tepu", "fine_tune", "mmd"}:
        return JointShiftPUBaseline(
            strategy=method,
            mmd_weight=mmd_weight,
            learning_rate=classifier_learning_rate,
            **shared,
        )
    dynamic_options: dict[str, Any] = {
        "alpha": alpha,
        "classifier_learning_rate": classifier_learning_rate,
        "weight_learning_rate": weight_learning_rate,
        **shared,
    }
    if method == "two_step":
        dynamic_options["training_mode"] = "two_step"
    elif method == "without_weight_correction":
        dynamic_options["weight_correction"] = False
    elif method == "without_classifier_correction":
        dynamic_options["classifier_correction"] = False
    elif method == "without_both_corrections":
        dynamic_options["weight_correction"] = False
        dynamic_options["classifier_correction"] = False
    return DynamicJointShiftPUClassifier(**dynamic_options)
