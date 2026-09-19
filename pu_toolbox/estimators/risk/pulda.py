# ruff: noqa: N803, N806
"""Label-distribution alignment PU learning (PULDA).

This module clean-room implements the loss and two-stage training flow exposed
by the authors' public code.  The default dense MLP is a toolbox adapter; the
upstream experiment uses a CIFAR CNN and is not reproduced here.
"""

from __future__ import annotations

import math

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
from ...core.validation import check_scalar_in_range, validate_pu_X_y


def symmetric_softplus_distance(left, right, *, temperature: float):
    """Return PULDA's symmetric softplus distance between scalar moments."""
    import torch.nn.functional as functional

    difference = left - right
    return functional.softplus(difference, beta=temperature) + functional.softplus(
        -difference, beta=temperature
    )


def pulda_distribution_alignment(
    logits,
    labels,
    *,
    class_prior: float,
    temperature: float,
    unlabeled_expectation=None,
    ema_correction: float = 1.0,
):
    """Compute the label-distribution term in the released PULDA code.

    ``unlabeled_expectation`` may carry an EMA value whose current-batch path
    remains differentiable. ``ema_correction`` is ``1-alpha_U`` after the
    first EMA update and one for the initial batch.
    """
    import torch

    labels = labels.reshape(-1)
    scores = torch.sigmoid(logits.reshape(-1))
    positive_scores = scores[labels == 1]
    unlabeled_scores = scores[labels == 0]
    if not positive_scores.numel() or not unlabeled_scores.numel():
        raise ValueError("PULDA batches must contain positive and unlabeled rows")
    positive_term = 1.0 - positive_scores.mean()
    expectation = (
        unlabeled_scores.mean() if unlabeled_expectation is None else unlabeled_expectation
    )
    target = torch.as_tensor(class_prior, dtype=logits.dtype, device=logits.device)
    unlabeled_term = (
        symmetric_softplus_distance(expectation, target, temperature=temperature) / ema_correction
    )
    return 2.0 * class_prior * positive_term + unlabeled_term, positive_term, unlabeled_term


def pulda_two_way_margin(
    logits,
    labels,
    *,
    class_prior: float,
    margin: float,
    temperature: float,
    positive_negative_expectation=None,
    unlabeled_negative_expectation=None,
    ema_correction: float = 1.0,
):
    """Compute the margin-based two-way sigmoid term from the public code."""
    import torch

    labels = labels.reshape(-1)
    logits = logits.reshape(-1)
    positive_logits = logits[labels == 1]
    unlabeled_logits = logits[labels == 0]
    if not positive_logits.numel() or not unlabeled_logits.numel():
        raise ValueError("PULDA batches must contain positive and unlabeled rows")

    def positive_cost(values):
        return torch.sigmoid(values) * torch.sigmoid(-(values - margin))

    def negative_cost(values):
        return torch.sigmoid(values + margin) * torch.sigmoid(-values)

    positive_plus = positive_cost(positive_logits).mean()
    positive_minus = (
        negative_cost(positive_logits).mean()
        if positive_negative_expectation is None
        else positive_negative_expectation
    )
    unlabeled_minus = (
        negative_cost(unlabeled_logits).mean()
        if unlabeled_negative_expectation is None
        else unlabeled_negative_expectation
    )
    target = class_prior * positive_minus
    distribution = (
        symmetric_softplus_distance(unlabeled_minus, target, temperature=temperature)
        / ema_correction
    )
    return class_prior * positive_plus + distribution, positive_plus, distribution


class _PULDAEMA:
    """Stateful EMA adapter matching the released loss modules."""

    def __init__(self, alpha_unlabeled: float, alpha_margin: float) -> None:
        self.alpha_unlabeled = alpha_unlabeled
        self.alpha_margin = alpha_margin
        self.unlabeled = None
        self.positive_negative = None
        self.unlabeled_negative = None

    def moments(self, logits, labels, margin):
        import torch

        labels = labels.reshape(-1)
        flat = logits.reshape(-1)
        scores = torch.sigmoid(flat)
        current_unlabeled = scores[labels == 0].mean()
        positive_negative = (
            torch.sigmoid(flat[labels == 1] + margin) * torch.sigmoid(-flat[labels == 1])
        ).mean()
        unlabeled_negative = (
            torch.sigmoid(flat[labels == 0] + margin) * torch.sigmoid(-flat[labels == 0])
        ).mean()
        first = self.unlabeled is None
        if not first:
            current_unlabeled = (
                self.alpha_unlabeled * self.unlabeled
                + (1 - self.alpha_unlabeled) * current_unlabeled
            )
            positive_negative = (
                self.alpha_margin * self.positive_negative
                + (1 - self.alpha_margin) * positive_negative
            )
            unlabeled_negative = (
                self.alpha_margin * self.unlabeled_negative
                + (1 - self.alpha_margin) * unlabeled_negative
            )
        self.unlabeled = current_unlabeled.detach()
        self.positive_negative = positive_negative.detach()
        self.unlabeled_negative = unlabeled_negative.detach()
        return (
            current_unlabeled,
            positive_negative,
            unlabeled_negative,
            1.0 if first else 1 - self.alpha_unlabeled,
            1.0 if first else 1 - self.alpha_margin,
        )


class PULDAClassifier(BasePUClassifier):
    """Fit the PULDA objective on dense two-dimensional PU features."""

    family = AlgorithmFamily.RISK_ESTIMATION
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_RELATED
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp"})
    input_ndims = frozenset({2})
    encoder_parameter = None
    trains_encoder = False

    def __init__(
        self,
        class_prior: float,
        *,
        hidden_dim: int = 64,
        depth: int = 2,
        warmup_epochs: int = 60,
        pu_epochs: int = 60,
        positive_batch_size: int = 16,
        unlabeled_batch_size: int = 128,
        warmup_learning_rate: float = 1e-4,
        learning_rate: float = 1e-3,
        warmup_weight_decay: float = 5e-4,
        weight_decay: float = 1e-4,
        temperature: float = 3.5,
        unlabeled_ema: float = 0.85,
        margin_ema: float = 0.5,
        margin: float = 0.6,
        mixup_weight: float = 4.2,
        mixup_alpha: float = 11.0,
        random_state: int | None = 0,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.warmup_epochs = warmup_epochs
        self.pu_epochs = pu_epochs
        self.positive_batch_size = positive_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.warmup_learning_rate = warmup_learning_rate
        self.learning_rate = learning_rate
        self.warmup_weight_decay = warmup_weight_decay
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.unlabeled_ema = unlabeled_ema
        self.margin_ema = margin_ema
        self.margin = margin
        self.mixup_weight = mixup_weight
        self.mixup_alpha = mixup_alpha
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        X,
        y_pu,
        *,
        class_prior=None,
        sample_weight=None,
        epoch_callback=None,
    ) -> PULDAClassifier:
        """Run distribution-alignment warmup followed by pseudo-label MixUp."""
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("PULDAClassifier requires the optional 'torch' dependency") from exc

        if sample_weight is not None:
            raise NotImplementedError("PULDA does not implement sample_weight")
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="PULDAClassifier")
        X = _finite_features(X, name="X")
        prior = self.class_prior if class_prior is None else class_prior
        check_scalar_in_range(prior, 0.0, 1.0, "class_prior", inclusive=False)
        self._validate_parameters()

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        device = resolve_device(self.device)
        layers = []
        width = X.shape[1]
        for _ in range(self.depth):
            layers.extend((nn.Linear(width, self.hidden_dim), nn.ReLU()))
            width = self.hidden_dim
        layers.append(nn.Linear(width, 1))
        self.model_ = nn.Sequential(*layers).to(device)
        tx = torch.as_tensor(X, device=device)
        ty = torch.as_tensor(y_pu, device=device)
        positive_indices = np.flatnonzero(y_pu == 1)
        unlabeled_indices = np.flatnonzero(y_pu == 0)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self._X_shape_ = X.shape
        self._class_prior = float(prior)
        self.device_ = device
        self._is_fitted = False
        self.optimizer_steps_ = 0
        self.history_ = {
            "epoch": [],
            "phase": [],
            "distribution_loss": [],
            "margin_loss": [],
            "mixup_loss": [],
            "train_loss": [],
            "optimizer_steps": [],
        }

        epoch_number = 0
        warmup_optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.warmup_learning_rate,
            weight_decay=self.warmup_weight_decay,
        )
        warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            warmup_optimizer, max(1, self.pu_epochs)
        )
        ema = _PULDAEMA(self.unlabeled_ema, self.margin_ema)
        for _ in range(self.warmup_epochs):
            totals = self._train_epoch(
                tx,
                ty,
                positive_indices,
                unlabeled_indices,
                rng,
                warmup_optimizer,
                ema,
                prior,
                pseudo_labels=None,
            )
            warmup_scheduler.step()
            self._record_epoch(epoch_number, "warmup", totals, epoch_callback)
            epoch_number += 1

        with torch.no_grad():
            pseudo_labels = torch.sigmoid(self.model_(tx).flatten().clamp(-10, 10)).detach()
        pu_optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        pu_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            pu_optimizer, max(1, self.pu_epochs), eta_min=0.7 * self.learning_rate
        )
        ema = _PULDAEMA(self.unlabeled_ema, self.margin_ema)
        for _ in range(self.pu_epochs):
            totals = self._train_epoch(
                tx,
                ty,
                positive_indices,
                unlabeled_indices,
                rng,
                pu_optimizer,
                ema,
                prior,
                pseudo_labels=pseudo_labels,
            )
            pu_scheduler.step()
            self._record_epoch(epoch_number, "pu_mixup", totals, epoch_callback)
            epoch_number += 1

        self.pseudo_labels_ = pseudo_labels.detach().cpu().numpy()
        self._is_fitted = True
        return self

    def _train_epoch(
        self,
        tx,
        ty,
        positive_indices,
        unlabeled_indices,
        rng,
        optimizer,
        ema,
        prior,
        *,
        pseudo_labels,
    ):
        import torch
        from torch.nn import functional

        self.model_.train()
        order = rng.permutation(unlabeled_indices)
        steps = max(1, math.ceil(len(order) / self.unlabeled_batch_size))
        totals = np.zeros(4, dtype=np.float64)
        for step in range(steps):
            u_idx = order[step * self.unlabeled_batch_size : (step + 1) * self.unlabeled_batch_size]
            p_idx = rng.choice(positive_indices, size=self.positive_batch_size, replace=True)
            indices = np.concatenate((p_idx, u_idx))
            batch_x = tx[indices]
            batch_y = ty[indices]
            logits = self.model_(batch_x).flatten().clamp(-10, 10)
            u_exp, p_neg, u_neg, u_correction, margin_correction = ema.moments(
                logits, batch_y, self.margin
            )
            distribution, _, _ = pulda_distribution_alignment(
                logits,
                batch_y,
                class_prior=prior,
                temperature=self.temperature,
                unlabeled_expectation=u_exp,
                ema_correction=u_correction,
            )
            margin_loss, _, _ = pulda_two_way_margin(
                logits,
                batch_y,
                class_prior=prior,
                margin=self.margin,
                temperature=1.0,
                positive_negative_expectation=p_neg,
                unlabeled_negative_expectation=u_neg,
                ema_correction=margin_correction,
            )
            mixup_loss = logits.new_zeros(())
            if pseudo_labels is not None:
                targets = pseudo_labels[indices].clone()
                targets[: len(p_idx)] = 1.0
                mixing = float(rng.beta(self.mixup_alpha, self.mixup_alpha))
                permutation = torch.as_tensor(
                    rng.permutation(len(indices)), dtype=torch.long, device=self.device_
                )
                mixed_x = mixing * batch_x + (1 - mixing) * batch_x[permutation]
                mixed_target = mixing * targets + (1 - mixing) * targets[permutation]
                mixed_logits = self.model_(mixed_x).flatten().clamp(-10, 10)
                mixup_loss = functional.binary_cross_entropy_with_logits(mixed_logits, mixed_target)
            loss = distribution + margin_loss + self.mixup_weight * mixup_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("PULDA objective became non-finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.optimizer_steps_ += 1
            if pseudo_labels is not None:
                with torch.no_grad():
                    pseudo_labels[indices] = torch.sigmoid(logits.detach())
            totals += [
                float(distribution.detach().cpu()),
                float(margin_loss.detach().cpu()),
                float(mixup_loss.detach().cpu()),
                float(loss.detach().cpu()),
            ]
        return totals / steps

    def _record_epoch(self, epoch, phase, totals, epoch_callback):
        self.history_["epoch"].append(epoch)
        self.history_["phase"].append(phase)
        for key, value in zip(
            ("distribution_loss", "margin_loss", "mixup_loss", "train_loss"),
            totals,
            strict=True,
        ):
            self.history_[key].append(float(value))
        self.history_["optimizer_steps"].append(self.optimizer_steps_)
        self._is_fitted = True
        if epoch_callback is not None:
            epoch_callback(epoch, self)

    def _validate_parameters(self):
        for name in (
            "hidden_dim",
            "positive_batch_size",
            "unlabeled_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("depth", "warmup_epochs", "pu_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.warmup_epochs + self.pu_epochs < 1:
            raise ValueError("at least one of warmup_epochs or pu_epochs must be positive")
        for name in (
            "warmup_learning_rate",
            "learning_rate",
            "temperature",
            "mixup_alpha",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("warmup_weight_decay", "weight_decay", "margin", "mixup_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("unlabeled_ema", "margin_ema"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")

    def _decision_function(self, X):
        import torch

        X = _finite_features(X, name="prediction X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("prediction X has the wrong feature dimension")
        self.model_.eval()
        with torch.no_grad():
            return (
                self.model_(torch.as_tensor(X, device=self.device_)).flatten().cpu().numpy()
                if len(X)
                else np.empty(0, dtype=np.float32)
            )

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        self._check_is_fitted()
        scores = self._decision_function(X)
        positive = 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
        return np.column_stack((1 - positive, positive))


def _finite_features(X, *, name):
    X = np.asarray(X)
    if X.ndim != 2 or not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"{name} must be a dense 2-D numeric array")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.asarray(X, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} overflows float32")
    return result
