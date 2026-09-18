"""GradPU: gradient-penalized, positive-upweighted PU learning.

This is a paper-derived, tabular MLP implementation of Dai et al. (AAAI 2023),
Equations 5--7 and Algorithm 1. The paper's image CNN and published benchmark
numbers are not claimed by this implementation.
"""

# ruff: noqa: N803, N806

from __future__ import annotations

import copy

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


def gradpu_positive_weight(raw_scores, beta: float):
    """Equation 6: ``1 - beta * log((1 + tanh(raw_scores)) / 2)``.

    ``logsigmoid(2 * raw_scores)`` is the numerically stable equivalent.
    The resulting weight is at least one and is *not* normalized by its sum.
    """
    from torch.nn import functional

    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    return 1.0 - beta * functional.logsigmoid(2.0 * raw_scores)


def gradpu_objective(
    raw_positive,
    raw_unlabeled,
    *,
    beta: float,
    alpha: float,
    interpolated_inputs=None,
    raw_interpolated=None,
):
    """Equations 5--7 using bounded tanh scores and raw-score input gradients.

    Return ``(total_loss, components)`` with differentiable components. The
    gradient penalty is the mean squared input-gradient norm. When ``alpha``
    is zero, interpolation inputs are unnecessary.
    """
    import torch

    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if raw_positive.numel() == 0 or raw_unlabeled.numel() == 0:
        raise ValueError("positive and unlabeled score batches must be non-empty")
    positive_score = torch.tanh(raw_positive)
    unlabeled_score = torch.tanh(raw_unlabeled)
    positive = (gradpu_positive_weight(raw_positive, beta) * (1.0 - positive_score)).mean()
    unlabeled = (1.0 + unlabeled_score).mean()
    penalty = torch.zeros((), dtype=positive.dtype, device=positive.device)
    if alpha > 0:
        if interpolated_inputs is None or raw_interpolated is None:
            raise ValueError("gradient penalty requires interpolated inputs and raw scores")
        gradient = torch.autograd.grad(
            raw_interpolated.sum(), interpolated_inputs, create_graph=True
        )[0]
        penalty = gradient.flatten(start_dim=1).square().sum(dim=1).mean()
    return positive + unlabeled + alpha * penalty, {
        "positive_loss": positive,
        "unlabeled_loss": unlabeled,
        "gradient_penalty": penalty,
    }


class GradPUClassifier(BasePUClassifier):
    """Paper-derived GradPU classifier for dense 2-D PU data.

    A user-supplied ``model`` must map one batch to one *raw* score per row.
    The fitted ``model_`` appends tanh so checkpoint snapshots and normal
    predictions share the paper's bounded score convention. The default is
    one-hidden-layer MLP128, a benchmark-adapted architecture rather than the
    paper's four-layer MLP300 or image CNN.

    ``class_prior`` is accepted by ``fit`` for the toolbox API but is unused:
    the paper objective treats U as negative without a prior correction.
    Non-None ``sample_weight`` is rejected rather than silently ignored.
    """

    family = AlgorithmFamily.DEEP_PU
    label_semantics = "pu"
    assumption = (Assumption.SCAR, Assumption.SAR)
    scenario = (Scenario.CASE_CONTROL, Scenario.SELECTION_BIASED)
    requires_class_prior = False
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.NOT_FOUND
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp"})
    input_ndims = frozenset({2})
    encoder_parameter = None
    trains_encoder = False

    def __init__(
        self,
        model=None,
        *,
        hidden_dim: int = 128,
        alpha: float = 0.1,
        beta_max: float = 1.0,
        batch_size: int = 256,
        max_epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta_max = beta_max
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
    ) -> GradPUClassifier:
        """Fit with independent P/U minibatches and linear beta annealing."""
        import torch

        if sample_weight is not None:
            raise NotImplementedError("GradPU does not implement sample_weight")
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="GradPUClassifier")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("X must contain finite numeric values")
        for name in ("hidden_dim", "batch_size", "max_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("alpha", "beta_max", "weight_decay"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if class_prior is not None and (not np.isfinite(class_prior) or not 0 < class_prior < 1):
            raise ValueError("class_prior, if supplied, must be in (0, 1)")

        positive = np.asarray(X[y_pu == 1], dtype=np.float32)
        unlabeled = np.asarray(X[y_pu == 0], dtype=np.float32)
        if len(unlabeled) == 0:
            raise ValueError("GradPU needs at least one unlabeled sample")
        if not np.isfinite(positive).all() or not np.isfinite(unlabeled).all():
            raise ValueError("X must remain finite after conversion to float32")
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        device = resolve_device(self.device)
        if self.model is None:
            raw_model = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, 1),
            )
        else:
            if not isinstance(self.model, torch.nn.Module):
                raise TypeError("model must be a torch.nn.Module")
            raw_model = copy.deepcopy(self.model)
        if any(
            isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
            for layer in raw_model.modules()
        ):
            raise ValueError("GradPU gradient penalty does not support BatchNorm layers")
        raw_model.to(device=device, dtype=torch.float32)
        if not any(parameter.requires_grad for parameter in raw_model.parameters()):
            raise ValueError("model must contain trainable parameters")
        with torch.no_grad():
            probe = raw_model(torch.as_tensor(positive[:1], device=device))
        if probe.shape not in ((1,), (1, 1)):
            raise ValueError("model must output one raw score per input row")

        self.raw_model_ = raw_model
        self.model_ = torch.nn.Sequential(raw_model, torch.nn.Tanh()).to(device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        p_data = torch.as_tensor(positive)
        u_data = torch.as_tensor(unlabeled)
        n_steps = max(
            (len(positive) + self.batch_size - 1) // self.batch_size,
            (len(unlabeled) + self.batch_size - 1) // self.batch_size,
        )
        self.history_ = {
            "epoch": [],
            "positive_loss": [],
            "unlabeled_loss": [],
            "gradient_penalty": [],
            "train_risk": [],
            "beta": [],
            "optimizer_steps": [],
        }
        self.optimizer_steps_ = 0
        self.n_positive_ = len(positive)
        self.n_unlabeled_ = len(unlabeled)
        self.n_features_in_ = X.shape[1]
        self._X_shape_ = X.shape
        self._class_prior = None  # accepted for API compatibility, never used by Eq. 7
        self._is_fitted = False

        for epoch in range(self.max_epochs):
            p_order = rng.permutation(len(positive))
            u_order = rng.permutation(len(unlabeled))
            p_batches = [
                p_order[i : i + self.batch_size] for i in range(0, len(p_order), self.batch_size)
            ]
            u_batches = [
                u_order[i : i + self.batch_size] for i in range(0, len(u_order), self.batch_size)
            ]
            sums = {
                key: 0.0
                for key in ("positive_loss", "unlabeled_loss", "gradient_penalty", "train_risk")
            }
            self.model_.train()
            for step in range(n_steps):
                p_index, u_index = (
                    p_batches[step % len(p_batches)],
                    u_batches[step % len(u_batches)],
                )
                p_batch = p_data[p_index].to(device)
                u_batch = u_data[u_index].to(device)
                beta = self.beta_max * (epoch * n_steps + step + 1) / (self.max_epochs * n_steps)
                raw_p = _one_score_per_row(raw_model(p_batch), len(p_batch))
                raw_u = _one_score_per_row(raw_model(u_batch), len(u_batch))
                if self.alpha > 0:
                    size = max(len(p_batch), len(u_batch))
                    p_repeat_index = torch.as_tensor(
                        rng.choice(len(p_batch), size=size, replace=size > len(p_batch)),
                        dtype=torch.long,
                        device=device,
                    )
                    u_repeat_index = torch.as_tensor(
                        rng.choice(len(u_batch), size=size, replace=size > len(u_batch)),
                        dtype=torch.long,
                        device=device,
                    )
                    p_repeated = p_batch[p_repeat_index]
                    u_repeated = u_batch[u_repeat_index]
                    mix_shape = (size,) + (1,) * (X.ndim - 1)
                    mixing = torch.as_tensor(
                        rng.uniform(size=mix_shape), dtype=torch.float32, device=device
                    )
                    mixed = (mixing * p_repeated + (1.0 - mixing) * u_repeated).detach()
                    mixed.requires_grad_(True)
                    raw_mixed = _one_score_per_row(raw_model(mixed), size)
                else:
                    mixed = raw_mixed = None
                loss, parts = gradpu_objective(
                    raw_p,
                    raw_u,
                    beta=beta,
                    alpha=self.alpha,
                    interpolated_inputs=mixed,
                    raw_interpolated=raw_mixed,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("GradPU objective became non-finite")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.optimizer_steps_ += 1
                for key in ("positive_loss", "unlabeled_loss", "gradient_penalty"):
                    sums[key] += float(parts[key].detach().cpu())
                sums["train_risk"] += float(loss.detach().cpu())
            self.history_["epoch"].append(epoch)
            for key, value in sums.items():
                self.history_[key].append(value / n_steps)
            self.history_["beta"].append(float(beta))
            self.history_["optimizer_steps"].append(self.optimizer_steps_)
            if epoch_callback is not None:
                epoch_callback(epoch, self)

        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("GradPU prediction X must have the fitted 2-D feature shape")
        if not np.isfinite(X).all():
            raise ValueError("GradPU prediction X must contain finite values")
        self.model_.eval()
        device = next(self.model_.parameters()).device
        with torch.no_grad():
            scores = self.model_(torch.as_tensor(X, dtype=torch.float32, device=device))
        return _one_score_per_row(scores, len(X)).cpu().numpy()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self._decision_function(X) >= 0.0).astype(int)


def _one_score_per_row(scores, batch_size: int):
    if scores.shape == (batch_size, 1):
        return scores[:, 0]
    if scores.shape == (batch_size,):
        return scores
    raise ValueError("model must output one raw score per input row")
