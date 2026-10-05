"""Robust-PU: nnPU warm-up followed by self-paced P/U correction.

Clean-room tabular adaptation of Zhu et al., arXiv:2308.00279. The
published image backbone and clean-label validation selection are not used.
"""

# ruff: noqa: N803, N806, N812

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
from ...core.training_views import build_training_view
from ...core.validation import validate_pu_X_y
from ...losses.nnpu import _nnpu_train_step


def self_paced_weights(losses, threshold: float, kind: str = "welsch"):
    """Return detached weights for the source's hard, linear or Welsch SPL rule."""
    import torch

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if kind == "hard":
        result = (losses < threshold).to(losses.dtype)
    elif kind == "linear":
        result = torch.clamp(1.0 - losses / threshold, min=0.0)
    elif kind == "welsch":
        result = torch.exp(-losses / (threshold * threshold))
    else:
        raise ValueError("spl_type must be hard, linear or welsch")
    return result.detach()


def _scores(model, X):
    scores = model(X)
    if scores.shape == (len(X), 1):
        return scores[:, 0]
    if scores.shape == (len(X),):
        return scores
    raise ValueError("model must output one raw score per input row")


def _episode_weights(logits, pu_labels, *, threshold_p, threshold_n, temper_p, temper_n, kind):
    import torch
    from torch.nn import functional as F

    positive_hardness = F.softplus(-logits / temper_p)
    negative_hardness = F.softplus(logits / temper_n)
    return torch.where(
        pu_labels == 1,
        self_paced_weights(positive_hardness, threshold_p, kind),
        self_paced_weights(negative_hardness, threshold_n, kind),
    )


class RobustPUClassifier(BasePUClassifier):
    """PU-only Robust-PU adaptation for dense tabular features.

    ``class_prior`` is the positive fraction in the unlabeled distribution,
    as in the source's case-control sampling protocol. It is used only by
    nnPU warm-up. Later episodes weight the BCE of P=1 and U=0 pseudo-labels.
    """

    family = AlgorithmFamily.DEEP_PU
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = True
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
        class_prior: float | None = None,
        *,
        model=None,
        hidden_dim: int = 100,
        pretrain_epochs: int = 10,
        episodes: int = 20,
        inner_epochs: int = 1,
        batch_size: int = 64,
        pretrain_lr: float = 1e-3,
        learning_rate: float = 1e-4,
        alpha_p: float = 0.1,
        alpha_n: float = 0.1,
        max_thresh_p: float = 2.0,
        max_thresh_n: float = 2.0,
        grow_steps: int = 10,
        spl_type: str = "welsch",
        temper_p: float = 1.0,
        temper_n: float = 1.0,
        phi: float = 0.0,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.model = model
        self.hidden_dim = hidden_dim
        self.pretrain_epochs = pretrain_epochs
        self.episodes = episodes
        self.inner_epochs = inner_epochs
        self.batch_size = batch_size
        self.pretrain_lr = pretrain_lr
        self.learning_rate = learning_rate
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n
        self.max_thresh_p = max_thresh_p
        self.max_thresh_n = max_thresh_n
        self.grow_steps = grow_steps
        self.spl_type = spl_type
        self.temper_p = temper_p
        self.temper_n = temper_n
        self.phi = phi
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
        os_or_ts: str = "os",
    ) -> RobustPUClassifier:
        import torch
        from torch.nn import functional as F

        if sample_weight is not None:
            raise NotImplementedError("Robust-PU does not implement sample_weight")
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="RobustPUClassifier")
        view = build_training_view(X, y_pu, requested_view=os_or_ts)
        prior = self.class_prior if class_prior is None else class_prior
        if prior is None or not np.isfinite(prior) or not 0 < prior < 1:
            raise ValueError("class_prior must be in (0, 1)")
        for name in (
            "hidden_dim",
            "pretrain_epochs",
            "episodes",
            "inner_epochs",
            "batch_size",
            "grow_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "pretrain_epochs" else 1):
                raise ValueError(f"{name} must be a valid non-negative/positive integer")
        if view.calibration_applied and self.pretrain_epochs == 0:
            raise ValueError("os_or_ts='ts' requires pretrain_epochs > 0 for nnPU calibration")
        for name in (
            "pretrain_lr",
            "learning_rate",
            "alpha_p",
            "alpha_n",
            "max_thresh_p",
            "max_thresh_n",
            "temper_p",
            "temper_n",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.phi) or not 0 <= self.phi < 1:
            raise ValueError("phi must be in [0, 1)")
        if self.spl_type not in {"hard", "linear", "welsch"}:
            raise ValueError("spl_type must be hard, linear or welsch")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("X must contain finite numeric values")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X must remain finite after float32 conversion")
        # Torch's NumPy index bridge expects writable arrays; the view owns
        # read-only positions, so make local index copies for later indexing.
        p_idx = np.array(view.positive_positions, copy=True)
        u_idx = np.array(view.native_unlabeled_positions, copy=True)
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        device = resolve_device(self.device)
        model = (
            torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, 1),
            )
            if self.model is None
            else copy.deepcopy(self.model)
        )
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        model.to(device=device, dtype=torch.float32)
        if not any(param.requires_grad for param in model.parameters()):
            raise ValueError("model must contain trainable parameters")
        with torch.no_grad():
            _scores(model, torch.as_tensor(X[:1], device=device))
        data = torch.as_tensor(X, device=device)
        labels = torch.as_tensor(y_pu.astype(np.float32), device=device)
        self.model_ = model
        self._class_prior = float(prior)
        self._X_shape_ = X.shape
        self.n_features_in_ = X.shape[1]
        self.n_positive_ = len(p_idx)
        self.n_unlabeled_ = len(u_idx)
        self.n_loss_unlabeled_ = len(view.loss_unlabeled_positions)
        self.training_view_ = os_or_ts
        self.calibration_applied_ = view.calibration_applied
        self.history_ = {
            "pretrain_risk": [],
            "episode_loss": [],
            "positive_weight": [],
            "unlabeled_weight": [],
            "threshold_p": [],
            "threshold_n": [],
        }
        self._is_fitted = False
        callback_epoch = 0
        pre_optimizer = torch.optim.Adam(model.parameters(), lr=self.pretrain_lr)
        for _ in range(self.pretrain_epochs):
            p_order, u_order = rng.permutation(p_idx), rng.permutation(u_idx)
            steps = max(
                (len(p_order) + self.batch_size - 1) // self.batch_size,
                (len(u_order) + self.batch_size - 1) // self.batch_size,
            )
            risks = []
            model.train()
            for step in range(steps):
                p = p_order[
                    (step * self.batch_size) % len(p_order) : (step * self.batch_size)
                    % len(p_order)
                    + self.batch_size
                ]
                u = u_order[
                    (step * self.batch_size) % len(u_order) : (step * self.batch_size)
                    % len(u_order)
                    + self.batch_size
                ]
                pos, unl = _scores(model, data[p]), _scores(model, data[u])
                # Only the nnPU marginal-risk role is calibrated.  Self-paced
                # pseudo-negative episodes below retain the original U rows.
                loss_unl = torch.cat((unl, pos)) if view.calibration_applied else unl
                loss, info = _nnpu_train_step(
                    torch.sigmoid(-pos).mean(),
                    torch.sigmoid(pos).mean(),
                    torch.sigmoid(loss_unl).mean(),
                    class_prior=prior,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Robust-PU pretrain loss became non-finite")
                pre_optimizer.zero_grad()
                loss.backward()
                pre_optimizer.step()
                risks.append(info["nnpu_risk"])
            self.history_["pretrain_risk"].append(float(np.mean(risks)))
            self._is_fitted = True
            if epoch_callback is not None:
                epoch_callback(callback_epoch, self)
            callback_epoch += 1

        moving = None
        for episode in range(self.episodes):
            threshold_p = self.alpha_p + (self.max_thresh_p - self.alpha_p) * min(
                episode / self.grow_steps, 1.0
            )
            threshold_n = self.alpha_n + (self.max_thresh_n - self.alpha_n) * min(
                episode / self.grow_steps, 1.0
            )
            model.eval()
            with torch.no_grad():
                raw = torch.cat([_scores(model, batch) for batch in data.split(self.batch_size)])
                weights = _episode_weights(
                    raw,
                    labels,
                    threshold_p=threshold_p,
                    threshold_n=threshold_n,
                    temper_p=self.temper_p,
                    temper_n=self.temper_n,
                    kind=self.spl_type,
                )
                moving = weights if moving is None else self.phi * moving + (1 - self.phi) * weights
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            losses = []
            model.train()
            for _ in range(self.inner_epochs):
                for indices in np.array_split(
                    rng.permutation(len(X)),
                    max(1, (len(X) + self.batch_size - 1) // self.batch_size),
                ):
                    logits = _scores(model, data[indices])
                    loss = (
                        F.binary_cross_entropy_with_logits(
                            logits, labels[indices], reduction="none"
                        )
                        * moving[indices]
                    ).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Robust-PU episode loss became non-finite")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
            self.history_["episode_loss"].append(float(np.mean(losses)))
            self.history_["positive_weight"].append(float(moving[p_idx].mean()))
            self.history_["unlabeled_weight"].append(float(moving[u_idx].mean()))
            self.history_["threshold_p"].append(float(threshold_p))
            self.history_["threshold_n"].append(float(threshold_n))
            self._is_fitted = True
            if epoch_callback is not None:
                epoch_callback(callback_epoch, self)
            callback_epoch += 1
        self.classes_ = np.array([0, 1])
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Robust-PU prediction X must have fitted 2-D feature shape")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("Robust-PU prediction X must be finite numeric")
        self.model_.eval()
        device = next(self.model_.parameters()).device
        with torch.no_grad():
            values = _scores(self.model_, torch.as_tensor(X, dtype=torch.float32, device=device))
        return values.cpu().numpy()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self._decision_function(X) >= 0).astype(int)
