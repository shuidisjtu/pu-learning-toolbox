# ruff: noqa: N803, N806
"""Prior-free variational PU learning for dense tabular feature vectors.

The objective follows Chen et al., NeurIPS 2020, Equations (6)--(9). The
default MLP is a toolbox-sized technical adapter, not the paper's seven-layer
tabular network or CIFAR CNN. PyTorch remains an optional fit-time dependency.
"""

from __future__ import annotations

import math

import numpy as np

from ...core.base import BasePUClassifier
from ...core.device import resolve_device
from ...core.labels import normalize_pu_labels
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


def vpu_objective(
    log_phi_positive,
    log_phi_marginal,
    log_phi_mixed,
    mixing,
    *,
    regularization_weight: float,
):
    """Return the paper's variational loss plus logarithmic MixUp penalty.

    ``log_phi_marginal`` is evaluated on the full P/U training pool, not only
    the rows marked unlabeled. The MixUp target retains its gradient, matching
    the locked official implementation rather than silently detaching it.
    """
    import torch

    variational = (
        torch.logsumexp(log_phi_marginal, dim=0)
        - math.log(log_phi_marginal.numel())
        - log_phi_positive.mean()
    )
    log_target = torch.logaddexp(torch.log(mixing), torch.log1p(-mixing) + log_phi_marginal)
    consistency = (log_target - log_phi_mixed).square().mean()
    return variational + regularization_weight * consistency, variational, consistency


class VPUClassifier(BasePUClassifier):
    """Fit a prior-free VPU classifier on dense two-dimensional PU data."""

    family = AlgorithmFamily.RISK_ESTIMATION
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = False
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
        *,
        hidden_dim: int = 64,
        depth: int = 2,
        max_epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 3e-4,
        regularization_weight: float = 0.03,
        mixup_alpha: float = 0.3,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization_weight = regularization_weight
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
        pu_validation_data=None,
        epoch_callback=None,
    ) -> VPUClassifier:
        """Minimize the variational objective using P and the full P/U pool.

        ``class_prior`` is accepted only for the common estimator interface;
        it never enters VPU's loss. ``pu_validation_data`` may be a PU-view
        DatasetPart or ``(X_val, y_pu_val)`` and is used only for epoch risk.
        """
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("VPUClassifier requires the optional 'torch' dependency") from exc

        if sample_weight is not None:
            raise NotImplementedError("VPU does not implement sample_weight")
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="VPUClassifier")
        X = _finite_features(X, name="X")
        if class_prior is not None and (not np.isfinite(class_prior) or not 0 < class_prior < 1):
            raise ValueError("class_prior, if supplied, must be in (0, 1)")
        self._validate_parameters()
        positive = X[y_pu == 1]
        if not np.any(y_pu == 0):
            raise ValueError("VPU needs at least one unlabeled sample")
        validation = _prepare_pu_validation(pu_validation_data, X.shape[1])

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        device = resolve_device(self.device)
        layers = []
        width = X.shape[1]
        for _ in range(self.depth):
            layers.extend((nn.Linear(width, self.hidden_dim), nn.ReLU()))
            width = self.hidden_dim
        layers.extend((nn.Linear(width, 1), nn.LogSigmoid()))
        calibration = nn.Linear(1, 1)
        with torch.no_grad():
            calibration.weight.fill_(1.0)
            calibration.bias.fill_(-math.log(0.5))
        calibration.requires_grad_(False)
        self.model_ = nn.Sequential(*layers, calibration).to(device)
        phi = self.model_[:-1]
        optimizer = torch.optim.Adam(phi.parameters(), lr=self.learning_rate, betas=(0.5, 0.99))
        x_data = torch.as_tensor(X, device=device)
        p_data = torch.as_tensor(positive, device=device)
        val_data = None
        if validation is not None:
            val_X, val_y = validation
            val_data = (
                torch.as_tensor(val_X, device=device),
                torch.as_tensor(val_X[val_y == 1], device=device),
            )
        steps = max(math.ceil(len(X) / self.batch_size), math.ceil(len(positive) / self.batch_size))
        batch_size = min(self.batch_size, len(X))
        self.history_ = {
            "epoch": [],
            "variational_loss": [],
            "regularization_loss": [],
            "train_risk": [],
            "val_risk": [],
            "optimizer_steps": [],
        }
        self.optimizer_steps_ = 0
        self.n_positive_ = len(positive)
        self.n_unlabeled_ = int(np.sum(y_pu == 0))
        self.n_features_in_ = X.shape[1]
        self._X_shape_ = X.shape
        self._class_prior = None
        self._is_fitted = False
        self.device_ = device

        for epoch in range(self.max_epochs):
            totals = np.zeros(3, dtype=np.float64)
            self.model_.train()
            for _ in range(steps):
                p_idx = rng.choice(len(positive), size=batch_size, replace=True)
                x_idx = rng.choice(len(X), size=batch_size, replace=True)
                p_batch = p_data[p_idx]
                x_batch = x_data[x_idx]
                mixing = torch.as_tensor(
                    np.clip(rng.beta(self.mixup_alpha, self.mixup_alpha), 1e-6, 1 - 1e-6),
                    dtype=torch.float32,
                    device=device,
                )
                mixed = mixing * p_batch + (1.0 - mixing) * x_batch
                log_p = phi(p_batch).flatten()
                log_x = phi(x_batch).flatten()
                log_mixed = phi(mixed).flatten()
                total, variational, consistency = vpu_objective(
                    log_p,
                    log_x,
                    log_mixed,
                    mixing,
                    regularization_weight=self.regularization_weight,
                )
                if not torch.isfinite(total):
                    raise FloatingPointError("VPU objective became non-finite")
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                self.optimizer_steps_ += 1
                totals += [
                    float(variational.detach().cpu()),
                    float(consistency.detach().cpu()),
                    float(total.detach().cpu()),
                ]

            self.model_.eval()
            with torch.no_grad():
                max_log_phi = max(
                    float(phi(x_data[start : start + self.batch_size]).max().cpu())
                    for start in range(0, len(X), self.batch_size)
                )
                calibration.bias.fill_(-max_log_phi - math.log(0.5))
                self.max_log_phi_ = max_log_phi
                if val_data is not None:
                    val_x, val_p = val_data
                    val_log_x = phi(val_x).flatten()
                    val_log_p = phi(val_p).flatten()
                    val_risk = float(
                        (
                            torch.logsumexp(val_log_x, dim=0)
                            - math.log(len(val_x))
                            - val_log_p.mean()
                        ).cpu()
                    )
                    self.history_["val_risk"].append(val_risk)
            self.history_["epoch"].append(epoch)
            for key, value in zip(
                ("variational_loss", "regularization_loss", "train_risk"),
                totals / steps,
                strict=True,
            ):
                self.history_[key].append(float(value))
            self.history_["optimizer_steps"].append(self.optimizer_steps_)
            self._is_fitted = True
            if epoch_callback is not None:
                epoch_callback(epoch, self)

        self.classes_ = np.array([0, 1])
        return self

    def _validate_parameters(self) -> None:
        for name in ("hidden_dim", "max_epochs", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.regularization_weight) or self.regularization_weight < 0:
            raise ValueError("regularization_weight must be finite and non-negative")
        if not np.isfinite(self.mixup_alpha) or self.mixup_alpha <= 0:
            raise ValueError("mixup_alpha must be finite and positive")

    def _decision_function(self, X) -> np.ndarray:
        import torch

        X = _finite_features(X, name="prediction X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("prediction X has the wrong feature dimension")
        self.model_.eval()
        with torch.no_grad():
            return (
                np.concatenate(
                    [
                        self.model_(
                            torch.as_tensor(X[start : start + self.batch_size], device=self.device_)
                        )
                        .flatten()
                        .cpu()
                        .numpy()
                        for start in range(0, len(X), self.batch_size)
                    ]
                )
                if len(X)
                else np.empty(0, dtype=np.float32)
            )

    def _predict(self, X) -> np.ndarray:
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        self._check_is_fitted()
        score = self._decision_function(X)
        positive = np.exp(np.clip(score + math.log(0.5), -80, 0))
        return np.column_stack((1.0 - positive, positive))


def _finite_features(X, *, name: str) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2 or not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"{name} must be a dense 2-D numeric array")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.asarray(X, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} overflows float32")
    return result


def _prepare_pu_validation(value, n_features: int):
    if value is None:
        return None
    if hasattr(value, "view"):
        if value.view != "pu":
            raise ValueError("pu_validation_data must carry a PU view, not clean labels")
        X_val, y_val = value.X, value.labels
    elif isinstance(value, tuple) and len(value) == 2:
        X_val, y_val = value
    else:
        raise ValueError("pu_validation_data must be a PU-view part or (X_val, y_pu_val)")
    X_val = _finite_features(X_val, name="pu_validation_data X")
    y_val = normalize_pu_labels(np.asarray(y_val))
    if len(X_val) != len(y_val) or X_val.shape[1] != n_features:
        raise ValueError("pu_validation_data shape does not match training X")
    if not np.any(y_val == 1) or not np.any(y_val == 0):
        raise ValueError("pu_validation_data needs positive and unlabeled rows")
    return X_val, y_val
