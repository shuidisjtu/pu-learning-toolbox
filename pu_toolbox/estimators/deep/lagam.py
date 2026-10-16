"""LaGAM feature/image adaptation with an explicit clean support boundary.

The source method requires clean labeled support examples for a second-order
meta-label update. This estimator fails closed when none are supplied; it is
not a PA-eligible PU-only learner.
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
from ...core.validation import validate_pu_X_y


def _network(n_features, hidden_dim):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


def _score(model, X):
    import torch

    values = model(X)
    if values.shape != (len(X), 1):
        raise ValueError("LaGAM model must output one raw logit per row")
    if not torch.isfinite(values).all():
        raise FloatingPointError("LaGAM model produced non-finite logits")
    return values[:, 0]


def _balanced_soft_bce(logits, targets):
    """Stable balanced BCE adaptation, not the source's probability clipping."""
    from torch.nn import functional as F

    positives = targets.sum().clamp_min(1.0)
    negatives = (1 - targets).sum().clamp_min(1.0)
    return (-targets * F.logsigmoid(logits)).sum() / positives + (
        -(1 - targets) * F.logsigmoid(-logits)
    ).sum() / negatives


def lagam_meta_labels(
    model, features, observed, support_features, support_labels, *, meta_lr, update_bias=True
):
    """One-step classifier-only meta-gradient label disambiguation.

    The sign of the clean support loss gradient w.r.t. a training label
    perturbation chooses 0/1. Observed positives are forced to remain 1.
    The encoder is held fixed in this virtual step, as the source updates
    the named classifier layer only. update_bias=False matches its weight-only
    virtual parameter scope; the default retains the existing toolbox variant.
    """
    import torch
    from torch.nn import functional as F

    if type(update_bias) is not bool:
        raise ValueError("update_bias must be a boolean")
    epsilon = torch.zeros_like(observed, requires_grad=True)
    weight, bias = model[-1].weight, model[-1].bias
    logits = F.linear(features.detach(), weight, bias).reshape(-1)
    labels = observed + epsilon
    virtual_loss = _balanced_soft_bce(logits, labels)
    parameters = (weight, bias) if update_bias else (weight,)
    gradients = torch.autograd.grad(virtual_loss, parameters, create_graph=True)
    grad_weight = gradients[0]
    updated_weight = weight - meta_lr * grad_weight
    updated_bias = bias - meta_lr * gradients[1] if update_bias else bias
    support_logits = F.linear(support_features.detach(), updated_weight, updated_bias).reshape(-1)
    support_loss = _balanced_soft_bce(support_logits, support_labels)
    gradient = torch.autograd.grad(support_loss, epsilon)[0]
    detected = (gradient < 0).float()
    return torch.where(observed == 1, torch.ones_like(detected), detected).detach()


def lagam_contrastive_loss(q, k, groups=None, *, temperature=0.07):
    """Instance InfoNCE plus same-latent-group contrastive objective."""
    import torch
    from torch.nn import functional as F

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    n = len(q)
    logits = q @ torch.cat([q, k], dim=0).T / temperature
    self_mask = torch.arange(n, device=q.device)
    logits[self_mask, self_mask] = -torch.inf
    log_prob = F.log_softmax(logits, dim=1)
    instance = -log_prob[self_mask, n + self_mask].mean()
    if groups is None:
        return instance
    if groups.shape != (n,):
        raise ValueError("groups must contain one cluster per row")
    group_mask = (groups[:, None] == torch.cat([groups, groups])[None, :]).float()
    group_mask[self_mask, self_mask] = 0
    group_terms = torch.where(group_mask.bool(), log_prob, torch.zeros_like(log_prob))
    group_loss = -group_terms.sum(dim=1) / group_mask.sum(dim=1).clamp_min(1)
    return instance + group_loss.mean()


class LaGAMClassifier(BasePUClassifier):
    """Latent group-aware meta disambiguation, requiring clean support data."""

    family = AlgorithmFamily.DEEP_PU
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    # Pinned author experiments remove labeled P from U (single training set).
    # This describes the released sampling recipe, not theoretical exclusivity.
    scenario = (Scenario.SINGLE_TRAINING_SET,)
    requires_class_prior = False
    requires_clean_support = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
    checkpoint_stages = ("warmup", "meta")

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        warmup_epochs: int = 2,
        max_epochs: int = 20,
        batch_size: int = 64,
        support_batch_size: int = 32,
        num_clusters: int = 5,
        learning_rate: float = 1e-3,
        meta_lr: float = 1e-3,
        mix_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        temperature: float = 0.07,
        noise_std: float = 0.05,
        rho_start: float = 0.95,
        rho_end: float = 0.8,
        random_state: int | None = None,
        device: str | None = None,
        encoder=None,
        data_device_policy: str = "auto",
        meta_update_bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.support_batch_size = support_batch_size
        self.num_clusters = num_clusters
        self.learning_rate = learning_rate
        self.meta_lr = meta_lr
        self.mix_weight = mix_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.noise_std = noise_std
        self.rho_start = rho_start
        self.rho_end = rho_end
        self.random_state = random_state
        self.device = device
        self.encoder = encoder
        self.data_device_policy = data_device_policy
        self.meta_update_bias = meta_update_bias

    @property
    def checkpoint_epoch_count(self):
        """Warmup is included in max_epochs, never added a second time."""
        return self.max_epochs

    @property
    def checkpoint_prediction_batch_size(self):
        """Streamed inference snapshots retain the estimator's chunk size."""
        return (
            self.batch_size
            if self.encoder is not None or self.data_device_policy == "cpu_stream"
            else 256
        )

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
        support_data: tuple[np.ndarray, np.ndarray] | None = None,
        epoch_callback=None,
    ) -> LaGAMClassifier:
        """Train with explicitly supplied independent clean support, never clean_val.

        Images require an encoder and CPU-streamed inputs. Auto preserves the
        old resident 2-D variant. Array-alias rejection catches only physical
        overlap: the caller must still prove role/index independence.
        """
        self._is_fitted = False
        try:
            return self._fit_impl(
                X,
                y_pu,
                class_prior=class_prior,
                sample_weight=sample_weight,
                support_data=support_data,
                epoch_callback=epoch_callback,
            )
        except Exception:
            self._is_fitted = False
            raise

    def _fit_impl(self, X, y_pu, *, class_prior, sample_weight, support_data, epoch_callback):
        import torch
        from sklearn.cluster import KMeans

        if sample_weight is not None:
            raise NotImplementedError("LaGAM does not implement sample_weight")
        if support_data is None:
            raise ValueError(
                "LaGAM requires explicit clean support_data=(X_support, y_clean); PA-ineligible"
            )
        if self.encoder is not None and not isinstance(self.encoder, torch.nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        if not isinstance(self.data_device_policy, str) or self.data_device_policy not in {
            "auto",
            "resident",
            "cpu_stream",
        }:
            raise ValueError("data_device_policy must be auto, resident or cpu_stream")
        if self.encoder is not None and self.data_device_policy == "resident":
            raise ValueError("injected encoders require CPU-streamed data, not resident")
        if type(self.meta_update_bias) is not bool:
            raise ValueError("meta_update_bias must be a boolean")
        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            allow_nd=self.encoder is not None,
            estimator_name="LaGAMClassifier",
        )
        if not np.any(y_pu == 0):
            raise ValueError("LaGAM needs unlabeled samples")
        _validate_input(X, allow_nd=self.encoder is not None)
        if not isinstance(support_data, tuple) or len(support_data) != 2:
            raise ValueError("support_data must be (X_support, y_clean)")
        support_X, support_y = (np.asarray(part) for part in support_data)
        if support_X.shape[1:] != X.shape[1:] or support_X.ndim != X.ndim or not len(support_X):
            raise ValueError("support_data feature shape must match X")
        _validate_input(support_X, allow_nd=self.encoder is not None)
        if np.shares_memory(X, support_X):
            raise ValueError("support_data must be independent: features share training memory")
        if support_y.shape != (len(support_X),) or set(np.unique(support_y)) != {0, 1}:
            raise ValueError("support_data labels must contain both clean classes {0, 1}")
        if class_prior is not None and (not np.isfinite(class_prior) or not 0 < class_prior < 1):
            raise ValueError("class_prior, if supplied, must be in (0, 1)")
        for name in (
            "hidden_dim",
            "max_epochs",
            "batch_size",
            "support_batch_size",
            "num_clusters",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_epochs) is not int or not 0 <= self.warmup_epochs < self.max_epochs:
            raise ValueError("warmup_epochs must be in [0, max_epochs)")
        if self.num_clusters > len(X):
            raise ValueError("num_clusters cannot exceed training sample count")
        for name in ("learning_rate", "meta_lr", "temperature"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("mix_weight", "contrastive_weight", "noise_std"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 <= self.rho_start < 1 or not 0 <= self.rho_end < 1:
            raise ValueError("rho_start and rho_end must be in [0, 1)")
        train_X = np.asarray(X, dtype=np.float32)
        support_X = np.asarray(support_X, dtype=np.float32)
        if not np.isfinite(train_X).all() or not np.isfinite(support_X).all():
            raise ValueError("features must remain finite after float32 conversion")
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        device = resolve_device(self.device)
        stream = self.encoder is not None or self.data_device_policy == "cpu_stream"
        self.data_device_policy_ = "cpu_stream" if stream else "resident"
        data_device = torch.device("cpu") if stream else device
        self.training_data_device_ = str(data_device)
        data = torch.as_tensor(train_X, device=data_device)
        labels = torch.as_tensor(y_pu.astype(np.float32), device=data_device)
        support_features = torch.as_tensor(support_X, device=data_device)
        support_labels = torch.as_tensor(support_y.astype(np.float32), device=data_device)
        width = X.shape[1]
        self.encoder_ = None
        if self.encoder is not None:
            from ._validation import validate_encoder_features

            self.encoder_ = copy.deepcopy(self.encoder).to(device=device, dtype=torch.float32)
            self.encoder_.requires_grad_(True).eval()
            with torch.no_grad():
                features = self.encoder_(torch.as_tensor(train_X[:1], device=device))
                width = validate_encoder_features(features, encoder_param_name="encoder")
                if features.shape[0] != 1:
                    raise ValueError("encoder must preserve the batch dimension")
        model = _network(width, self.hidden_dim)
        if self.encoder_ is not None:
            model = torch.nn.Sequential(self.encoder_, *model.children())
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.model_ = model
        self.n_features_in_ = X.shape[1]
        self.input_shape_ = tuple(X.shape[1:])
        self._X_shape_ = X.shape
        self._class_prior = None  # LaGAM does not use a class prior
        self.n_support_ = len(support_X)
        self.pseudo_labels_ = labels.detach().clone()
        self.history_ = {"train_loss": [], "contrastive_loss": [], "meta_positive_fraction": []}
        self.optimizer_steps_ = 0
        self.virtual_steps_ = 0
        self.clustering_passes_ = 0
        self.epoch_provenance_ = []
        self._is_fitted = False

        for epoch in range(self.max_epochs):
            groups = None
            if epoch >= self.warmup_epochs and self.contrastive_weight > 0:
                model.eval()
                with torch.no_grad():
                    if stream:
                        embeddings = np.concatenate(
                            [
                                _features(model, data[start : start + self.batch_size].to(device))
                                .cpu()
                                .numpy()
                                for start in range(0, len(data), self.batch_size)
                            ]
                        )
                    else:
                        embeddings = _features(model, data).cpu().numpy()
                cluster = KMeans(
                    n_clusters=self.num_clusters, random_state=self.random_state, n_init=10
                )
                groups = torch.as_tensor(cluster.fit_predict(embeddings), device=data_device)
                self.clustering_passes_ += 1
            model.train()
            order = rng.permutation(len(X))
            epoch_losses, contrastive_losses = [], []
            for offset in range(0, len(X), self.batch_size):
                indices = order[offset : offset + self.batch_size]
                batch = data[indices].to(device)
                observed = labels[indices].to(device)
                if epoch < self.warmup_epochs:
                    targets = observed
                else:
                    chosen = rng.choice(
                        len(support_X),
                        size=min(self.support_batch_size, len(support_X)),
                        replace=False,
                    )
                    support_batch = support_features[chosen].to(device)
                    if self.encoder_ is None:
                        batch_features = _features(model, batch)
                        support_batch_features = _features(model, support_batch)
                    else:
                        # Virtual head-only step must not update the real CNN's
                        # BN buffers from support images or draw dropout masks.
                        batch_features, support_batch_features = _virtual_features(
                            model, batch, support_batch
                        )
                    detected = lagam_meta_labels(
                        model,
                        batch_features,
                        observed,
                        support_batch_features,
                        support_labels[chosen].to(device),
                        meta_lr=self.meta_lr,
                        update_bias=self.meta_update_bias,
                    )
                    self.virtual_steps_ += 1
                    rho = self.rho_start + (self.rho_end - self.rho_start) * epoch / max(
                        self.max_epochs - 1, 1
                    )
                    with torch.no_grad():
                        self.pseudo_labels_[indices] = rho * self.pseudo_labels_[indices] + (
                            1 - rho
                        ) * detected.to(data_device)
                        self.pseudo_labels_[indices] = torch.where(
                            labels[indices] == 1,
                            torch.ones_like(labels[indices]),
                            self.pseudo_labels_[indices],
                        )
                    targets = self.pseudo_labels_[indices].detach().to(device)
                logits = _score(model, batch)
                classification = _balanced_soft_bce(logits, targets)
                mix_loss = torch.zeros((), device=device)
                if self.mix_weight > 0 and len(batch) > 1:
                    raw_ratio = float(rng.beta(4, 4))
                    ratio = max(raw_ratio, 1 - raw_ratio)
                    permutation = torch.as_tensor(rng.permutation(len(batch)), device=device)
                    mixed_X = ratio * batch + (1 - ratio) * batch[permutation]
                    mixed_y = ratio * targets + (1 - ratio) * targets[permutation]
                    mix_loss = _balanced_soft_bce(_score(model, mixed_X), mixed_y)
                contrastive = torch.zeros((), device=device)
                if self.contrastive_weight > 0 and len(batch) > 1:
                    weak = batch + torch.randn_like(batch) * self.noise_std
                    strong = batch + torch.randn_like(batch) * (2 * self.noise_std)
                    q, k = _features(model, weak), _features(model, strong)
                    batch_groups = groups[indices].to(device) if groups is not None else None
                    contrastive = lagam_contrastive_loss(
                        q, k, batch_groups, temperature=self.temperature
                    )
                loss = (
                    classification
                    + self.mix_weight * mix_loss
                    + self.contrastive_weight * contrastive
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("LaGAM objective became non-finite")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.optimizer_steps_ += 1
                epoch_losses.append(float(loss.detach()))
                contrastive_losses.append(float(contrastive.detach()))
            self.history_["train_loss"].append(float(np.mean(epoch_losses)))
            self.history_["contrastive_loss"].append(float(np.mean(contrastive_losses)))
            self.history_["meta_positive_fraction"].append(
                float(self.pseudo_labels_[labels == 0].mean().detach())
            )
            stage = "warmup" if epoch < self.warmup_epochs else "meta"
            self.checkpoint_stage_ = stage
            self.checkpoint_stage_epoch_ = (
                epoch + 1 if stage == "warmup" else epoch - self.warmup_epochs + 1
            )
            self.checkpoint_round_ = None
            self.epoch_provenance_.append(
                dict(
                    stage=stage,
                    stage_epoch=self.checkpoint_stage_epoch_,
                    round_index=None,
                    optimizer_steps=self.optimizer_steps_,
                )
            )
            self._is_fitted = True
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        self.classes_ = np.array([0, 1])
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X)
        _validate_input(X, allow_nd=self.encoder is not None)
        if tuple(X.shape[1:]) != self.input_shape_:
            raise ValueError("LaGAM prediction requires fitted feature/image shape")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("LaGAM prediction features overflow float32")
        self.model_.eval()
        device = next(self.model_.parameters()).device
        with torch.no_grad():
            if not len(X):
                return np.empty(0, dtype=np.float32)
            if self.data_device_policy_ == "cpu_stream":
                return np.concatenate(
                    [
                        _score(
                            self.model_,
                            torch.as_tensor(X[start : start + self.batch_size], device=device),
                        )
                        .cpu()
                        .numpy()
                        for start in range(0, len(X), self.batch_size)
                    ]
                )
            return (
                _score(self.model_, torch.as_tensor(X, dtype=torch.float32, device=device))
                .cpu()
                .numpy()
            )

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self._decision_function(X) >= 0).astype(int)


def _validate_input(X, *, allow_nd):
    if X.ndim not in ({2, 4} if allow_nd else {2}) or any(size == 0 for size in X.shape[1:]):
        raise ValueError("LaGAM requires 2-D features or 4-D images with an encoder")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError("LaGAM features must be finite numeric")


def _features(model, batch):
    from ._validation import validate_encoder_features

    features = model[:-1](batch)
    validate_encoder_features(features, encoder_param_name="classifier features")
    if features.shape[0] != len(batch):
        raise ValueError("LaGAM features must preserve the batch dimension")
    return features


def _virtual_features(model, batch, support_batch):
    """Frozen eval features for the new CNN virtual step, restoring all modes."""
    import torch

    modules = list(model[:-1].modules())
    modes = [module.training for module in modules]
    try:
        for module in modules:
            module.training = False
        with torch.no_grad():
            return _features(model, batch), _features(model, support_batch)
    finally:
        for module, mode in zip(modules, modes, strict=True):
            module.training = mode
