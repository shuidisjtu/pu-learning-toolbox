"""Clean-room neural LBE: observed-label likelihood and detached generalized EM.

Default schedule follows the separately identified user-supplied MLP variant,
not the older official archive. No upstream code, reader, labels or test-based
diagnostics are imported. The existing linear ``lbe`` remains unchanged.
"""

# ruff: noqa: N803, N806

from __future__ import annotations

import copy
from numbers import Real

import numpy as np
from scipy.special import expit

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
from ._validation import validate_encoder_features


def _vectors(*values):
    import torch

    if any(not torch.is_tensor(value) or value.ndim != 1 for value in values):
        raise ValueError("LBE objectives require one-dimensional tensors")
    if not values[0].numel() or any(value.shape != values[0].shape for value in values):
        raise ValueError("LBE objective tensors must be nonempty and matched")
    if any(value.device != values[0].device or not torch.isfinite(value).all() for value in values):
        raise ValueError("LBE objective tensors must be finite and on the same device")


def _log_joints(class_logits, propensity_logits, observed):
    import torch
    from torch.nn import functional

    _vectors(class_logits, propensity_logits, observed)
    if not torch.is_floating_point(class_logits) or not torch.is_floating_point(propensity_logits):
        raise ValueError("LBE logits must be floating point")
    if not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("LBE observed labels must be binary PU labels")
    positive = functional.logsigmoid(class_logits) + torch.where(
        observed == 1,
        functional.logsigmoid(propensity_logits),
        functional.logsigmoid(-propensity_logits),
    )
    negative = functional.logsigmoid(-class_logits)
    return positive, negative


def lbe_latent_positive_posterior(class_logits, propensity_logits, observed):
    """E-step in log space; labeled P stays one and the posterior is detached."""
    import torch

    positive, negative = _log_joints(class_logits, propensity_logits, observed)
    posterior = torch.exp(positive - torch.logaddexp(positive, negative))
    return torch.where(observed == 1, torch.ones_like(posterior), posterior).detach()


def lbe_neural_objective(class_logits, propensity_logits, observed, posterior):
    """Negative expected complete likelihood: soft class BCE + q-weighted label BCE."""
    import torch
    from torch.nn import functional

    _log_joints(class_logits, propensity_logits, observed)
    _vectors(class_logits, posterior)
    if torch.any((posterior < 0) | (posterior > 1)) or torch.any(posterior[observed == 1] != 1):
        raise ValueError("LBE posterior must lie in [0,1] and equal one for labeled P")
    q = posterior.detach().to(class_logits)
    labels = observed.detach().to(propensity_logits)
    classifier = functional.binary_cross_entropy_with_logits(class_logits, q)
    propensity = (
        q * functional.binary_cross_entropy_with_logits(propensity_logits, labels, reduction="none")
    ).mean()
    return classifier + propensity, {"classifier_loss": classifier, "propensity_loss": propensity}


def lbe_observed_log_likelihood(class_logits, propensity_logits, observed):
    """Unpenalized sum log P(S|X), evaluated without any hidden class labels."""
    import torch

    positive, negative = _log_joints(class_logits, propensity_logits, observed)
    return torch.where(observed == 1, positive, torch.logaddexp(positive, negative)).sum()


class NeuralLBEClassifier(BasePUClassifier):
    """Experimental two-network LBE, distinct from the linear ``LBEClassifier``.

    The default MLP100/tanh and 20+20 pretrain / 20 EM x 40 M updates follow
    the identified uploaded variant. Stable logits, CPU-resident data,
    gradient accumulation and injected CNNs are engineering adaptations, not
    a claim to reproduce either archive's published numerical results.
    """

    family = AlgorithmFamily.BIAS_AWARE
    label_semantics = "pu"
    assumption = (Assumption.SAR,)
    scenario = (Scenario.SINGLE_TRAINING_SET, Scenario.SELECTION_BIASED)
    requires_class_prior = False
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_RELATED
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
    checkpoint_stages = ("em",)

    def __init__(
        self,
        model=None,
        *,
        propensity_model=None,
        encoder=None,
        hidden_dim=100,
        pretrain_steps=20,
        n_em_iter=20,
        m_steps=40,
        sample_fraction=0.1,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-3,
        random_state=0,
        device=None,
    ):
        super().__init__()
        self.model = model
        self.propensity_model = propensity_model
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.pretrain_steps = pretrain_steps
        self.n_em_iter = n_em_iter
        self.m_steps = m_steps
        self.sample_fraction = sample_fraction
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    @property
    def checkpoint_epoch_count(self):
        """One classifier snapshot per EM round; M updates are not snapshot epochs."""
        return self.n_em_iter

    @property
    def checkpoint_prediction_batch_size(self):
        """Keep inference chunks identical when classifier snapshots are restored."""
        return self.batch_size

    def fit(
        self, X, y_pu, *, class_prior=None, sample_weight=None, epoch_callback=None, os_or_ts="os"
    ):
        """Fit from observed PU labels only; test and validation data are not inputs.

        class_prior, if supplied, is validated but not used by this source
        recipe. Each pretrain/M optimizer update accumulates row-weighted
        gradients over CPU-resident chunks; an EM posterior stays fixed across
        all M updates in that round. sample_weight and TS are refused.
        """
        import torch

        self._is_fitted = False
        if sample_weight is not None:
            raise NotImplementedError("Neural LBE does not implement sample_weight")
        if os_or_ts != "os":
            raise NotImplementedError(
                "Neural LBE is an OS joint-likelihood method; TS is not applicable"
            )
        for name in ("hidden_dim", "n_em_iter", "m_steps", "batch_size", "pretrain_steps"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < (0 if name == "pretrain_steps" else 1)
            ):
                raise ValueError(
                    f"{name} must be an integer in its allowed positive/nonnegative range"
                )
        for name, lower, upper in (
            ("learning_rate", 0, None),
            ("weight_decay", 0, None),
            ("sample_fraction", 0, 1),
            ("class_prior", 0, 1),
        ):
            value = class_prior if name == "class_prior" else getattr(self, name)
            if name == "class_prior" and value is None:
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite scalar")
            if (
                value < lower
                or (value == lower and name != "weight_decay")
                or (
                    upper is not None
                    and (value > upper or (name == "class_prior" and value == upper))
                )
            ):
                raise ValueError(f"{name} is outside its allowed range")
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, (int, np.integer))
            or not 0 <= self.random_state < 2**32
        ):
            raise ValueError("random_state must be None or an integer in [0, 2**32)")
        X, labels = validate_pu_X_y(
            X, y_pu, accept_sparse=False, allow_nd=True, estimator_name=type(self).__name__
        )
        if X.ndim not in {2, 4} or (X.ndim == 4 and self.encoder is None):
            raise ValueError(
                "Neural LBE requires 2-D features, or 4-D images with an explicit encoder"
            )
        if not np.any(labels == 0):
            raise ValueError("Neural LBE requires both observed P and U")
        for name in ("model", "propensity_model", "encoder"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, torch.nn.Module):
                raise TypeError(f"{name} must be a torch.nn.Module")
        with np.errstate(over="ignore", invalid="ignore"):
            data = np.asarray(X, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("X must remain finite after float32 conversion")
        self.input_shape_ = tuple(data.shape[1:])
        self.n_features_in_ = data.shape[1]
        self._X_shape_ = data.shape
        self.training_view_ = "os"
        self.calibration_applied_ = False
        self.recipe_variant_ = "uploaded_mlp_20261012_clean_room"
        rng = np.random.RandomState(self.random_state)
        device = resolve_device(self.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Neural LBE currently supports CPU or CUDA devices")
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            seed = int(rng.randint(0, 2**31))
            # torch.manual_seed also seeds unrelated CUDA devices. Seed only
            # the CPU generator and the selected device that fork_rng saves.
            torch.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            self._fit_models(
                torch.as_tensor(data),
                torch.as_tensor(labels, dtype=torch.float32),
                device,
                rng,
                epoch_callback,
            )
        self.classes_ = np.array([0, 1])
        self._is_fitted = True
        return self

    def _network(self, head, data, device):
        import torch

        width = data.shape[1]
        encoder = None
        if self.encoder is not None:
            encoder = copy.deepcopy(self.encoder).to(device=device, dtype=torch.float32)
            encoder.requires_grad_(True).eval()
            with torch.no_grad():
                width = validate_encoder_features(
                    encoder(data[:1].to(device)), encoder_param_name="encoder"
                )
        network = (
            copy.deepcopy(head)
            if head is not None
            else torch.nn.Sequential(
                torch.nn.Linear(width, self.hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(self.hidden_dim, 1),
            )
        )
        if encoder is not None:
            network = torch.nn.Sequential(encoder, network)
        network.to(device=device, dtype=torch.float32).eval()
        if not any(parameter.requires_grad for parameter in network.parameters()):
            raise ValueError("Neural LBE networks must have trainable parameters")
        with torch.no_grad():
            self._scores(network, data[:1].to(device))
        return network

    @staticmethod
    def _scores(network, batch):
        import torch

        scores = network(batch)
        if not torch.is_tensor(scores) or scores.shape not in {(len(batch),), (len(batch), 1)}:
            raise ValueError("Neural LBE network must emit one raw logit per row")
        scores = scores.reshape(-1)
        if not torch.is_floating_point(scores):
            raise ValueError("Neural LBE network logits must be floating point")
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Neural LBE network produced non-finite logits")
        return scores

    def _logits(self, network, data):
        import torch

        training = network.training
        network.eval()
        try:
            with torch.no_grad():
                return (
                    torch.cat(
                        [
                            self._scores(
                                network,
                                data[start : start + self.batch_size].to(
                                    next(network.parameters()).device
                                ),
                            ).cpu()
                            for start in range(0, len(data), self.batch_size)
                        ]
                    )
                    if len(data)
                    else torch.empty(0)
                )
        finally:
            network.train(training)

    def _update(self, optimizer, data, indices, loss_fn):
        import torch

        optimizer.zero_grad()
        device = next(self.model_.parameters()).device
        for start in range(0, len(indices), self.batch_size):
            index = indices[start : start + self.batch_size]
            loss = loss_fn(data[index].to(device), index)
            if not torch.isfinite(loss):
                raise FloatingPointError("Neural LBE objective became non-finite")
            (loss * (len(index) / len(indices))).backward()
        optimizer.step()
        self.optimizer_steps_ += 1

    def _fit_models(self, data, observed, device, rng, callback):
        import torch
        from torch.nn import functional

        self.model_ = self._network(self.model, data, device)
        self.propensity_model_ = self._network(self.propensity_model, data, device)
        self.model_.train()
        self.propensity_model_.train()
        self.optimizer_steps_ = 0
        all_rows = np.arange(len(data))
        frequency = float(observed.mean())
        classifier_optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        def pretrain_class(batch, index):
            labels = observed[index].to(device)
            weights = labels * (1 - frequency) + (1 - labels) * frequency
            return (
                weights
                * functional.binary_cross_entropy_with_logits(
                    self._scores(self.model_, batch), labels, reduction="none"
                )
            ).mean()

        for _ in range(self.pretrain_steps):
            self._update(classifier_optimizer, data, all_rows, pretrain_class)
        weights = observed + (1 - observed) * self._logits(self.model_, data).sigmoid()
        propensity_optimizer = torch.optim.Adam(
            self.propensity_model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def pretrain_propensity(batch, index):
            return (
                weights[index].to(device)
                * functional.binary_cross_entropy_with_logits(
                    self._scores(self.propensity_model_, batch),
                    observed[index].to(device),
                    reduction="none",
                )
            ).mean()

        for _ in range(self.pretrain_steps):
            self._update(propensity_optimizer, data, all_rows, pretrain_propensity)
        optimizer = torch.optim.Adam(
            [
                *self.model_.parameters(),
                *self.propensity_model_.parameters(),
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.history_ = {
            key: []
            for key in (
                "epoch",
                "train_risk",
                "classifier_loss",
                "propensity_loss",
                "observed_log_likelihood",
                "optimizer_steps",
                "sample_size",
            )
        }
        sample_size = max(1, int(len(data) * self.sample_fraction))
        for epoch in range(self.n_em_iter):
            q = lbe_latent_positive_posterior(
                self._logits(self.model_, data),
                self._logits(self.propensity_model_, data),
                observed,
            )

            def em_loss(batch, index, posterior=q):
                return lbe_neural_objective(
                    self._scores(self.model_, batch),
                    self._scores(self.propensity_model_, batch),
                    observed[index].to(device),
                    posterior[index].to(device),
                )[0]

            self.model_.train()
            self.propensity_model_.train()
            for _ in range(self.m_steps):
                self._update(optimizer, data, rng.permutation(len(data))[:sample_size], em_loss)
            logits, propensity = (
                self._logits(self.model_, data),
                self._logits(self.propensity_model_, data),
            )
            risk, parts = lbe_neural_objective(logits, propensity, observed, q)
            likelihood = lbe_observed_log_likelihood(logits, propensity, observed)
            self.history_["epoch"].append(epoch)
            self.history_["train_risk"].append(float(risk))
            for key, value in parts.items():
                self.history_[key].append(float(value))
            self.history_["observed_log_likelihood"].append(float(likelihood))
            self.history_["optimizer_steps"].append(self.optimizer_steps_)
            self.history_["sample_size"].append(sample_size)
            self.n_iter_ = epoch + 1
            self.checkpoint_stage_ = "em"
            self.checkpoint_stage_epoch_ = epoch + 1
            self.checkpoint_round_ = epoch + 1
            if callback is not None:
                callback(epoch, self)
        logits, propensity = (
            self._logits(self.model_, data),
            self._logits(self.propensity_model_, data),
        )
        self._latent_positive_probability_ = lbe_latent_positive_posterior(
            logits, propensity, observed
        ).numpy()
        self._class_prior = float(logits.sigmoid().mean())
        self.observed_log_likelihood_history_ = list(self.history_["observed_log_likelihood"])
        self.pretraining_optimizer_steps_ = 2 * self.pretrain_steps
        self.em_optimizer_steps_ = self.n_em_iter * self.m_steps
        self.model_.eval()
        self.propensity_model_.eval()

    def _prediction_logits(self, X, network):
        import torch

        data = np.asarray(X)
        if data.ndim != len(self.input_shape_) + 1 or tuple(data.shape[1:]) != self.input_shape_:
            raise ValueError("Neural LBE prediction shape must match the fitted input shape")
        with np.errstate(over="ignore", invalid="ignore"):
            data = np.asarray(data, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("Neural LBE prediction input must be finite float32")
        return self._logits(network, torch.as_tensor(data)).numpy()

    def _decision_function(self, X):
        return self._prediction_logits(X, self.model_)

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        """Class posterior; not the probability of observing a positive label."""
        self._check_is_fitted()
        positive = expit(self._decision_function(X))
        return np.column_stack((1 - positive, positive))

    def predict_label_proba(self, X):
        """Return P(S=1|X) as class posterior times positive-label propensity."""
        self._check_is_fitted()
        return expit(self._decision_function(X)) * expit(
            self._prediction_logits(X, self.propensity_model_)
        )
