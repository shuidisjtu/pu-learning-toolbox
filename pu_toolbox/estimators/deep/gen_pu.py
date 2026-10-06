# ruff: noqa: N803, N806
"""Paper-objective MLP/pixel-generative CNN GenPU, IJCAI 2018 equations (3)--(11).

The author demo uses different unweighted D_u / non-saturating G losses.
This explicit paper-objective adapter is not an author-demo replay.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

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
from ._validation import validate_encoder_features


def genpu_discriminator_losses(
    dp_real,
    dp_fake,
    dn_real,
    dn_fake,
    du_real,
    du_p,
    du_n,
    *,
    class_prior,
    positive_weight,
    negative_weight,
    unlabeled_weight,
):
    """D maximizes each GAN value; D_n also maximizes, not minimizes."""
    from torch.nn import functional

    prior = class_prior
    return (
        -prior
        * positive_weight
        * (functional.logsigmoid(dp_real).mean() + functional.logsigmoid(-dp_fake).mean()),
        -(1 - prior)
        * negative_weight
        * (functional.logsigmoid(dn_real).mean() + functional.logsigmoid(-dn_fake).mean()),
        -unlabeled_weight
        * (
            functional.logsigmoid(du_real).mean()
            + prior * functional.logsigmoid(-du_p).mean()
            + (1 - prior) * functional.logsigmoid(-du_n).mean()
        ),
    )


def genpu_generator_losses(
    dp_fake, du_p, dn_fake, du_n, *, class_prior, positive_weight, negative_weight, unlabeled_weight
):
    """Minimax G_p; G_n minimizes U value but maximizes N value (anti-GAN)."""
    from torch.nn import functional

    return (
        class_prior
        * (
            positive_weight * functional.logsigmoid(-dp_fake).mean()
            + unlabeled_weight * functional.logsigmoid(-du_p).mean()
        ),
        (1 - class_prior)
        * (
            unlabeled_weight * functional.logsigmoid(-du_n).mean()
            - negative_weight * functional.logsigmoid(-dn_fake).mean()
        ),
    )


class GenPUClassifier(BasePUClassifier):
    """Two generators, three discriminators, then a synthetic PN classifier.

    Finite Adam training does not certify GAN convergence or class recovery.
    Dense unbounded outputs suit tabular features, unlike the demo's tanh pixels.
    No clean labels or validation/test data are consumed inside fit.
    """

    family = AlgorithmFamily.DEEP_PU
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_RELATED
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
    checkpoint_stages = ("synthetic_pn",)

    @property
    def checkpoint_prediction_batch_size(self):
        """Preserve inference chunks; changing them can change float32 scores."""
        return self.batch_size

    def __init__(
        self,
        *,
        class_prior=None,
        hidden_dim=128,
        latent_dim=100,
        max_epochs=100,
        classifier_epochs=100,
        batch_size=64,
        learning_rate=3e-4,
        positive_weight=1.0,
        negative_weight=1.0,
        unlabeled_weight=1.0,
        encoder=None,
        generator_output="identity",
        random_state=0,
        device=None,
    ):
        super().__init__()
        self.class_prior = class_prior
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_epochs = max_epochs
        self.classifier_epochs = classifier_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.unlabeled_weight = unlabeled_weight
        self.encoder = encoder
        self.generator_output = generator_output
        self.random_state = random_state
        self.device = device

    @property
    def checkpoint_epoch_count(self):
        """Only synthetic-PN classifier epochs have deployable classifier weights."""
        return self.classifier_epochs

    def fit(
        self, X, y_pu, *, class_prior=None, sample_weight=None, os_or_ts="os", epoch_callback=None
    ):
        import torch
        from torch import nn
        from torch.nn import functional

        self._is_fitted = False
        if sample_weight is not None:
            raise NotImplementedError("GenPU does not implement sample_weight")
        prior = self.class_prior if class_prior is None else class_prior
        if prior is None or not np.isfinite(prior) or not 0 < prior < 1:
            raise ValueError("GenPU requires population class_prior in (0, 1)")
        if (
            self.class_prior is not None
            and class_prior is not None
            and self.class_prior != class_prior
        ):
            raise ValueError("fit class_prior differs from constructor class_prior")
        for name in ("hidden_dim", "latent_dim", "max_epochs", "classifier_epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "positive_weight", "negative_weight", "unlabeled_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.generator_output, str) or self.generator_output not in {
            "identity",
            "tanh",
        }:
            raise ValueError("generator_output must be identity or tanh")
        X, y_pu = validate_pu_X_y(
            X, y_pu, accept_sparse=False, allow_nd=True, estimator_name="GenPUClassifier"
        )
        if X.ndim not in (2, 4):
            raise ValueError("GenPU supports 2-D features or 4-D NCHW images")
        if X.ndim == 4 and self.encoder is None:
            raise ValueError("GenPU 4-D images require an explicit encoder; no input flattening")
        if self.encoder is not None and not isinstance(self.encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X must remain finite after float32 conversion")
        if self.generator_output == "tanh" and np.any((X < -1) | (X > 1)):
            raise ValueError("tanh generator_output requires X already scaled into [-1, 1]")
        view = build_training_view(X, y_pu, requested_view=os_or_ts)
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        self.device_ = resolve_device(self.device)
        width = X.shape[1]
        self.input_shape_ = tuple(X.shape[1:])

        def mlp(input_dim, output_dim, depth):
            layers = []
            for _ in range(depth):
                layers.extend([nn.Linear(input_dim, self.hidden_dim), nn.LeakyReLU(0.2)])
                input_dim = self.hidden_dim
            return nn.Sequential(*layers, nn.Linear(input_dim, output_dim)).to(self.device_)

        def generator():
            dense = mlp(self.latent_dim, int(np.prod(self.input_shape_)), 2)
            if self.generator_output == "tanh":
                dense.append(nn.Tanh())
            if X.ndim == 4:
                # The paper's image generators are dense pixel networks. Shape
                # their OUTPUT, never flatten real images into the old MLP path.
                dense.append(nn.Unflatten(1, self.input_shape_))
            return dense

        def score_network(depth):
            if self.encoder is None:
                return mlp(width, 1, depth), None
            own_encoder = copy.deepcopy(self.encoder).to(self.device_, dtype=torch.float32)
            own_encoder.requires_grad_(True).eval()
            with torch.no_grad():
                features = own_encoder(torch.as_tensor(X[:1], device=self.device_))
            feature_width = validate_encoder_features(features, encoder_param_name="encoder")
            if features.shape[0] != 1:
                raise ValueError("encoder must preserve the input batch dimension")
            network = nn.Sequential(own_encoder, mlp(feature_width, 1, depth)).to(self.device_)
            network.train()
            return network, own_encoder

        self.positive_generator_ = generator()
        self.negative_generator_ = generator()
        self.positive_discriminator_, self.positive_discriminator_encoder_ = score_network(0)
        self.negative_discriminator_, self.negative_discriminator_encoder_ = score_network(0)
        self.unlabeled_discriminator_, self.unlabeled_discriminator_encoder_ = score_network(2)
        discriminators = [
            self.positive_discriminator_,
            self.negative_discriminator_,
            self.unlabeled_discriminator_,
        ]
        generators = [self.positive_generator_, self.negative_generator_]
        optimizers = [
            torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            for net in discriminators + generators
        ]
        weights = dict(
            class_prior=float(prior),
            positive_weight=self.positive_weight,
            negative_weight=self.negative_weight,
            unlabeled_weight=self.unlabeled_weight,
        )
        data = torch.as_tensor(X)
        p_indices = view.positive_positions.copy()
        u_indices = view.loss_unlabeled_positions.copy()
        steps = (len(u_indices) + self.batch_size - 1) // self.batch_size
        self.optimizer_steps_ = 0
        stages = (
            "positive_discriminator",
            "negative_discriminator",
            "unlabeled_discriminator",
            "positive_generator",
            "negative_generator",
            "synthetic_pn",
        )
        self.stage_optimizer_steps_ = dict.fromkeys(stages, 0)
        self.history_ = {"gan_discriminator_loss": [], "gan_generator_loss": [], "pn_loss": []}
        self.history_["optimizer_steps"] = []
        self.n_positive_ = len(p_indices)
        self.n_unlabeled_ = len(view.native_unlabeled_positions)
        self.n_loss_unlabeled_ = len(u_indices)
        self.training_view_ = os_or_ts
        self.calibration_applied_ = view.calibration_applied
        self._class_prior = float(prior)
        self.classes_ = np.array([0, 1])
        self.n_features_in_, self._X_shape_ = width, X.shape
        self.objective_variant_ = "paper_minimax_prior_weighted"
        self.generator_variant_ = (
            f"dense_pixel_{self.generator_output}_NCHW"
            if X.ndim == 4
            else f"dense_feature_{self.generator_output}"
        )
        self.generator_output_ = self.generator_output

        def latent():
            # NumPy RNG is local and stable across CPU/device transfers.
            return torch.as_tensor(
                rng.normal(size=(self.batch_size, self.latent_dim)),
                dtype=torch.float32,
                device=self.device_,
            )

        def update(loss, optimizer, stage):
            if not torch.isfinite(loss):
                raise ValueError("GenPU loss became non-finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.optimizer_steps_ += 1
            self.stage_optimizer_steps_[stage] += 1

        dp, dn, du = discriminators
        gp, gn = generators
        for _ in range(self.max_epochs):
            d_values, g_values = [], []
            for _ in range(steps):
                p = data[rng.choice(p_indices, self.batch_size)].to(self.device_)
                u = data[rng.choice(u_indices, self.batch_size)].to(self.device_)
                with torch.no_grad():
                    fake_p, fake_n = self._generate(gp, latent()), self._generate(gn, latent())
                d_losses = genpu_discriminator_losses(
                    self._logits(dp, p),
                    self._logits(dp, fake_p),
                    self._logits(dn, p),
                    self._logits(dn, fake_n),
                    self._logits(du, u),
                    self._logits(du, fake_p),
                    self._logits(du, fake_n),
                    **weights,
                )
                for loss, optimizer, stage in zip(
                    d_losses, optimizers[:3], stages[:3], strict=True
                ):
                    update(loss, optimizer, stage)
                with self._frozen_discriminators(discriminators):
                    fake_p, fake_n = self._generate(gp, latent()), self._generate(gn, latent())
                    g_losses = genpu_generator_losses(
                        self._logits(dp, fake_p),
                        self._logits(du, fake_p),
                        self._logits(dn, fake_n),
                        self._logits(du, fake_n),
                        **weights,
                    )
                    for loss, optimizer, stage in zip(
                        g_losses, optimizers[3:], stages[3:5], strict=True
                    ):
                        update(loss, optimizer, stage)
                d_values.append(sum(float(loss.detach().cpu()) for loss in d_losses))
                g_values.append(sum(float(loss.detach().cpu()) for loss in g_losses))
            self.history_["gan_discriminator_loss"].append(float(np.mean(d_values)))
            self.history_["gan_generator_loss"].append(float(np.mean(g_values)))
        for net in generators + discriminators:
            net.eval()
        self.model_, self.encoder_ = score_network(2)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        # Prior-weighted PN risk on fresh generated streams; no real clean labels.
        self.checkpoint_stage_ = "synthetic_pn"
        self.checkpoint_round_ = None
        for epoch in range(self.classifier_epochs):
            self.model_.train()
            losses = []
            for _ in range(steps):
                with torch.no_grad():
                    fake_p, fake_n = self._generate(gp, latent()), self._generate(gn, latent())
                loss = (
                    prior * functional.softplus(-self._logits(self.model_, fake_p)).mean()
                    + (1 - prior) * functional.softplus(self._logits(self.model_, fake_n)).mean()
                )
                update(loss, optimizer, "synthetic_pn")
                losses.append(float(loss.detach().cpu()))
            self.history_["pn_loss"].append(float(np.mean(losses)))
            self.history_["optimizer_steps"].append(self.optimizer_steps_)
            self.model_.eval()
            self.checkpoint_stage_epoch_ = epoch + 1
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        self.model_.eval()
        self._is_fitted = True
        return self

    def _generate(self, network, latent):
        """Finite samples in the same feature/image coordinates as real training."""
        import torch

        samples = network(latent)
        if not torch.is_tensor(samples) or samples.shape != (len(latent), *self.input_shape_):
            raise ValueError("GenPU generator must preserve batch and training input shape")
        if not torch.isfinite(samples).all():
            raise ValueError("GenPU generated samples became non-finite")
        return samples

    @staticmethod
    @contextmanager
    def _frozen_discriminators(networks):
        """G gradients pass through D, but D weights/BN/dropout do not change."""
        modes = [(layer, layer.training) for network in networks for layer in network.modules()]
        flags = [
            (parameter, parameter.requires_grad)
            for network in networks
            for parameter in network.parameters()
        ]
        try:
            for network in networks:
                network.requires_grad_(False).eval()
            yield
        finally:
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            for layer, training in modes:
                layer.training = training

    @staticmethod
    def _logits(network, batch):
        """One finite logit per row; singleton BN uses running stats, no dropped rows."""
        import torch

        batch_norm = [
            layer
            for layer in network.modules()
            if len(batch) == 1
            and isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
            and layer.training
        ]
        try:
            for layer in batch_norm:
                layer.eval()
            logits = network(batch)
        finally:
            for layer in batch_norm:
                layer.train()
        if not torch.is_tensor(logits) or logits.shape != (len(batch), 1):
            raise ValueError("GenPU score network must return one logit per input row")
        if not torch.isfinite(logits).all():
            raise ValueError("GenPU logits became non-finite")
        return logits

    def _decision_function(self, X):
        import torch
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=np.float32, allow_nd=True, ensure_min_samples=0)
        if tuple(X.shape[1:]) != self.input_shape_:
            raise ValueError("GenPU input shape differs from training")
        if not len(X):
            return np.empty(0, dtype=np.float32)
        modes = [(layer, layer.training) for layer in self.model_.modules()]
        self.model_.eval()
        try:
            with torch.no_grad():
                return np.concatenate(
                    [
                        self._logits(
                            self.model_,
                            torch.as_tensor(X[i : i + self.batch_size], device=self.device_),
                        )
                        .reshape(-1)
                        .cpu()
                        .numpy()
                        for i in range(0, len(X), self.batch_size)
                    ]
                )
        finally:
            for layer, training in modes:
                layer.training = training

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        from scipy.special import expit

        self._check_is_fitted()
        positive = expit(self._decision_function(X))
        return np.column_stack((1 - positive, positive))
