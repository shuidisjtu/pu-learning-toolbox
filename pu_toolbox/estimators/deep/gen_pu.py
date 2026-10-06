# ruff: noqa: N803, N806
"""Paper-objective dense GenPU, IJCAI 2018 equations (3)--(11).

The author demo uses different unweighted D_u / non-saturating G losses.
This explicit paper-objective adapter is not an author-demo replay.
"""

from __future__ import annotations

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
    native_architectures = frozenset({"mlp"})
    input_ndims = frozenset({2})
    encoder_parameter = None
    trains_encoder = False
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
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="GenPUClassifier")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X must remain finite after float32 conversion")
        view = build_training_view(X, y_pu, requested_view=os_or_ts)
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        self.device_ = resolve_device(self.device)
        width = X.shape[1]

        def mlp(input_dim, output_dim, depth):
            layers = []
            for _ in range(depth):
                layers.extend([nn.Linear(input_dim, self.hidden_dim), nn.LeakyReLU(0.2)])
                input_dim = self.hidden_dim
            return nn.Sequential(*layers, nn.Linear(input_dim, output_dim)).to(self.device_)

        self.positive_generator_ = mlp(self.latent_dim, width, 2)
        self.negative_generator_ = mlp(self.latent_dim, width, 2)
        self.positive_discriminator_ = mlp(width, 1, 0)
        self.negative_discriminator_ = mlp(width, 1, 0)
        self.unlabeled_discriminator_ = mlp(width, 1, 2)
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
        self.history_ = {"gan_discriminator_loss": [], "gan_generator_loss": [], "pn_loss": []}
        self.n_positive_ = len(p_indices)
        self.n_unlabeled_ = len(view.native_unlabeled_positions)
        self.n_loss_unlabeled_ = len(u_indices)
        self.training_view_ = os_or_ts
        self.calibration_applied_ = view.calibration_applied
        self._class_prior = float(prior)
        self.classes_ = np.array([0, 1])
        self.n_features_in_, self._X_shape_ = width, X.shape
        self.objective_variant_ = "paper_minimax_prior_weighted"

        def latent():
            # NumPy RNG is local and stable across CPU/device transfers.
            return torch.as_tensor(
                rng.normal(size=(self.batch_size, self.latent_dim)),
                dtype=torch.float32,
                device=self.device_,
            )

        def update(loss, optimizer):
            if not torch.isfinite(loss):
                raise ValueError("GenPU loss became non-finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.optimizer_steps_ += 1

        dp, dn, du = discriminators
        gp, gn = generators
        for _ in range(self.max_epochs):
            d_values, g_values = [], []
            for _ in range(steps):
                p = data[rng.choice(p_indices, self.batch_size)].to(self.device_)
                u = data[rng.choice(u_indices, self.batch_size)].to(self.device_)
                with torch.no_grad():
                    fake_p, fake_n = gp(latent()), gn(latent())
                d_losses = genpu_discriminator_losses(
                    dp(p), dp(fake_p), dn(p), dn(fake_n), du(u), du(fake_p), du(fake_n), **weights
                )
                for loss, optimizer in zip(d_losses, optimizers[:3], strict=True):
                    update(loss, optimizer)
                for net in discriminators:
                    net.requires_grad_(False)
                try:
                    fake_p, fake_n = gp(latent()), gn(latent())
                    g_losses = genpu_generator_losses(
                        dp(fake_p), du(fake_p), dn(fake_n), du(fake_n), **weights
                    )
                    for loss, optimizer in zip(g_losses, optimizers[3:], strict=True):
                        update(loss, optimizer)
                finally:
                    for net in discriminators:
                        net.requires_grad_(True)
                d_values.append(sum(float(loss.detach().cpu()) for loss in d_losses))
                g_values.append(sum(float(loss.detach().cpu()) for loss in g_losses))
            self.history_["gan_discriminator_loss"].append(float(np.mean(d_values)))
            self.history_["gan_generator_loss"].append(float(np.mean(g_values)))
        for net in generators + discriminators:
            net.eval()
        self.model_ = mlp(width, 1, 2)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        # Prior-weighted PN risk on fresh generated streams; no real clean labels.
        self.checkpoint_stage_ = "synthetic_pn"
        self.checkpoint_round_ = None
        for epoch in range(self.classifier_epochs):
            losses = []
            for _ in range(steps):
                with torch.no_grad():
                    fake_p, fake_n = gp(latent()), gn(latent())
                loss = (
                    prior * functional.softplus(-self.model_(fake_p)).mean()
                    + (1 - prior) * functional.softplus(self.model_(fake_n)).mean()
                )
                update(loss, optimizer)
                losses.append(float(loss.detach().cpu()))
            self.history_["pn_loss"].append(float(np.mean(losses)))
            self.model_.eval()
            self.checkpoint_stage_epoch_ = epoch + 1
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        self.model_.eval()
        self._is_fitted = True
        return self

    def _decision_function(self, X):
        import torch
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("GenPU feature count differs from training")
        with torch.no_grad():
            return np.concatenate(
                [
                    self.model_(torch.as_tensor(X[i : i + self.batch_size], device=self.device_))
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    for i in range(0, len(X), self.batch_size)
                ]
            )

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        from scipy.special import expit

        self._check_is_fitted()
        positive = expit(self._decision_function(X))
        return np.column_stack((1 - positive, positive))
