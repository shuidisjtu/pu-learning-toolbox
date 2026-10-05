# ruff: noqa: N803, N806
"""Paper-derived PAN (AAAI 2021), Equation 7 and Algorithm 1.

MLP or injected end-to-end CNN, not a reproduction of the paper's networks.
The paper explicitly uses single-training-set sampling (p. 7810, footnote 5).
"""

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
from ._validation import validate_encoder_features


def pan_discriminator_loss(positive_logits, unlabeled_logits, classifier_logits, *, weight):
    """Negative of Eq. 7 for D ascent; C is held fixed, P/U means separate."""
    from torch.nn import functional

    positive = positive_logits.reshape(-1)
    unlabeled = unlabeled_logits.reshape(-1)
    classifier = classifier_logits.reshape(-1).detach()
    if not positive.numel() or not unlabeled.numel() or classifier.shape != unlabeled.shape:
        raise ValueError("PAN requires nonempty P/U and matched unlabeled logits")
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("weight must be finite and positive")
    # log(1-sigmoid(c))-log(sigmoid(c)) = -c; stable even at extreme logits.
    game = (-classifier * (2 * unlabeled.sigmoid() - 1)).mean()
    return (
        -functional.logsigmoid(positive).mean()
        - functional.logsigmoid(-unlabeled).mean()
        - weight * game
    )


def pan_classifier_loss(classifier_logits, discriminator_logits, *, weight):
    """Eq. 7 minimized over C only, using the updated, detached D reward."""
    classifier = classifier_logits.reshape(-1)
    discriminator = discriminator_logits.reshape(-1).detach()
    if not classifier.numel() or classifier.shape != discriminator.shape:
        raise ValueError("PAN requires nonempty matched C/D logits")
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("weight must be finite and positive")
    return weight * (-classifier * (2 * discriminator.sigmoid() - 1)).mean()


class PANClassifier(BasePUClassifier):
    """Alternating predictive adversarial learning with an OS PU training set."""

    family = AlgorithmFamily.DEEP_PU
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.SINGLE_TRAINING_SET,)
    requires_class_prior = False
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.NOT_FOUND
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True

    def __init__(
        self,
        *,
        hidden_dim=128,
        max_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        adversarial_weight=1.0,
        model=None,
        discriminator=None,
        encoder=None,
        random_state=0,
        device=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.adversarial_weight = adversarial_weight
        self.model = model
        self.discriminator = discriminator
        self.encoder = encoder
        self.random_state = random_state
        self.device = device

    def fit(
        self, X, y_pu, *, class_prior=None, sample_weight=None, epoch_callback=None, os_or_ts="os"
    ):
        import torch
        from torch import nn

        self._is_fitted = False
        if sample_weight is not None:
            raise NotImplementedError("PAN does not implement sample_weight")
        if os_or_ts != "os":
            raise ValueError("PAN is native OS; ts risk substitution is not applicable")
        X, y_pu = validate_pu_X_y(
            X, y_pu, accept_sparse=False, allow_nd=True, estimator_name="PANClassifier"
        )
        if X.ndim not in (2, 4):
            raise ValueError("PAN supports 2-D features or 4-D NCHW images")
        if X.ndim == 4 and self.encoder is None:
            raise ValueError("PAN 4-D images require an explicit encoder; no silent flattening")
        if self.encoder is not None and not isinstance(self.encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        view = build_training_view(X, y_pu, requested_view=os_or_ts)
        for name in ("hidden_dim", "max_epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "adversarial_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if class_prior is not None and (not np.isfinite(class_prior) or not 0 < class_prior < 1):
            raise ValueError("class_prior, if supplied, must be in (0, 1)")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X must remain finite after float32 conversion")
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        self.device_ = resolve_device(self.device)

        def network(supplied):
            if supplied is not None and not isinstance(supplied, nn.Module):
                raise TypeError("model/discriminator must be a torch.nn.Module")
            if self.encoder is not None:
                # C and D own independent encoder copies and BN state. Never
                # share a fitted module across adversaries or CV folds.
                encoder = copy.deepcopy(self.encoder).to(device=self.device_, dtype=torch.float32)
                encoder.requires_grad_(True).eval()
                with torch.no_grad():
                    representation = encoder(torch.as_tensor(X[:1], device=self.device_))
                width = validate_encoder_features(representation, encoder_param_name="encoder")
                head = copy.deepcopy(supplied) if supplied is not None else nn.Linear(width, 1)
                result = nn.Sequential(encoder, head)
            elif supplied is not None:
                result = copy.deepcopy(supplied)
            else:
                result = nn.Sequential(
                    nn.Linear(X.shape[1], self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)
                )
            result.to(device=self.device_, dtype=torch.float32).eval()
            with torch.no_grad():
                scores = result(torch.as_tensor(X[:1], device=self.device_))
            if scores.shape not in ((1,), (1, 1)) or not torch.isfinite(scores).all():
                raise ValueError("PAN networks must return one finite logit per input")
            if not any(p.requires_grad for p in result.parameters()):
                raise ValueError("PAN networks must contain trainable parameters")
            return result

        self.model_, self.discriminator_ = network(self.model), network(self.discriminator)
        self.encoder_ = self.model_[0] if self.encoder is not None else None
        self.discriminator_encoder_ = self.discriminator_[0] if self.encoder is not None else None
        c_optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        d_optimizer = torch.optim.Adam(self.discriminator_.parameters(), lr=self.learning_rate)
        positive, unlabeled = view.positive_positions, view.native_unlabeled_positions
        data = torch.as_tensor(X)
        n_steps = (len(unlabeled) + self.batch_size - 1) // self.batch_size
        self.history_ = {"epoch": [], "discriminator_loss": [], "classifier_loss": []}
        self.optimizer_steps_ = 0
        self.training_view_ = "os"
        self.calibration_applied_ = False
        self.n_positive_, self.n_unlabeled_ = len(positive), len(unlabeled)
        self.n_features_in_, self._X_shape_ = X.shape[1], X.shape
        self.input_shape_ = tuple(X.shape[1:])
        self._class_prior = None
        self.classes_ = np.array([0, 1])
        for epoch in range(self.max_epochs):
            d_losses, c_losses = [], []
            self.model_.train()
            self.discriminator_.train()
            for _ in range(n_steps):
                p = data[rng.choice(positive, self.batch_size)].to(self.device_)
                u = data[rng.choice(unlabeled, self.batch_size)].to(self.device_)
                # The inactive network is evaluated without updating BN/dropout state.
                self.model_.eval()
                with torch.no_grad():
                    c_fixed = self.model_(u)
                self.model_.train()
                d_optimizer.zero_grad()
                d_loss = pan_discriminator_loss(
                    self.discriminator_(p),
                    self.discriminator_(u),
                    c_fixed,
                    weight=self.adversarial_weight,
                )
                if not torch.isfinite(d_loss):
                    raise ValueError("PAN discriminator loss became non-finite")
                d_loss.backward()
                d_optimizer.step()
                # Algorithm 1 uses a fresh U minibatch for C after D's update.
                u = data[rng.choice(unlabeled, self.batch_size)].to(self.device_)
                self.discriminator_.eval()
                with torch.no_grad():
                    d_fixed = self.discriminator_(u)
                self.discriminator_.train()
                c_optimizer.zero_grad()
                c_loss = pan_classifier_loss(
                    self.model_(u), d_fixed, weight=self.adversarial_weight
                )
                if not torch.isfinite(c_loss):
                    raise ValueError("PAN classifier loss became non-finite")
                c_loss.backward()
                c_optimizer.step()
                self.optimizer_steps_ += 2
                d_losses.append(float(d_loss.detach().cpu()))
                c_losses.append(float(c_loss.detach().cpu()))
            self.history_["epoch"].append(epoch)
            self.history_["discriminator_loss"].append(float(np.mean(d_losses)))
            self.history_["classifier_loss"].append(float(np.mean(c_losses)))
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        self.model_.eval()
        self.discriminator_.eval()
        self._is_fitted = True
        return self

    def _decision_function(self, X):
        import torch
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=np.float32, allow_nd=True, ensure_min_samples=0)
        if tuple(X.shape[1:]) != self.input_shape_:
            raise ValueError("PAN input shape differs from training")
        if not len(X):
            return np.empty(0, dtype=np.float32)
        was_training = self.model_.training
        self.model_.eval()
        try:
            with torch.no_grad():
                return np.concatenate(
                    [
                        self.model_(
                            torch.as_tensor(X[i : i + self.batch_size], device=self.device_)
                        )
                        .reshape(-1)
                        .cpu()
                        .numpy()
                        for i in range(0, len(X), self.batch_size)
                    ]
                )
        finally:
            self.model_.train(was_training)

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        from scipy.special import expit

        self._check_is_fitted()
        probabilities = expit(self._decision_function(X))
        return np.column_stack((1 - probabilities, probabilities))
