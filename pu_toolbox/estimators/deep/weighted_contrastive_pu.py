# ruff: noqa: N803, N806, N812

"""Weighted contrastive learning with hard-negative mining for PU data."""

from __future__ import annotations

import copy

import numpy as np

from ...core.base import BasePUClassifier
from ...core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    Scenario,
    SourceStatus,
)
from ...core.validation import validate_pu_X_y


def embedding_dissimilarity(query, keys):
    """Paper Eq. 13: normalized dissimilarity in [0, 1]."""
    from torch.nn import functional as F

    query = F.normalize(query, dim=-1)
    keys = F.normalize(keys, dim=-1)
    return 0.25 * (query.unsqueeze(0) - keys).square().sum(dim=-1)


class WeightedContrastivePUClassifier(BasePUClassifier):
    """Clean-room WConPU core with prototypes, SAT and a momentum queue."""

    family = AlgorithmFamily.DEEP_PU
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.NOT_FOUND
    backend = Backend.TORCH
    maturity = Maturity.RESEARCH

    def __init__(
        self,
        class_prior: float,
        *,
        encoder=None,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        queue_size: int = 8192,
        temperature: float = 0.07,
        momentum: float = 0.999,
        pseudo_label_momentum: float = 0.9,
        contrastive_weight: float = 0.1,
        distribution_weight: float = 0.1,
        hard_negative_quantile: float = 0.25,
        weak_augmentation=None,
        strong_augmentation=None,
        batch_size: int = 256,
        max_epochs: int = 800,
        learning_rate: float = 1e-2,
        random_state: int | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.temperature = temperature
        self.momentum = momentum
        self.pseudo_label_momentum = pseudo_label_momentum
        self.contrastive_weight = contrastive_weight
        self.distribution_weight = distribution_weight
        self.hard_negative_quantile = hard_negative_quantile
        self.weak_augmentation = weak_augmentation
        self.strong_augmentation = strong_augmentation
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device

    def _validate_parameters(self, class_prior: float) -> None:
        if not 0.0 < class_prior < 1.0:
            raise ValueError("class_prior must be in (0, 1)")
        for name, value in (
            ("hidden_dim", self.hidden_dim),
            ("embedding_dim", self.embedding_dim),
            ("queue_size", self.queue_size),
            ("batch_size", self.batch_size),
            ("max_epochs", self.max_epochs),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.temperature <= 0 or self.learning_rate <= 0:
            raise ValueError("temperature and learning_rate must be positive")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 <= self.pseudo_label_momentum < 1.0:
            raise ValueError("pseudo_label_momentum must be in [0, 1)")
        if not 0.0 <= self.hard_negative_quantile <= 1.0:
            raise ValueError("hard_negative_quantile must be in [0, 1]")

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> WeightedContrastivePUClassifier:
        """Fit the WConPU collaborative contrastive/classification objective."""
        try:
            import torch
            from torch import nn
            from torch.nn import functional as F
        except ImportError as exc:
            raise ImportError(
                "WeightedContrastivePUClassifier requires optional dependency 'torch'"
            ) from exc

        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            estimator_name="WeightedContrastivePUClassifier",
        )
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values")
        pi = self.class_prior if class_prior is None else class_prior
        self._validate_parameters(pi)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)
            if sample_weight.shape != (len(X),):
                raise ValueError("sample_weight must have shape (n_samples,)")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)
        device = torch.device(self.device)
        self.encoder_ = (
            nn.Sequential(nn.Linear(X.shape[1], self.hidden_dim), nn.ReLU())
            if self.encoder is None
            else copy.deepcopy(self.encoder)
        ).to(device)
        with torch.no_grad():
            probe = self.encoder_(torch.as_tensor(X[:1], device=device))
        feature_dim = int(probe.shape[-1])
        self.projector_ = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        ).to(device)
        self.classifier_head_ = nn.Linear(feature_dim, 2).to(device)
        self.key_encoder_ = copy.deepcopy(self.encoder_).to(device)
        self.key_projector_ = copy.deepcopy(self.projector_).to(device)
        for parameter in list(self.key_encoder_.parameters()) + list(
            self.key_projector_.parameters()
        ):
            parameter.requires_grad_(False)

        optimizer = torch.optim.SGD(
            list(self.encoder_.parameters())
            + list(self.projector_.parameters())
            + list(self.classifier_head_.parameters()),
            lr=self.learning_rate,
            momentum=0.9,
        )
        tx = torch.as_tensor(X, dtype=torch.float32, device=device)
        ty = torch.as_tensor(y_pu, dtype=torch.long, device=device)
        pseudo = torch.empty((len(X), 2), dtype=torch.float32, device=device)
        pseudo[:, 0] = 1.0 - pi
        pseudo[:, 1] = pi
        pseudo[ty == 1] = torch.tensor([0.0, 1.0], device=device)
        prototypes = F.normalize(
            torch.randn(2, self.embedding_dim, device=device),
            dim=1,
        )
        queue_embeddings = torch.empty((0, self.embedding_dim), device=device)
        queue_labels = torch.empty(0, dtype=torch.long, device=device)
        queue_confidence = torch.empty(0, device=device)
        queue_observed = torch.empty(0, dtype=torch.long, device=device)
        sat_global = torch.tensor(0.5, device=device)
        sat_classwise = torch.full((2,), 0.5, device=device)
        self.history_ = {
            "loss": [],
            "classification_loss": [],
            "contrastive_loss": [],
            "distribution_loss": [],
        }

        def augment(batch, *, strong: bool):
            function = self.strong_augmentation if strong else self.weak_augmentation
            if function is not None:
                result = function(batch)
                return torch.as_tensor(result, dtype=batch.dtype, device=device)
            if strong:
                return batch + 0.05 * torch.randn_like(batch)
            return batch

        for _ in range(self.max_epochs):
            permutation = rng.permutation(len(X))
            epoch_values = np.zeros(4, dtype=float)
            steps = 0
            for start in range(0, len(X), self.batch_size):
                indices_np = permutation[start : start + self.batch_size]
                indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
                batch = tx[indices]
                batch_labels = ty[indices]
                weak = augment(batch, strong=False)
                strong = augment(batch, strong=True)

                features = self.encoder_(weak)
                query = F.normalize(self.projector_(features), dim=1)
                logits = self.classifier_head_(features)
                probabilities = torch.softmax(logits, dim=1)
                with torch.no_grad():
                    key = F.normalize(
                        self.key_projector_(self.key_encoder_(strong)),
                        dim=1,
                    )
                    sat_global.mul_(self.momentum).add_(
                        probabilities.max(dim=1).values.mean() * (1.0 - self.momentum)
                    )
                    sat_classwise.mul_(self.momentum).add_(
                        probabilities.mean(dim=0) * (1.0 - self.momentum)
                    )
                    thresholds = sat_global * sat_classwise / sat_classwise.max()

                    predicted = probabilities.argmax(dim=1)
                    for row, predicted_class in enumerate(predicted):
                        class_index = int(predicted_class.item())
                        prototypes[class_index] = F.normalize(
                            self.momentum * prototypes[class_index]
                            + (1.0 - self.momentum) * query[row].detach(),
                            dim=0,
                        )
                    nearest = (query.detach() @ prototypes.T).argmax(dim=1)
                    hard_labels = F.one_hot(nearest, num_classes=2).float()
                    unlabeled = batch_labels == 0
                    pseudo[indices[unlabeled]] = (
                        self.pseudo_label_momentum * pseudo[indices[unlabeled]]
                        + (1.0 - self.pseudo_label_momentum) * hard_labels[unlabeled]
                    )
                    pseudo[indices[batch_labels == 1]] = torch.tensor(
                        [0.0, 1.0],
                        device=device,
                    )

                pool_embeddings = torch.cat([key, queue_embeddings], dim=0)
                pool_labels = torch.cat(
                    [probabilities.detach().argmax(dim=1), queue_labels],
                    dim=0,
                )
                pool_confidence = torch.cat(
                    [probabilities.detach().max(dim=1).values, queue_confidence],
                    dim=0,
                )
                pool_observed = torch.cat([batch_labels, queue_observed], dim=0)
                contrastive_terms = []
                for row in range(len(indices)):
                    anchor_label = int(probabilities[row].argmax())
                    if int(batch_labels[row]) == 1:
                        positive_mask = (pool_observed == 1) | (
                            (pool_labels == 1) & (pool_confidence >= thresholds[1])
                        )
                    else:
                        positive_mask = (pool_labels == anchor_label) & (
                            pool_confidence >= thresholds[anchor_label]
                        )
                    positive_mask[row] = True
                    similarity = (query[row] @ pool_embeddings.T) / self.temperature
                    dissimilarity = embedding_dissimilarity(
                        query[row],
                        pool_embeddings,
                    )
                    prototype_labels = (pool_embeddings @ prototypes.T).argmax(dim=1)
                    candidate_negative = prototype_labels != nearest[row]
                    weights = torch.ones_like(similarity)
                    if candidate_negative.any():
                        cutoff = torch.quantile(
                            dissimilarity,
                            self.hard_negative_quantile,
                        )
                        hard_negative = candidate_negative & (dissimilarity <= cutoff)
                        weights[hard_negative] = (
                            dissimilarity[hard_negative].clamp_min(1e-4).reciprocal()
                        )
                    denominator = torch.logsumexp(
                        similarity + torch.log(weights),
                        dim=0,
                    )
                    contrastive_terms.append(-(similarity[positive_mask] - denominator).mean())
                contrastive_loss = torch.stack(contrastive_terms).mean()

                classification_terms = -(pseudo[indices] * torch.log_softmax(logits, dim=1)).sum(
                    dim=1
                )
                if sample_weight is None:
                    classification_loss = classification_terms.mean()
                else:
                    batch_weights = torch.as_tensor(
                        sample_weight[indices_np],
                        device=device,
                    )
                    classification_loss = (
                        classification_terms * batch_weights
                    ).sum() / batch_weights.sum().clamp_min(1e-12)
                p_mask = batch_labels == 1
                u_mask = ~p_mask
                positive_alignment = (
                    (probabilities[p_mask, 1].mean() - 1.0).abs()
                    if p_mask.any()
                    else probabilities.new_zeros(())
                )
                unlabeled_alignment = (
                    (probabilities[u_mask, 1].mean() - pi).abs()
                    if u_mask.any()
                    else probabilities.new_zeros(())
                )
                distribution_loss = 2.0 * pi * positive_alignment + unlabeled_alignment
                loss = (
                    classification_loss
                    + self.contrastive_weight * contrastive_loss
                    + self.distribution_weight * distribution_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    query_parameters = list(self.encoder_.parameters()) + list(
                        self.projector_.parameters()
                    )
                    key_parameters = list(self.key_encoder_.parameters()) + list(
                        self.key_projector_.parameters()
                    )
                    for key_parameter, query_parameter in zip(
                        key_parameters,
                        query_parameters,
                        strict=True,
                    ):
                        key_parameter.mul_(self.momentum).add_(
                            query_parameter * (1.0 - self.momentum)
                        )
                    queue_embeddings = torch.cat(
                        [queue_embeddings, key.detach()],
                        dim=0,
                    )[-self.queue_size :]
                    queue_labels = torch.cat(
                        [queue_labels, probabilities.detach().argmax(dim=1)],
                        dim=0,
                    )[-self.queue_size :]
                    queue_confidence = torch.cat(
                        [
                            queue_confidence,
                            probabilities.detach().max(dim=1).values,
                        ],
                        dim=0,
                    )[-self.queue_size :]
                    queue_observed = torch.cat(
                        [queue_observed, batch_labels],
                        dim=0,
                    )[-self.queue_size :]
                epoch_values += [
                    float(loss.detach().cpu()),
                    float(classification_loss.detach().cpu()),
                    float(contrastive_loss.detach().cpu()),
                    float(distribution_loss.detach().cpu()),
                ]
                steps += 1

            for key, value in zip(self.history_, epoch_values / steps, strict=True):
                self.history_[key].append(float(value))

        self.prototypes_ = prototypes.detach().cpu().numpy()
        self.pseudo_labels_ = pseudo.detach().cpu().numpy()
        self.queue_embeddings_ = queue_embeddings.detach().cpu().numpy()
        self.queue_labels_ = queue_labels.detach().cpu().numpy()
        self.sat_global_ = float(sat_global.cpu())
        self.sat_classwise_ = sat_classwise.cpu().numpy()
        self.class_prior_ = float(pi)
        self.classes_ = np.array([0, 1])
        self.device_ = device
        self._class_prior = self.class_prior_
        self._X_shape_ = X.shape
        self._is_fitted = True
        return self

    def _logits(self, X: np.ndarray):
        import torch

        X = np.asarray(X, dtype=np.float32)
        self.encoder_.eval()
        self.classifier_head_.eval()
        with torch.no_grad():
            features = self.encoder_(torch.as_tensor(X, dtype=torch.float32, device=self.device_))
            return self.classifier_head_(features).cpu().numpy()

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(X)
        return logits[:, 1] - logits[:, 0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._logits(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(X)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
