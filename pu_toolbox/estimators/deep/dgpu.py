# ruff: noqa: N803, N806, N812

"""Discriminative-generative PU orchestration with a generator protocol."""

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
    Scenario,
    SourceStatus,
)
from ...core.validation import check_scalar_in_range, validate_pu_X_y


class DGPUClassifier(BasePUClassifier):
    """DGPU classifier using an explicit user-supplied conditional generator."""

    family = AlgorithmFamily.DEEP_PU
    assumption = (Assumption.SCAR, Assumption.SAR)
    scenario = (Scenario.CASE_CONTROL, Scenario.SELECTION_BIASED)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.NOT_FOUND
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL

    def __init__(
        self,
        class_prior: float,
        generator,
        *,
        model=None,
        hidden_dim: int = 128,
        rounds: int = 3,
        initialization_epochs: int = 100,
        annotation_epochs: int = 100,
        generated_samples: int = 5000,
        pseudo_label_fraction: float = 0.1,
        confidence_threshold: float = 0.95,
        debias_strength: float = 0.8,
        distribution_momentum: float = 0.999,
        batch_size: int = 256,
        learning_rate: float = 1e-4,
        weak_augmentation=None,
        strong_augmentation=None,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.generator = generator
        self.model = model
        self.hidden_dim = hidden_dim
        self.rounds = rounds
        self.initialization_epochs = initialization_epochs
        self.annotation_epochs = annotation_epochs
        self.generated_samples = generated_samples
        self.pseudo_label_fraction = pseudo_label_fraction
        self.confidence_threshold = confidence_threshold
        self.debias_strength = debias_strength
        self.distribution_momentum = distribution_momentum
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weak_augmentation = weak_augmentation
        self.strong_augmentation = strong_augmentation
        self.random_state = random_state
        self.device = device

    def _validate_parameters(self, class_prior: float) -> None:
        check_scalar_in_range(class_prior, 0.0, 1.0, "class_prior", inclusive=False)
        if self.generator is None:
            raise ValueError(
                "generator is required and must implement fit(X, y, warm_start=...) "
                "and sample(n_samples, class_label=..., random_state=...)"
            )
        if not callable(getattr(self.generator, "fit", None)) or not callable(
            getattr(self.generator, "sample", None)
        ):
            raise TypeError("generator must implement callable fit and sample methods")
        for name, value in (
            ("hidden_dim", self.hidden_dim),
            ("rounds", self.rounds),
            ("initialization_epochs", self.initialization_epochs),
            ("annotation_epochs", self.annotation_epochs),
            ("generated_samples", self.generated_samples),
            ("batch_size", self.batch_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 < self.pseudo_label_fraction <= 1.0:
            raise ValueError("pseudo_label_fraction must be in (0, 1]")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.distribution_momentum < 1.0:
            raise ValueError("distribution_momentum must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> DGPUClassifier:
        """Fit DGPU's discriminative-generative collaborative loop."""
        try:
            import torch
            from torch import nn
            from torch.nn import functional as F
        except ImportError as exc:
            raise ImportError("DGPUClassifier requires optional dependency 'torch'") from exc

        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            estimator_name="DGPUClassifier",
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
            if not np.isfinite(sample_weight).all() or np.any(sample_weight < 0):
                raise ValueError("sample_weight must be finite and non-negative")
        else:
            sample_weight = np.ones(len(X), dtype=np.float32)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)
        device = resolve_device(self.device)
        self.model_ = (
            nn.Sequential(
                nn.Linear(X.shape[1], self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            if self.model is None
            else copy.deepcopy(self.model)
        ).to(device)
        self.generator_ = copy.deepcopy(self.generator)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        tx = torch.as_tensor(X, dtype=torch.float32, device=device)
        ty = torch.as_tensor(y_pu, dtype=torch.long, device=device)
        sample_weight_t = torch.as_tensor(sample_weight, dtype=torch.float32, device=device)
        p_mask = ty == 1
        u_mask = ~p_mask

        def scalar_logits(model, batch):
            result = model(batch)
            if result.ndim == 2 and result.shape[1] == 1:
                result = result[:, 0]
            if result.ndim != 1:
                raise ValueError("DGPU model must output shape (n,) or (n, 1)")
            return result

        def augment(batch, *, strong: bool):
            function = self.strong_augmentation if strong else self.weak_augmentation
            if function is not None:
                result = function(batch)
                return torch.as_tensor(result, dtype=batch.dtype, device=device)
            if strong:
                return batch + 0.05 * torch.randn_like(batch)
            return batch

        self.history_ = {
            "initialization_loss": [],
            "supervised_loss": [],
            "unsupervised_loss": [],
        }
        for _ in range(self.initialization_epochs):
            logits = scalar_logits(self.model_, augment(tx, strong=False))
            probabilities = torch.sigmoid(logits)
            positive_mean = (
                probabilities[p_mask] * sample_weight_t[p_mask]
            ).sum() / sample_weight_t[p_mask].sum().clamp_min(1e-12)
            unlabeled_mean = (
                probabilities[u_mask] * sample_weight_t[u_mask]
            ).sum() / sample_weight_t[u_mask].sum().clamp_min(1e-12)
            positive_alignment = (positive_mean - 1.0).abs()
            unlabeled_alignment = (unlabeled_mean - pi).abs()
            loss = 2.0 * pi * positive_alignment + unlabeled_alignment
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.history_["initialization_loss"].append(float(loss.detach().cpu()))

        predicted_distribution = torch.tensor(
            [1.0 - pi, pi],
            dtype=torch.float32,
            device=device,
        )
        self.generated_counts_ = []
        self.pseudo_labeled_indices_ = []

        def probabilities_for(model, values):
            with torch.no_grad():
                scores = scalar_logits(model, values)
                positive = torch.sigmoid(scores)
                return torch.stack([1.0 - positive, positive], dim=1)

        def select_pseudo_labels():
            probabilities = probabilities_for(self.model_, tx[u_mask]).cpu().numpy()
            predicted = probabilities.argmax(axis=1)
            confidence = probabilities.max(axis=1)
            original_indices = np.flatnonzero(y_pu == 0)
            target_per_class = max(
                1,
                int(np.ceil(self.pseudo_label_fraction * len(original_indices))),
            )
            chosen_indices = []
            chosen_labels = []
            for class_label in (0, 1):
                candidates = np.flatnonzero(predicted == class_label)
                if len(candidates) == 0:
                    continue
                count = min(target_per_class, len(candidates))
                weights = confidence[candidates]
                weights = weights / weights.sum()
                selected = rng.choice(
                    candidates,
                    size=count,
                    replace=False,
                    p=weights,
                )
                chosen_indices.extend(original_indices[selected].tolist())
                chosen_labels.extend([class_label] * count)
            return np.asarray(chosen_indices, dtype=int), np.asarray(chosen_labels, dtype=int)

        for round_index in range(self.rounds):
            pseudo_indices, pseudo_labels = select_pseudo_labels()
            self.pseudo_labeled_indices_.append(pseudo_indices.copy())
            generator_X = np.concatenate([X[y_pu == 1], X[pseudo_indices]], axis=0)
            generator_y = np.concatenate(
                [np.ones(np.sum(y_pu == 1), dtype=int), pseudo_labels],
                axis=0,
            )
            self.generator_.fit(
                generator_X,
                generator_y,
                warm_start=round_index > 0,
            )
            positive_count = int(round(self.generated_samples * pi))
            negative_count = self.generated_samples - positive_count
            generated_negative = np.asarray(
                self.generator_.sample(
                    negative_count,
                    class_label=0,
                    random_state=None
                    if self.random_state is None
                    else self.random_state + 2 * round_index,
                ),
                dtype=np.float32,
            )
            generated_positive = np.asarray(
                self.generator_.sample(
                    positive_count,
                    class_label=1,
                    random_state=None
                    if self.random_state is None
                    else self.random_state + 2 * round_index + 1,
                ),
                dtype=np.float32,
            )
            generated_X = np.concatenate(
                [generated_negative, generated_positive],
                axis=0,
            )
            generated_y = np.concatenate(
                [
                    np.zeros(negative_count, dtype=np.float32),
                    np.ones(positive_count, dtype=np.float32),
                ]
            )
            if generated_X.shape != (self.generated_samples, X.shape[1]):
                raise ValueError(
                    "generator.sample returned incompatible shape; expected "
                    f"({self.generated_samples}, {X.shape[1]}) in total"
                )
            self.generated_counts_.append({"negative": negative_count, "positive": positive_count})

            labeled_X = np.concatenate([X[y_pu == 1], generated_X], axis=0)
            labeled_y = np.concatenate(
                [np.ones(np.sum(y_pu == 1), dtype=np.float32), generated_y],
                axis=0,
            )
            labeled_sample_weight = np.concatenate(
                [
                    sample_weight[y_pu == 1],
                    np.ones(self.generated_samples, dtype=np.float32),
                ]
            )
            labeled_t = torch.as_tensor(labeled_X, dtype=torch.float32, device=device)
            labeled_y_t = torch.as_tensor(labeled_y, dtype=torch.float32, device=device)
            labeled_sample_weight_t = torch.as_tensor(
                labeled_sample_weight,
                dtype=torch.float32,
                device=device,
            )
            previous_model = copy.deepcopy(self.model_).eval()
            for parameter in previous_model.parameters():
                parameter.requires_grad_(False)

            for _ in range(self.annotation_epochs):
                previous_probabilities = probabilities_for(
                    previous_model,
                    augment(labeled_t, strong=False),
                )
                confidence_weights = previous_probabilities.max(dim=1).values.sqrt()
                supervised_logits = scalar_logits(
                    self.model_,
                    augment(labeled_t, strong=False),
                )
                supervised_terms = F.binary_cross_entropy_with_logits(
                    supervised_logits,
                    labeled_y_t,
                    reduction="none",
                )
                effective_weights = confidence_weights * labeled_sample_weight_t
                supervised_loss = (
                    effective_weights * supervised_terms
                ).sum() / effective_weights.sum().clamp_min(1e-12)

                weak_logits = scalar_logits(
                    self.model_,
                    augment(tx[u_mask], strong=False),
                )
                weak_binary_logits = torch.stack(
                    [torch.zeros_like(weak_logits), weak_logits],
                    dim=1,
                )
                weak_probabilities = torch.softmax(weak_binary_logits, dim=1)
                predicted_distribution = self.distribution_momentum * predicted_distribution + (
                    1.0 - self.distribution_momentum
                ) * weak_probabilities.detach().mean(dim=0)
                predicted_distribution = (
                    predicted_distribution / predicted_distribution.sum()
                ).clamp_min(1e-6)
                predicted_distribution = predicted_distribution / predicted_distribution.sum()
                debiased_logits = weak_binary_logits - self.debias_strength * torch.log(
                    predicted_distribution
                )
                debiased_probabilities = torch.softmax(debiased_logits, dim=1)
                pseudo_targets = weak_probabilities.detach().argmax(dim=1)
                mask = debiased_probabilities.detach().max(dim=1).values > self.confidence_threshold

                strong_logits = scalar_logits(
                    self.model_,
                    augment(tx[u_mask], strong=True),
                )
                strong_binary_logits = torch.stack(
                    [torch.zeros_like(strong_logits), strong_logits],
                    dim=1,
                )
                strong_probabilities = torch.softmax(strong_binary_logits, dim=1)
                marginal_logits = strong_probabilities + self.debias_strength * torch.log(
                    predicted_distribution
                )
                dml = F.cross_entropy(
                    marginal_logits,
                    pseudo_targets,
                    reduction="none",
                )
                if mask.any():
                    unlabeled_weights = sample_weight_t[u_mask][mask]
                    unsupervised_loss = (
                        dml[mask] * unlabeled_weights
                    ).sum() / unlabeled_weights.sum().clamp_min(1e-12)
                else:
                    unsupervised_loss = dml.new_zeros(())
                loss = supervised_loss + unsupervised_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.history_["supervised_loss"].append(float(supervised_loss.detach().cpu()))
                self.history_["unsupervised_loss"].append(float(unsupervised_loss.detach().cpu()))

        self.predicted_distribution_ = predicted_distribution.detach().cpu().numpy()
        self.class_prior_ = float(pi)
        self.classes_ = np.array([0, 1])
        self.device_ = device
        self._class_prior = self.class_prior_
        self._X_shape_ = X.shape
        self._is_fitted = True
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            result = self.model_(torch.as_tensor(X, dtype=torch.float32, device=self.device_))
        return result.reshape(-1).cpu().numpy()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self._decision_function(X) >= 0.0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        score = 1.0 / (1.0 + np.exp(-np.clip(self._decision_function(X), -40, 40)))
        return np.column_stack([1.0 - score, score])
