"""P3Mix and its E/C variants, based on Li et al. (ICLR 2022).

The paper's algorithm and equations are implemented as an OS PU classifier.
Unspecified optimizer/backbone/batching details are exposed as engineering
choices; this is not a claim of author-code or benchmark reproduction.
"""

# ruff: noqa: N803, N806

from __future__ import annotations

from numbers import Real

import numpy as np

from ...core.base import BasePUClassifier
from ...core.device import resolve_device
from ...core.fit_state import invalidate_failed_fit
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


def p3mix_candidate_positions(positive_probabilities, *, pool_size):
    """Top-k binary entropy among known positives; ties use stable row order."""
    from scipy.special import xlogy

    probabilities = np.asarray(positive_probabilities, dtype=float)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("candidate probabilities must be nonempty and one-dimensional")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("candidate probabilities must be finite and in [0,1]")
    if type(pool_size) is not int or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    entropy = -xlogy(probabilities, probabilities) - xlogy(1 - probabilities, 1 - probabilities)
    return np.argsort(-entropy, kind="stable")[: min(pool_size, len(probabilities))]


def _is_finite_real(value):
    """Reject bools and nonnumeric scalars before range comparisons."""
    return (
        not isinstance(value, bool | np.bool_)
        and isinstance(value, Real | np.integer | np.floating)
        and bool(np.isfinite(value))
    )


def p3mix_training_candidate_pool(
    features, pu_labels, predicted_probabilities, sample_indices, *, pool_size
):
    """Build a candidate pool exclusively from this training partition's P.

    Labels must be observed PU labels, not hidden PN truth. Identities are
    preserved for downstream provenance checks; no clean validation/test
    input is accepted. This does not implement the paper's pool update schedule.
    """
    features = np.asarray(features)
    labels = np.asarray(pu_labels)
    probabilities = np.asarray(predicted_probabilities, dtype=float)
    indices = np.asarray(sample_indices)
    if (
        features.ndim < 2
        or not len(features)
        or any(d < 1 for d in features.shape[1:])
        or not np.isfinite(features).all()
    ):
        raise ValueError("training features must be nonempty finite batch-first arrays")
    if labels.shape != (len(features),) or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("training pu_labels must be matched binary observed labels")
    if (
        probabilities.shape != labels.shape
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("training predictions must be matched finite [0,1]")
    if (
        indices.shape != labels.shape
        or indices.dtype.kind not in "iu"
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("training sample_indices must be matched unique integer identities")
    positives = np.flatnonzero(labels == 1)
    if not len(positives):
        raise ValueError("candidate pool requires observed labeled training positives")
    positions = positives[p3mix_candidate_positions(probabilities[positives], pool_size=pool_size)]
    return {
        "features": features[positions].copy(),
        "sample_indices": indices[positions].copy(),
        "training_positions": positions.copy(),
    }


def p3mix_heuristic_batch(
    features,
    pu_labels,
    predicted_probabilities,
    candidate_features,
    *,
    gamma=0.9,
    alpha=0.5,
    random_state=0,
):
    """Equations (4)-(6): heuristic partners and lambda'=max(lambda,1-lambda).

    candidate_features must come exclusively from labeled P in the training
    partition; this pure array helper cannot verify that provenance by itself.
    Marginal U has inclusive 1-gamma <= probability <= gamma; all other
    partners are drawn uniformly from this minibatch's P+U, including self.
    Features and candidates share a floating dtype before partner selection,
    preserving fractional candidate values even when features are integers.
    Returns diagnostics as well as mixed features and soft labels.
    """
    features = np.asarray(features)
    labels = np.asarray(pu_labels)
    probabilities = np.asarray(predicted_probabilities, dtype=float)
    candidates = np.asarray(candidate_features)
    if (
        features.ndim < 2
        or not len(features)
        or features.dtype.kind not in "iuf"
        or not np.isfinite(features).all()
    ):
        raise ValueError("features must be nonempty finite batch-first arrays")
    if labels.shape != (len(features),) or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("pu_labels must be matched binary observed labels")
    if (
        probabilities.shape != labels.shape
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("predicted probabilities must be matched finite [0,1]")
    if (
        candidates.ndim != features.ndim
        or not len(candidates)
        or candidates.shape[1:] != features.shape[1:]
        or candidates.dtype.kind not in "iuf"
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("candidate_features must be finite and match feature shape")
    if not _is_finite_real(gamma) or not 0.5 <= gamma <= 1:
        raise ValueError("gamma must be in [0.5,1]")
    if not _is_finite_real(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    # The paper explicitly states that gamma=0.5 yields an empty set, even
    # though the closed interval in Eq. (6) collapses to the singleton 0.5.
    marginal = (
        np.zeros(len(labels), dtype=bool)
        if gamma == 0.5
        else (labels == 0) & (probabilities >= 1 - gamma) & (probabilities <= gamma)
    )
    rng = np.random.RandomState(random_state)
    mix_dtype = np.result_type(features.dtype, candidates.dtype, np.float32)
    features = features.astype(mix_dtype, copy=False)
    candidates = candidates.astype(mix_dtype, copy=False)
    partners = rng.randint(0, len(features), size=len(features))
    partner_features = features[partners].copy()
    partner_labels = labels[partners].astype(float)
    candidate_positions = np.full(len(features), -1, dtype=int)
    selected = rng.randint(0, len(candidates), size=np.count_nonzero(marginal))
    candidate_positions[marginal] = selected
    partner_features[marginal] = candidates[selected]
    partner_labels[marginal] = 1.0
    lambdas = rng.beta(alpha, alpha, size=len(features))
    lambdas = np.maximum(lambdas, 1 - lambdas)
    feature_lambdas = lambdas.astype(mix_dtype).reshape(
        (len(lambdas),) + (1,) * (features.ndim - 1)
    )
    return {
        "features": feature_lambdas * features + (1 - feature_lambdas) * partner_features,
        "soft_labels": lambdas * labels + (1 - lambdas) * partner_labels,
        "lambdas": lambdas,
        "marginal_mask": marginal,
        "minibatch_partner_positions": np.where(marginal, -1, partners),
        "candidate_partner_positions": candidate_positions,
    }


def p3mix_batch_loss(logits, soft_labels, original_pu_labels, *, unlabeled_weight):
    """Paper Eq. (2)'s separate mean_P + beta*mean_U, not a pooled-size mean."""
    import torch
    from torch.nn import functional

    logits, soft_labels = logits.reshape(-1), soft_labels.reshape(-1)
    observed = original_pu_labels.reshape(-1)
    if not logits.numel() or logits.shape != soft_labels.shape or logits.shape != observed.shape:
        raise ValueError("P3Mix loss requires matched nonempty logits/labels")
    if (
        not torch.isfinite(logits).all()
        or not torch.isfinite(soft_labels).all()
        or not torch.all((soft_labels >= 0) & (soft_labels <= 1))
    ):
        raise ValueError("P3Mix logits/soft labels must be finite with labels in [0,1]")
    if (
        not torch.all((observed == 0) | (observed == 1))
        or not (observed == 1).any()
        or not (observed == 0).any()
    ):
        raise ValueError("P3Mix loss requires both observed P and U")
    if not _is_finite_real(unlabeled_weight) or unlabeled_weight <= 0:
        raise ValueError("unlabeled_weight must be finite and positive")
    losses = functional.binary_cross_entropy_with_logits(
        logits, soft_labels.detach(), reduction="none"
    )
    return losses[observed == 1].mean() + unlabeled_weight * losses[observed == 0].mean()


def _p3mix_logits(model, X):
    import torch

    # A custom network may start with an in-place operation. Keep raw x for
    # Eq. (4), teacher inputs and caller-owned scoring arrays unchanged.
    logits = model(X.clone())
    if logits.shape not in ((len(X),), (len(X), 1)):
        raise ValueError("P3Mix model must return one logit per row")
    logits = logits.reshape(-1)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("P3Mix model produced nonfinite logits")
    return logits


def _update_teacher(student, teacher, decay):
    import torch

    with torch.no_grad():
        student_state, teacher_state = student.state_dict(), teacher.state_dict()
        for name, teacher_value in teacher_state.items():
            student_value = student_state[name].to(teacher_value.device)
            if teacher_value.is_floating_point():
                teacher_value.mul_(decay).add_(student_value, alpha=1 - decay)
            else:
                teacher_value.copy_(student_value)


def p3mix_early_learning_regularizer(student_logits, teacher_logits):
    """Equation (8): mean log(1 - probability-vector agreement)."""
    import torch

    student_logits = student_logits.reshape(-1)
    teacher_logits = teacher_logits.reshape(-1).detach()
    if not student_logits.numel() or student_logits.shape != teacher_logits.shape:
        raise ValueError("P3Mix-E needs nonempty matched student/teacher logits")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError("P3Mix-E logits must be finite")
    student_positive = student_logits.sigmoid()
    teacher_positive = teacher_logits.sigmoid()
    student_targets = torch.stack((1 - student_positive, student_positive), dim=1)
    teacher_targets = torch.stack((1 - teacher_positive, teacher_positive), dim=1)
    agreement = (student_targets * teacher_targets).sum(dim=1).clamp(max=1 - 1e-7)
    return torch.log1p(-agreement).mean()


class P3MixClassifier(BasePUClassifier):
    """Positive/unlabeled classifier with heuristic positive-partner mixup.

    ``variant`` selects the paper's base method, P3Mix-E (mean teacher), or
    P3Mix-C (high-confidence pseudo-negative correction). The paper defines
    these as independent extensions, so ``e`` and ``c`` are not combined.
    This implementation uses the paper's OS P/U objective and never reads
    clean validation/test labels.
    """

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
    checkpoint_stages = ("train",)

    def __init__(
        self,
        *,
        variant="base",
        hidden_dim=128,
        max_epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        beta=1.0,
        gamma=0.9,
        pool_size=100,
        mixup_alpha=1.0,
        early_learning_weight=1.0,
        teacher_decay=0.99,
        model=None,
        encoder=None,
        random_state=0,
        device=None,
    ):
        super().__init__()
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.pool_size = pool_size
        self.mixup_alpha = mixup_alpha
        self.early_learning_weight = early_learning_weight
        self.teacher_decay = teacher_decay
        self.model = model
        self.encoder = encoder
        self.random_state = random_state
        self.device = device

    @property
    def checkpoint_epoch_count(self):
        return self.max_epochs

    @property
    def checkpoint_prediction_batch_size(self):
        return self.batch_size

    @invalidate_failed_fit
    def fit(
        self,
        X,
        y_pu,
        *,
        class_prior=None,
        sample_weight=None,
        epoch_callback=None,
        os_or_ts="os",
    ):
        import copy

        import torch
        from torch import nn
        from torch.nn import functional

        self._is_fitted = False
        if sample_weight is not None:
            raise NotImplementedError("P3Mix does not implement sample_weight")
        if os_or_ts != "os":
            raise ValueError("P3Mix implements the paper's OS P/U objective; TS is not hooked")
        if epoch_callback is not None and not callable(epoch_callback):
            raise TypeError("epoch_callback must be callable or None")
        if self.variant not in {"base", "e", "c"}:
            raise ValueError("variant must be 'base', 'e', or 'c'")
        if class_prior is not None and (
            isinstance(class_prior, bool | np.bool_)
            or not isinstance(class_prior, Real)
            or not np.isfinite(class_prior)
            or not 0 < class_prior < 1
        ):
            raise ValueError("class_prior, if supplied, must be in (0, 1)")
        if self.encoder is not None and not isinstance(self.encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            allow_nd=self.encoder is not None,
            estimator_name="P3MixClassifier",
        )
        if X.ndim not in ({2, 4} if self.encoder is not None else {2}):
            raise ValueError("P3Mix accepts 2-D features or 4-D images with an explicit encoder")
        view = build_training_view(X, y_pu, requested_view=os_or_ts)
        for name in ("hidden_dim", "max_epochs", "batch_size", "pool_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a valid positive integer")
        for name in ("learning_rate", "beta", "mixup_alpha"):
            value = getattr(self, name)
            if not _is_finite_real(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not _is_finite_real(self.gamma) or not 0.5 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0.5, 1]")
        if not _is_finite_real(self.early_learning_weight) or self.early_learning_weight < 0:
            raise ValueError("early_learning_weight must be finite and nonnegative")
        if not _is_finite_real(self.teacher_decay) or not 0 <= self.teacher_decay < 1:
            raise ValueError("teacher_decay must be in [0, 1)")
        if self.variant == "e" and self.early_learning_weight <= 0:
            raise ValueError("P3Mix-E requires early_learning_weight > 0")
        X = np.asarray(X)
        if X.dtype.kind not in "iuf" or not np.isfinite(X).all():
            raise ValueError("X must contain finite numeric values")
        with np.errstate(over="ignore"):
            X = X.astype(np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X must remain finite after float32 conversion")
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        self.device_ = resolve_device(self.device)
        self.input_shape_ = tuple(X.shape[1:])
        self.n_features_in_ = X.shape[1]
        self._X_shape_ = X.shape
        self.classes_ = np.array([0, 1])

        if self.encoder is not None:
            from ._validation import validate_encoder_features

            encoder = copy.deepcopy(self.encoder).to(self.device_, dtype=torch.float32)
            encoder.requires_grad_(True).eval()
            with torch.no_grad():
                width = validate_encoder_features(
                    encoder(torch.as_tensor(X[:1].copy(), device=self.device_)),
                    encoder_param_name="encoder",
                )
            head = (
                copy.deepcopy(self.model)
                if self.model is not None
                else nn.Sequential(
                    nn.Linear(width, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)
                )
            )
            self.model_ = nn.Sequential(encoder, head).to(self.device_, dtype=torch.float32)
        elif self.model is not None:
            if not isinstance(self.model, nn.Module):
                raise TypeError("model must be a torch.nn.Module")
            self.model_ = copy.deepcopy(self.model).to(self.device_, dtype=torch.float32)
        else:
            self.model_ = nn.Sequential(
                nn.Linear(X.shape[1], self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)
            ).to(self.device_)
        if not any(parameter.requires_grad for parameter in self.model_.parameters()):
            raise ValueError("model must contain trainable parameters")
        self.model_.eval()
        with torch.no_grad():
            _p3mix_logits(self.model_, torch.as_tensor(X[:1], device=self.device_))
        self.encoder_ = self.model_[0] if self.encoder is not None else None
        self.teacher_ = copy.deepcopy(self.model_).eval() if self.variant == "e" else None
        if self.teacher_ is not None:
            self.teacher_.requires_grad_(False)
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        data = torch.as_tensor(X, device="cpu")
        labels = np.asarray(y_pu, dtype=np.int64)
        p_idx, u_idx = view.positive_positions.copy(), view.native_unlabeled_positions.copy()
        self.train_positive_indices_, self.train_unlabeled_indices_ = p_idx, u_idx
        self.optimizer_steps_, self.epochs_completed_ = 0, 0
        self.positive_rows_seen_, self.unlabeled_rows_seen_ = 0, 0
        self.history_ = {
            key: []
            for key in (
                "loss",
                "mixup_loss",
                "early_learning_loss",
                "marginal_count",
                "corrected_count",
                "candidate_count",
            )
        }
        self.epoch_provenance_ = []
        initial = rng.choice(p_idx, size=min(self.pool_size, len(p_idx)), replace=False)
        candidate_idx = np.asarray(initial, dtype=int)
        self.candidate_training_positions_ = candidate_idx.copy()
        self.data_device_policy_ = "cpu_stream"
        self.training_provenance_ = {
            "method": "p3mix",
            "variant": self.variant,
            "run_view": "os",
            "calibration_applied": view.calibration_applied,
            "candidate_source": "observed_labeled_train_positive_only",
            "candidate_initialization": "uniform_random_without_replacement",
            "candidate_refresh": "top_k_binary_entropy_after_each_epoch",
            "batch_construction": "independent_shuffled_P_and_U_batches_with_shorter_role_cycled",
            "batch_construction_is_engineering_choice": True,
            "observed_positive_rows": len(p_idx),
            "observed_unlabeled_rows": len(u_idx),
            "positive_rows_seen": 0,
            "unlabeled_rows_seen": 0,
            "configured_batch_size_per_role": self.batch_size,
            "maximum_rows_per_training_forward": 2 * self.batch_size,
            "forward_input_isolation": "private_tensor_clone",
            "beta": float(self.beta),
            "gamma": float(self.gamma),
            "mixup_alpha": float(self.mixup_alpha),
            "pool_size": self.pool_size,
            "pseudo_negative_correction": self.variant == "c",
            "mean_teacher_regularization": self.variant == "e",
            "early_learning_weight": float(self.early_learning_weight),
            "teacher_decay": float(self.teacher_decay),
            "teacher_decay_is_engineering_choice": self.variant == "e",
            "teacher_update_frequency": "end_of_epoch",
            "teacher_initialization": "copy_of_student",
            "learning_rate": float(self.learning_rate),
            "max_epochs": self.max_epochs,
            "class_prior_used": False,
            "clean_labels_read": False,
            "external_selection_data_read": False,
            "formal_admission": False,
            "data_device_policy": "cpu_stream",
            "source_pdf": "30_who_is_your_right_mixup_partne.pdf",
        }
        self.checkpoint_stage_ = "train"
        self.checkpoint_stage_epoch_ = 0
        self.checkpoint_round_ = None
        self.model_.train()
        for epoch in range(1, self.max_epochs + 1):
            p_batches = np.array_split(
                rng.permutation(p_idx), range(self.batch_size, len(p_idx), self.batch_size)
            )
            u_batches = np.array_split(
                rng.permutation(u_idx), range(self.batch_size, len(u_idx), self.batch_size)
            )
            epoch_losses, mix_losses, elr_losses = [], [], []
            marginal_total = corrected_total = 0
            positive_rows_seen = unlabeled_rows_seen = 0
            for step in range(max(len(p_batches), len(u_batches))):
                p = p_batches[step % len(p_batches)]
                u = u_batches[step % len(u_batches)]
                row_idx = np.r_[p, u]
                batch = data[row_idx].to(self.device_)
                batch_labels = labels[row_idx].copy()
                with torch.no_grad():
                    current_logits = _p3mix_logits(self.model_, batch)
                    probabilities = current_logits.sigmoid().cpu().numpy()
                corrected = (
                    (batch_labels == 0) & (probabilities > self.gamma)
                    if self.variant == "c"
                    else np.zeros(len(row_idx), dtype=bool)
                )
                batch_labels[corrected] = 1
                corrected_total += int(corrected.sum())
                mixed = p3mix_heuristic_batch(
                    batch.detach().cpu().numpy(),
                    batch_labels,
                    probabilities,
                    data[candidate_idx].numpy(),
                    gamma=self.gamma,
                    alpha=self.mixup_alpha,
                    random_state=int(rng.randint(0, 2**31)),
                )
                marginal_total += int(mixed["marginal_mask"].sum())
                mixed_x = torch.as_tensor(
                    mixed["features"], dtype=torch.float32, device=self.device_
                )
                mixed_y = torch.as_tensor(
                    mixed["soft_labels"], dtype=torch.float32, device=self.device_
                )
                logits = _p3mix_logits(self.model_, mixed_x)
                # The two means in Eq. (2) follow the original P/U roles even
                # when P3Mix-C changes a U row's soft target to positive.
                p_role = torch.arange(len(p), device=self.device_)
                u_role = torch.arange(len(p), len(row_idx), device=self.device_)
                losses = functional.binary_cross_entropy_with_logits(
                    logits, mixed_y.detach(), reduction="none"
                )
                mix_loss = losses[p_role].mean() + self.beta * losses[u_role].mean()
                early_loss = torch.zeros((), device=self.device_)
                if self.variant == "e":
                    with torch.no_grad():
                        teacher_logits = _p3mix_logits(self.teacher_, mixed_x)
                    early_loss = p3mix_early_learning_regularizer(logits, teacher_logits)
                loss = mix_loss + (
                    self.early_learning_weight * early_loss if self.variant == "e" else 0
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("P3Mix objective became nonfinite")
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
                self.optimizer_steps_ += 1
                positive_rows_seen += len(p)
                unlabeled_rows_seen += len(u)
                epoch_losses.append(float(loss.detach().cpu()))
                mix_losses.append(float(mix_loss.detach().cpu()))
                elr_losses.append(float(early_loss.detach().cpu()))
            self.model_.eval()
            with torch.no_grad():
                positive_probabilities = np.concatenate(
                    [
                        _p3mix_logits(
                            self.model_,
                            data[p_idx[start : start + self.batch_size]].to(self.device_),
                        )
                        .sigmoid()
                        .cpu()
                        .numpy()
                        for start in range(0, len(p_idx), self.batch_size)
                    ]
                )
            positions = p3mix_candidate_positions(positive_probabilities, pool_size=self.pool_size)
            candidate_idx = p_idx[positions]
            self.candidate_training_positions_ = candidate_idx.copy()
            if self.teacher_ is not None:
                _update_teacher(self.model_, self.teacher_, self.teacher_decay)
            self.model_.train()
            self.epochs_completed_ = epoch
            self.checkpoint_stage_epoch_ = epoch
            self.positive_rows_seen_ += positive_rows_seen
            self.unlabeled_rows_seen_ += unlabeled_rows_seen
            self.training_provenance_.update(
                positive_rows_seen=self.positive_rows_seen_,
                unlabeled_rows_seen=self.unlabeled_rows_seen_,
            )
            self.history_["loss"].append(float(np.mean(epoch_losses)))
            self.history_["mixup_loss"].append(float(np.mean(mix_losses)))
            self.history_["early_learning_loss"].append(float(np.mean(elr_losses)))
            self.history_["marginal_count"].append(marginal_total)
            self.history_["corrected_count"].append(corrected_total)
            self.history_["candidate_count"].append(len(candidate_idx))
            self.epoch_provenance_.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": self.optimizer_steps_,
                    "positive_rows_seen": positive_rows_seen,
                    "unlabeled_rows_seen": unlabeled_rows_seen,
                    "positive_rows_reused": positive_rows_seen - len(p_idx),
                    "unlabeled_rows_reused": unlabeled_rows_seen - len(u_idx),
                }
            )
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        self.model_.eval()
        self._is_fitted = True
        return self

    def _decision_function(self, X):
        import torch
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=np.float32, allow_nd=True, ensure_min_samples=0)
        if tuple(X.shape[1:]) != self.input_shape_:
            raise ValueError("P3Mix prediction input shape differs from training")
        if not len(X):
            return np.empty(0, dtype=np.float32)
        modes = [(module, module.training) for module in self.model_.modules()]
        try:
            self.model_.eval()
            with torch.no_grad():
                return np.concatenate(
                    [
                        _p3mix_logits(
                            self.model_,
                            torch.as_tensor(
                                X[start : start + self.batch_size], device=self.device_
                            ),
                        )
                        .cpu()
                        .numpy()
                        for start in range(0, len(X), self.batch_size)
                    ]
                )
        finally:
            for module, training in modes:
                module.training = training

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        from scipy.special import expit

        self._check_is_fitted()
        probabilities = expit(self._decision_function(X))
        return np.column_stack((1 - probabilities, probabilities))
