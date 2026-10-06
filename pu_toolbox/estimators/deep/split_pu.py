"""Split-PU: source-derived easy/hard distillation for PU data.

This adaptation preserves the nnPU teacher, prediction-disagreement split,
Jensen-Shannon easy loss and dual-source hard consistency of Xu et al. It
supports injected CNN representations but does not reproduce the official
image augmentation, SimSiam or full CNN experiment.
"""

# ruff: noqa: N803, N806, N812

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
from ...losses.nnpu import _nnpu_train_step


def splitpu_js_loss(student_logits, teacher_logits, *, teacher_weight: float = 0.7):
    """Official weighted Jensen-Shannon loss for Bernoulli soft targets."""
    import torch
    from torch.nn import functional as F

    if not 0 < teacher_weight < 1:
        raise ValueError("teacher_weight must be in (0, 1)")
    teacher = torch.sigmoid(teacher_logits.detach())
    student = torch.sigmoid(student_logits)
    mixture = teacher_weight * teacher + (1 - teacher_weight) * student
    eps = torch.finfo(mixture.dtype).eps
    teacher = teacher.clamp(eps, 1 - eps)
    student = student.clamp(eps, 1 - eps)
    mixture = mixture.clamp(eps, 1 - eps)
    kl_teacher = F.kl_div(mixture.log(), teacher, reduction="none") + F.kl_div(
        torch.log1p(-mixture), 1 - teacher, reduction="none"
    )
    kl_student = F.kl_div(mixture.log(), student, reduction="none") + F.kl_div(
        torch.log1p(-mixture), 1 - student, reduction="none"
    )
    scale = -1.0 / ((1 - teacher_weight) * np.log(1 - teacher_weight))
    return (scale * (teacher_weight * kl_teacher + (1 - teacher_weight) * kl_student)).mean()


def _network(n_features, hidden_dim, *, encoder=None, feature_layer=None):
    import torch

    head = torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )
    if encoder is None:
        return head
    model = torch.nn.Sequential(copy.deepcopy(encoder), *list(head.children()))
    model._splitpu_feature_layer = feature_layer
    return model


@contextmanager
def _singleton_batchnorm(model, X):
    """Use running BN statistics for one-row groups without dropping rows.

    A valid hard/easy group can contain one row and CNN spatial maps can
    shrink to 1x1. BN affine/encoder gradients remain enabled. Restore all
    layer modes even when forward raises; larger groups are unchanged.
    """
    import torch

    layers = (
        [
            layer
            for layer in model.modules()
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) and layer.training
        ]
        if len(X) == 1
        else []
    )
    try:
        for layer in layers:
            layer.eval()
        yield
    finally:
        for layer in layers:
            layer.train()


def _score(model, X):
    with _singleton_batchnorm(model, X):
        scores = model(X)
    if scores.shape == (len(X), 1):
        return scores[:, 0]
    raise ValueError("model must output one raw logit per row")


def _features(model, X):
    if hasattr(model, "_splitpu_feature_layer"):
        import torch

        outputs = []
        layer = model[0].get_submodule(model._splitpu_feature_layer)
        # Preserve the layer value before a downstream in-place activation.
        # clone keeps gradients, unlike detach, and is only batch-sized.
        handle = layer.register_forward_hook(
            lambda _module, _args, output: outputs.append(
                output.clone() if isinstance(output, torch.Tensor) else output
            )
        )
        try:
            with _singleton_batchnorm(model[0], X):
                high = model[0](X)
        finally:
            handle.remove()
        if len(outputs) != 1:
            raise ValueError("encoder_feature_layer must execute exactly once per encoder forward")
        from ._validation import validate_encoder_features

        validate_encoder_features(high, encoder_param_name="encoder")
        if high.shape[0] != len(X):
            raise ValueError("encoder output must preserve the input batch size")
        low = outputs[0]
        if (
            not isinstance(low, torch.Tensor)
            or low.ndim < 2
            or low.shape[0] != len(X)
            or low[0].numel() == 0
            or not torch.isfinite(low).all()
        ):
            raise ValueError("encoder_feature_layer must return finite batched features")
        return low, high
    low = model[:2](X)
    high = model[2:4](low)
    return low, high


def _batched_score(model, data, batch_size, device):
    """Detached CPU scores without a full-data GPU tensor or BN updates."""
    import torch

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return torch.cat(
                [_score(model, batch.to(device)).cpu() for batch in data.split(batch_size)]
            )
    finally:
        model.train(was_training)


class SplitPUClassifier(BasePUClassifier):
    """PU-only Split-PU adaptation with independent MLP/CNN training stages."""

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
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
    checkpoint_stages = ("teacher", "splitter", "student")

    @property
    def checkpoint_prediction_batch_size(self):
        """Preserve inference chunks; changing them can change float32 scores."""
        return self.batch_size

    @property
    def checkpoint_epoch_count(self):
        """Upper bound across teacher, early-stopped splitter and student rounds."""
        return self.teacher_epochs + self.split_epochs + self.rounds * self.student_epochs

    def __init__(
        self,
        class_prior: float | None = None,
        *,
        encoder=None,
        encoder_feature_layer: str | None = None,
        hidden_dim: int = 100,
        teacher_epochs: int = 10,
        split_epochs: int = 10,
        student_epochs: int = 10,
        rounds: int = 2,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        agreement_threshold: float = 0.92,
        js_teacher_weight: float = 0.7,
        hard_weight: float = 0.3,
        feature_weight: float = 0.3,
        similarity_weight: float = 0.1,
        noise_std: float = 0.05,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.encoder = encoder
        self.encoder_feature_layer = encoder_feature_layer
        self.hidden_dim = hidden_dim
        self.teacher_epochs = teacher_epochs
        self.split_epochs = split_epochs
        self.student_epochs = student_epochs
        self.rounds = rounds
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.agreement_threshold = agreement_threshold
        self.js_teacher_weight = js_teacher_weight
        self.hard_weight = hard_weight
        self.feature_weight = feature_weight
        self.similarity_weight = similarity_weight
        self.noise_std = noise_std
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
    ) -> SplitPUClassifier:
        import torch
        from torch.nn import functional as F

        if sample_weight is not None:
            raise NotImplementedError("Split-PU does not implement sample_weight")
        self._is_fitted = False
        X, y_pu = validate_pu_X_y(
            X, y_pu, accept_sparse=False, allow_nd=True, estimator_name="SplitPUClassifier"
        )
        if X.ndim not in (2, 4):
            raise ValueError("Split-PU requires 2-D features or 4-D images")
        if X.ndim == 4 and self.encoder is None:
            raise ValueError("Split-PU 4-D images require encoder; images are not flattened")
        if self.encoder is not None and not isinstance(self.encoder, torch.nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        if self.encoder_feature_layer is not None and self.encoder is None:
            raise ValueError("encoder_feature_layer requires encoder")
        view = build_training_view(X, y_pu, requested_view=os_or_ts)
        prior = self.class_prior if class_prior is None else class_prior
        if prior is None or not np.isfinite(prior) or not 0 < prior < 1:
            raise ValueError("class_prior must be in (0, 1)")
        for name in (
            "hidden_dim",
            "teacher_epochs",
            "split_epochs",
            "student_epochs",
            "rounds",
            "batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        for name in ("agreement_threshold", "js_teacher_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be in (0, 1)")
        for name in ("hard_weight", "feature_weight", "similarity_weight", "noise_std"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("X must be finite numeric")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X must remain finite after float32 conversion")
        p_idx = np.array(view.positive_positions, copy=True)
        u_idx = np.array(view.native_unlabeled_positions, copy=True)
        if len(u_idx) < 2:
            raise ValueError("Split-PU requires at least two original U rows for easy/hard groups")
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        device = resolve_device(self.device)
        data = torch.as_tensor(X)
        width, feature_layer = X.shape[1], None
        encoder_template = None
        if self.encoder is not None:
            from ._validation import validate_encoder_features

            encoder_template = copy.deepcopy(self.encoder).to(device=device, dtype=torch.float32)
            encoder_template.requires_grad_(True).eval()
            if self.encoder_feature_layer is not None:
                if (
                    not isinstance(self.encoder_feature_layer, str)
                    or not self.encoder_feature_layer
                ):
                    raise ValueError("encoder_feature_layer must be a nonempty module path")
                feature_layer = self.encoder_feature_layer
                try:
                    encoder_template.get_submodule(feature_layer)
                except (AttributeError, KeyError) as exc:
                    raise ValueError(
                        "encoder_feature_layer does not identify an encoder module"
                    ) from exc
            else:
                modules = list(encoder_template.named_modules())
                feature_layer = next(
                    (name for name, module in modules if isinstance(module, torch.nn.MaxPool2d)),
                    None,
                )
                if feature_layer is None:
                    feature_layer = next(
                        (
                            name
                            for name, module in modules
                            if name and isinstance(module, torch.nn.Conv2d | torch.nn.Linear)
                        ),
                        None,
                    )
                if feature_layer is None:
                    raise ValueError("encoder requires an explicit encoder_feature_layer")
            with torch.no_grad():
                width = validate_encoder_features(
                    encoder_template(data[:1].to(device)), encoder_param_name="encoder"
                )
                probe_model = (
                    _network(
                        width,
                        self.hidden_dim,
                        encoder=encoder_template,
                        feature_layer=feature_layer,
                    )
                    .to(device)
                    .eval()
                )
                _features(probe_model, data[:1].to(device))
            del probe_model
        self.encoder_feature_layer_ = feature_layer
        self.input_shape_ = X.shape[1:]
        self.n_features_in_ = X.shape[1]
        self.n_positive_, self.n_unlabeled_ = len(p_idx), len(u_idx)
        self.n_loss_unlabeled_ = len(view.loss_unlabeled_positions)
        self.training_view_ = os_or_ts
        self.calibration_applied_ = view.calibration_applied
        self._X_shape_ = X.shape
        self._class_prior = float(prior)
        self.optimizer_steps_ = 0
        self.history_ = {
            "teacher_risk": [],
            "split_agreement": [],
            "student_loss": [],
            "round_weights": [],
        }
        self._is_fitted = False
        epoch = 0

        teacher = _network(
            width, self.hidden_dim, encoder=encoder_template, feature_layer=feature_layer
        ).to(device)
        self.encoder_ = teacher[0] if encoder_template is not None else None
        self.model_ = teacher
        optimizer = torch.optim.Adam(teacher.parameters(), lr=self.learning_rate)
        for teacher_epoch in range(self.teacher_epochs):
            teacher.train()
            p_order, u_order = rng.permutation(p_idx), rng.permutation(u_idx)
            steps = max(
                int(np.ceil(len(p_idx) / self.batch_size)),
                int(np.ceil(len(u_idx) / self.batch_size)),
            )
            risks = []
            for step in range(steps):
                p = p_order[(step * self.batch_size) % len(p_idx) :][: self.batch_size]
                u = u_order[(step * self.batch_size) % len(u_idx) :][: self.batch_size]
                pos = _score(teacher, data[p].to(device))
                unl = _score(teacher, data[u].to(device))
                # The teacher's nnPU marginal term is the only unlabeled risk
                # role.  Easy/hard splitting below must use original U only.
                loss_unl = torch.cat((unl, pos)) if view.calibration_applied else unl
                loss, info = _nnpu_train_step(
                    torch.sigmoid(-pos).mean(),
                    torch.sigmoid(pos).mean(),
                    torch.sigmoid(loss_unl).mean(),
                    class_prior=prior,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Split-PU teacher loss became non-finite")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.optimizer_steps_ += 1
                risks.append(info["nnpu_risk"])
            self.history_["teacher_risk"].append(float(np.mean(risks)))
            self._is_fitted = True
            self.checkpoint_stage_ = "teacher"
            self.checkpoint_stage_epoch_ = teacher_epoch + 1
            self.checkpoint_round_ = None
            if epoch_callback is not None:
                epoch_callback(epoch, self)
            epoch += 1
        self.teacher_ = copy.deepcopy(teacher).eval()
        for param in self.teacher_.parameters():
            param.requires_grad_(False)
        self.teacher_encoder_ = self.teacher_[0] if encoder_template is not None else None

        splitter = _network(
            width, self.hidden_dim, encoder=encoder_template, feature_layer=feature_layer
        ).to(device)
        self.splitter_encoder_ = splitter[0] if encoder_template is not None else None
        self.encoder_ = self.splitter_encoder_
        self.splitter_ = splitter
        self.model_ = splitter
        optimizer = torch.optim.SGD(splitter.parameters(), lr=self.learning_rate)
        # The teacher is frozen, so its targets are computed once on CPU.
        target = (_batched_score(self.teacher_, data, self.batch_size, device) > 0).float()
        for split_epoch in range(self.split_epochs):
            splitter.train()
            for indices in _batches(rng, len(X), self.batch_size):
                loss = F.binary_cross_entropy_with_logits(
                    _score(splitter, data[indices].to(device)), target[indices].to(device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.optimizer_steps_ += 1
            splitter.eval()
            split_scores = _batched_score(splitter, data[u_idx], self.batch_size, device)
            agreement = float(((split_scores > 0).float() == target[u_idx]).float().mean())
            self.history_["split_agreement"].append(agreement)
            self._is_fitted = True
            self.checkpoint_stage_ = "splitter"
            self.checkpoint_stage_epoch_ = split_epoch + 1
            self.checkpoint_round_ = None
            if epoch_callback is not None:
                epoch_callback(epoch, self)
            epoch += 1
            if agreement >= self.agreement_threshold:
                break

        teacher_scores = _batched_score(self.teacher_, data[u_idx], self.batch_size, device)
        teacher_sign = teacher_scores > 0
        splitter_sign = split_scores > 0
        hard_mask = (teacher_sign != splitter_sign).numpy()
        # A tiny/noisy dataset may have perfect agreement; retain one
        # lowest-margin U for the hard branch instead of dividing by zero.
        margins = teacher_scores.abs().numpy()
        if not np.any(hard_mask):
            hard_mask[int(np.argmin(margins))] = True
        if np.all(hard_mask):
            hard_mask[int(np.argmax(margins))] = False
        easy_idx, hard_idx = u_idx[~hard_mask], u_idx[hard_mask]
        self.n_easy_, self.n_hard_ = len(easy_idx), len(hard_idx)
        current_teacher = self.teacher_
        for round_index in range(self.rounds):
            student = _network(
                width, self.hidden_dim, encoder=encoder_template, feature_layer=feature_layer
            ).to(device)
            self.encoder_ = student[0] if encoder_template is not None else None
            self.model_ = student
            optimizer = torch.optim.Adam(student.parameters(), lr=self.learning_rate)
            # Official main.py: first (hard=.3, sim=.1, feat=.3), then
            # (hard=.01, sim=0, feat=0). Keep user tuning on the first pass.
            hard_weight = self.hard_weight if round_index == 0 else 0.01
            feature_weight = self.feature_weight if round_index == 0 else 0.0
            similarity_weight = self.similarity_weight if round_index == 0 else 0.0
            self.history_["round_weights"].append(
                {"hard": hard_weight, "feature": feature_weight, "similarity": similarity_weight}
            )
            for student_epoch in range(self.student_epochs):
                student.train()
                losses = []
                for p, easy, hard in _three_group_batches(
                    rng, p_idx, easy_idx, hard_idx, self.batch_size
                ):
                    pos = data[p].to(device)
                    easy_data, hard_data = data[easy].to(device), data[hard].to(device)
                    with torch.no_grad():
                        teacher_easy = _score(current_teacher, easy_data)
                        teacher_low, _ = _features(current_teacher, hard_data)
                    positive_loss = F.binary_cross_entropy_with_logits(
                        _score(student, pos), torch.ones(len(pos), device=device)
                    )
                    easy_loss = splitpu_js_loss(
                        _score(student, easy_data),
                        teacher_easy,
                        teacher_weight=self.js_teacher_weight,
                    )
                    weak = hard_data + torch.randn_like(hard_data) * self.noise_std
                    strong = hard_data + torch.randn_like(hard_data) * (2 * self.noise_std)
                    weak_logits, strong_logits = _score(student, weak), _score(student, strong)
                    weak_prob = torch.sigmoid(weak_logits.detach())
                    mask = (weak_prob >= 0.95).float()
                    hard_loss = (
                        F.binary_cross_entropy_with_logits(
                            strong_logits, weak_prob, reduction="none"
                        )
                        * mask
                    ).mean()
                    low, high = _features(student, weak)
                    _, high_strong = _features(student, strong)
                    feature_loss = F.mse_loss(low, teacher_low.detach())
                    sim_loss = 1 - F.cosine_similarity(high, high_strong, dim=1).mean()
                    loss = (
                        positive_loss
                        + easy_loss
                        + hard_weight * hard_loss
                        + feature_weight * feature_loss
                        + similarity_weight * sim_loss
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Split-PU student loss became non-finite")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.optimizer_steps_ += 1
                    losses.append(float(loss.detach()))
                self.history_["student_loss"].append(float(np.mean(losses)))
                self._is_fitted = True
                self.checkpoint_stage_ = "student"
                self.checkpoint_stage_epoch_ = student_epoch + 1
                self.checkpoint_round_ = round_index + 1
                if epoch_callback is not None:
                    epoch_callback(epoch, self)
                epoch += 1
            current_teacher = copy.deepcopy(student).eval()
            for param in current_teacher.parameters():
                param.requires_grad_(False)
        self.classes_ = np.array([0, 1])
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X)
        if X.ndim != len(self.input_shape_) + 1 or X.shape[1:] != self.input_shape_:
            raise ValueError("Split-PU prediction requires fitted 2-D/4-D feature shape")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("Split-PU prediction must be finite numeric")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("Split-PU prediction must remain finite after float32 conversion")
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        device = next(self.model_.parameters()).device
        return _batched_score(self.model_, torch.as_tensor(X), self.batch_size, device).numpy()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self._decision_function(X) >= 0).astype(int)


def _batches(rng, n, size):
    order = rng.permutation(n)
    return [order[i : i + size] for i in range(0, n, size)]


def _three_group_batches(rng, positive, easy, hard, size):
    steps = max(
        int(np.ceil(len(positive) / size)),
        int(np.ceil(len(easy) / size)),
        int(np.ceil(len(hard) / size)),
    )
    groups = [rng.permutation(group) for group in (positive, easy, hard)]
    for step in range(steps):
        yield tuple(group[(step * size) % len(group) :][:size] for group in groups)
