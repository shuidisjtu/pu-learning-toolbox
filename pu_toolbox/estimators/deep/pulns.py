# ruff: noqa: N803, N806
"""Paper-derived MLP/CNN PULNS with independent clean reward support.

Luo et al., AAAI 2021, pp. 8786--8788. REINFORCE negative selection is
not PU-only: clean positive/negative support accuracy affects training.
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


def pulns_intermediate_reward(classifier_logits, actions):
    """Paper's action-dependent clipped log odds; 1 selects a negative."""
    import torch

    logits, actions = classifier_logits.reshape(-1), actions.reshape(-1)
    if logits.shape != actions.shape or not logits.numel():
        raise ValueError("PULNS rewards need nonempty matched logits/actions")
    if not torch.isfinite(logits).all() or not torch.all((actions == 0) | (actions == 1)):
        raise ValueError("PULNS rewards require finite logits and binary actions")
    return ((1 - 2 * actions) * logits).clamp(-1, 1).detach()


def pulns_discounted_returns(rewards, terminal_reward, *, discount, terminal_weight):
    """v_i=sum_{t>=i} beta^(t-i) r_t + alpha r_terminal (terminal not discounted)."""
    import torch

    if not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("discount must be in [0, 1]")
    if not np.isfinite(terminal_weight) or terminal_weight < 0:
        raise ValueError("terminal_weight must be finite and nonnegative")
    if not np.isfinite(terminal_reward) or not rewards.numel() or not torch.isfinite(rewards).all():
        raise ValueError("PULNS returns need nonempty finite rewards")
    rewards = rewards.reshape(-1).detach()
    result = torch.empty_like(rewards)
    future = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for i in range(len(rewards) - 1, -1, -1):
        future = rewards[i] + discount * future
        result[i] = future + terminal_weight * terminal_reward
    return result


class PULNSClassifier(BasePUClassifier):
    """Sequential RL selection with an explicitly budgeted clean support set.

    support_data is consumed for terminal rewards and internal best-model
    choice. It must not be the experiment's PA/OA selection or test partition.
    Sample IDs can verify disjointness; absent IDs leave that claim unverified.
    """

    family = AlgorithmFamily.DEEP_PU
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = False
    requires_clean_support = True
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
        hidden_dim=32,
        pretrain_epochs=10,
        episodes=20,
        classifier_epochs=1,
        batch_size=128,
        learning_rate=1e-3,
        selector_learning_rate=1e-3,
        discount=0.9,
        terminal_weight=1.0,
        encoder=None,
        random_state=0,
        device=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pretrain_epochs = pretrain_epochs
        self.episodes = episodes
        self.classifier_epochs = classifier_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.selector_learning_rate = selector_learning_rate
        self.discount = discount
        self.terminal_weight = terminal_weight
        self.encoder = encoder
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        X,
        y_pu,
        *,
        class_prior=None,
        sample_weight=None,
        os_or_ts="os",
        support_data=None,
        train_indices=None,
        support_indices=None,
    ):
        import torch
        from torch import nn

        self._is_fitted = False
        if sample_weight is not None:
            raise NotImplementedError("PULNS does not implement sample_weight")
        if support_data is None:
            raise ValueError("PULNS requires independent clean support_data; PA-ineligible")
        if os_or_ts != "os":
            raise ValueError(
                "PULNS has no justified ts risk substitution; clean reward budget required"
            )
        X, y_pu = validate_pu_X_y(
            X, y_pu, accept_sparse=False, allow_nd=True, estimator_name="PULNSClassifier"
        )
        if X.ndim not in (2, 4):
            raise ValueError("PULNS supports 2-D features or 4-D NCHW images")
        if X.ndim == 4 and self.encoder is None:
            raise ValueError("PULNS 4-D images require an explicit encoder; no flattening")
        if self.encoder is not None and not isinstance(self.encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        view = build_training_view(X, y_pu, requested_view="os")
        if not isinstance(support_data, tuple) or len(support_data) != 2:
            raise ValueError("support_data must be (X_support, y_clean)")
        support_X, support_y = map(np.asarray, support_data)
        if support_X.ndim != X.ndim or support_X.shape[1:] != X.shape[1:] or not len(support_X):
            raise ValueError("support_data feature shape must match X")
        if support_y.shape != (len(support_X),) or set(np.unique(support_y)) != {0, 1}:
            raise ValueError("support_data labels must contain both clean classes {0, 1}")
        X, support_X = np.asarray(X, dtype=np.float32), np.asarray(support_X, dtype=np.float32)
        if not np.isfinite(X).all() or not np.isfinite(support_X).all():
            raise ValueError("PULNS features must remain finite in float32")
        self.support_isolation_status_ = "unverified_caller_responsibility"
        if (train_indices is None) != (support_indices is None):
            raise ValueError("provide both train_indices and support_indices to verify isolation")
        if train_indices is not None:
            train_ids, support_ids = np.asarray(train_indices), np.asarray(support_indices)
            if train_ids.shape != (len(X),) or support_ids.shape != (len(support_X),):
                raise ValueError("sample identity shapes must match their respective inputs")
            if len(np.unique(train_ids)) != len(X) or len(np.unique(support_ids)) != len(support_X):
                raise ValueError("sample identities must be unique within each partition")
            if np.intersect1d(train_ids, support_ids).size:
                raise ValueError("clean support identities overlap PU training")
            self.support_isolation_status_ = "train_support_ids_disjoint"
        for name in (
            "hidden_dim",
            "pretrain_epochs",
            "episodes",
            "classifier_epochs",
            "batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "selector_learning_rate"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if class_prior is not None and (not np.isfinite(class_prior) or not 0 < class_prior < 1):
            raise ValueError("class_prior, if supplied, must be in (0, 1)")
        # Validate discount/alpha before any model training.
        pulns_discounted_returns(
            torch.zeros(1), 0.0, discount=self.discount, terminal_weight=self.terminal_weight
        )
        self.device_ = resolve_device(self.device)
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        width = X.shape[1]
        self.encoder_ = None
        if self.encoder is not None:
            self.encoder_ = copy.deepcopy(self.encoder).to(self.device_, dtype=torch.float32)
            self.encoder_.requires_grad_(True).eval()
            with torch.no_grad():
                features = self.encoder_(torch.as_tensor(X[:1], device=self.device_))
            width = validate_encoder_features(features, encoder_param_name="encoder")
            if features.shape[0] != 1:
                raise ValueError("encoder must preserve the input batch dimension")
        head = [nn.Linear(width, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)]
        # Last hidden layer remains the selector representation in both paths.
        layers = head if self.encoder_ is None else [self.encoder_, *head]
        self.model_ = nn.Sequential(*layers).to(self.device_)
        self.selector_ = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(self.device_)
        selector_optimizer = torch.optim.Adam(
            self.selector_.parameters(), lr=self.selector_learning_rate
        )
        data = torch.as_tensor(X)
        self.optimizer_steps_ = 0
        self.stage_optimizer_steps_ = dict.fromkeys(
            ("pretrain", "reward_probe", "policy", "classifier"), 0
        )
        self._training_stage_ = "pretrain"
        self.input_shape_ = tuple(X.shape[1:])
        self._train_network(self.model_, data, y_pu, rng, self.pretrain_epochs)
        positive, unlabeled = view.positive_positions, view.native_unlabeled_positions
        # TrainingView owns read-only identity arrays; torch indexing must not borrow them.
        positive, unlabeled = positive.copy(), unlabeled.copy()
        self.initial_support_accuracy_ = self._support_accuracy(self.model_, support_X, support_y)
        self.best_support_accuracy_ = self.initial_support_accuracy_
        self.reward_baseline_ = self.initial_support_accuracy_
        best = copy.deepcopy(self.model_.state_dict())
        self.history_ = {
            "episode": [],
            "probe_support_accuracy": [],
            "terminal_reward": [],
            "support_accuracy": [],
            "policy_loss": [],
            "selected_negatives": [],
            "optimizer_steps": [],
        }
        self.empty_negative_episodes_ = 0
        self.n_features_in_, self._X_shape_ = X.shape[1], X.shape
        self._class_prior = None
        self.n_support_ = len(support_X)
        self.training_view_, self.calibration_applied_ = "os", False
        self.classes_ = np.array([0, 1])
        for episode in range(self.episodes):
            self.model_.eval()
            representation, raw_u = self._episode_features(data, unlabeled)
            p_mean = representation[positive].mean(0)
            u_mean = representation[unlabeled].mean(0)
            selected, log_policy, rewards = self._select(
                representation, positive, unlabeled, p_mean, u_mean, raw_u, rng, record=True
            )
            probe = copy.deepcopy(self.model_)
            self._training_stage_ = "reward_probe"
            self._train_selected(probe, data, positive, selected, rng)
            probe_accuracy = self._support_accuracy(probe, support_X, support_y)
            del probe
            terminal = probe_accuracy - self.reward_baseline_
            returns = pulns_discounted_returns(
                torch.stack(rewards),
                terminal,
                discount=self.discount,
                terminal_weight=self.terminal_weight,
            )
            loss = -(torch.stack(log_policy) * returns).sum()
            if not torch.isfinite(loss):
                raise ValueError("PULNS policy loss became non-finite")
            selector_optimizer.zero_grad()
            loss.backward()
            selector_optimizer.step()
            self.optimizer_steps_ += 1
            self.stage_optimizer_steps_["policy"] += 1
            # Resample using the updated policy, as Algorithm 1 line 13 requires.
            with torch.no_grad():
                selected, _, _ = self._select(
                    representation, positive, unlabeled, p_mean, u_mean, raw_u, rng, record=False
                )
            self.selected_negative_indices_ = np.asarray(selected, dtype=int)
            if not len(selected):
                self.empty_negative_episodes_ += 1  # do not fabricate a forced negative
            self._training_stage_ = "classifier"
            self._train_selected(self.model_, data, positive, selected, rng)
            accuracy = self._support_accuracy(self.model_, support_X, support_y)
            if accuracy > self.best_support_accuracy_:
                best = copy.deepcopy(self.model_.state_dict())
            self.best_support_accuracy_ = max(self.best_support_accuracy_, accuracy)
            # z_l belongs to the reward probe, not the post-policy classifier.
            self.reward_baseline_ = max(self.reward_baseline_, probe_accuracy)
            self.history_["episode"].append(episode)
            self.history_["probe_support_accuracy"].append(probe_accuracy)
            self.history_["terminal_reward"].append(terminal)
            self.history_["support_accuracy"].append(accuracy)
            self.history_["policy_loss"].append(float(loss.detach().cpu()))
            self.history_["selected_negatives"].append(len(selected))
            self.history_["optimizer_steps"].append(self.optimizer_steps_)
        self.model_.load_state_dict(best)
        self.model_.eval()
        self.selector_.eval()
        self._is_fitted = True
        return self

    def _select(self, representation, positive, unlabeled, p_mean, u_mean, raw_u, rng, *, record):
        import torch
        from torch.nn import functional

        selected, logs, rewards = [], [], []
        centroid_sum = torch.zeros_like(u_mean)
        for offset in rng.permutation(len(unlabeled)):
            index = int(unlabeled[offset])
            centroid = centroid_sum / len(selected) if selected else u_mean
            state = torch.cat((representation[index], centroid, p_mean)).to(self.device_)
            logit = self.selector_(state).reshape(())
            action = int(rng.uniform() < float(logit.detach().sigmoid().cpu()))
            if action:
                selected.append(index)
                centroid_sum = centroid_sum + representation[index]
            if record:
                logs.append(functional.logsigmoid(logit if action else -logit))
                rewards.append(
                    pulns_intermediate_reward(
                        raw_u[offset : offset + 1].to(self.device_),
                        torch.tensor([action], device=self.device_),
                    ).reshape(())
                )
        return selected, logs, rewards

    def _train_selected(self, network, data, positive, selected, rng):
        if not len(selected):
            return
        indices = np.r_[positive, selected]
        labels = np.r_[np.ones(len(positive)), np.zeros(len(selected))]
        self._train_network(network, data[indices], labels, rng, self.classifier_epochs)

    def _train_network(self, network, data, labels, rng, epochs):
        import torch
        from torch.nn import functional

        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        targets = torch.as_tensor(labels, dtype=torch.float32)
        network.train()
        for _ in range(epochs):
            order = rng.permutation(len(labels))
            for start in range(0, len(labels), self.batch_size):
                indices = order[start : start + self.batch_size]
                optimizer.zero_grad()
                logits = self._logits(network, data[indices].to(self.device_))
                loss = functional.binary_cross_entropy_with_logits(
                    logits, targets[indices].to(self.device_)
                )
                if not torch.isfinite(loss):
                    raise ValueError("PULNS classifier loss became non-finite")
                loss.backward()
                optimizer.step()
                self.optimizer_steps_ += 1
                self.stage_optimizer_steps_[self._training_stage_] += 1
        network.eval()

    def _support_accuracy(self, network, X, y):
        return float(np.mean((self._scores(network, X) >= 0) == y))

    def _episode_features(self, data, unlabeled):
        """Detached last-hidden state; CNN caches rows on CPU, never whole images on GPU.

        Legacy MLP preserves its whole-array forward for numerical compatibility.
        Sequential policy autograd still scales with |U|; this is not a constant
        memory claim for the complete REINFORCE episode.
        """
        import torch

        with torch.no_grad():
            if self.encoder_ is None:
                return (
                    self.model_[:-1](data.to(self.device_)),
                    self._logits(self.model_, data[unlabeled].to(self.device_)),
                )
            representation = torch.cat(
                [
                    self.model_[:-1](data[start : start + self.batch_size].to(self.device_)).cpu()
                    for start in range(0, len(data), self.batch_size)
                ]
            )
            if representation.shape != (len(data), self.hidden_dim):
                raise ValueError("PULNS classifier must preserve the input batch dimension")
            if not torch.isfinite(representation).all():
                raise ValueError("PULNS hidden representations must remain finite")
            return representation, torch.as_tensor(
                self._scores(self.model_, data[unlabeled].numpy())
            )

    @staticmethod
    def _logits(network, batch):
        """Validate one logit per row; retain singleton tails using BN running stats."""
        import torch

        single_row_bn = [
            layer
            for layer in network.modules()
            if len(batch) == 1
            and isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
            and layer.training
        ]
        try:
            for layer in single_row_bn:
                layer.eval()
            logits = network(batch)
        finally:
            for layer in single_row_bn:
                layer.train()
        if not torch.is_tensor(logits) or logits.shape != (len(batch), 1):
            raise ValueError("PULNS classifier must return one logit per input row")
        if not torch.isfinite(logits).all():
            raise ValueError("PULNS logits became non-finite")
        return logits[:, 0]

    def _scores(self, network, X):
        import torch

        if not len(X):
            return np.empty(0, dtype=np.float32)
        modes = [(layer, layer.training) for layer in network.modules()]
        network.eval()
        try:
            with torch.no_grad():
                return np.concatenate(
                    [
                        self._logits(
                            network,
                            torch.as_tensor(X[i : i + self.batch_size], device=self.device_),
                        )
                        .cpu()
                        .numpy()
                        for i in range(0, len(X), self.batch_size)
                    ]
                )
        finally:
            for layer, training in modes:
                layer.training = training

    def _decision_function(self, X):
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=np.float32, allow_nd=True, ensure_min_samples=0)
        if tuple(X.shape[1:]) != self.input_shape_:
            raise ValueError("PULNS input shape differs from training")
        return self._scores(self.model_, X)

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        from scipy.special import expit

        self._check_is_fitted()
        probabilities = expit(self._decision_function(X))
        return np.column_stack((1 - probabilities, probabilities))
