# ruff: noqa: N803, N806
"""Assembly for versioned survey runs, including a verified frozen adapter cache.

Design notes: random frozen ResNet features are shared across methods and c,
never trained on clean labels. Oracle paths are isolated supervised estimators;
CNN support does not expand the historical frozen experiment matrix.
See docs/research/pu_survey/survey_execution_plan.md, P2.0a.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .bundle import DatasetBundle, validate_bundle
from .feature_adapter import _encoder_state_sha256, adapt_image_bundle_to_features
from .image import (
    build_survey_image_encoder,
    fit_survey_image_preprocessing,
    transform_survey_images,
)
from .survey_protocol import ROLES, array_digest, digest, validate_parameters


def score_model(input_dim: int, family: str):
    """Construct the shared score blueprint, without importing torch at module load."""
    from torch import nn

    if family == "resnet18_linear":
        return nn.Linear(input_dim, 1)
    return nn.Sequential(nn.Flatten(), nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1))


def validate_score_model(network, family: str) -> None:
    from torch import nn

    if family == "resnet18_linear":
        valid = (
            type(network) is nn.Linear and network.in_features == 512 and network.out_features == 1
        )
    else:
        valid = (
            type(network) is nn.Sequential
            and len(network) == 4
            and type(network[0]) is nn.Flatten
            and type(network[1]) is nn.Linear
            and network[1].out_features == 128
            and type(network[2]) is nn.ReLU
            and type(network[3]) is nn.Linear
            and network[3].in_features == 128
            and network[3].out_features == 1
        )
    if not valid:
        raise ValueError("score model disagrees with the protocol-locked backbone blueprint")


class PilotOracleMLP(ClassifierMixin, BaseEstimator):
    """Isolated torch supervised MLP; no PU prior or real validation labels in fit.

    Epoch callbacks expose fixed-budget weights to offline OA selection;
    no validation labels enter training through the callback.
    """

    input_ndims = frozenset({2})
    native_architectures = frozenset({"mlp"})
    label_semantics = "pn"

    def __init__(
        self, *, max_epochs=200, batch_size=256, learning_rate=1e-3, random_state=None, device="cpu"
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device

    def fit(self, X, y, *, epoch_callback=None):
        import torch
        from torch import nn

        if self.max_epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("oracle epochs/batch_size/learning_rate must be positive")
        if X.ndim not in self.input_ndims or set(np.unique(y)) != {0, 1}:
            raise ValueError("pilot oracle requires supported inputs and both real binary classes")
        if len(X) != len(y) or np.asarray(y).ndim != 1:
            raise ValueError("oracle requires one binary label per input")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state or 0)
            self.model_ = self._build_network(X).to(self.device)
        # Keep the whole image dataset off the GPU; transfer only each batch.
        tx = torch.as_tensor(np.asarray(X, dtype=np.float32))
        ty = torch.as_tensor(np.asarray(y, dtype=np.float32))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.random_state or 0)
        self.loss_history_ = []
        self.model_.train()
        for epoch in range(self.max_epochs):
            self.model_.train()  # callbacks may temporarily switch to eval mode
            order = torch.randperm(len(X), generator=generator)
            losses = []
            for start in range(0, len(X), self.batch_size):
                indices = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self.model_(tx[indices].to(self.device)).reshape(-1),
                    ty[indices].to(self.device),
                )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.loss_history_.append(float(np.mean(losses)))
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        self.model_.eval()
        self.classes_ = np.array([0, 1])
        return self

    def _build_network(self, X):
        return score_model(X.shape[1], "mlp128")

    def decision_function(self, X):
        import torch

        was_training = self.model_.training
        self.model_.eval()
        try:
            with torch.no_grad():
                scores = [
                    self.model_(
                        torch.as_tensor(
                            np.asarray(X[start : start + self.batch_size], dtype=np.float32),
                            device=self.device,
                        )
                    )
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    for start in range(0, len(X), self.batch_size)
                ]
            return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
        finally:
            self.model_.train(was_training)

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


class PilotOracleCNN(PilotOracleMLP):
    """PN end-to-end image oracle: trainable encoder plus binary score head.

    CleanLabelGenerator/SupervisedTrainer and OA-only selection remain the
    runner's responsibility. An encoder is mandatory; no image flatten fallback.
    """

    input_ndims = frozenset({4})
    native_architectures = frozenset({"cnn"})

    def __init__(
        self,
        *,
        encoder,
        max_epochs=200,
        batch_size=256,
        learning_rate=1e-3,
        random_state=None,
        device="cpu",
    ):
        super().__init__(
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=random_state,
            device=device,
        )
        self.encoder = encoder

    def _build_network(self, X):
        import torch
        from torch import nn

        if self.encoder is None:
            raise ValueError("CNN oracle requires an explicit image encoder")
        encoder = copy.deepcopy(self.encoder).to("cpu")
        encoder.eval()
        with torch.no_grad():
            features = encoder(torch.as_tensor(np.asarray(X[:1], dtype=np.float32)))
        if features.ndim != 2 or features.shape[1] < 1:
            raise ValueError("CNN oracle encoder must return a 2-D feature matrix")
        encoder.requires_grad_(True)
        return nn.Sequential(encoder, nn.Linear(features.shape[1], 1))


class SourceSpaceGenerator:
    """Generate SAR labels on the original image space, not method-specific features."""

    def __init__(self, generator, source: DatasetBundle, adapted: DatasetBundle):
        self.base_generator = generator
        self.output_view = generator.output_view
        self._sources = {
            id(getattr(adapted, role).X): getattr(source, role).X for role in ("train", "pu_val")
        }

    def generate(self, X, y_true, c, seed=None):
        if id(X) not in self._sources:
            raise ValueError("source-space generator received an unbound partition")
        source = self._sources[id(X)]
        labels, metadata = self.base_generator.generate(source, y_true, c, seed)
        return labels, {
            **metadata,
            "sampling_space": "original_input",
            "sampling_input_sha256": array_digest(source),
        }


def prepare_image_bundle(bundle: DatasetBundle, protocol: dict, seed: int):
    """Fit only train statistics; normalization stays inside the encoder."""
    import torch

    spec = protocol["backbone_specs"]["image"]
    train, preprocessing = fit_survey_image_preprocessing(
        bundle.train.X, dataset="cifar10", train_augmentation=spec["train_augmentation"]
    )
    if list(preprocessing.input_size) != spec["input_size"]:
        raise ValueError("CIFAR input size disagrees with the locked image specification")
    if preprocessing.train_augmentation != "none":
        raise ValueError(
            "pilot assembly has no augmentation hook; non-none augmentation is unsupported"
        )
    prepared = DatasetBundle(
        **{
            role: replace(
                getattr(bundle, role),
                X=train
                if role == "train"
                else transform_survey_images(getattr(bundle, role).X, preprocessing, role=role),
            )
            for role in ROLES
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = build_survey_image_encoder(preprocessing)
    metadata = preprocessing.to_manifest()
    metadata["initialization_seed"] = seed
    metadata["encoder_state_sha256"] = _encoder_state_sha256(encoder)
    return prepared, encoder, metadata


def cached_adapter(
    bundle: DatasetBundle,
    encoder,
    image_manifest: dict,
    *,
    cache_dir: Path | None,
    batch_size=64,
    device="cpu",
):
    """Content-addressed, label-free cache; damaged entries fail rather than retrain."""
    cache_key = digest(
        {
            "schema_version": "1.0",
            "encoder_state_sha256": _encoder_state_sha256(encoder),
            "image": image_manifest,
            "inputs": {role: array_digest(getattr(bundle, role).X) for role in ROLES},
            "indices": {role: np.asarray(getattr(bundle, role).indices).tolist() for role in ROLES},
            "device": device,
            "extraction_batch_size": batch_size,
        }
    )
    entry = None if cache_dir is None else Path(cache_dir) / cache_key
    if entry is not None and entry.exists():
        from .feature_adapter import _array_sha256

        manifest = json.loads((entry / "adapter.json").read_text(encoding="utf-8"))
        if manifest.get("cache_key") != cache_key or manifest[
            "encoder_state_sha256"
        ] != _encoder_state_sha256(encoder):
            raise ValueError("adapter cache encoder/key mismatch")
        with np.load(entry / "features.npz", allow_pickle=False) as payload:
            features = {role: payload[role] for role in ROLES}
        if any(_array_sha256(features[role]) != manifest["feature_sha256"][role] for role in ROLES):
            raise ValueError("adapter cache feature hash mismatch")
        adapted = DatasetBundle(
            **{role: replace(getattr(bundle, role), X=features[role]) for role in ROLES}
        )
        validate_bundle(adapted)
        return adapted, {**manifest, "cache_hit": True}
    adapted, manifest = adapt_image_bundle_to_features(
        bundle,
        encoder,
        feature_version="survey-v1/random-frozen-resnet18",
        backbone_manifest=image_manifest,
        batch_size=batch_size,
        device=device,
        encoder_fit_scope="fixed_external",
    )
    manifest.update(
        {
            "cache_key": cache_key,
            "cache_schema_version": "1.0",
            "encoder_training": "none; seeded_random_frozen",
            "cache_hit": False,
        }
    )
    if entry is not None:
        entry.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="adapter-", dir=entry.parent) as temporary:
            stage = Path(temporary) / "entry"
            stage.mkdir()
            np.savez(stage / "features.npz", **{role: getattr(adapted, role).X for role in ROLES})
            (stage / "adapter.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            try:
                os.rename(stage, entry)
            except OSError:
                # A competing writer completed first; validate its entry through the read path.
                # Linux can report ENOTEMPTY instead of EEXIST for directory rename.
                if not entry.is_dir():
                    raise
                return cached_adapter(
                    bundle,
                    encoder,
                    image_manifest,
                    cache_dir=cache_dir,
                    batch_size=batch_size,
                    device=device,
                )
    return adapted, manifest


def assemble_model(
    protocol: dict,
    row: dict,
    input_dim: int,
    *,
    seed: int,
    params: dict,
    class_prior: float | None,
    device: str,
    encoder=None,
):
    """Instantiate matrix defaults; constructor/candidate overrides cannot change locks."""
    profile = protocol["method_profiles"][row["method"]]
    validate_parameters(params, profile)
    constructor = {**profile["params"], **params}
    if row["method"] == "pn_oracle":
        if row["training_path"] == "native_cnn":
            if encoder is None:
                raise ValueError("CNN oracle requires an explicit image encoder")
            return PilotOracleCNN(encoder=encoder, **constructor, random_state=seed, device=device)
        return PilotOracleMLP(**constructor, random_state=seed, device=device)
    from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

    register_all_builtin_methods()
    cls = get_algorithm(row["method"])
    import inspect

    accepted = inspect.signature(cls).parameters
    if "random_state" in accepted:
        constructor["random_state"] = seed
    if "device" in accepted:
        constructor["device"] = device
    if "class_prior" in accepted:
        constructor["class_prior"] = class_prior
    if row["method"] in {"nnpu", "self_pu"}:
        import torch

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = score_model(512 if encoder is not None else input_dim, row["model_family"])
        constructor["model" if row["method"] == "nnpu" else "backbone"] = network
    if row["training_path"] == "native_cnn":
        constructor["encoder"] = encoder
    return cls(**constructor)
