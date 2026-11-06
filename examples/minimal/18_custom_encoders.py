"""Train nnPU with a copied encoder, then fit uPU on frozen features on CPU."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pu_toolbox.estimators.deep import build_encoder
from pu_toolbox.estimators.risk import NonNegativePUClassifier, UPUClassifier
from pu_toolbox.experiment import DatasetBundle, DatasetPart, adapt_image_bundle_to_features
from pu_toolbox.experiment.feature_adapter import encoder_state_sha256
from pu_toolbox.preprocessing import make_scar_labels
from pu_toolbox.prior import population_to_native_unlabeled


def make_bundle(seed: int = 7, image_size: int = 8) -> DatasetBundle:
    """Create balanced, disjoint image roles with a known population prior of 0.5."""
    rng = np.random.RandomState(seed)
    parts = []
    offset = 0
    for role, size in (("train", 32), ("pu_val", 12), ("clean_val", 12), ("test", 12)):
        labels = rng.permutation(np.arange(size) % 2)
        images = rng.normal(0, 0.12, (size, 1, image_size, image_size))
        images += 0.25 + 0.5 * labels[:, None, None, None]
        parts.append(
            DatasetPart(
                X=images.clip(0, 1).astype(np.float32),
                labels=labels,
                view="clean",
                indices=np.arange(offset, offset + size),
                for_selection=role != "test",
            )
        )
        offset += size
    return DatasetBundle(*parts)


def make_encoder(backbone: str = "tiny", seed: int = 7) -> nn.Module:
    """Build a fresh template; shared backbones use their built-in normalization."""
    torch.manual_seed(seed)
    if backbone == "tiny":
        return nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16, 4),
        )
    return build_encoder("cnn", backbone=backbone, in_channels=1)


def run_demo(*, seed: int = 7, backbone: str = "tiny", max_epochs: int = 2) -> dict:
    """Return predictors, role data and the sampling/representation audit record."""
    bundle = make_bundle(seed, image_size=8 if backbone == "tiny" else 32)
    template = make_encoder(backbone, seed)
    initial_hash = encoder_state_sha256(template)
    observed = make_scar_labels(bundle.train.labels, c=0.5, random_state=seed)
    labeled_fraction = float(observed.mean())
    native_prior = population_to_native_unlabeled(0.5, labeled_fraction, input_sampling="os")

    native = NonNegativePUClassifier(
        encoder=template,
        class_prior=native_prior,
        batch_size=8,
        max_epochs=max_epochs,
        random_state=seed,
        device="cpu",
    ).fit(bundle.train.X, observed)
    features, representation = adapt_image_bundle_to_features(
        bundle,
        template,
        feature_version=f"synthetic-{backbone}-seed-{seed}-v1",
        backbone_manifest={
            "name": backbone,
            "initialization": "pytorch_default",
            "seed": seed,
            "pretraining": "none",
            "input_range": [0, 1],
        },
        batch_size=8,
        device="cpu",
        encoder_fit_scope="fixed_external",
    )
    frozen = UPUClassifier(class_prior=native_prior, fit_intercept=False, random_state=seed).fit(
        features.train.X, observed
    )
    scores = native.decision_function(bundle.test.X)
    restored = pickle.loads(pickle.dumps(native))
    np.testing.assert_array_equal(restored.decision_function(bundle.test.X), scores)
    template_unchanged = encoder_state_sha256(template) == initial_hash
    encoder_updated = encoder_state_sha256(native.encoder_) != initial_hash
    assert template_unchanged and encoder_updated
    record = {
        "input_sampling": "os",
        "population_class_prior": 0.5,
        "labeled_fraction": labeled_fraction,
        "training_prior": {"target": "native_unlabeled", "value": native_prior},
        "native_cnn": {
            "backbone": backbone,
            "budget": {"n_candidates": 1, "max_epochs": max_epochs, "batch_size": 8},
            "initial_encoder_sha256": initial_hash,
            "fitted_encoder_sha256": encoder_state_sha256(native.encoder_),
            "template_unchanged": template_unchanged,
            "encoder_updated": encoder_updated,
            "pickle_predictions_equal": True,
            "test_accuracy": float(np.mean(native.predict(bundle.test.X) == bundle.test.labels)),
        },
        "cnn_feature_adapter": {
            **representation,
            "budget": {"n_candidates": 1, "fit_calls": 1},
            "classifier_parameters": {
                "loss": frozen.loss,
                "basis": frozen.basis,
                "reg_lambda": frozen.reg_lambda,
                "fit_intercept": frozen.fit_intercept,
            },
            "test_accuracy": float(np.mean(frozen.predict(features.test.X) == bundle.test.labels)),
        },
        "selection": "fixed budget; default zero-score decision threshold",
    }
    return {
        "record": record,
        "bundle": bundle,
        "features": features,
        "template": template,
        "native": native,
        "frozen": frozen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backbone", choices=("tiny", "cnn13_no_bn", "resnet18"), default="tiny")
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_demo(seed=args.seed, backbone=args.backbone, max_epochs=args.max_epochs)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "settings.json").write_text(
            json.dumps(result["record"], indent=2, allow_nan=False), encoding="utf-8"
        )
        with (args.output_dir / "native.pkl").open("wb") as handle:
            pickle.dump(result["native"], handle)
        with (args.output_dir / "frozen.pkl").open("wb") as handle:
            pickle.dump((result["template"], result["frozen"]), handle)
    print(json.dumps(result["record"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
