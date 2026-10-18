"""Opt-in PULDA/GradPU image assembly, not a frozen Survey recipe or admission.

The returned manifest describes construction-time facts. Callers must persist it
with the actual run parameters, selection policy and measured resource record.
No existing pilot, feature cache or leaderboard is modified by this helper.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

from .feature_adapter import encoder_state_sha256
from .image import (
    SurveyImagePreprocessing,
    build_survey_image_augmentation,
    build_survey_image_encoder,
)


def build_candidate_image_classifier(
    method: str,
    preprocessing: SurveyImagePreprocessing,
    *,
    seed: int,
    class_prior: float | None = None,
    device: str = "cpu",
):
    """Return a default-budget estimator and explicit, unapproved image metadata.

    PULDA uses the shared train-fitted ResNet-18 and optional train augmentation.
    GradPU uses a private CNN13 without BN; shared augmentation is deliberately
    refused until a method-specific policy is decided. Each construction resets
    the CPU torch seed in a fork, independent of previous methods' RNG use.
    """
    if not isinstance(method, str) or method not in {"pulda", "gradpu"}:
        raise ValueError("method must be 'pulda' or 'gradpu'")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if method == "pulda" and class_prior is None:
        raise ValueError("PULDA requires the population class_prior")
    if method == "pulda":
        if (
            isinstance(class_prior, bool)
            or not isinstance(class_prior, Real)
            or not math.isfinite(float(class_prior))
            or not 0 < float(class_prior) < 1
        ):
            raise ValueError("population class_prior must be a finite real number in (0, 1)")
        class_prior = float(class_prior)
    if method == "gradpu" and class_prior is not None:
        raise ValueError("GradPU does not use a class_prior")
    if method == "gradpu" and preprocessing.train_augmentation != "none":
        raise ValueError("GradPU private image setup requires train_augmentation='none'")
    if method == "gradpu" and min(preprocessing.input_size) < 8:
        raise ValueError("cnn13_no_bn requires spatial input size >= 8")

    import torch

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        if method == "pulda":
            from pu_toolbox.estimators.risk.pulda import PULDAClassifier

            encoder = build_survey_image_encoder(preprocessing)
            augmentation = build_survey_image_augmentation(preprocessing, role="train")
            estimator = PULDAClassifier(
                class_prior=class_prior,
                encoder=encoder,
                train_transform=augmentation,
                random_state=int(seed),
                device=device,
            )
        else:
            from pu_toolbox.estimators.deep.grad_pu import GradPUClassifier
            from pu_toolbox.estimators.deep.vision import build_wconpu_backbone

            encoder = build_wconpu_backbone(
                "cnn13_no_bn",
                in_channels=preprocessing.in_channels,
                normalization_mean=preprocessing.normalization_mean,
                normalization_std=preprocessing.normalization_std,
            )
            estimator = GradPUClassifier(encoder=encoder, random_state=int(seed), device=device)

    image = preprocessing.to_manifest()
    if method == "gradpu":
        image["backbone"] = {
            "name": "cnn13_no_bn",
            "weights": None,
            "initialization": "random",
            "base_channels": 64,
            "batch_norm": False,
            "first_layer": {
                "in_channels": preprocessing.in_channels,
                "out_channels": 64,
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "bias": False,
            },
        }
    preprocessing_digest = image.pop("configuration_sha256")
    return estimator, {
        "schema_version": "candidate-image-setup-1",
        "method": method,
        "execution_mode": "technical_preintegration",
        "formal_admission": False,
        "approved_representation_ref": None,
        "training_path": "native_cnn",
        "comparison_group": "end-to-end CNN" if method == "pulda" else "method-private/no-BN CNN",
        "standard_resnet18_leaderboard_eligible": False,
        "source_reproduction": False,
        "adaptation": "benchmark-adapted",
        "source_identity": "official_related" if method == "pulda" else "paper-only",
        "initialization_seed": int(seed),
        "encoder_initial_state_sha256": encoder_state_sha256(encoder),
        "encoder_trainable": True,
        "batch_norm_policy": "train" if method == "pulda" else "forbidden",
        "image_preprocessing": image,
        "preprocessing_configuration_sha256": preprocessing_digest,
        "input_order": ["train_only_augmentation", "input_mixup", "encoder_normalization"]
        if method == "pulda"
        else ["raw_image_interpolation_for_GP", "encoder_normalization"],
        "augmentation_sampling": "fresh_per_training_batch_both_phases"
        if method == "pulda"
        else "none",
        "evaluation_augmentation": "none",
        "prior_contract": {
            "source": "caller_supplied_population_prior" if method == "pulda" else "not_used",
            "value": class_prior,
            "actual_calibration_applied": "record_from_fitted_run_not_construction",
        },
        "head_specification": "method-private; shared encoder does not imply shared MLP head",
        "resource_profile": {"status": "not_measured", "formal_budget_approved": False},
    }
