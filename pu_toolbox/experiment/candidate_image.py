"""Opt-in PULDA/GradPU image assembly, not a frozen Survey recipe or admission.

The returned manifest describes construction-time facts. Callers must persist it
with the actual run parameters, selection policy and measured resource record.
No existing pilot, feature cache or leaderboard is modified by this helper.
"""

from __future__ import annotations

import copy
import json
import math
import time
from numbers import Integral, Real
from pathlib import Path

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
        "train_transform_description": repr(augmentation)
        if method == "pulda" and augmentation is not None
        else None,
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


def fit_candidate_image_classifier(
    estimator, image_setup, train_images, s_train, *, os_or_ts="ts", epoch_callback=None
):
    """Fit only a bound technical image setup and return an actual training receipt.

    Inputs must be the exact float32 train pixels returned by preprocessing,
    with observed PU labels. No clean validation/test data is accepted. This
    helper neither selects checkpoints nor writes artifacts or authorizes runs.
    Hashing scans CPU inputs twice; it is opt-in and not part of default fit.
    A rejected/failed receipt does not roll back estimator or callback artifacts.
    """
    import numpy as np

    from pu_toolbox.core.validation import validate_pu_X_y
    from pu_toolbox.estimators.deep.grad_pu import GradPUClassifier
    from pu_toolbox.estimators.risk.pulda import PULDAClassifier
    from pu_toolbox.utils.serialization import array_hash, file_hash

    started = time.perf_counter()
    if not isinstance(image_setup, dict):
        raise ValueError("image_setup must be a construction record")
    setup = copy.deepcopy(image_setup)
    method = setup.get("method")
    classes = {"pulda": PULDAClassifier, "gradpu": GradPUClassifier}
    expected_group = "end-to-end CNN" if method == "pulda" else "method-private/no-BN CNN"
    if (
        not isinstance(method, str)
        or method not in classes
        or type(estimator) is not classes[method]
    ):
        raise ValueError("estimator and image setup method identity differ")
    if (
        setup.get("schema_version") != "candidate-image-setup-1"
        or setup.get("execution_mode") != "technical_preintegration"
        or setup.get("formal_admission") is not False
        or "approved_representation_ref" not in setup
        or setup.get("approved_representation_ref") is not None
        or setup.get("standard_resnet18_leaderboard_eligible") is not False
        or setup.get("source_reproduction") is not False
        or setup.get("comparison_group") != expected_group
        or setup.get("training_path") != "native_cnn"
        or setup.get("encoder_trainable") is not True
        or setup.get("adaptation") != "benchmark-adapted"
    ):
        raise ValueError("image setup identity/group/admission fields are not a technical setup")
    if estimator.random_state != setup["initialization_seed"]:
        raise ValueError("estimator seed changed after image construction")
    if (
        estimator.encoder is None
        or encoder_state_sha256(estimator.encoder) != setup["encoder_initial_state_sha256"]
    ):
        raise ValueError("encoder template changed after image construction")
    if "train_transform_description" not in setup:
        raise ValueError("rebuild image setup to bind its train transform")
    transform = estimator.train_transform if method == "pulda" else None
    if (repr(transform) if transform is not None else None) != setup["train_transform_description"]:
        raise ValueError("train transform changed after image construction")
    if method == "pulda" and estimator.class_prior != setup["prior_contract"]["value"]:
        raise ValueError("PULDA population class_prior changed after image construction")
    if method == "gradpu" and estimator.model is not None:
        raise ValueError("custom score models require a separate image setup")

    features, labels = validate_pu_X_y(
        train_images,
        s_train,
        accept_sparse=False,
        allow_nd=True,
        estimator_name="candidate image fit",
    )
    image = setup["image_preprocessing"]
    if image["backbone"]["name"] != ("resnet18" if method == "pulda" else "cnn13_no_bn"):
        raise ValueError("image backbone name differs from the bound method path")
    expected_shape = (image["in_channels"], *image["input_size"])
    if (
        features.dtype != np.float32
        or features.ndim != 4
        or tuple(features.shape[1:]) != expected_shape
    ):
        raise ValueError("fit requires unchanged preprocessed float32 NCHW train pixels")
    input_hash = array_hash(features)
    if input_hash != image["train_data_sha256"]:
        raise ValueError("training pixels differ from the train-fitted image preprocessing")
    labels = np.asarray(labels, dtype=np.int64)
    label_hash = array_hash(labels)
    parameters = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in estimator.get_params(deep=False).items()
        if key not in {"encoder", "train_transform", "model"}
    }
    # Refuse unsupported/nonfinite parameter encodings before starting fit.
    json.dumps(parameters, allow_nan=False)
    estimator.fit(features, labels, os_or_ts=os_or_ts, epoch_callback=epoch_callback)
    if array_hash(features) != input_hash or array_hash(labels) != label_hash:
        raise RuntimeError("training inputs changed during fit; no valid receipt produced")
    source = "estimators/risk/pulda.py" if method == "pulda" else "estimators/deep/grad_pu.py"
    receipt = {
        "schema_version": "candidate-image-training-receipt-1",
        "image_setup": setup,
        "parameters_at_fit_entry": parameters,
        "train_input_sha256": input_hash,
        "observed_PU_labels_sha256": label_hash,
        "input_shape": list(features.shape),
        "training_label_semantics": "caller_declared_observed_PU",
        "label_budget_externally_verified": False,
        "four_role_isolation_verified": False,
        "training_view": estimator.training_view_,
        "calibration_applied": bool(estimator.calibration_applied_),
        "population_mixture_restoration_verified": False,
        "completed_epochs": len(estimator.history_["epoch"]),
        "optimizer_steps": estimator.optimizer_steps_,
        "actual_device": str(next(estimator.model_.parameters()).device),
        "encoder_trainable": all(p.requires_grad for p in estimator.encoder_.parameters()),
        "algorithm_training": copy.deepcopy(getattr(estimator, "training_provenance_", None)),
        "implementation_sha256": file_hash(Path(__file__).resolve().parents[1] / source),
        "helper_sha256": file_hash(Path(__file__)),
        "elapsed_seconds_including_input_hashing": time.perf_counter() - started,
        "selection_performed": False,
        "checkpoint_persistence_verified": False,
        "resource_profile_completed": False,
        "runtime_environment_lock_verified": False,
        "protocol_registry_bound": False,
        "formal_admission": False,
        "ready_for_formal_run": False,
    }
    json.dumps(receipt, allow_nan=False)
    return estimator, receipt
