"""Opt-in candidate image assembly, not a frozen recipe or admission.

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
    PAN also uses that encoder template, cloned independently by its C/D fit;
    it retains native OS and requires augmentation=none in this candidate path.
    VPU uses a separate copy with its native MLP/log-Phi head and TS marginal.
    P3Mix uses a separate copy with its native OS mixup head; this ResNet-18
    is an engineering alternative, not the paper's dataset-specific backbone.
    GradPU uses a private CNN13 without BN; shared augmentation is deliberately
    refused until a method-specific policy is decided. Each construction resets
    the CPU torch seed in a fork, independent of previous methods' RNG use.
    """
    if not isinstance(method, str) or method not in {"pulda", "gradpu", "pan", "vpu", "p3mix"}:
        raise ValueError("method must be 'pulda', 'gradpu', 'pan', 'vpu' or 'p3mix'")
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
    if method in {"gradpu", "pan", "vpu", "p3mix"} and class_prior is not None:
        raise ValueError(f"{method} does not use a class_prior")
    if method == "pan" and preprocessing.train_augmentation != "none":
        raise ValueError("PAN candidate image setup requires train_augmentation='none'")
    if method == "vpu" and preprocessing.train_augmentation != "none":
        raise ValueError("VPU candidate image setup requires train_augmentation='none'")
    if method == "p3mix" and preprocessing.train_augmentation != "none":
        raise ValueError("P3Mix candidate image setup requires train_augmentation='none'")
    if method == "gradpu" and preprocessing.train_augmentation != "none":
        raise ValueError("GradPU private image setup requires train_augmentation='none'")
    if method == "gradpu" and min(preprocessing.input_size) < 8:
        raise ValueError("cnn13_no_bn requires spatial input size >= 8")

    import torch

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        if method in {"pulda", "pan", "vpu", "p3mix"}:
            from pu_toolbox.estimators.risk.pulda import PULDAClassifier

            encoder = build_survey_image_encoder(preprocessing)
            augmentation = build_survey_image_augmentation(preprocessing, role="train")
            if method == "pulda":
                estimator = PULDAClassifier(
                    class_prior=class_prior,
                    encoder=encoder,
                    train_transform=augmentation,
                    random_state=int(seed),
                    device=device,
                )
            elif method == "pan":
                from pu_toolbox.estimators.deep.pan import PANClassifier

                estimator = PANClassifier(encoder=encoder, random_state=int(seed), device=device)
            elif method == "vpu":
                from pu_toolbox.estimators.risk.vpu import VPUClassifier

                estimator = VPUClassifier(
                    encoder=encoder,
                    random_state=int(seed),
                    device=device,
                    data_device_policy="cpu_stream",
                )
            else:
                from pu_toolbox.estimators.deep.p3mix import P3MixClassifier

                estimator = P3MixClassifier(encoder=encoder, random_state=int(seed), device=device)
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
    record = {
        "schema_version": "candidate-image-setup-1",
        "method": method,
        "execution_mode": "technical_preintegration",
        "formal_admission": False,
        "approved_representation_ref": None,
        "training_path": "native_cnn",
        "comparison_group": "method-private/no-BN CNN" if method == "gradpu" else "end-to-end CNN",
        "standard_resnet18_leaderboard_eligible": False,
        "source_reproduction": False,
        "adaptation": "benchmark-adapted",
        "source_identity": "paper-only" if method in {"gradpu", "p3mix"} else "official_related",
        "initialization_seed": int(seed),
        "encoder_initial_state_sha256": encoder_state_sha256(encoder),
        "encoder_trainable": True,
        "train_transform_description": repr(augmentation)
        if method == "pulda" and augmentation is not None
        else None,
        "batch_norm_policy": "forbidden" if method == "gradpu" else "train",
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
    if method == "pan":
        record.update(
            input_order=["encoder_normalization"],
            training_view="os",
            network_instances=2,
            selection_components=["classifier_C"],
            adversarial_state_policy="independent_C_D; inactive_eval_no_grad",
        )
    if method == "vpu":
        record.update(
            input_order=[
                "raw_image_mixup",
                "encoder_normalization",
                "train_marginal_max_log_phi_normalization",
            ],
            training_view="ts",
            marginal_pool="train_P_union_U",
            selection_components=["normalized_phi_model"],
            normalization_scope="completed_epoch_training_marginal_only",
            validation_diagnostic_selection_eligible=False,
        )
    if method == "p3mix":
        record.update(
            input_order=["raw_image_heuristic_mixup", "encoder_normalization"],
            training_view="os",
            candidate_pool_scope="observed_labeled_train_positive_only",
            selection_components=["student"],
            teacher_selection_eligible=False,
            backbone_scope="shared_resnet18_engineering_alternative_not_paper_backbone",
        )
    return estimator, record


def _validate_candidate_image_fit(estimator, image_setup, train_images, s_train):
    """Share the existing bound-input checks with the checkpoint runner."""
    import numpy as np

    from pu_toolbox.core.validation import validate_pu_X_y
    from pu_toolbox.estimators.deep.grad_pu import GradPUClassifier
    from pu_toolbox.estimators.deep.p3mix import P3MixClassifier
    from pu_toolbox.estimators.deep.pan import PANClassifier
    from pu_toolbox.estimators.risk.pulda import PULDAClassifier
    from pu_toolbox.estimators.risk.vpu import VPUClassifier
    from pu_toolbox.utils.serialization import array_hash

    if not isinstance(image_setup, dict):
        raise ValueError("image_setup must be a construction record")
    setup = copy.deepcopy(image_setup)
    method = setup.get("method")
    classes = {
        "pulda": PULDAClassifier,
        "gradpu": GradPUClassifier,
        "pan": PANClassifier,
        "vpu": VPUClassifier,
        "p3mix": P3MixClassifier,
    }
    expected_group = "method-private/no-BN CNN" if method == "gradpu" else "end-to-end CNN"
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
    if method == "pan" and (
        estimator.model is not None
        or estimator.discriminator is not None
        or setup.get("training_view") != "os"
        or setup.get("network_instances") != 2
        or setup.get("selection_components") != ["classifier_C"]
        or setup.get("adversarial_state_policy") != "independent_C_D; inactive_eval_no_grad"
    ):
        raise ValueError("PAN requires the bound native OS independent C/D image setup")
    if method == "vpu" and (
        setup.get("training_view") != "ts"
        or setup.get("marginal_pool") != "train_P_union_U"
        or setup.get("selection_components") != ["normalized_phi_model"]
        or setup.get("normalization_scope") != "completed_epoch_training_marginal_only"
        or setup.get("validation_diagnostic_selection_eligible") is not False
    ):
        raise ValueError("VPU requires native TS and train-marginal-only normalization")
    if method == "p3mix" and (
        estimator.model is not None
        or setup.get("training_view") != "os"
        or setup.get("candidate_pool_scope") != "observed_labeled_train_positive_only"
        or setup.get("selection_components") != ["student"]
        or setup.get("teacher_selection_eligible") is not False
        or setup.get("backbone_scope")
        != "shared_resnet18_engineering_alternative_not_paper_backbone"
    ):
        raise ValueError("P3Mix requires native OS, train-P pool and student-only selection")

    features, labels = validate_pu_X_y(
        train_images,
        s_train,
        accept_sparse=False,
        allow_nd=True,
        estimator_name="candidate image fit",
    )
    image = setup["image_preprocessing"]
    if image["backbone"]["name"] != ("cnn13_no_bn" if method == "gradpu" else "resnet18"):
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
        if key not in {"encoder", "train_transform", "model", "discriminator"}
    }
    # Refuse unsupported/nonfinite parameter encodings before starting fit.
    json.dumps(parameters, allow_nan=False)
    return setup, features, labels, input_hash, label_hash, parameters


def _vpu_normalization_facts(estimator):
    """Inspect completed epoch state without repeating its normalization scan."""
    import torch

    normalizer = estimator.model_[-1]
    maximum = float(estimator.max_log_phi_)
    expected_bias = -maximum - math.log(0.5)
    if (
        not math.isfinite(expected_bias)
        or not torch.equal(normalizer.weight.detach().cpu(), torch.ones((1, 1)))
        or normalizer.weight.requires_grad
        or normalizer.bias.requires_grad
        or not math.isclose(
            float(normalizer.bias.detach().cpu()), expected_bias, rel_tol=1e-6, abs_tol=1e-7
        )
    ):
        raise ValueError("VPU inference normalization does not match its training marginal")
    return dict(
        max_log_phi=maximum,
        normalization_bias=float(normalizer.bias.detach().cpu()),
        normalization_embedded_in_model_state=True,
    )


def _p3mix_epoch_facts(estimator, labels, epoch):
    """Verify completed OS epochs and identity-bound pool without new forwards."""
    import numpy as np
    import torch

    p = np.flatnonzero(labels == 1)
    u = np.flatnonzero(labels == 0)
    batch = estimator.batch_size
    merged = estimator.batching_policy == "merged_shuffle"
    steps = (
        math.ceil(len(labels) / batch)
        if merged
        else max(math.ceil(len(p) / batch), math.ceil(len(u) / batch))
    )

    def exposure(count):
        batches = math.ceil(count / batch)
        cycles, remainder = divmod(steps, batches)
        return count if merged else cycles * count + min(count, remainder * batch)

    expected = {
        "epoch": epoch,
        "optimizer_steps": epoch * steps,
        "positive_rows_seen": exposure(len(p)),
        "unlabeled_rows_seen": exposure(len(u)),
        "positive_rows_reused": exposure(len(p)) - len(p),
        "unlabeled_rows_reused": exposure(len(u)) - len(u),
    }
    algorithm = estimator.training_provenance_
    pool = estimator.candidate_training_positions_
    if (
        estimator.epochs_completed_ != epoch
        or estimator.optimizer_steps_ != epoch * steps
        or estimator.positive_rows_seen_ != epoch * exposure(len(p))
        or estimator.unlabeled_rows_seen_ != epoch * exposure(len(u))
        or algorithm.get("run_view") != "os"
        or algorithm.get("calibration_applied") is not False
        or not np.array_equal(estimator.train_positive_indices_, p)
        or not np.array_equal(estimator.train_unlabeled_indices_, u)
        or len(estimator.epoch_provenance_) != epoch
        or estimator.epoch_provenance_[-1] != expected
        or len(pool) != min(estimator.pool_size, len(p))
        or len(np.unique(pool)) != len(pool)
        or not np.isin(pool, p).all()
    ):
        raise ValueError("P3Mix completed epoch/pool/role exposure cost mismatch")
    teacher = estimator.teacher_
    if (teacher is not None) != (estimator.variant == "e"):
        raise ValueError("P3Mix teacher does not match its variant")
    if teacher is not None:
        student_tensors = list(estimator.model_.parameters()) + list(estimator.model_.buffers())
        teacher_tensors = list(teacher.parameters()) + list(teacher.buffers())
        if (
            any(parameter.requires_grad for parameter in teacher.parameters())
            or {t.data_ptr() for t in student_tensors} & {t.data_ptr() for t in teacher_tensors}
            or any(not torch.isfinite(t).all() for t in teacher_tensors)
        ):
            raise ValueError("P3Mix teacher must be finite, frozen and storage-independent")
    return {
        **expected,
        "student_training_forward_batches": 2 * steps,
        "teacher_forward_batches": steps if teacher is not None else 0,
        "positive_pool_refresh_forward_batches": math.ceil(len(p) / batch),
        "forward_counts_scope": "derived_from_verified_updates_excludes_probes_and_selection",
        "teacher_ema_updates_completed": epoch if teacher is not None else 0,
    }


def _candidate_image_training_receipt(estimator, bound_inputs, started):
    """Verify inputs after training and retain the original receipt semantics."""
    from pu_toolbox.utils.serialization import array_hash, file_hash

    setup, features, labels, input_hash, label_hash, parameters = bound_inputs
    method = setup["method"]
    if array_hash(features) != input_hash or array_hash(labels) != label_hash:
        raise RuntimeError("training inputs changed during fit; no valid receipt produced")
    source = {
        "pulda": "estimators/risk/pulda.py",
        "gradpu": "estimators/deep/grad_pu.py",
        "pan": "estimators/deep/pan.py",
        "vpu": "estimators/risk/vpu.py",
        "p3mix": "estimators/deep/p3mix.py",
    }[method]
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
        "training_view": estimator.training_provenance_["run_view"]
        if method == "p3mix"
        else estimator.training_view_,
        "calibration_applied": bool(estimator.training_provenance_["calibration_applied"])
        if method == "p3mix"
        else bool(estimator.calibration_applied_),
        "population_mixture_restoration_verified": False,
        "completed_epochs": estimator.epochs_completed_
        if method == "p3mix"
        else len(estimator.history_["epoch"]),
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
    if method == "pan":
        import torch

        if estimator.training_view_ != "os" or estimator.calibration_applied_:
            raise ValueError("PAN image run must retain native OS without TS calibration")
        if encoder_state_sha256(estimator.encoder) != setup["encoder_initial_state_sha256"]:
            raise ValueError("PAN fitted through or mutated the shared encoder template")
        networks = [estimator.model_, estimator.discriminator_]
        tensors = [list(n.parameters()) + list(n.buffers()) for n in networks]
        if {t.data_ptr() for t in tensors[0]} & {t.data_ptr() for t in tensors[1]}:
            raise ValueError("PAN C/D cannot share parameter or buffer storage")
        if not all(p.requires_grad for p in estimator.discriminator_encoder_.parameters()):
            raise ValueError("PAN discriminator encoder was not trainable")
        if any(not torch.isfinite(t).all() for group in tensors for t in group):
            raise ValueError("PAN C/D finished with non-finite state")
        receipt.update(
            independent_C_D_storage_verified=True,
            discriminator_encoder_trainable=True,
            classifier_only_inference=True,
            target_identity="paper_Eq7_not_related_source_max_margin",
        )
    if method == "vpu":
        if (
            estimator.training_view_ != "ts"
            or not estimator.calibration_applied_
            or estimator.n_loss_unlabeled_ != len(features)
            or estimator.training_data_device_ != "cpu"
        ):
            raise ValueError("VPU candidate fit must stream the complete train P-union-U marginal")
        receipt.update(_vpu_normalization_facts(estimator))
        receipt.update(
            training_marginal_rows=estimator.n_loss_unlabeled_,
            decision_score_semantics="log_Phi_normalized_over_0.5_not_calibrated_probability",
            normalization_forward_batches_per_epoch=math.ceil(
                estimator.n_loss_unlabeled_ / estimator.batch_size
            ),
            normalization_training_rows_per_epoch=estimator.n_loss_unlabeled_,
            optimizer_resets=estimator.optimizer_resets_,
            optimizer_schedule=estimator.optimizer_schedule,
            iterations_per_epoch=estimator.iterations_per_epoch,
        )
    if method == "p3mix":
        if encoder_state_sha256(estimator.encoder) != setup["encoder_initial_state_sha256"]:
            raise ValueError("P3Mix mutated the shared encoder template")
        receipt.update(_p3mix_epoch_facts(estimator, labels, estimator.epochs_completed_))
        receipt.update(
            student_only_inference=True,
            data_device_policy=estimator.data_device_policy_,
            backbone_is_paper_architecture=False,
        )
    json.dumps(receipt, allow_nan=False)
    return receipt


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
    started = time.perf_counter()
    bound = _validate_candidate_image_fit(estimator, image_setup, train_images, s_train)
    estimator.fit(bound[1], bound[2], os_or_ts=os_or_ts, epoch_callback=epoch_callback)
    return estimator, _candidate_image_training_receipt(estimator, bound, started)
