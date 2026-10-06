"""CNN checkpoint/storage probes, never formal budgets or paper reproduction.

Uses actual toolbox backbone topology with reduced synthetic training epochs.
Only temporary probe checkpoints are removed; user artifacts are not touched.
Prints strict JSON, with source hashes identifying uncommitted implementations.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

METHODS = ("pulda", "gradpu", "robust_pu", "split_pu", "pan", "pulns", "genpu", "holistic_pu")
ROOT = Path(__file__).resolve().parents[1]


def validate_probe_arguments(
    method,
    *,
    seed,
    base_channels,
    image_size,
    device,
    generator_output="identity",
    pseudo_pn_initialization="continue",
    warmup_selection="fixed",
):
    """Refuse malformed/irrelevant recipes before importing torch or training."""
    if method not in METHODS:
        raise ValueError(f"CNN probe supports {METHODS}")
    for name, value in (("base_channels", base_channels), ("image_size", image_size)):
        if type(value) is not int or value < (8 if name == "image_size" else 1):
            raise ValueError(f"{name} must be an integer >= {8 if name == 'image_size' else 1}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda (no automatic GPU fallback)")
    for name, value, choices, owner, default in (
        ("generator_output", generator_output, {"identity", "tanh"}, "genpu", "identity"),
        (
            "pseudo_pn_initialization",
            pseudo_pn_initialization,
            {"continue", "reinitialize"},
            "holistic_pu",
            "continue",
        ),
        (
            "warmup_selection",
            warmup_selection,
            {"fixed", "lzo_positive_loss"},
            "holistic_pu",
            "fixed",
        ),
    ):
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"invalid {name}")
        if method != owner and value != default:
            raise ValueError(f"{name} is only applicable to {owner}")


def host_resident_memory(status_path=Path("/proc/self/status")):
    """OS-reported process lifetime RSS, not a training-only peak or upper bound."""
    try:
        rows = status_path.read_text(encoding="utf-8").splitlines()
        fields = {}
        for row in rows:
            if row.startswith(("VmRSS:", "VmHWM:")):
                key, value, unit = row.split()
                if unit != "kB" or not value.isdecimal():
                    raise ValueError("unexpected proc RSS units")
                fields[key.rstrip(":")] = int(value) * 1024
        if set(fields) != {"VmRSS", "VmHWM"}:
            raise ValueError("missing proc RSS fields")
    except (OSError, ValueError, UnicodeError):
        fields = {"VmRSS": None, "VmHWM": None}
    return {"rss_bytes": fields["VmRSS"], "high_water_bytes": fields["VmHWM"]}


def retained_module_storage(model):
    """Deduplicate shared parameter/buffer storage; exclude grads and optimizers."""
    import torch

    modules = {
        name: value for name, value in vars(model).items() if isinstance(value, torch.nn.Module)
    }
    storages, parameters = {}, {}
    for module in modules.values():
        for parameter in module.parameters():
            parameters[id(parameter)] = parameter.numel()
        for tensor in (*module.parameters(), *module.buffers()):
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    devices = sorted({key[0] for key in storages})
    return {
        "module_attributes": {
            name: sum(p.numel() for p in net.parameters()) for name, net in modules.items()
        },
        "unique_parameter_count": sum(parameters.values()),
        "unique_storage_bytes_by_device": {
            device: sum(size for (owner, _), size in storages.items() if owner == device)
            for device in devices
        },
        "scope": "retained_nn_modules_including_template_no_grads_optimizer_or_temporary_probe",
    }


def profile_cnn_storage(
    method,
    *,
    seed=0,
    base_channels=64,
    image_size=8,
    device="cpu",
    generator_output="identity",
    pseudo_pn_initialization="continue",
    warmup_selection="fixed",
):
    validate_probe_arguments(
        method,
        seed=seed,
        base_channels=base_channels,
        image_size=image_size,
        device=device,
        generator_output=generator_output,
        pseudo_pn_initialization=pseudo_pn_initialization,
        warmup_selection=warmup_selection,
    )

    import torch

    from pu_toolbox.estimators.deep.vision import build_wconpu_backbone
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer
    from pu_toolbox.registry import get_algorithm, register_all_builtin_methods
    from pu_toolbox.utils.serialization import file_hash

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA probe but CUDA is unavailable")
    register_all_builtin_methods()
    rng = np.random.RandomState(seed)
    features = rng.normal(size=(24, 3, image_size, image_size)).astype(np.float32)
    if method == "genpu" and generator_output == "tanh":
        features = np.tanh(features)  # synthetic recipe, not fitted preprocessing
    labels = np.r_[np.ones(8, int), np.zeros(16, int)]
    torch.manual_seed(seed)
    backbone = "cnn13_no_bn" if method == "gradpu" else "cnn13"
    encoder = build_wconpu_backbone(backbone, base_channels=base_channels)
    kwargs = dict(encoder=encoder, hidden_dim=4, random_state=seed, device=device)
    if method == "pulda":
        kwargs.update(
            class_prior=0.4,
            depth=1,
            warmup_epochs=1,
            pu_epochs=1,
            positive_batch_size=4,
            unlabeled_batch_size=8,
        )
    elif method == "gradpu":
        kwargs.update(max_epochs=1, batch_size=8)
    elif method == "robust_pu":
        kwargs.update(class_prior=0.4, pretrain_epochs=1, episodes=1, batch_size=8)
    elif method == "split_pu":
        kwargs.update(
            class_prior=0.4,
            teacher_epochs=1,
            split_epochs=1,
            student_epochs=1,
            rounds=2,
            batch_size=8,
        )
    elif method == "pan":
        kwargs.update(max_epochs=2, batch_size=8)
    elif method == "genpu":
        kwargs.update(
            class_prior=0.4,
            latent_dim=2,
            max_epochs=1,
            classifier_epochs=2,
            batch_size=8,
            generator_output=generator_output,
        )
    elif method == "pulns":
        kwargs.update(pretrain_epochs=1, episodes=2, classifier_epochs=1, batch_size=8)
    else:
        kwargs.update(
            warmup_epochs=3,
            max_epochs=1,
            batch_size=8,
            pseudo_pn_initialization=pseudo_pn_initialization,
            warmup_selection=warmup_selection,
            lzo_validation_size=5 if warmup_selection == "lzo_positive_loss" else None,
        )
    model = get_algorithm(method)(**kwargs)
    view = "os" if method in {"pan", "pulns", "holistic_pu"} else "ts"
    fit_kwargs, support_roles = {"os_or_ts": view}, None
    if method == "pulns":
        fit_kwargs.update(
            support_data=(
                rng.normal(size=(8, 3, image_size, image_size)).astype(np.float32),
                np.tile([0, 1], 4),
            ),
            train_indices=np.arange(24),
            support_indices=np.arange(24, 32),
        )
        support_roles = {
            "rows": 8,
            "labels": "clean_pn",
            "train_ids": list(range(24)),
            "support_ids": list(range(24, 32)),
            "disjoint_ids": True,
            "selection_test_labels_provided": False,
            "formal_support_budget_approved": False,
        }
    source_paths = [
        Path(__file__).relative_to(ROOT),
        Path("pu_toolbox/estimators/deep/vision.py"),
        Path("pu_toolbox/experiment/checkpoints.py"),
        Path("pu_toolbox/utils/serialization.py"),
        Path("pu_toolbox/core/training_views.py"),
        Path("pu_toolbox/core/device.py"),
        Path("pu_toolbox/estimators/deep/_validation.py"),
        Path(__import__(model.__module__, fromlist=["__file__"]).__file__).relative_to(ROOT),
    ]
    source_hashes = {str(path): file_hash(ROOT / path) for path in source_paths}
    if device == "cuda":
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    host_before_fit = host_resident_memory()
    with tempfile.TemporaryDirectory(prefix="pu-cnn-storage-probe-") as root:
        if method == "pulns":
            model.fit(features, labels, **fit_kwargs)
            trajectory = None  # no epoch callback; never fabricate deployable candidates
        else:
            trajectory = EpochCheckpointTrainer(checkpoint_dir=Path(root)).fit(
                model, features, labels, **fit_kwargs
            )
        if device == "cuda":
            torch.cuda.synchronize()
        fit_elapsed = time.perf_counter() - started
        host_after_fit = host_resident_memory()
        peak = (
            {
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
                "scope": "this_process_torch_allocator_during_synthetic_fit",
            }
            if device == "cuda"
            else None
        )
        snapshots = trajectory.checkpoints if trajectory is not None else []
        sizes = [Path(snapshot.path).stat().st_size for snapshot in snapshots]
        if trajectory is not None and not sizes:
            raise AssertionError("CNN probe requires actual epoch checkpoints")
        expected = model.decision_function(features)
        for snapshot in snapshots:
            scores = snapshot.restore(device="cpu").decision_function(features)
            if scores.shape != (len(labels),) or not np.isfinite(scores).all():
                raise AssertionError("snapshot replay did not produce finite per-row scores")
        if snapshots:
            np.testing.assert_allclose(
                snapshots[-1].restore(device="cpu").decision_function(features),
                expected,
                atol=1e-5,
                rtol=1e-5,
            )
        serialized = pickle.dumps(model, protocol=5)
        # Only bytes created in this function are deserialized.
        restored = pickle.loads(serialized)
        np.testing.assert_allclose(restored.decision_function(features), expected, atol=1e-6)
        host_after_verification = host_resident_memory()
        verification_peak = None
        if device == "cuda":
            torch.cuda.synchronize()
            verification_peak = {
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
                "scope": "this_process_fit_and_restore_verification_torch_allocator",
            }

    parameter_count = sum(parameter.numel() for parameter in model.model_.parameters())
    stage_steps = getattr(model, "stage_optimizer_steps_", None)
    if stage_steps is not None and sum(stage_steps.values()) != model.optimizer_steps_:
        raise AssertionError("stage optimizer cost does not sum to full paid cost")
    if source_hashes != {str(path): file_hash(ROOT / path) for path in source_paths}:
        raise RuntimeError("probe source changed during training; refuse ambiguous receipt")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "schema_version": 2,
        "status": "synthetic_cnn_storage_probe_not_formal_admission",
        "method": method,
        "seed": seed,
        "device": device,
        "training_view": view,
        "input_shape": list(features.shape),
        "input_coordinates": "tanh_of_synthetic_standard_normal"
        if method == "genpu" and generator_output == "tanh"
        else "synthetic_standard_normal",
        "calibration_applied": getattr(model, "calibration_applied_", False),
        "clean_support": support_roles,
        "encoder": {
            "backbone": backbone,
            "base_channels": base_channels,
            "matches_toolbox_default_width": base_channels == 64,
            "normalization_mean": [0.5, 0.5, 0.5],
            "normalization_std": [0.5, 0.5, 0.5],
        },
        "parameters": {
            name: value for name, value in model.get_params().items() if name != "encoder"
        },
        "inference_model_parameter_count": parameter_count,
        "retained_module_storage": retained_module_storage(model),
        "optimizer_steps": model.optimizer_steps_,
        "stage_optimizer_steps": stage_steps,
        "fit_elapsed_seconds": fit_elapsed,
        "epoch_weights_bytes": {
            "count": len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "total": sum(sizes),
            "all_snapshots_replayed": True,
            "final_scores_match": True,
        }
        if sizes
        else None,
        "epoch_checkpoint_status": "inference_only_not_training_resume"
        if snapshots
        else "unavailable_no_epoch_callback",
        "checkpoint_training_contexts": [
            snapshot.reference().get("training_context") for snapshot in snapshots
        ],
        "holistic_terminal_selection": {
            name: getattr(model, name)
            for name in (
                "selected_warmup_epoch_",
                "executed_warmup_epochs_",
                "discarded_warmup_optimizer_steps_",
                "lzo_evaluated_rows_",
                "lzo_forward_batches_",
                "pseudo_pn_optimizer_reset_",
            )
        }
        if method == "holistic_pu"
        else None,
        "estimator_pickle_bytes": len(serialized),
        "trusted_pickle_roundtrip_verified": True,
        "cuda_peak": peak,
        "cuda_peak_including_verification": verification_peak,
        "host_resident_memory": {
            "before_fit": host_before_fit,
            "after_fit": host_after_fit,
            "after_verification": host_after_verification,
            "scope": "OS_process_lifetime_highwater_including_imports_and_prior_calls",
            "precise_upper_bound": False,
        },
        "process_id": os.getpid(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "instance": platform.node(),
            "torch_num_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
        },
        "cuda_identity": {
            "logical_index": torch.cuda.current_device(),
            "name": torch.cuda.get_device_name(),
            "total_memory_bytes": torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory,
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        }
        if device == "cuda"
        else None,
        "code_commit": commit,
        "source_files_sha256": source_hashes,
        "formal_budget_approved": False,
        "limitations": [
            "Synthetic images, 24 rows, reduced head/epochs; not paper or pilot experiments",
            "Reported disk bytes are measurements, not upper bounds for formal checkpoints",
            "Pickle includes auxiliary networks; inference snapshots include selected model only",
            "No training resume, formal candidates/retries, clean selection or frozen-lock audit",
            "CPU mode has no GPU memory evidence; CUDA peak is process-local, not formal audit",
            "Host RSS highwater is OS approximate and process-lifetime, not a fit-only bound",
            "PULNS needs separate clean reward labels and has no per-epoch checkpoint interface",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="pulda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--generator-output", choices=["identity", "tanh"], default="identity")
    parser.add_argument(
        "--pseudo-pn-initialization", choices=["continue", "reinitialize"], default="continue"
    )
    parser.add_argument(
        "--warmup-selection", choices=["fixed", "lzo_positive_loss"], default="fixed"
    )
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    methods = METHODS if args.method == "all" else (args.method,)
    for method in methods:
        for seed in args.seeds:
            validate_probe_arguments(
                method,
                seed=seed,
                base_channels=args.base_channels,
                image_size=args.image_size,
                device=args.device,
                generator_output=args.generator_output,
                pseudo_pn_initialization=args.pseudo_pn_initialization,
                warmup_selection=args.warmup_selection,
            )
    print(
        json.dumps(
            {
                "schema_version": 2,
                "profiles": [
                    profile_cnn_storage(
                        method,
                        seed=seed,
                        base_channels=args.base_channels,
                        image_size=args.image_size,
                        device=args.device,
                        generator_output=args.generator_output,
                        pseudo_pn_initialization=args.pseudo_pn_initialization,
                        warmup_selection=args.warmup_selection,
                    )
                    for method in methods
                    for seed in args.seeds
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
