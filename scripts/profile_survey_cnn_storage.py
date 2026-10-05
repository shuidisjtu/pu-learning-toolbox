"""CNN checkpoint/storage probes, never formal budgets or paper reproduction.

Uses actual toolbox backbone topology with reduced synthetic training epochs.
Only temporary probe checkpoints are removed; user artifacts are not touched.
Prints strict JSON, with source hashes identifying uncommitted implementations.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

METHODS = ("pulda", "gradpu", "robust_pu", "split_pu", "pan")
ROOT = Path(__file__).resolve().parents[1]


def profile_cnn_storage(method, *, seed=0, base_channels=64, image_size=8, device="cpu"):
    if method not in METHODS:
        raise ValueError(f"CNN probe supports {METHODS}")
    for name, value in (("base_channels", base_channels), ("image_size", image_size)):
        if type(value) is not int or value < (8 if name == "image_size" else 1):
            raise ValueError(f"{name} must be an integer >= {8 if name == 'image_size' else 1}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda (no automatic GPU fallback)")

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
    else:
        kwargs.update(max_epochs=2, batch_size=8)
    model = get_algorithm(method)(**kwargs)
    view = "os" if method == "pan" else "ts"
    source_paths = [
        Path(__file__).relative_to(ROOT),
        Path("pu_toolbox/estimators/deep/vision.py"),
        Path("pu_toolbox/experiment/checkpoints.py"),
        Path("pu_toolbox/utils/serialization.py"),
        Path(__import__(model.__module__, fromlist=["__file__"]).__file__).relative_to(ROOT),
    ]
    source_hashes = {str(path): file_hash(ROOT / path) for path in source_paths}
    if device == "cuda":
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pu-cnn-storage-probe-") as root:
        trajectory = EpochCheckpointTrainer(checkpoint_dir=Path(root)).fit(
            model, features, labels, os_or_ts=view
        )
        if device == "cuda":
            torch.cuda.synchronize()
        fit_elapsed = time.perf_counter() - started
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
        sizes = [Path(snapshot.path).stat().st_size for snapshot in trajectory.checkpoints]
        if not sizes:
            raise AssertionError("CNN probe requires actual epoch checkpoints")
        expected = model.decision_function(features)
        for snapshot in trajectory.checkpoints:
            scores = snapshot.restore(device="cpu").decision_function(features)
            if scores.shape != (len(labels),) or not np.isfinite(scores).all():
                raise AssertionError("snapshot replay did not produce finite per-row scores")
        np.testing.assert_allclose(
            trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
            expected,
            atol=1e-5,
            rtol=1e-5,
        )
        serialized = pickle.dumps(model, protocol=5)
        # Only bytes created in this function are deserialized.
        restored = pickle.loads(serialized)
        np.testing.assert_allclose(restored.decision_function(features), expected, atol=1e-6)

    parameter_count = sum(parameter.numel() for parameter in model.model_.parameters())
    if source_hashes != {str(path): file_hash(ROOT / path) for path in source_paths}:
        raise RuntimeError("probe source changed during training; refuse ambiguous receipt")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "schema_version": 1,
        "status": "synthetic_cnn_storage_probe_not_formal_admission",
        "method": method,
        "seed": seed,
        "device": device,
        "training_view": view,
        "input_shape": list(features.shape),
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
        "optimizer_steps": model.optimizer_steps_,
        "fit_elapsed_seconds": fit_elapsed,
        "epoch_weights_bytes": {
            "count": len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "total": sum(sizes),
            "all_snapshots_replayed": True,
            "final_scores_match": True,
        },
        "estimator_pickle_bytes": len(serialized),
        "trusted_pickle_roundtrip_verified": True,
        "cuda_peak": peak,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "code_commit": commit,
        "source_files_sha256": source_hashes,
        "formal_budget_approved": False,
        "limitations": [
            "Synthetic images, 24 rows, reduced head/epochs; not paper or pilot experiments",
            "Reported disk bytes are measurements, not upper bounds for formal checkpoints",
            "Pickle includes auxiliary networks; inference snapshots include selected model only",
            "No training resume, formal candidates/retries, clean selection or frozen-lock audit",
            "CPU mode has no GPU memory evidence; CUDA peak is process-local, not formal audit",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="pulda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    methods = METHODS if args.method == "all" else (args.method,)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": [
                    profile_cnn_storage(
                        method,
                        seed=seed,
                        base_channels=args.base_channels,
                        image_size=args.image_size,
                        device=args.device,
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
