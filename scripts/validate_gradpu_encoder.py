"""Full-training-set encoder validation; separate from frozen survey admission.

Uses all train rows, the frozen 200-epoch/batch-256 budget, and a fixed clean-val
diagnostic (never test or checkpoint selection). Initialization/LR arms are
explicit experimental deviations, not paper reproduction. Split GP execution
retains the optimizer batch; it does not authorize release of paused fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.diagnose_gradpu_encoder import ARMS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_indices(labels, seed):
    """Fixed balanced train probe; never reorder or subsample training itself."""
    rng = np.random.RandomState(seed)
    return np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == y), min(128, sum(labels == y)), replace=False)
            for y in (0, 1)
        ]
    )


def observe(estimator, images, labels=None):
    """Bounded inference batches; preserve module modes and training RNG state."""
    import torch
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    modes = [(layer, layer.training) for layer in estimator.model_.modules()]
    device = next(estimator.model_.parameters()).device
    features, scores = [], []
    try:
        estimator.model_.eval()
        with torch.no_grad():
            for start in range(0, len(images), 16):
                batch = torch.as_tensor(images[start : start + 16], device=device)
                features.append(estimator.encoder_(batch).cpu().numpy())
                scores.append(estimator.model_(batch).flatten().cpu().numpy())
    finally:
        for layer, mode in modes:
            layer.training = mode
    features, scores = np.concatenate(features), np.concatenate(scores)
    if not np.isfinite(features).all() or not np.isfinite(scores).all():
        raise FloatingPointError("nonfinite encoder features or model scores")
    result = dict(
        feature_std=float(features.std(axis=0, dtype=np.float64).mean()),
        score_std=float(scores.std(dtype=np.float64)),
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        saturated_fraction=float(np.mean(np.abs(scores) >= 0.9999)),
        positive_prediction_fraction=float(np.mean(scores >= 0)),
    )
    if labels is not None:
        result.update(
            auc=float(roc_auc_score(labels, scores)),
            balanced_accuracy=float(balanced_accuracy_score(labels, scores >= 0)),
        )
    return result


def collapsed(metrics):
    """Explicit numerical-collapse gate, not a statistical performance claim."""
    return metrics["score_std"] < 1e-7 or metrics["feature_std"] < 1e-8


class CollapseDetectedError(RuntimeError):
    """An invalid constant-output trajectory is not a completed formal fit."""


def run(
    data_dir,
    output,
    *,
    dataset,
    arm,
    seed=0,
    c=0.1,
    device="cuda",
    backward_mode="split",
    gp_chunk_size=128,
):
    import torch

    from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier
    from pu_toolbox.experiment.image import fit_survey_image_preprocessing, transform_survey_images
    from pu_toolbox.experiment.strategies import SCARGenerator

    choices = {name: (initialization, rate) for name, initialization, rate in ARMS}
    if arm not in choices or not 0 < c <= 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid fixed arm, seed or labeling frequency")
    if backward_mode not in {"joint", "split"} or (
        backward_mode == "joint" and gp_chunk_size is not None
    ):
        raise ValueError("GP chunking requires split backward")
    if gp_chunk_size is not None and (type(gp_chunk_size) is not int or gp_chunk_size < 1):
        raise ValueError("invalid GP chunk size")
    if output.exists():
        raise ValueError("output exists; never overwrite")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no CPU fallback")
    root = Path(__file__).resolve().parents[1]
    registry_file = root / "pu_toolbox/experiment/survey_recipe_registry_v1_1.json"
    registry = json.loads(registry_file.read_text())
    profiles = registry["profiles"]
    profiles = profiles.values() if isinstance(profiles, dict) else profiles
    profile = next(p for p in profiles if p["method"] == "gradpu")
    params = dict(profile["candidate_pool"][0]["resolved_snapshot"]["params"])
    if params["batch_size"] != 256 or params["max_epochs"] != 200:
        raise ValueError("frozen reference budget changed")
    initialization, rate = choices[arm]
    params.update(learning_rate=rate, backward_mode=backward_mode, gp_chunk_size=gp_chunk_size)
    train_file, val_file = data_dir / "train.npz", data_dir / "clean_val.npz"
    with np.load(train_file, allow_pickle=False) as archive:
        raw, y = archive["X"], archive["y"]
    images, preprocessing = fit_survey_image_preprocessing(
        raw, dataset=dataset, train_augmentation="none"
    )
    del raw
    with np.load(val_file, allow_pickle=False) as archive:
        validation = transform_survey_images(archive["X"], preprocessing, role="clean_val")
        y_val = archive["y"]
    if set(np.unique(y)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
        raise ValueError("both partitions must have both binary classes")
    observed, generation = SCARGenerator().generate(images, y, c, seed)
    indices = probe_indices(y, seed)
    estimator, setup = build_candidate_image_classifier(
        "gradpu", preprocessing, seed=seed, device=device, gradpu_initialization=initialization
    )
    estimator.set_params(**params)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        diagnostic_only=True,
        formal_admission=False,
        completed_formal_fit=False,
        dataset=dataset,
        seed=seed,
        c=c,
        arm=arm,
        params=params,
        initialization=initialization,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
        ),
        script_sha256=file_sha(Path(__file__)),
        registry_sha256=file_sha(registry_file),
        inputs={
            p.name: dict(path=str(p.resolve()), sha256=file_sha(p)) for p in (train_file, val_file)
        },
        train_shape=list(images.shape),
        validation_shape=list(validation.shape),
        full_train=True,
        training_view="ts",
        generation=generation,
        image_setup=setup,
        test_set_read=False,
        validation_use="fixed diagnostics only; no selection or feedback",
        inference_batch_size=16,
        probe_indices=indices.tolist(),
        collapse_rule=(
            "score std < 1e-7 OR mean feature std < 1e-8; "
            "5 consecutive epochs, earliest stop epoch 10"
        ),
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        environment={
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMPY_MADVISE_HUGEPAGE",
            )
        },
        status="running",
        epochs_completed=0,
    )
    write_json(output / "configuration.json", record)
    begin = time.monotonic()
    consecutive = 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        with (output / "epochs.jsonl").open("x") as stream:

            def callback(epoch, fitted):
                nonlocal consecutive
                train = observe(fitted, images[indices])
                consecutive = consecutive + 1 if collapsed(train) else 0
                row = dict(
                    epoch=epoch + 1,
                    seconds=time.monotonic() - begin,
                    train_probe=train,
                    losses={k: v[-1] for k, v in fitted.history_.items() if k != "epoch"},
                )
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    row["clean_val"] = observe(fitted, validation, y_val)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
                record["epochs_completed"] = epoch + 1
                record["last_observation"] = row
                write_json(output / "progress.json", dict(status="running", **row))
                if epoch + 1 >= 10 and consecutive >= 5:
                    raise CollapseDetectedError("persistent collapsed train-probe signal")

            estimator.fit(images, observed, os_or_ts="ts", epoch_callback=callback)
        record["status"] = "full_budget_completed_not_formal"
        record["final_clean_val"] = observe(estimator, validation, y_val)
        torch.save(estimator.model_.state_dict(), output / "final_weights.pt")
    except CollapseDetectedError as error:
        record.update(status="failed_numerical_collapse", reason=str(error))
    except Exception as error:
        record.update(status="failed_runtime", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - begin
        if device.startswith("cuda"):
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(output / "report.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dataset", required=True, choices=("cifar10", "adni", "mnist", "fashion_mnist")
    )
    parser.add_argument("--arm", required=True, choices=[a[0] for a in ARMS])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backward-mode", choices=("joint", "split"), default="split")
    parser.add_argument("--gp-chunk-size", type=int, default=None)
    args = parser.parse_args()
    run(
        args.data_dir,
        args.output,
        dataset=args.dataset,
        arm=args.arm,
        seed=args.seed,
        c=args.c,
        device=args.device,
        backward_mode=args.backward_mode,
        gp_chunk_size=args.gp_chunk_size,
    )


if __name__ == "__main__":
    main()
