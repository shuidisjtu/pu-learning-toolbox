"""Train-only, explicitly non-formal GradPU encoder ablation; never selects on test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import numpy as np

ARMS = (
    ("legacy", "pytorch_default", 1e-3),
    ("relu_gain", "relu_gain", 1e-3),
    ("relu_gain_lr1e4", "relu_gain", 1e-4),
)


class DiagnosticStopError(Exception):
    """Stop a diagnostic without shortening the nominal beta schedule."""


def subset_indices(labels, samples, seed):
    """Stratify only training labels and retain both classes for PU generation."""
    if samples < 4 or samples > len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("diagnostic needs binary train labels and 4 <= samples <= train size")
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    count = int(round(samples * len(positive) / len(labels)))
    count = min(len(positive), samples - 1, max(1, samples - len(negative), count))
    rng = np.random.RandomState(seed)
    indices = np.r_[
        rng.choice(positive, count, replace=False),
        rng.choice(negative, samples - count, replace=False),
    ]
    rng.shuffle(indices)
    return indices


def signal_metrics(encoder, images, model=None):
    """Measure feature spread and score saturation on fixed training probes."""
    import torch

    target = encoder if model is None else model
    modes = [(layer, layer.training) for layer in target.modules()]
    try:
        target.eval()
        with torch.no_grad():
            features = encoder(images)
            result = dict(
                feature_std=float(features.std(dim=0, unbiased=False).mean()),
                feature_rms=float(features.square().mean().sqrt()),
            )
            if model is not None:
                scores = model(images).flatten()
                result.update(
                    score_std=float(scores.std(unbiased=False)),
                    saturated_fraction=float((scores.abs() >= 0.9999).float().mean()),
                )
            return result
    finally:
        for layer, mode in modes:
            layer.training = mode


def run(
    train_file,
    output,
    *,
    dataset,
    seeds,
    samples=128,
    batch_size=32,
    stop_epochs=5,
    c=0.1,
    device="cpu",
):
    import torch

    from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier
    from pu_toolbox.experiment.image import fit_survey_image_preprocessing
    from pu_toolbox.experiment.strategies import SCARGenerator

    if train_file.name != "train.npz":
        raise ValueError("only an explicit train.npz is accepted; no validation/test input")
    if not 1 <= stop_epochs <= 200 or batch_size < 1 or not 0 < c <= 1:
        raise ValueError("invalid diagnostic budget or label frequency")
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no CPU fallback")
    if output.exists():
        raise ValueError("diagnostic output exists; never overwrite")
    with np.load(train_file, allow_pickle=False) as archive:
        raw, labels = archive["X"], archive["y"]
    inputs, preprocessing = fit_survey_image_preprocessing(
        raw, dataset=dataset, train_augmentation="none"
    )
    del raw
    subsets = {seed: subset_indices(labels, samples, seed) for seed in seeds}
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sha = hashlib.sha256()
    with train_file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            sha.update(block)
    record = dict(
        diagnostic_only=True,
        formal_admission=False,
        completed_formal_fit=False,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
        ),
        train_file=str(train_file.resolve()),
        train_sha256=sha.hexdigest(),
        dataset=dataset,
        samples=samples,
        batch_size=batch_size,
        c=c,
        device=device,
        nominal_max_epochs=200,
        stop_epochs=stop_epochs,
        torch_version=torch.__version__,
        training_view="ts",
        test_set_read=False,
        arms=[],
    )
    (output / "configuration.json").write_text(json.dumps(record, indent=2) + "\n")
    for seed, indices in subsets.items():
        x = inputs[indices].copy()
        observed, generation = SCARGenerator().generate(x, labels[indices], c, seed)
        if not (observed == 1).any() or not (observed == 0).any():
            raise ValueError("diagnostic subset must contain labeled P and U; increase samples")
        probe = torch.as_tensor(x[: min(16, len(x))], device=device)
        for name, initialization, rate in ARMS:
            estimator, metadata = build_candidate_image_classifier(
                "gradpu",
                preprocessing,
                seed=seed,
                device=device,
                gradpu_initialization=initialization,
            )
            estimator.set_params(batch_size=batch_size, learning_rate=rate, max_epochs=200)
            arm = dict(
                name=name,
                seed=seed,
                learning_rate=rate,
                initialization=initialization,
                train_indices=indices.tolist(),
                generation=generation,
                image_setup=metadata,
                epochs=[],
            )
            begin = time.monotonic()
            with (output / f"{seed}-{name}.jsonl").open("x") as stream:

                def callback(
                    epoch, fitted, *, begin=begin, probe=probe, arm=arm, name=name, seed=seed
                ):
                    row = dict(
                        epoch=epoch + 1,
                        seconds=time.monotonic() - begin,
                        **{k: v[-1] for k, v in fitted.history_.items() if k != "epoch"},
                        **signal_metrics(fitted.encoder_, probe, fitted.model_),
                    )
                    arm["epochs"].append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(json.dumps(dict(arm=name, seed=seed, **row)), flush=True)
                    if epoch + 1 == stop_epochs:
                        raise DiagnosticStopError()

                with suppress(DiagnosticStopError):
                    estimator.fit(x, observed, os_or_ts="ts", epoch_callback=callback)
            arm["status"] = "diagnostic_complete_not_formal"
            record["arms"].append(arm)
            del estimator
    (output / "report.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset", required=True, choices=("cifar10", "mnist", "fashion_mnist", "adni")
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stop-epochs", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(
        args.train_file,
        args.output,
        dataset=args.dataset,
        seeds=args.seed or [0],
        samples=args.samples,
        batch_size=args.batch_size,
        stop_epochs=args.stop_epochs,
        c=args.c,
        device=args.device,
    )


if __name__ == "__main__":
    main()
