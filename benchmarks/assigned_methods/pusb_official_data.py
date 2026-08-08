"""Run the official-data PUSB adapter benchmark with provenance artifacts."""

# The benchmark follows the public estimator's X/y convention.
# ruff: noqa: N803, N806

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from pu_toolbox.estimators.bias_aware import PUSBKernelClassifier
from pu_toolbox.estimators.bias_aware.pusb_kernel import prior_quantile_predict


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an official-data PUSB benchmark config."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    if config.get("protocol") != "pusb_official_data":
        raise ValueError("config protocol must be 'pusb_official_data'")
    if config.get("paper_claim") is not False:
        raise ValueError("PUSB adapter runs must explicitly set paper_claim=false")
    if config.get("fidelity_level") not in {"smoke", "paper_protocol"}:
        raise ValueError("fidelity_level must be 'smoke' or 'paper_protocol'")
    experiment = config.get("experiment", {})
    if not experiment.get("seeds") or not experiment.get("class_priors"):
        raise ValueError("experiment seeds and class_priors must be non-empty")
    if not experiment.get("unlabeled_sizes"):
        raise ValueError("experiment unlabeled_sizes must be non-empty")
    density_ratio = config.get("density_ratio", {"enabled": False})
    if not isinstance(density_ratio.get("enabled"), bool):
        raise ValueError("density_ratio.enabled must be boolean")
    if density_ratio.get("enabled") and not isinstance(density_ratio.get("parameters", {}), dict):
        raise ValueError("density_ratio.parameters must be an object")
    return config


def load_ijcnn1(path: str | Path, *, expected_sha256: str | None = None):
    """Load the complete 49,990-row LIBSVM IJCNN1 file and map +1 to positive."""
    path = Path(path)
    actual_hash = _sha256(path)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError(
            f"IJCNN1 SHA-256 mismatch: expected {expected_sha256}, found {actual_hash}"
        )
    X_sparse, labels = load_svmlight_file(path)
    X = X_sparse.toarray().astype(float, copy=False)
    y = (np.asarray(labels) == 1).astype(int)
    if X.shape != (49_990, 22):
        raise ValueError(f"expected complete IJCNN1 shape (49990, 22), found {X.shape}")
    if np.unique(y).size != 2:
        raise ValueError("IJCNN1 must contain both positive and negative classes")
    maxima = np.max(X, axis=0)
    if np.any(maxima == 0):
        raise ValueError("official max normalization is undefined for a zero-maximum feature")
    return X / maxima, y, actual_hash


def construct_official_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    class_prior: float,
    unlabeled_size: int,
    positive_size: int = 400,
    test_size: int = 1000,
    holdout_size: int = 3000,
    selection_probability_power: float = 20.0,
) -> dict[str, np.ndarray]:
    """Reproduce the released PUSB script's selected-positive data construction."""
    if not 0.0 < class_prior < 1.0:
        raise ValueError("class_prior must be in (0, 1)")
    if min(unlabeled_size, positive_size, test_size, holdout_size) <= 0:
        raise ValueError("all sample sizes must be positive")
    if holdout_size >= len(X):
        raise ValueError("holdout_size must be smaller than the dataset")
    rng = np.random.RandomState(seed)

    # The released script appends a constant feature while sklearn also fits an intercept.
    pn_design = np.column_stack((X, np.ones(len(X), dtype=float)))
    pn_classifier = LogisticRegression(C=0.01, penalty="l2", max_iter=1000)
    pn_classifier.fit(pn_design, y)

    permutation = rng.permutation(len(X))
    train_indices = permutation[:-holdout_size]
    holdout_indices = permutation[-holdout_size:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_holdout, y_holdout = X[holdout_indices], y[holdout_indices]

    positive_pool = X_train[y_train == 1]
    positive_design = np.column_stack((positive_pool, np.ones(len(positive_pool), dtype=float)))
    selection_probability = pn_classifier.predict_proba(positive_design)[:, 1]
    selection_probability = selection_probability**selection_probability_power
    selection_probability /= np.max(selection_probability)

    selected_batches = []
    n_selected = 0
    for _ in range(10_000):
        accepted = positive_pool[selection_probability > rng.uniform(size=len(positive_pool))]
        if len(accepted):
            selected_batches.append(accepted)
            n_selected += len(accepted)
        if n_selected >= positive_size:
            break
    if n_selected < positive_size:
        raise RuntimeError("selection rejection sampling did not produce enough positives")
    selected_positive = np.concatenate(selected_batches, axis=0)
    selected_positive = selected_positive[rng.permutation(len(selected_positive))[:positive_size]]

    n_unlabeled_positive = int(unlabeled_size * class_prior)
    n_unlabeled_negative = unlabeled_size - n_unlabeled_positive
    unlabeled_positive_pool = X_train[y_train == 1]
    unlabeled_negative_pool = X_train[y_train == 0]
    if n_unlabeled_positive > len(unlabeled_positive_pool) or n_unlabeled_negative > len(
        unlabeled_negative_pool
    ):
        raise ValueError("requested unlabeled mixture exceeds the available class pools")
    unlabeled_positive = unlabeled_positive_pool[
        rng.permutation(len(unlabeled_positive_pool))[:n_unlabeled_positive]
    ]
    unlabeled_negative = unlabeled_negative_pool[
        rng.permutation(len(unlabeled_negative_pool))[:n_unlabeled_negative]
    ]
    unlabeled = np.concatenate((unlabeled_positive, unlabeled_negative), axis=0)
    X_pu = np.concatenate((selected_positive, unlabeled), axis=0)
    y_pu = np.r_[
        np.ones(positive_size, dtype=int),
        np.zeros(unlabeled_size, dtype=int),
    ]

    n_test_positive = int(test_size * class_prior)
    n_test_negative = test_size - n_test_positive
    test_positive_pool = X_holdout[y_holdout == 1]
    test_negative_pool = X_holdout[y_holdout == 0]
    if n_test_positive > len(test_positive_pool) or n_test_negative > len(test_negative_pool):
        raise ValueError("holdout split does not contain enough examples for the requested prior")
    test_positive = test_positive_pool[rng.permutation(len(test_positive_pool))[:n_test_positive]]
    test_negative = test_negative_pool[rng.permutation(len(test_negative_pool))[:n_test_negative]]
    X_test = np.concatenate((test_positive, test_negative), axis=0)
    y_test = np.r_[
        np.ones(n_test_positive, dtype=int),
        np.zeros(n_test_negative, dtype=int),
    ]
    return {
        "X_pu": X_pu,
        "y_pu": y_pu,
        "X_test": X_test,
        "y_test": y_test,
        "selection_probability": selection_probability,
    }


def _density_ratio_metrics(
    split: dict[str, np.ndarray],
    *,
    class_prior: float,
    seed: int,
    parameters: dict[str, Any],
    fitter=None,
) -> dict[str, Any]:
    """Fit the released uLSIF comparator and evaluate its quantile decision."""
    if fitter is None:
        try:
            from densratio import densratio as fitter
        except ImportError as error:
            raise RuntimeError(
                "density-ratio comparison requires densratio==0.3.0; "
                "install the project's research extra"
            ) from error

    positive = split["X_pu"][split["y_pu"] == 1]
    unlabeled = split["X_pu"][split["y_pu"] == 0]
    sigma_range = parameters.get("sigma_range", "auto")
    lambda_range = parameters.get("lambda_range", "auto")
    if isinstance(sigma_range, list):
        sigma_range = np.asarray(sigma_range, dtype=float)
    if isinstance(lambda_range, list):
        lambda_range = np.asarray(lambda_range, dtype=float)

    # densratio 0.3.0 draws centers from NumPy's module-level RNG.
    random_state = np.random.get_state()
    started = time.perf_counter()
    try:
        np.random.seed(seed)
        result = fitter(
            positive,
            unlabeled,
            alpha=0,
            sigma_range=sigma_range,
            lambda_range=lambda_range,
            kernel_num=int(parameters.get("kernel_num", 100)),
            verbose=bool(parameters.get("verbose", False)),
        )
    finally:
        np.random.set_state(random_state)

    scores = np.asarray(result.compute_density_ratio(split["X_test"]), dtype=float)
    if scores.shape != (len(split["X_test"]),) or not np.isfinite(scores).all():
        raise RuntimeError("densratio produced invalid test scores")
    predictions, threshold = prior_quantile_predict(scores, class_prior)
    return {
        "density_ratio_sigma": float(result.kernel_info.sigma),
        "density_ratio_reg_lambda": float(result.lambda_),
        "density_ratio_threshold": threshold,
        "density_ratio_accuracy": float(accuracy_score(split["y_test"], predictions)),
        "density_ratio_balanced_accuracy": float(
            balanced_accuracy_score(split["y_test"], predictions)
        ),
        "density_ratio_roc_auc": float(roc_auc_score(split["y_test"], scores)),
        "density_ratio_elapsed_seconds": time.perf_counter() - started,
    }


def run_trials(
    config: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    density_ratio_fitter=None,
    completed_keys: set[tuple[int, float, int]] | None = None,
    on_trial: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Run every configured prior, U size, and seed."""
    experiment = config["experiment"]
    model_parameters = config["model"]["parameters"]
    rows = []
    completed_keys = completed_keys or set()
    for unlabeled_size in experiment["unlabeled_sizes"]:
        for class_prior in experiment["class_priors"]:
            for seed in experiment["seeds"]:
                trial_key = (int(seed), float(class_prior), int(unlabeled_size))
                if trial_key in completed_keys:
                    continue
                started = time.perf_counter()
                split = construct_official_split(
                    X,
                    y,
                    seed=int(seed),
                    class_prior=float(class_prior),
                    unlabeled_size=int(unlabeled_size),
                    positive_size=int(experiment["positive_size"]),
                    test_size=int(experiment["test_size"]),
                    holdout_size=int(experiment["holdout_size"]),
                    selection_probability_power=float(experiment["selection_probability_power"]),
                )
                model = PUSBKernelClassifier(
                    random_state=int(seed),
                    **model_parameters,
                ).fit(split["X_pu"], split["y_pu"], class_prior=float(class_prior))
                scores = model.decision_function(split["X_test"])
                quantile_predictions, threshold = prior_quantile_predict(scores, float(class_prior))
                zero_predictions = (scores > 0.0).astype(int)
                density_ratio_metrics = {}
                density_ratio_config = config.get("density_ratio", {"enabled": False})
                if density_ratio_config.get("enabled"):
                    density_ratio_metrics = _density_ratio_metrics(
                        split,
                        class_prior=float(class_prior),
                        seed=int(seed),
                        parameters=density_ratio_config.get("parameters", {}),
                        fitter=density_ratio_fitter,
                    )
                row = {
                    "protocol": config["protocol"],
                    "fidelity_level": config["fidelity_level"],
                    "paper_claim": False,
                    "implementation_variant": config["model"]["variant"],
                    "seed": int(seed),
                    "class_prior": float(class_prior),
                    "positive_size": int(experiment["positive_size"]),
                    "unlabeled_size": int(unlabeled_size),
                    "test_size": len(split["y_test"]),
                    "sigma": model.sigma_,
                    "reg_lambda": model.reg_lambda_,
                    "cv_all_converged": bool(model.cv_convergence_.all()),
                    "cv_converged_fraction": float(model.cv_convergence_.mean()),
                    "optimizer_success": bool(model.optimization_result_.success),
                    "quantile_threshold": threshold,
                    "quantile_accuracy": float(
                        accuracy_score(split["y_test"], quantile_predictions)
                    ),
                    "quantile_balanced_accuracy": float(
                        balanced_accuracy_score(split["y_test"], quantile_predictions)
                    ),
                    "zero_threshold_accuracy": float(
                        accuracy_score(split["y_test"], zero_predictions)
                    ),
                    "roc_auc": float(roc_auc_score(split["y_test"], scores)),
                    "elapsed_seconds": time.perf_counter() - started,
                    **density_ratio_metrics,
                }
                rows.append(row)
                if on_trial is not None:
                    on_trial(row)
    return pd.DataFrame(rows)


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _completed_trial_keys(trials: pd.DataFrame) -> set[tuple[int, float, int]]:
    required = {"seed", "class_prior", "unlabeled_size"}
    if not required.issubset(trials.columns):
        raise ValueError("existing trials.csv is missing resume key columns")
    if trials.duplicated(list(required)).any():
        raise ValueError("existing trials.csv contains duplicate trial keys")
    return {
        (int(row.seed), float(row.class_prior), int(row.unlabeled_size))
        for row in trials.itertuples(index=False)
    }


def run_benchmark(
    config: dict[str, Any],
    *,
    data_root: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> pd.DataFrame:
    """Load verified data, execute trials, and write reproducibility artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output / "resolved_config.json"
    trials_path = output / "trials.csv"
    if resume:
        if not resolved_config_path.is_file():
            raise ValueError("cannot resume without resolved_config.json")
        existing_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise ValueError("resume config differs from the existing resolved_config.json")
        existing_trials = pd.read_csv(trials_path) if trials_path.is_file() else pd.DataFrame()
    else:
        existing_trials = pd.DataFrame()
        for stale_name in ("trials.csv", "summary.csv", "run_manifest.json"):
            stale_path = output / stale_name
            if stale_path.exists():
                stale_path.unlink()
        resolved_config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    completed_keys = _completed_trial_keys(existing_trials) if not existing_trials.empty else set()

    def checkpoint(row: dict[str, Any]) -> None:
        nonlocal existing_trials
        existing_trials = pd.concat(
            [existing_trials, pd.DataFrame([row])], ignore_index=True, sort=False
        )
        _write_csv_atomic(existing_trials, trials_path)

    dataset_path = Path(data_root) / config["dataset"]["path"]
    X, y, dataset_hash = load_ijcnn1(
        dataset_path,
        expected_sha256=config["dataset"].get("sha256"),
    )
    new_trials = run_trials(
        config,
        X,
        y,
        completed_keys=completed_keys,
        on_trial=checkpoint,
    )
    if existing_trials.empty and new_trials.empty:
        raise ValueError("benchmark configuration produced no trials")
    trials = existing_trials if not existing_trials.empty else new_trials
    _write_csv_atomic(trials, trials_path)
    metric_columns = [
        "cv_converged_fraction",
        "optimizer_success",
        "quantile_accuracy",
        "quantile_balanced_accuracy",
        "zero_threshold_accuracy",
        "roc_auc",
        "elapsed_seconds",
    ]
    metric_columns.extend(
        column
        for column in (
            "density_ratio_accuracy",
            "density_ratio_balanced_accuracy",
            "density_ratio_roc_auc",
            "density_ratio_elapsed_seconds",
        )
        if column in trials
    )
    summary = (
        trials.groupby(["class_prior", "unlabeled_size"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(output / "summary.csv", index=False)

    project_root = Path(__file__).resolve().parents[2]
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    status = _git_value(project_root, "status", "--porcelain")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol"],
        "fidelity_level": config["fidelity_level"],
        "paper_claim": False,
        "config_sha256": config_hash,
        "dataset": {
            "path": config["dataset"]["path"],
            "sha256": dataset_hash,
            "shape": list(X.shape),
        },
        "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "git_worktree_dirty": bool(status) if status is not None else None,
        "runner_sha256": _sha256(Path(__file__)),
        "n_trials": len(trials),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "scipy": _package_version("scipy"),
            "densratio": _package_version("densratio"),
        },
        "limitations": config.get("limitations", []),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trials = run_benchmark(
        config,
        data_root=args.data_root,
        output_dir=args.output,
        resume=args.resume,
    )
    print(f"Wrote {len(trials)} PUSB official-data trial(s) to {args.output}")


if __name__ == "__main__":
    main()
