"""Shared synthetic multi-seed runner for the first five assigned papers."""

# Public estimator calls follow the project's conventional X/y naming.
# ruff: noqa: N803, N806

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from pu_toolbox.estimators.bias_aware import LBEClassifier, PUSBClassifier
from pu_toolbox.estimators.risk import DistPUClassifier
from pu_toolbox.prior import ClassPriorEstimator, ReCPEEstimator

SUPPORTED_METHODS = (
    "class_prior_estimation",
    "recpe",
    "dist_pu",
    "pusb",
    "lbe",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark JSON configuration."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("benchmark config schema_version must be 1")
    if config.get("protocol") != "clean_room":
        raise ValueError("this runner accepts protocol='clean_room' only")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be a non-empty list of non-negative integers")
    methods = config.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("methods must be a non-empty object")
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"unsupported benchmark methods: {unknown}")
    return config


def _canonical_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _case_control_data(
    rng: np.random.Generator,
    data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_positive = int(data["n_positive"])
    n_unlabeled = int(data["n_unlabeled"])
    n_test = int(data["n_test"])
    n_features = int(data["n_features"])
    prior = float(data["class_prior"])
    separation = float(data["separation"])
    if min(n_positive, n_unlabeled, n_test, n_features) < 1 or not 0 < prior < 1:
        raise ValueError("case-control sizes must be positive and class_prior must be in (0, 1)")

    positive_mean = np.full(n_features, separation / 2)
    negative_mean = -positive_mean
    positive = rng.normal(loc=positive_mean, size=(n_positive, n_features))
    y_unlabeled = rng.binomial(1, prior, size=n_unlabeled)
    unlabeled = rng.normal(
        loc=np.where(y_unlabeled[:, None] == 1, positive_mean, negative_mean),
        size=(n_unlabeled, n_features),
    )
    X_train = np.vstack([positive, unlabeled])
    y_pu = np.r_[np.ones(n_positive, dtype=int), np.zeros(n_unlabeled, dtype=int)]

    y_test = rng.binomial(1, prior, size=n_test)
    X_test = rng.normal(
        loc=np.where(y_test[:, None] == 1, positive_mean, negative_mean),
        size=(n_test, n_features),
    )
    return X_train, y_pu, X_test, y_test


def _calibrate_propensity_intercept(scores: np.ndarray, target: float) -> float:
    low, high = -30.0, 30.0
    for _ in range(80):
        midpoint = (low + high) / 2
        if expit(scores + midpoint).mean() < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def _sar_data(
    rng: np.random.Generator,
    data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(data["n_train"])
    n_test = int(data["n_test"])
    n_features = int(data["n_features"])
    prior = float(data["class_prior"])
    separation = float(data["separation"])
    label_rate = float(data["label_rate"])
    strength = float(data["sar_strength"])
    if min(n_train, n_test, n_features) < 1 or not 0 < prior < 1 or not 0 < label_rate < 1:
        raise ValueError("SAR sizes must be positive and rates must be in (0, 1)")

    def sample(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_true = rng.binomial(1, prior, size=n_samples)
        direction = np.zeros(n_features)
        direction[0] = separation / 2
        X = rng.normal(loc=np.where(y_true[:, None] == 1, direction, -direction))
        raw = strength * X[:, 0]
        positive_raw = raw[y_true == 1]
        intercept = (
            _calibrate_propensity_intercept(positive_raw, label_rate)
            if len(positive_raw)
            else logit(label_rate)
        )
        propensity = expit(raw + intercept)
        return X, y_true, propensity

    X_train, y_train, propensity_train = sample(n_train)
    observed = (y_train * rng.binomial(1, propensity_train)).astype(int)
    if observed.sum() == 0:
        positive_indices = np.flatnonzero(y_train == 1)
        if not len(positive_indices):
            raise ValueError("SAR sample contains no positive examples")
        observed[positive_indices[np.argmax(propensity_train[positive_indices])]] = 1
    X_test, y_test, propensity_test = sample(n_test)
    return X_train, observed, X_test, y_test, propensity_test


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "predicted_positive_rate": float(predictions.mean()),
    }


def _run_prior_method(
    method: str,
    parameters: dict[str, Any],
    X_train: np.ndarray,
    y_pu: np.ndarray,
    true_prior: float,
) -> dict[str, float]:
    if method == "class_prior_estimation":
        estimator = ClassPriorEstimator(**parameters)
    else:
        estimator = ReCPEEstimator(**parameters)
    estimate = float(estimator.fit(X_train, y_pu).estimate())
    return {
        "class_prior_estimate": estimate,
        "prior_abs_error": abs(estimate - true_prior),
        "prior_signed_error": estimate - true_prior,
    }


def _run_classifier(
    method: str,
    parameters: dict[str, Any],
    seed: int,
    X_train: np.ndarray,
    y_pu: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    true_prior: float,
    propensity_test: np.ndarray | None,
) -> dict[str, float]:
    if method == "dist_pu":
        estimator = DistPUClassifier(
            class_prior=true_prior,
            random_state=seed,
            **parameters,
        )
        estimator.fit(X_train, y_pu)
    elif method == "pusb":
        estimator = PUSBClassifier(**parameters).fit(X_train, y_pu)
    else:
        estimator = LBEClassifier(**parameters).fit(
            X_train,
            y_pu,
            class_prior=true_prior,
        )

    scores = np.asarray(estimator.predict_proba(X_test)[:, 1], dtype=float)
    metrics = _classification_metrics(y_test, scores)
    metrics["score_brier"] = float(brier_score_loss(y_test, scores))
    if method == "lbe" and propensity_test is not None:
        estimated_propensity = estimator.propensity_model_.predict_proba(X_test)[:, 1]
        positive = y_test == 1
        if positive.any():
            metrics["propensity_mse_positive"] = float(
                np.mean((propensity_test[positive] - estimated_propensity[positive]) ** 2)
            )
            metrics["propensity_spearman_positive"] = float(
                spearmanr(
                    propensity_test[positive],
                    estimated_propensity[positive],
                ).statistic
            )
    return metrics


def run_trials(
    config: dict[str, Any],
    *,
    selected_methods: list[str] | None = None,
) -> pd.DataFrame:
    """Run all configured seeds and return one row per method/seed."""
    methods = list(config["methods"]) if selected_methods is None else selected_methods
    unknown = sorted(set(methods) - set(config["methods"]))
    if unknown:
        raise ValueError(f"methods are not configured: {unknown}")

    rows: list[dict[str, Any]] = []
    true_prior = float(config["data"]["class_prior"])
    for seed in config["seeds"]:
        for method in methods:
            rng = np.random.default_rng(seed)
            started = time.perf_counter()
            if method in {"class_prior_estimation", "recpe", "dist_pu"}:
                X_train, y_pu, X_test, y_test = _case_control_data(rng, config["data"])
                propensity_test = None
            else:
                X_train, y_pu, X_test, y_test, propensity_test = _sar_data(
                    rng,
                    config["data"],
                )
            parameters = config["methods"][method].get("parameters", {})
            if method in {"class_prior_estimation", "recpe"}:
                metrics = _run_prior_method(method, parameters, X_train, y_pu, true_prior)
            else:
                metrics = _run_classifier(
                    method,
                    parameters,
                    seed,
                    X_train,
                    y_pu,
                    X_test,
                    y_test,
                    true_prior,
                    propensity_test,
                )
            rows.append(
                {
                    "protocol": config["protocol"],
                    "method": method,
                    "implementation_variant": config["methods"][method]["variant"],
                    "seed": seed,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "true_class_prior": true_prior,
                    "observed_positive_rate": float(y_pu.mean()),
                    "elapsed_seconds": time.perf_counter() - started,
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "seed"]).reset_index(drop=True)


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every numeric metric by method without hiding raw trials."""
    identifiers = {
        "seed",
        "n_train",
        "n_test",
        "true_class_prior",
    }
    numeric = [
        column
        for column in trials.select_dtypes(include="number").columns
        if column not in identifiers
    ]
    rows = []
    for method, group in trials.groupby("method", sort=True):
        for metric in numeric:
            values = group[metric].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_dirty(project_root: Path) -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def run_benchmark(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    selected_methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute trials and persist raw, aggregate, and provenance artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trials = run_trials(config, selected_methods=selected_methods)
    summary = summarize_trials(trials)
    trials.to_csv(output / "trials.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)

    project_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol"],
        "paper_claim": False,
        "config_sha256": _canonical_hash(config),
        "git_commit": _git_commit(project_root),
        "git_worktree_dirty": _git_worktree_dirty(project_root),
        "runner_sha256": _file_sha256(Path(__file__)),
        "selected_methods": selected_methods or list(config["methods"]),
        "n_trials": len(trials),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "scipy": _package_version("scipy"),
            "torch": _package_version("torch"),
        },
        "limitations": config.get("limitations", []),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return trials, summary
