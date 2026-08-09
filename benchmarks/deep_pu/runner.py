# ruff: noqa: N803, N806

"""Unified clean-room benchmark runner for InfoMax PU, WConPU, and DGPU."""

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

from pu_toolbox.estimators.deep import (
    DGPUClassifier,
    InfoMaxPUClassifier,
    WeightedContrastivePUClassifier,
)

from .._common import canonical_hash

SUPPORTED_METHODS = ("infomax_pu", "weighted_contrastive_pu", "dgpu")


class GaussianConditionalGenerator:
    """Deterministic benchmark-only generator satisfying the DGPU protocol."""

    def __init__(self, *, noise_scale: float = 0.2) -> None:
        if noise_scale <= 0:
            raise ValueError("noise_scale must be positive.")
        self.noise_scale = noise_scale

    def fit(self, X: np.ndarray, y: np.ndarray, *, warm_start: bool = True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError("Gaussian generator X/y shapes do not align.")
        self.fit_calls_ = getattr(self, "fit_calls_", 0) + 1
        self.n_features_in_ = X.shape[1]
        self.global_mean_ = X.mean(axis=0)
        previous = getattr(self, "means_", {}) if warm_start else {}
        self.means_ = dict(previous)
        for class_label in (0, 1):
            if np.any(y == class_label):
                self.means_[class_label] = X[y == class_label].mean(axis=0)
            elif class_label not in self.means_:
                self.means_[class_label] = self.global_mean_.copy()
        return self

    def sample(
        self,
        n_samples: int,
        *,
        class_label: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        if not hasattr(self, "means_"):
            raise ValueError("GaussianConditionalGenerator must be fitted before sample().")
        if n_samples < 0 or class_label not in {0, 1}:
            raise ValueError("n_samples must be non-negative and class_label must be 0 or 1.")
        rng = np.random.RandomState(random_state)
        return self.means_[class_label] + self.noise_scale * rng.randn(
            n_samples, self.n_features_in_
        )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a deep-PU clean-room benchmark configuration."""
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
        raise ValueError(f"unsupported deep PU methods: {unknown}")
    data = config.get("data", {})
    required = {
        "n_positive",
        "n_unlabeled",
        "n_test",
        "n_features",
        "class_prior",
        "separation",
    }
    if not required <= set(data):
        raise ValueError(f"data is missing required fields: {sorted(required - set(data))}")
    if not 0.0 < float(data["class_prior"]) < 1.0:
        raise ValueError("data.class_prior must be in (0, 1)")
    if any(int(data[name]) < 1 for name in ("n_positive", "n_unlabeled", "n_test", "n_features")):
        raise ValueError("data sizes must be positive")
    return config


def make_case_control_data(
    seed: int,
    data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired train/test data and the test Bayes posterior."""
    rng = np.random.default_rng(seed)
    n_positive = int(data["n_positive"])
    n_unlabeled = int(data["n_unlabeled"])
    n_test = int(data["n_test"])
    n_features = int(data["n_features"])
    prior = float(data["class_prior"])
    separation = float(data["separation"])
    positive_mean = np.full(n_features, separation / 2.0)
    negative_mean = -positive_mean

    positive = rng.normal(positive_mean, 1.0, size=(n_positive, n_features))
    hidden_unlabeled = rng.binomial(1, prior, size=n_unlabeled)
    unlabeled = rng.normal(
        np.where(hidden_unlabeled[:, None] == 1, positive_mean, negative_mean),
        1.0,
    )
    X_train = np.vstack([positive, unlabeled]).astype(np.float32)
    y_pu = np.r_[np.ones(n_positive, dtype=int), np.zeros(n_unlabeled, dtype=int)]

    y_test = rng.binomial(1, prior, size=n_test)
    if len(np.unique(y_test)) < 2:
        y_test[:2] = [0, 1]
    X_test = rng.normal(
        np.where(y_test[:, None] == 1, positive_mean, negative_mean),
        1.0,
    ).astype(np.float32)
    posterior = expit(logit(prior) + separation * X_test.sum(axis=1))
    return X_train, y_pu, X_test, y_test.astype(int), posterior


def _classification_metrics(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    bayes_posterior: np.ndarray,
) -> dict[str, float]:
    probabilities = expit(np.clip(decision_scores, -50.0, 50.0))
    predictions = (decision_scores >= 0.0).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, decision_scores)),
        "average_precision": float(average_precision_score(y_true, decision_scores)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "predicted_positive_rate": float(predictions.mean()),
        "bayes_posterior_spearman": float(spearmanr(bayes_posterior, decision_scores).statistic),
    }


def _build_estimator(
    method: str,
    method_config: dict[str, Any],
    *,
    class_prior: float | None,
    seed: int,
):
    parameters = dict(method_config.get("parameters", {}))
    if "random_state" in parameters or "class_prior" in parameters:
        raise ValueError("random_state and class_prior are controlled by the runner")
    if method == "infomax_pu":
        prior_config = parameters.get("prior_estimator")
        if isinstance(prior_config, dict):
            from pu_toolbox.prior import KernelMeanPriorEstimator

            if prior_config.get("name") != "kernel_mean":
                raise ValueError("only the kernel_mean structured prior estimator is supported")
            parameters["prior_estimator"] = KernelMeanPriorEstimator(
                **prior_config.get("parameters", {})
            )
        return InfoMaxPUClassifier(
            class_prior=class_prior,
            random_state=seed,
            **parameters,
        )
    if method == "weighted_contrastive_pu":
        if class_prior is None:
            raise ValueError("weighted_contrastive_pu requires a known class_prior")
        vision_config = parameters.pop("vision", None)
        if vision_config is not None:
            from pu_toolbox.estimators.deep import (
                build_wconpu_augmentation,
                build_wconpu_backbone,
            )

            if not isinstance(vision_config, dict):
                raise ValueError("weighted_contrastive_pu vision config must be an object")
            backbone_config = dict(vision_config.get("backbone", {}))
            backbone_name = backbone_config.pop("name", None)
            if backbone_name is None:
                raise ValueError("weighted_contrastive_pu vision.backbone.name is required")
            parameters["encoder"] = build_wconpu_backbone(
                backbone_name,
                **backbone_config,
            )
            for config_name, parameter_name in (
                ("weak_augmentation", "weak_augmentation"),
                ("strong_augmentation", "strong_augmentation"),
            ):
                augmentation_config = dict(vision_config.get(config_name, {}))
                augmentation_name = augmentation_config.pop("name", None)
                if augmentation_name is None:
                    raise ValueError(
                        f"weighted_contrastive_pu vision.{config_name}.name is required"
                    )
                parameters[parameter_name] = build_wconpu_augmentation(
                    augmentation_name,
                    **augmentation_config,
                )
        return WeightedContrastivePUClassifier(
            class_prior,
            random_state=seed,
            **parameters,
        )
    if class_prior is None:
        raise ValueError("dgpu requires a known class_prior")
    generator_config = method_config.get("generator", {})
    generator = GaussianConditionalGenerator(
        noise_scale=float(generator_config.get("noise_scale", 0.2))
    )
    return DGPUClassifier(
        class_prior,
        generator,
        random_state=seed,
        **parameters,
    )


def _method_diagnostics(method: str, estimator: Any) -> dict[str, float]:
    if method == "infomax_pu":
        objective = estimator.representation_.history_["objective"]
        return {
            "class_prior_used": float(estimator.class_prior_),
            "representation_objective_first": float(objective[0]),
            "representation_objective_last": float(objective[-1]),
        }
    if method == "weighted_contrastive_pu":
        return {
            "class_prior_used": float(estimator.class_prior_),
            "queue_final_size": float(len(estimator.queue_embeddings_)),
            "classification_loss_last": float(estimator.history_["classification_loss"][-1]),
            "contrastive_loss_last": float(estimator.history_["contrastive_loss"][-1]),
        }
    return {
        "class_prior_used": float(estimator.class_prior_),
        "generator_fit_calls": float(estimator.generator_.fit_calls_),
        "generated_positive_total": float(
            sum(item["positive"] for item in estimator.generated_counts_)
        ),
        "predicted_positive_distribution": float(estimator.predicted_distribution_[1]),
    }


def run_trials(
    config: dict[str, Any],
    *,
    selected_methods: list[str] | None = None,
) -> pd.DataFrame:
    """Execute every configured method/seed and return raw trial rows."""
    methods = list(config["methods"]) if selected_methods is None else selected_methods
    unknown = sorted(set(methods) - set(config["methods"]))
    if unknown:
        raise ValueError(f"methods are not configured: {unknown}")

    rows: list[dict[str, Any]] = []
    prior = float(config["data"]["class_prior"])
    for seed in config["seeds"]:
        X_train, y_pu, X_test, y_test, posterior = make_case_control_data(seed, config["data"])
        for method in methods:
            started = time.perf_counter()
            estimator = _build_estimator(
                method,
                config["methods"][method],
                class_prior=prior,
                seed=seed,
            ).fit(X_train, y_pu)
            scores = np.asarray(estimator.decision_function(X_test), dtype=float)
            rows.append(
                {
                    "protocol": "clean_room",
                    "paper_claim": False,
                    "method": method,
                    "implementation_variant": config["methods"][method]["variant"],
                    "seed": seed,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "true_class_prior": prior,
                    "observed_positive_rate": float(y_pu.mean()),
                    "elapsed_seconds": time.perf_counter() - started,
                    **_classification_metrics(y_test, scores, posterior),
                    **_method_diagnostics(method, estimator),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "seed"]).reset_index(drop=True)


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each numeric metric by method, retaining raw trials separately."""
    identifiers = {"seed", "n_train", "n_test", "true_class_prior"}
    numeric = [
        column
        for column in trials.select_dtypes(include="number").columns
        if column not in identifiers and column != "paper_claim"
    ]
    rows: list[dict[str, Any]] = []
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


def _git_value(project_root: Path, command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
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


def run_benchmark(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    selected_methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run trials and persist raw, aggregate, configuration, and provenance files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trials = run_trials(config, selected_methods=selected_methods)
    summary = summarize_trials(trials)
    trials.to_csv(output / "trials.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)

    project_root = Path(__file__).resolve().parents[2]
    dirty = _git_value(project_root, ["git", "status", "--porcelain"])
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "clean_room",
        "paper_claim": False,
        "config_sha256": canonical_hash(config),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": _git_value(project_root, ["git", "rev-parse", "HEAD"]),
        "git_worktree_dirty": None if dirty is None else bool(dirty),
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
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trials, summary
