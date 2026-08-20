# ruff: noqa: N803, N806

"""Multi-seed public-data benchmark for joint-shift PU methods.

The bundled sklearn Breast Cancer Wisconsin dataset keeps the smoke protocol
download-free. The protocol is not the AISTATS paper's original dataset and
therefore always emits ``paper_claim=false``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import t
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pu_toolbox.estimators.research import JOINT_SHIFT_METHODS, build_joint_shift_estimator
from pu_toolbox.utils.serialization import json_safe

DEFAULT_METHODS = ("dynamic", "trpu", "tepu", "fine_tune", "mmd")

__all__ = ["JointShiftBenchmarkReport", "run_joint_shift_benchmark", "summarize_trials"]


@dataclass(frozen=True)
class JointShiftBenchmarkReport:
    """Trials and aggregate confidence intervals from a locked protocol."""

    configuration: dict[str, Any]
    trials: tuple[dict[str, Any], ...]
    summary: tuple[dict[str, Any], ...]
    split_audit: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "schema_version": "1.0",
                "analysis_type": "joint_shift_public_benchmark",
                "paper_claim": False,
                "configuration": self.configuration,
                "summary": list(self.summary),
                "split_audit": list(self.split_audit),
                "provenance": {
                    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
                    "dataset_loader": "sklearn.datasets.load_breast_cancer",
                    "sklearn_version": sklearn.__version__,
                    "original_paper_protocol": False,
                },
            }
        )

    def to_markdown(self) -> str:
        lines = [
            "# Joint-shift Public-data Benchmark",
            "",
            "- Dataset: `Breast Cancer Wisconsin (Diagnostic)`",
            f"- Seeds: `{len(self.configuration['seeds'])}`",
            "- Paper claim: `false` (toolbox smoke protocol, not the original paper protocol)",
            "",
            "| Method | Metric | Mean | Std | 95% CI | n |",
            "|---|---|---:|---:|---|---:|",
        ]
        for row in self.summary:
            lines.append(
                f"| `{row['method']}` | `{row['metric']}` | {row['mean']:.6f} | "
                f"{row['std']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | "
                f"{row['n']} |"
            )
        return "\n".join(lines) + "\n"

    def save(self, out_dir: str | Path) -> Path:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "summary.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (destination / "summary.md").write_text(self.to_markdown(), encoding="utf-8")
        pd.DataFrame(self.trials).to_csv(destination / "trials.csv", index=False)
        (destination / "config.json").write_text(
            json.dumps(self.configuration, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination


def _confidence_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    mean = float(values.mean())
    if len(values) < 2 or float(values.std(ddof=1)) == 0.0:
        return mean, mean
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical = float(t.ppf((1.0 + confidence) / 2.0, df=len(values) - 1))
    return mean - critical * sem, mean + critical * sem


def summarize_trials(
    trials: Sequence[dict[str, Any]], *, confidence: float = 0.95
) -> tuple[dict[str, Any], ...]:
    """Aggregate method metrics with Student-t confidence intervals."""
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1).")
    frame = pd.DataFrame(trials)
    required = {"method", "seed", "accuracy", "roc_auc"}
    if not required.issubset(frame.columns):
        raise ValueError(f"trials must contain columns {sorted(required)}.")
    rows: list[dict[str, Any]] = []
    for method in sorted(frame["method"].unique()):
        method_rows = frame[frame["method"] == method]
        if method_rows["seed"].duplicated().any():
            raise ValueError(f"method {method!r} contains duplicate seeds.")
        for metric in ("accuracy", "roc_auc"):
            values = method_rows[metric].to_numpy(dtype=float)
            if len(values) < 2:
                raise ValueError(
                    "Each method requires at least two seeds for uncertainty reporting."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"metric {metric!r} contains non-finite values.")
            low, high = _confidence_interval(values, confidence)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "ci_level": float(confidence),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return tuple(rows)


def _pu_labels(y_true: np.ndarray, propensity: float, rng: np.random.Generator) -> np.ndarray:
    labels = ((y_true == 1) & (rng.random(len(y_true)) < propensity)).astype(int)
    if labels.sum() < 2:
        positive_rows = np.flatnonzero(y_true == 1)
        labels[positive_rows[: min(2, len(positive_rows))]] = 1
    return labels


def _protocol_split(seed: int, target_pu_size: int) -> dict[str, Any]:
    dataset = load_breast_cancer()
    X = np.asarray(dataset.data, dtype=float)
    y = 1 - np.asarray(dataset.target, dtype=int)  # malignant is the positive class
    ids = np.arange(len(y), dtype=int)
    source_ids, target_pool_ids = train_test_split(
        ids, test_size=0.4, stratify=y, random_state=seed
    )
    if target_pu_size >= len(target_pool_ids) - 20:
        raise ValueError("target_pu_size must leave at least 20 target evaluation rows.")
    target_train_ids, target_test_ids = train_test_split(
        target_pool_ids,
        train_size=target_pu_size,
        stratify=y[target_pool_ids],
        random_state=seed + 10_000,
    )
    scaler = StandardScaler().fit(X[source_ids])
    X_source = scaler.transform(X[source_ids])
    X_target_train = scaler.transform(X[target_train_ids])
    X_target_test = scaler.transform(X[target_test_ids])
    source_truth = y[source_ids].copy()
    # Public-data concept-shift protocol: source relation differs in one stable feature region.
    flip_region = X_source[:, 0] > np.median(X_source[:, 0])
    source_truth[flip_region] = 1 - source_truth[flip_region]
    target_train_truth = y[target_train_ids]
    target_test_truth = y[target_test_ids]
    rng = np.random.default_rng(seed)
    source_pu = _pu_labels(source_truth, 0.5, rng)
    target_pu = _pu_labels(target_train_truth, 0.5, rng)
    return {
        "X_source": X_source,
        "y_source_pu": source_pu,
        "source_prior": float(source_truth.mean()),
        "X_target": X_target_train,
        "y_target_pu": target_pu,
        "target_prior": float(target_train_truth.mean()),
        "X_test": X_target_test,
        "y_test": target_test_truth,
        "source_ids": source_ids,
        "target_train_ids": target_train_ids,
        "target_test_ids": target_test_ids,
    }


def run_joint_shift_benchmark(
    *,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    methods: Sequence[str] = DEFAULT_METHODS,
    target_pu_size: int = 100,
    max_epochs: int = 50,
    hidden_dim: int = 32,
    feature_dim: int = 16,
    alpha: float = 0.1,
    beta: float = 0.5,
    mmd_weight: float = 0.1,
    confidence: float = 0.95,
    device: str = "cpu",
) -> JointShiftBenchmarkReport:
    """Run all methods on identical non-overlapping public-data splits."""
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if len(normalized_seeds) < 2 or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must contain at least two unique integers.")
    normalized_methods = tuple(methods)
    if not normalized_methods or len(set(normalized_methods)) != len(normalized_methods):
        raise ValueError("methods must be a non-empty unique sequence.")
    unknown = sorted(set(normalized_methods) - set(JOINT_SHIFT_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}.")
    trials: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for seed in normalized_seeds:
        split = _protocol_split(seed, target_pu_size)
        source_ids = set(split["source_ids"].tolist())
        target_train_ids = set(split["target_train_ids"].tolist())
        target_test_ids = set(split["target_test_ids"].tolist())
        audits.append(
            {
                "seed": seed,
                "n_source": len(source_ids),
                "n_target_pu": len(target_train_ids),
                "n_target_test": len(target_test_ids),
                "source_target_overlap": len(source_ids & target_train_ids),
                "train_test_overlap": len((source_ids | target_train_ids) & target_test_ids),
            }
        )
        for method in normalized_methods:
            estimator = build_joint_shift_estimator(
                method,
                alpha=alpha,
                beta=beta,
                mmd_weight=mmd_weight,
                hidden_dim=hidden_dim,
                feature_dim=feature_dim,
                max_epochs=max_epochs,
                random_state=seed,
                device=device,
            ).fit(
                split["X_source"],
                split["y_source_pu"],
                X_target=split["X_target"],
                y_target_pu=split["y_target_pu"],
                class_prior=split["source_prior"],
                target_class_prior=split["target_prior"],
            )
            prediction = estimator.predict(split["X_test"])
            score = estimator.decision_function(split["X_test"])
            trials.append(
                {
                    "seed": seed,
                    "method": method,
                    "accuracy": float(accuracy_score(split["y_test"], prediction)),
                    "roc_auc": float(roc_auc_score(split["y_test"], score)),
                    "source_class_prior": split["source_prior"],
                    "target_class_prior": split["target_prior"],
                    "n_target_test": len(split["y_test"]),
                }
            )
    configuration = {
        "dataset": "sklearn_breast_cancer",
        "protocol": "nonoverlap_concept_shift_smoke_v1",
        "paper_claim": False,
        "seeds": list(normalized_seeds),
        "methods": list(normalized_methods),
        "target_pu_size": target_pu_size,
        "max_epochs": max_epochs,
        "hidden_dim": hidden_dim,
        "feature_dim": feature_dim,
        "alpha": alpha,
        "beta": beta,
        "mmd_weight": mmd_weight,
        "confidence": confidence,
        "device": device,
    }
    return JointShiftBenchmarkReport(
        configuration=configuration,
        trials=tuple(trials),
        summary=summarize_trials(trials, confidence=confidence),
        split_audit=tuple(audits),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--target-pu-size", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = run_joint_shift_benchmark(
        seeds=[int(value) for value in args.seeds.split(",") if value],
        methods=[value for value in args.methods.split(",") if value],
        target_pu_size=args.target_pu_size,
        max_epochs=args.max_epochs,
        device=args.device,
    )
    report.save(args.out_dir)
    print(report.to_markdown())


if __name__ == "__main__":
    main()
