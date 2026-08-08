"""Validate PUSB Table 2 strict results and produce statistical reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

KEY_COLUMNS = ["dataset", "seed", "class_prior", "unlabeled_size"]
GROUP_COLUMNS = ["dataset", "class_prior", "unlabeled_size"]
METRICS = (
    "quantile_accuracy",
    "quantile_balanced_accuracy",
    "roc_auc",
    "density_ratio_accuracy",
    "density_ratio_balanced_accuracy",
    "density_ratio_roc_auc",
)


def validate_strict_trials(
    trials: pd.DataFrame,
    *,
    expected_trials: int | None = 4500,
    expected_repetitions: int | None = 100,
) -> None:
    """Reject incomplete, duplicated, truncated, or claim-unsafe strict results."""
    required = {
        *KEY_COLUMNS,
        "actual_unlabeled_size",
        "requested_test_size",
        "test_size",
        "paper_claim",
        "sampling_policy",
        "strictly_feasible_split",
        "cell_all_repetitions_feasible",
        "cv_all_converged",
        "optimizer_success",
        "elapsed_seconds",
        "density_ratio_elapsed_seconds",
        "sigma",
        "reg_lambda",
        *METRICS,
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"strict trials are missing columns: {missing}")
    if expected_trials is not None and len(trials) != expected_trials:
        raise ValueError(f"expected {expected_trials} strict trials, found {len(trials)}")
    if trials.duplicated(KEY_COLUMNS).any():
        raise ValueError("strict trials contain duplicate keys")
    if not (trials["actual_unlabeled_size"] == trials["unlabeled_size"]).all():
        raise ValueError("strict trials contain undersized unlabeled samples")
    if not (trials["test_size"] == trials["requested_test_size"]).all():
        raise ValueError("strict trials contain undersized test samples")
    if trials["paper_claim"].astype(bool).any():
        raise ValueError("strict feasible subset must keep paper_claim=false")
    if set(trials["sampling_policy"]) != {"strict_complete_cells"}:
        raise ValueError("unexpected sampling policy in strict trials")
    if not trials["strictly_feasible_split"].astype(bool).all():
        raise ValueError("strict trials contain infeasible splits")
    if not trials["cell_all_repetitions_feasible"].astype(bool).all():
        raise ValueError("strict trials contain incomplete-feasibility cells")
    finite_columns = [*METRICS, "elapsed_seconds", "density_ratio_elapsed_seconds"]
    if not np.isfinite(trials[finite_columns].astype(float).to_numpy()).all():
        raise ValueError("strict trials contain non-finite metrics or timings")
    if expected_repetitions is not None:
        counts = trials.groupby(GROUP_COLUMNS).size()
        if not (counts == expected_repetitions).all():
            raise ValueError("strict cells do not all contain the expected repetitions")


def _statistics(values: pd.Series) -> dict[str, float | int]:
    clean = values.astype(float).dropna()
    count = len(clean)
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if count > 1 else 0.0
    half_width = (
        float(student_t.ppf(0.975, count - 1) * std / math.sqrt(count)) if count > 1 else 0.0
    )
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_half_width": half_width,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_statistical_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cell t intervals and paired PUSB-minus-uLSIF differences."""
    rows = []
    for key, group in trials.groupby(GROUP_COLUMNS, sort=True):
        row: dict[str, Any] = dict(zip(GROUP_COLUMNS, key, strict=True))
        row["n"] = len(group)
        for metric in METRICS:
            stats = _statistics(group[metric])
            for statistic in ("mean", "std", "ci95_half_width", "ci95_lower", "ci95_upper"):
                row[f"{metric}_{statistic}"] = stats[statistic]
        for name, left, right in (
            (
                "paired_accuracy_difference",
                "quantile_accuracy",
                "density_ratio_accuracy",
            ),
            ("paired_roc_auc_difference", "roc_auc", "density_ratio_roc_auc"),
        ):
            difference = group[left].astype(float) - group[right].astype(float)
            stats = _statistics(difference)
            for statistic in ("mean", "std", "ci95_half_width", "ci95_lower", "ci95_upper"):
                row[f"{name}_{statistic}"] = stats[statistic]
            row[f"{name}_win_rate"] = float((difference > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _format_estimate(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_ci95_half_width']:.4f}"


def _render_markdown(summary: pd.DataFrame, report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# PUSB Table 2 严格可行子集报告",
        "",
        "## 声明边界",
        "",
        "本报告只覆盖 100 个官方 seed 均满足声明 P/U/test 样本量的 45 个单元。",
        "27 个不可行单元被排除，并保持 `paper_claim=false`；因此这不是论文 Table 2 的",
        "完整复现。",
        "",
        "## 完整性验证",
        "",
        f"- Trial 数：{report['validation']['n_trials']}",
        f"- 单元数：{report['validation']['n_cells']}",
        f"- 数据集数：{report['validation']['n_datasets']}",
        f"- CV 全部收敛：{report['convergence']['cv_all_converged']}",
        f"- 最终优化成功：{report['convergence']['optimizer_success']}",
        "- 区间：每个单元 100 次配对重复的双侧 95% Student-t 置信区间。",
        "",
        "## 聚合诊断",
        "",
        "以下为 4500 个选中 trial 的微平均，不包含论文协议中被排除的单元：",
        "",
        f"- PUSB accuracy：{aggregate['micro_average']['quantile_accuracy']:.4f}",
        f"- uLSIF accuracy：{aggregate['micro_average']['density_ratio_accuracy']:.4f}",
        f"- PUSB ROC-AUC：{aggregate['micro_average']['roc_auc']:.4f}",
        f"- uLSIF ROC-AUC：{aggregate['micro_average']['density_ratio_roc_auc']:.4f}",
        "- Accuracy 配对 CI 方向（PUSB 正向 / uLSIF 负向 / 跨 0）："
        f"{aggregate['paired_accuracy_cell_directions']['positive']} / "
        f"{aggregate['paired_accuracy_cell_directions']['negative']} / "
        f"{aggregate['paired_accuracy_cell_directions']['overlap_zero']}",
        "- ROC-AUC 配对 CI 方向（PUSB 正向 / uLSIF 负向 / 跨 0）："
        f"{aggregate['paired_roc_auc_cell_directions']['positive']} / "
        f"{aggregate['paired_roc_auc_cell_directions']['negative']} / "
        f"{aggregate['paired_roc_auc_cell_directions']['overlap_zero']}",
        "",
        "## 单元结果",
        "",
        "| 数据集 | Prior | U | PUSB acc | uLSIF acc | 配对 acc 差 | "
        "PUSB AUC | uLSIF AUC | 配对 AUC 差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {dataset} | {prior:.1f} | {u:d} | {pa} | {da} | {pd} | {pr} | {dr} | {rd} |".format(
                dataset=row["dataset"],
                prior=row["class_prior"],
                u=int(row["unlabeled_size"]),
                pa=_format_estimate(row, "quantile_accuracy"),
                da=_format_estimate(row, "density_ratio_accuracy"),
                pd=_format_estimate(row, "paired_accuracy_difference"),
                pr=_format_estimate(row, "roc_auc"),
                dr=_format_estimate(row, "density_ratio_roc_auc"),
                rd=_format_estimate(row, "paired_roc_auc_difference"),
            )
        )
    lines.extend(
        [
            "",
            "数值为均值 +/- 95% CI 半宽。差值为配对的 PUSB 减 uLSIF；正值表示该指标",
            "倾向 PUSB。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    trials_path: str | Path,
    *,
    output_dir: str | Path,
    expected_trials: int = 4500,
    expected_repetitions: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate aggregated trials and write CSV, JSON, and Markdown reports."""
    trials_path = Path(trials_path)
    trials = pd.read_csv(trials_path)
    validate_strict_trials(
        trials,
        expected_trials=expected_trials,
        expected_repetitions=expected_repetitions,
    )
    summary = build_statistical_summary(trials)
    accuracy_lower = summary["paired_accuracy_difference_ci95_lower"]
    accuracy_upper = summary["paired_accuracy_difference_ci95_upper"]
    auc_lower = summary["paired_roc_auc_difference_ci95_lower"]
    auc_upper = summary["paired_roc_auc_difference_ci95_upper"]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "pusb_table2_strict_statistical_report",
        "paper_claim": False,
        "fidelity_level": "paper_protocol_strict_feasible_subset",
        "input": {
            "trials_path": str(trials_path),
            "trials_sha256": _sha256(trials_path),
            "reporter_sha256": _sha256(Path(__file__)),
        },
        "validation": {
            "n_trials": len(trials),
            "n_cells": len(summary),
            "n_datasets": int(trials["dataset"].nunique()),
            "all_declared_sample_sizes_satisfied": True,
            "duplicate_trial_keys": 0,
        },
        "convergence": {
            "cv_all_converged": int(trials["cv_all_converged"].astype(bool).sum()),
            "optimizer_success": int(trials["optimizer_success"].astype(bool).sum()),
        },
        "aggregate": {
            "scope": "micro-average over the 4,500 selected strict trials",
            "micro_average": {metric: float(trials[metric].mean()) for metric in METRICS},
            "paired_accuracy_cell_directions": {
                "positive": int((accuracy_lower > 0).sum()),
                "negative": int((accuracy_upper < 0).sum()),
                "overlap_zero": int(((accuracy_lower <= 0) & (accuracy_upper >= 0)).sum()),
            },
            "paired_roc_auc_cell_directions": {
                "positive": int((auc_lower > 0).sum()),
                "negative": int((auc_upper < 0).sum()),
                "overlap_zero": int(((auc_lower <= 0) & (auc_upper >= 0)).sum()),
            },
        },
        "compute": {
            "full_trial_seconds_sum": float(trials["elapsed_seconds"].sum()),
            "density_ratio_seconds_sum": float(trials["density_ratio_elapsed_seconds"].sum()),
        },
        "selected_hyperparameters": {
            "pusb_sigma_counts": {
                str(key): int(value) for key, value in trials["sigma"].value_counts().items()
            },
            "pusb_reg_lambda_counts": {
                str(key): int(value) for key, value in trials["reg_lambda"].value_counts().items()
            },
        },
        "interval": "two-sided Student-t 95% CI over paired repetitions within each cell",
        "limitations": [
            "Only 45 fully feasible cells are included; 27 released-protocol cells are excluded.",
            "The report is a strict feasible-subset benchmark and not a complete "
            "Table 2 reproduction.",
        ],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "statistical_summary.csv", index=False)
    (output / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(_render_markdown(summary, report), encoding="utf-8")
    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, report = write_report(args.trials, output_dir=args.output)
    print(
        f"Validated {report['validation']['n_trials']} trials and wrote "
        f"{len(summary)} PUSB Table 2 cell summaries"
    )


if __name__ == "__main__":
    main()
