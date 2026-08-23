"""Per-config statistics and paired confidence intervals (contract §6)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(
    trials: pd.DataFrame,
    metric_columns: list[str],
    group_cols: tuple[str, ...] = ("algorithm", "scenario"),
) -> pd.DataFrame:
    """Group statistics; success rows only feed metric means (failures counted in rates)."""
    rows: list[dict] = []
    for group_keys, group in trials.groupby(list(group_cols), dropna=False):
        grouped = dict(zip(group_cols, group_keys, strict=True))
        status = group["status"]
        success_mask = status == "success"
        n_total = int(len(group))
        n_success = int(success_mask.sum())
        n_nonconverged = int((status == "nonconverged").sum())
        n_failed = int((status == "failed").sum())
        n_timeout = int((status == "timeout").sum())
        n_nan_inf = int((status == "nan_inf").sum())
        elapsed = group["elapsed_seconds"].astype(float)
        row: dict = {
            **grouped,
            "count": n_total,
            "success": n_success,
            "failures": n_failed + n_timeout + n_nan_inf,
            "nonconverged": n_nonconverged,
            "success_rate": n_success / n_total if n_total else float("nan"),
            "mean_elapsed_seconds": float(elapsed.mean()) if n_total else float("nan"),
            "p95_elapsed_seconds": float(elapsed.quantile(0.95)) if n_total else float("nan"),
            "warning_count": int(group["warning_count"].sum()),
        }
        sliced = group.loc[success_mask]
        for col in metric_columns:
            vals = sliced[col].astype(float)
            row[f"mean_{col}"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"std_{col}"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            lo, hi = _ci95(vals)
            row[f"ci95_low_{col}"] = lo
            row[f"ci95_high_{col}"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _ci95(vals: np.ndarray) -> tuple[float, float]:
    """95% confidence interval of the mean (t distribution, normal approx fallback)."""
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size < 2 or float(vals.std(ddof=1)) == 0.0:
        return mean, mean
    try:
        from scipy.stats import t  # type: ignore[import-not-found, unused-ignore]

        critical = float(t.ppf(0.975, df=vals.size - 1))
    except ImportError:
        critical = 1.96
    half = critical * float(vals.std(ddof=1)) / np.sqrt(vals.size)
    return float(mean - half), float(mean + half)


def paired_diff_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float, float]:
    """Same-seed paired difference (candidate - baseline); returns (mean, ci_low, ci_high)."""
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if len(baseline) != len(candidate) or len(baseline) == 0:
        raise ValueError("paired_diff_ci requires equal non-empty seed-aligned arrays")
    diff = candidate - baseline
    mean = float(diff.mean())
    half = 0.0
    if len(diff) > 1 and float(diff.std(ddof=1)) > 0.0:
        try:
            from scipy.stats import t  # type: ignore[import-not-found, unused-ignore]

            critical = float(t.ppf(0.975, df=len(diff) - 1))
        except ImportError:
            critical = 1.96
        half = critical * float(diff.std(ddof=1)) / np.sqrt(len(diff))
    return mean, mean - half, mean + half
