"""Paired-candidate verdicts between two results dirs (contract §6).

Contract §6: performance comparisons are made on same-seed paired differences
with a 95% CI, never on two means alone.  ``confirmed_improvement`` requires
all four conditions:

1. paired 95% CI on the confirmation seeds supports improvement;
2. success rate / non-convergence rate do not deteriorate unacceptably;
3. mean/P95 runtime stays within the declared budget;
4. data protocol, evaluation threshold and metric semantics identical.

Usage::

    uv run python -m benchmarks.traditional_pu.compare \
      --baseline-dir benchmarks/traditional_pu/results/confirmation_v1 \
      --candidate-dir benchmarks/traditional_pu/results/confirmation_v1_candidates \
      --budget-seconds 120 \
      --metric pu_zero_one_risk \
      --out benchmarks/traditional_pu/results/m6_candidate_verdict.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.traditional_pu.statistics import paired_diff_ci

# Metrics where a lower score is better; everything else is higher-is-better.
LOWER_IS_BETTER = {"pu_zero_one_risk", "pu_negative_rate", "brier_score",
                   "expected_calibration_error"}


def load_trials(results_dir: Path) -> pd.DataFrame:
    """Read a results dir's trial table; raise on a missing/incomplete run."""
    trials_file = results_dir / "trials.csv"
    if not trials_file.exists():
        raise FileNotFoundError(f"{trials_file} not found (run aborted before completion?)")
    return pd.read_csv(trials_file)


def alignment_is_valid(baseline: pd.DataFrame, candidate: pd.DataFrame) -> bool:
    """Contract §6 condition 4: identical (scenario, seed) protocol keys."""
    base_keys = set(zip(baseline["scenario"], baseline["seed"], strict=True))
    cand_keys = set(zip(candidate["scenario"], candidate["seed"], strict=True))
    return base_keys == cand_keys


def pair_cells(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Same-seed paired (candidate - baseline) per (algorithm, scenario) cell.

    Only pairs where both sides are ``success`` feed the CI; they are counted
    in ``n_paired``.  Rows with an empty pair set record ``paired_available=False``
    and NaN diffs — the baseline side has no performance value to compare
    against (every non-success row stays out of means per contract §6).
    """
    rows: list[dict] = []
    keys = sorted(
        set(zip(baseline["algorithm"], baseline["scenario"], strict=True))
        | set(zip(candidate["algorithm"], candidate["scenario"], strict=True))
    )
    for method, scenario in keys:
        b = baseline[(baseline["algorithm"] == method) & (baseline["scenario"] == scenario)]
        c = candidate[(candidate["algorithm"] == method) & (candidate["scenario"] == scenario)]
        b_ok = b[b["status"] == "success"].set_index("seed")
        c_ok = c[c["status"] == "success"].set_index("seed")
        # NaN/Inf values also stay out of the pair set (sanitized like summarize)
        common = b_ok.index.intersection(c_ok.index)
        vals_b = b_ok.loc[common, metric].astype(float).replace([np.inf, -np.inf], np.nan)
        vals_c = c_ok.loc[common, metric].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = vals_b.notna() & vals_c.notna()
        paired = valid.sum()
        row: dict = {
            "algorithm": method,
            "scenario": scenario,
            "n_paired": int(paired) if len(vals_b) else 0,
            "baseline_success_rate": float((b["status"] == "success").mean()),
            "candidate_success_rate": float((c["status"] == "success").mean()),
            "baseline_mean_elapsed_seconds": float(b["elapsed_seconds"].mean()),
            "candidate_mean_elapsed_seconds": float(c["elapsed_seconds"].mean()),
            "baseline_p95_elapsed_seconds": float(b["elapsed_seconds"].quantile(0.95)),
            "candidate_p95_elapsed_seconds": float(c["elapsed_seconds"].quantile(0.95)),
            "paired_available": bool(paired > 0),
        }
        if paired > 0:
            row["baseline_mean"] = float(vals_b[valid].mean())
            row["candidate_mean"] = float(vals_c[valid].mean())
            mean, lo, hi = paired_diff_ci(vals_b[valid].to_numpy(), vals_c[valid].to_numpy())
            row["diff_mean"] = mean
            row["diff_ci95_low"] = lo
            row["diff_ci95_high"] = hi
            # Condition 1: CI supports improvement in the metric's direction.
            if metric in LOWER_IS_BETTER:
                row["cond1_improves"] = bool(hi < 0)
            else:
                row["cond1_improves"] = bool(lo > 0)
        else:
            row["baseline_mean"] = float("nan")
            row["candidate_mean"] = float("nan")
            row["diff_mean"] = float("nan")
            row["diff_ci95_low"] = float("nan")
            row["diff_ci95_high"] = float("nan")
            row["cond1_improves"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def verdicts(table: pd.DataFrame, *, budget_seconds: float) -> pd.DataFrame:
    """Contract §6 verdict columns; cond2/cond3 are evaluated from raw columns."""
    table = table.copy()
    # Condition 2: success rate must not deteriorate (5pp tolerance, recorded).
    table["cond2_success_ok"] = (table["candidate_success_rate"]
                                 >= table["baseline_success_rate"] - 0.05)
    # Condition 3: P95 runtime within the pre-declared budget.
    table["cond3_budget_ok"] = table["candidate_p95_elapsed_seconds"] <= budget_seconds
    table["confirmed_improvement"] = (
        table["cond1_improves"] & table["cond2_success_ok"] & table["cond3_budget_ok"]
        & table["paired_available"]
    )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--metric", default="pu_zero_one_risk")
    parser.add_argument("--budget-seconds", type=float, default=120.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    baseline = load_trials(args.baseline_dir)
    candidate = load_trials(args.candidate_dir)
    if not alignment_is_valid(baseline, candidate):
        print(
            "error: (scenario, seed) protocol keys differ between runs — "
            "cannot pair (conditions 1/4)",
            file=sys.stderr,
        )
        return 2
    table = verdicts(pair_cells(baseline, candidate, args.metric),
                     budget_seconds=args.budget_seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    n_improved = int(table["confirmed_improvement"].sum())
    print(f"wrote {args.out} ({len(table)} cells, {n_improved} confirmed_improvement)")
    return 0 if n_improved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
