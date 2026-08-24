"""Per-ACS-iteration trajectory probe for KLDCE (M6 follow-up verification).

Records the full outer-loop state so the convergence-criterion change can
be verified against the pre-fix stall signature: QP optimal at iter 1 but
the criterion never firing.

Grid (small tier, n = 400): seeds 0..4 x pi-list x reg-list.  The Task-3
brief fixed pi=0.1; the controller-mandated extension adds
``--pi-list`` / ``--reg-strength`` so the limit-cycle question can be
quantified across pi and the L2 regularisation lambda.

Trajectory CSVs carry every ``acs_history_`` key (``mu_change`` is already
recorded by kldce and is passed through directly) plus the derived
``rel_mu_change`` / ``rel_obj_change`` columns used to classify
non-converged cells (period-2 limit cycle vs slow drift vs other).

Usage:
    uv run python benchmarks/traditional_pu/probe_kldce_acs.py \
        --out-dir benchmarks/traditional_pu/results/kldce_probe
    uv run python benchmarks/traditional_pu/probe_kldce_acs.py \
        --pi-list 0.1,0.3,0.5 --reg-strength 0.1,1.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pu_toolbox.estimators.risk.kldce import KLDCEClassifier


def _make_pu_data(seed: int, n_pos: int = 40, n_neg: int = 80, h: float = 0.3, d: int = 5):
    rng = np.random.RandomState(seed)
    x_pos = rng.randn(n_pos, d) + 1.5
    x_neg = rng.randn(n_neg, d) - 1.5
    x = np.vstack([x_pos, x_neg])
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
    hide = rng.choice(n_pos, size=int(n_pos * h), replace=False)
    y_pu = y_true.copy()
    y_pu[hide] = 0
    return x, y_pu


def probe_cell(seed: int, pi: float, reg_strength: float, out_dir: Path) -> dict:
    x, y_pu = _make_pu_data(seed, n_pos=int(pi * 400), n_neg=400 - int(pi * 400))
    clf = KLDCEClassifier(
        flip_probability=0.3,
        sigma="scale",
        reg_strength=reg_strength,
        max_acs_iter=300,
        max_dual_variables=2000,
        tol=1e-6,
        random_state=42,
    )
    clf.fit(x, y_pu)
    m_hat_norm = float(np.linalg.norm(clf.centroid_hat_))
    rows = []
    prev_obj = None
    for entry in clf.acs_history_:
        rel_obj = 0.0
        if prev_obj is not None and abs(prev_obj) > 1e-15:
            rel_obj = abs(entry["dual_obj"] - prev_obj) / abs(prev_obj)
        prev_obj = entry["dual_obj"]
        rows.append(
            {
                "seed": seed,
                "pi": pi,
                "reg_strength": reg_strength,
                "iter": entry["iter"],
                "dual_obj": entry["dual_obj"],
                "rel_obj_change": rel_obj,
                "mu_change": entry["mu_change"],
                "rel_mu_change": entry["mu_change"] / (1.0 + m_hat_norm),
                "eq_residual": entry["eq_residual"],
                "box_violation": entry["box_violation"],
                "gradient_norm": entry["gradient_norm"],
                "kkt_residual": entry["kkt_residual"],
                "kkt_nu": entry["kkt_nu"],
                "kkt_n_free": entry["kkt_n_free"],
                "centroid_constraint_residual": entry["centroid_constraint_residual"],
                "centroid_violation": entry["centroid_violation"],
                "degenerate": entry["degenerate_centroid_step"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / f"traj_seed{seed}_pi{pi:.1f}_reg{reg_strength:.1f}.csv", index=False)
    return {
        "seed": seed,
        "pi": pi,
        "reg_strength": reg_strength,
        "converged": bool(clf.converged_),
        "n_acs_iter": clf.n_acs_iter_,
        "final_kkt": rows[-1]["kkt_residual"] if rows else None,
        "final_mu": float(np.asarray(clf.centroid_opt_, dtype=float).ravel().min()),
    }


def classify_trajectory(frame: pd.DataFrame, tol: float = 1e-6) -> str:
    """Classify a non-converged trajectory by its outer-loop signature.

    - period-2 limit cycle: the degenerate flag alternates every iteration
      (mu jumps between the ellipsoid boundary and m_hat, mu_change stays
      O(1) — soft drift guard cannot help);
    - slow mu drift: rel_mu stays below sqrt(tol) and the QP feasibility
      terms below tol (the only shape the brief's soft plan can cover);
    - other: anything else (early stall, irregular).
    """
    n = len(frame)
    if n < 3:
        return "short trajectory"
    d = frame["degenerate"].astype(int).to_numpy()
    period2 = bool((d[2:] == d[:-2]).all()) and bool((d[1:] != d[:-1]).any())
    rel_mu = frame["rel_mu_change"].to_numpy()
    others = np.maximum(
        frame["rel_obj_change"].to_numpy(),
        np.maximum(frame["eq_residual"].to_numpy(), frame["box_violation"].to_numpy()),
    )
    if period2:
        half = frame["mu_change"].to_numpy()[n // 2 :]
        return (
            f"period-2 limit cycle (degenerate alternation), "
            f"last-half mean mu_change={half.mean():.3g}, max rel_mu={rel_mu.max():.3g}"
        )
    if float(rel_mu.max()) < np.sqrt(tol) and float(others.max()) < tol:
        return "slow mu drift (soft-eligible)"
    return f"other (max rel_mu={rel_mu.max():.3g}, max others={others.max():.3g})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmarks/traditional_pu/results/kldce_probe"),
    )
    parser.add_argument(
        "--pi-list",
        default="0.1",
        help="comma-separated class priors (grid: seeds 0..4 x pi-list x reg-list)",
    )
    parser.add_argument(
        "--reg-strength",
        default="1.0",
        help="comma-separated L2 regularisation strengths (control)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pi_list = [float(x) for x in args.pi_list.split(",")]
    reg_list = [float(x) for x in args.reg_strength.split(",")]
    cells = [(s, pi, reg) for s in range(5) for pi in pi_list for reg in reg_list]

    summary = []
    for idx, (s, pi, reg) in enumerate(cells, start=1):
        t0 = time.perf_counter()
        row = probe_cell(s, pi, reg, args.out_dir)
        elapsed = time.perf_counter() - t0
        summary.append(row)
        print(
            f"[{idx}/{len(cells)}] seed={s} pi={pi:.1f} reg={reg:.1f} "
            f"converged={row['converged']} n_acs_iter={row['n_acs_iter']} ({elapsed:.1f}s)"
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out_dir / "summary.csv", index=False)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    n_conv = int(summary_df["converged"].sum())
    print(f"\n== probe verdict: converged {n_conv}/{len(summary)} ==")
    for _, row in summary_df.iterrows():
        if row["converged"]:
            continue
        traj = pd.read_csv(
            args.out_dir
            / f"traj_seed{int(row['seed'])}_pi{row['pi']:.1f}_reg{row['reg_strength']:.1f}.csv"
        )
        print(
            f"  seed={int(row['seed'])} pi={row['pi']:.1f} reg={row['reg_strength']:.1f}: "
            f"n_acs_iter={int(row['n_acs_iter'])} -> {classify_trajectory(traj)}"
        )
    print(f"probe written to {args.out_dir}")


if __name__ == "__main__":
    main()
