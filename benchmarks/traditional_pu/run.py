"""CLI entry for the traditional PU benchmark runner."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from benchmarks.traditional_pu.runner import (
    _iter_scenario_specs,
    _run_one_trial,
    load_config,
    run_trials,
    write_artifacts,
)


def _write_timeout_profile(config: dict, out_csv: str) -> None:
    """Log one minimal-scale trial per algorithm (seed=0), then exit without the grid."""
    data_cfg = config["data"]
    rows = []
    for method in config["methods"]:
        # First cell of the method: scar-small when the SCAR grid applies,
        # the first PNU cell for pnu-only configs (pnu has no scar cells).
        spec = next(s for m, _, s in _iter_scenario_specs(config) if m == method)
        row = _run_one_trial(method, spec, seed=0, config=config)
        rows.append(
            {
                "algorithm": method,
                "elapsed_seconds": row["elapsed_seconds"],
                "n_features": int(data_cfg["n_features"]),
                "n_samples": int(spec["n_samples"]),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pu-toolbox-traditional-benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--seed-set", choices=["development", "confirmation"], default="development"
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--timeout-profile",
        metavar="OUT_CSV",
        default=None,
        help="run one trial per algorithm to log runtime, then exit",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.timeout_profile is not None:
        _write_timeout_profile(config, args.timeout_profile)
        return 0
    trials, summary = run_trials(
        config, results_dir=args.results_dir, seed_set=args.seed_set, resume=not args.no_resume
    )
    write_artifacts(trials, summary, config, args.results_dir, seed_set=args.seed_set)
    n_success = int((trials["status"] == "success").sum())
    if len(trials) > 0 and n_success == 0:
        print("WARNING: no trial succeeded (all non-success)", file=sys.stderr)
        return 2
    print(f"done: {len(trials)} trials, {n_success} successful", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
