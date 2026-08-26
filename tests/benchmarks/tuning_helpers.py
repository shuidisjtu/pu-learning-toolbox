"""Shared builders for the traditional-PU tuning-round tests.

Not collected by pytest (no ``test_`` prefix), mirroring the top-level
``tests/helpers.py`` precedent.  Lives in ``tests/benchmarks/`` because
the builders are benchmark-specific (scar/sar scenario names, baseline
config shape).
"""

from __future__ import annotations

import json

import pandas as pd

SCENARIOS = [
    "scar-pi0.1-scalesmall",
    "scar-pi0.1-scalemid",
    "scar-pi0.3-scalesmall",
    "scar-pi0.3-scalemid",
    "scar-pi0.5-scalesmall",
    "scar-pi0.5-scalemid",
]

NNPU_BASE_PARAMS = {
    "model": None,
    "loss": "sigmoid",
    "beta": 0.0,
    "gamma": 1.0,
    "optimizer": None,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 20,
    "device": None,
}

UPU_BASE_PARAMS = {
    "loss": "double_hinge",
    "reg_lambda": 0.001,
    "basis": "linear",
    "kernel_width": None,
    "n_centers": None,
    "fit_intercept": True,
    "max_iter": 1000,
    "tol": 1e-6,
}


def base_config_path(tmp_path, *, methods=None, timeouts=None):
    """Persist a baseline-like config (locked kldce defaults) and return its path."""
    cfg = {
        "schema_version": 1,
        "protocol": "traditional_pu_baseline",
        "description": "base",
        "seed_set_development": [0, 1],
        "seed_set_confirmation": [2, 3],
        "data": {
            "n_samples_small": 50,
            "n_samples_mid": 50,
            "n_features": 5,
            "class_priors": [0.5],
            "separation": 2.0,
            "label_frequency": 0.5,
        },
        "methods": methods
        or {
            "kldce": {
                "sigma": "scale",
                "reg_strength": 1.0,
                "centroid_radius": 1.0,
                "mom_groups": 10,
                "covariance_ridge": 0.0,
                "max_acs_iter": 50,
                "max_inner_iter": 2000,
                "inner_tol": 1e-6,
                "tol": 1e-6,
            }
        },
        "timeouts": timeouts or {"kldce": 120},
        "limitations": [],
        "locks_source_defaults": True,
    }
    path = tmp_path / "base.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def make_row(scenario, seed, *, status="success", degenerate=False, risk=0.5, elapsed=1.0):
    return {
        "algorithm": "kldce",
        "scenario": scenario,
        "seed": seed,
        "status": status,
        "degenerate_prediction": degenerate,
        "pu_zero_one_risk": risk,
        "elapsed_seconds": elapsed,
    }


def write_candidate_trials(root, name, rows):
    """Persist a candidate's trials.csv under <root>/<name>/."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "trials.csv", index=False)


def full_grid(risk_per_scenario=None, *, degenerate_scenarios=(), failed_cells=()):
    """30 rows (6 scenarios × 5 seeds) with per-scenario risk and planted flaws."""
    risk_per_scenario = risk_per_scenario or {}
    rows = []
    for scenario in SCENARIOS:
        for seed in range(5):
            status = "failed" if (scenario, seed) in failed_cells else "success"
            rows.append(
                make_row(
                    scenario,
                    seed,
                    status=status,
                    degenerate=scenario in degenerate_scenarios and status == "success",
                    risk=risk_per_scenario.get(scenario, 0.5),
                )
            )
    return rows
