"""Round plumbing for the seven-method tuning protocol (optimization plan §8 step 5).

Config generation for one-parameter-cluster candidate rounds, dev-stage
ranking with the plan §4 elimination chain, and the degenerate-rate
comparison (plan §5 condition 3).  Verdicts (paired CIs, success/budget
conditions) stay in compare.py — this module never re-implements them.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pu_toolbox.estimators.classic.elkan_noto import ElkanNotoClassifier
from pu_toolbox.estimators.classic.llsvm import LLSVMClassifier
from pu_toolbox.estimators.risk.kldce import KLDCEClassifier
from pu_toolbox.estimators.risk.ldce import LDCEClassifier
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.estimators.risk.pnu import PNUClassifier
from pu_toolbox.estimators.risk.upu import UPUClassifier

# method -> estimator class, mirrors check_baseline_configs.py's registry
# (extended one entry per tuning round).
_ESTIMATOR_CLASSES = {
    "kldce": KLDCEClassifier,
    "ldce": LDCEClassifier,
    "nnpu": NonNegativePUClassifier,
    "upu": UPUClassifier,
    "llsvm": LLSVMClassifier,
    "elkan_noto": ElkanNotoClassifier,
    "pnu": PNUClassifier,
}
_RUNNER_INJECTED = {"flip_probability", "random_state", "class_prior"}
# Constructor params that a JSON config cannot meaningfully tune: torch
# module/optimizer objects, the device string, and the Elkan-Noto base
# estimator instance (plan §4: optimizer/lr stay outside the round-1
# cluster).  Their constructor defaults (None) are the correct values, so
# they are also excluded from the constructor-defaults coverage check.
_NON_TUNABLE = {"model", "optimizer", "device", "base_estimator"}


def generate_round_configs(
    base_config_path,
    out_dir,
    candidates,
    *,
    method="kldce",
):
    """Write one candidate config per (name, overrides) pair (plan §4).

    Each produced config inherits the base config's data/timeouts/
    limitations, restricts ``methods`` to *method* with the base method
    params merged with *overrides*, and records the candidate in
    ``description``.  ``locks_source_defaults`` is stripped — candidates
    are not locked baselines.  Rejections (unknown params, runner-injected
    params, under-specified params, duplicate names) raise ``ValueError``
    before anything is written.
    """
    base_config_path = Path(base_config_path)
    out_dir = Path(out_dir)
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    if "data" not in base or method not in base.get("methods", {}):
        raise ValueError(f"base config must contain 'data' and methods[{method!r}]")
    cls = _ESTIMATOR_CLASSES.get(method)
    if cls is None:
        raise ValueError(f"no estimator class registered for method {method!r}")
    constructor_defaults = {
        name: param.default
        for name, param in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
        and param.default is not inspect.Parameter.empty
        and name not in _RUNNER_INJECTED
        and name not in _NON_TUNABLE
    }
    names = [name for name, _ in candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate candidate names: {names}")
    for name, overrides in candidates:
        for key in overrides:
            if key in _RUNNER_INJECTED:
                raise ValueError(f"{key!r} is runner-injected and cannot be tuned via config")
            if key in _NON_TUNABLE:
                raise ValueError(f"{key!r} is not tunable via JSON config")
            if key not in constructor_defaults:
                raise ValueError(f"{key!r} is not a constructor parameter of {method!r}")
        merged = {**base["methods"][method], **overrides}
        missing = set(constructor_defaults) - set(merged)
        if missing:
            raise ValueError(
                f"merged params for {name!r} miss constructor defaults: {sorted(missing)}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    for name, overrides in candidates:
        cfg = {key: value for key, value in base.items() if key != "locks_source_defaults"}
        cfg["methods"] = {method: {**base["methods"][method], **overrides}}
        base_description = base.get("description", "")
        cfg["description"] = (
            f"{base_description} | {method} tuning candidate {name!r} "
            f"with overrides {json.dumps(overrides, sort_keys=True)}"
        ).strip()
        path = out_dir / f"{name}.json"
        path.write_text(
            json.dumps(cfg, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        produced.append(path)
    return produced


def rank_candidates(
    dev_root, names, *, metric="pu_zero_one_risk", n_select=3, higher_is_better=False
):
    """Rank dev-stage candidates with the plan §4 elimination chain.

    Elimination order: success rate below 100% first, then any
    degenerate prediction (§3.3) among success rows; survivors rank by the
    pooled *metric* mean over the 30 main-grid success rows (ascending by
    default, descending when *higher_is_better*), ties broken by name.
    ``selected`` marks the first *n_select* survivors.  The returned frame
    keeps the input *names* order.

    The main grid rows (``scar-`` or ``pnu-``, never mixed) feed the
    elimination chain and the risk mean; methods that enter the SAR
    diagnostic line (nnPU-class) carry 30 extra ``sar-`` rows reported as
    diagnostics (``sar_n_total``/``sar_n_success``).  For the PNU
    tri-label grid the caller must pass an oracle metric (``pu_auc_roc``):
    binary PU metrics are unavailable there (contract §2.2).
    """
    dev_root = Path(dev_root)
    rows = []
    for name in names:
        trials_path = dev_root / name / "trials.csv"
        if not trials_path.exists():
            raise FileNotFoundError(f"missing trials.csv for candidate {name!r}: {trials_path}")
        trials = pd.read_csv(trials_path)
        scenario_col = trials["scenario"].astype(str)
        scar = trials[scenario_col.str.startswith("scar-")]
        pnu = trials[scenario_col.str.startswith("pnu-")]
        sar = trials[scenario_col.str.startswith("sar-")]
        known = scenario_col.str.startswith("scar-") | scenario_col.str.startswith("sar-")
        known = known | scenario_col.str.startswith("pnu-")
        other = trials[~known]
        if len(other):
            prefixes = sorted({s.split("-", 1)[0] for s in scenario_col[other.index]})
            raise ValueError(
                f"candidate {name!r}: unknown scenario prefix in dev trials: {prefixes}"
            )
        if len(scar) and len(pnu):
            raise ValueError(f"candidate {name!r}: mixed scar and pnu main-grid rows in dev trials")
        main = pd.concat([scar, pnu])
        if len(main) != 30:
            raise ValueError(f"candidate {name!r}: expected 30 main-grid dev rows, got {len(main)}")
        if len(sar) not in (0, 30):
            raise ValueError(f"candidate {name!r}: expected 0 or 30 sar dev rows, got {len(sar)}")
        if "degenerate_prediction" not in trials.columns:
            raise ValueError(f"candidate {name!r}: trials lack the degenerate_prediction column")
        success = main[main["status"] == "success"]
        n_success = int(len(success))
        degenerate_mask = success["degenerate_prediction"].fillna(False).astype(bool)
        n_degenerate = int(degenerate_mask.sum())
        degenerate_cells = sorted(success.loc[degenerate_mask, "scenario"].astype(str).unique())
        risk_values = pd.to_numeric(success[metric], errors="coerce")
        elapsed = pd.to_numeric(trials["elapsed_seconds"], errors="coerce").dropna()
        if n_success < len(main):
            reason = "success_rate < 100%"
        elif n_degenerate > 0:
            reason = "degenerate_prediction"
        else:
            reason = None
        rows.append(
            {
                "name": name,
                "n_total": int(len(trials)),
                "n_success": n_success,
                "success_rate": n_success / len(main),
                "sar_n_total": int(len(sar)),
                "sar_n_success": int((sar["status"] == "success").sum()),
                "n_degenerate": n_degenerate,
                "degenerate_cells": degenerate_cells,
                "n_nonconverged": int((trials["status"] == "nonconverged").sum()),
                "n_timeout": int((trials["status"] == "timeout").sum()),
                "n_failed": int((trials["status"] == "failed").sum()),
                "risk_mean": float(risk_values.mean()),
                "risk_std": float(risk_values.std()),
                "p95_elapsed_seconds": float(np.percentile(elapsed, 95)),
                "eliminated_reason": reason,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["eliminated_reason"].isna()].sort_values(
        ["risk_mean", "name"], ascending=[not higher_is_better, True], na_position="last"
    )
    table["rank"] = np.nan
    table.loc[eligible.index, "rank"] = range(1, len(eligible) + 1)
    table["selected"] = table["rank"].notna() & (table["rank"] <= n_select)
    return table


def degenerate_rates(results_dir):
    """Per-(algorithm, scenario) degenerate counts, success rows only.

    The degenerate flag is only defined for success rows, so rates use
    n_success as denominator.  A missing ``degenerate_prediction`` column
    (pre-step-4 artifacts, e.g. baseline_v2) yields NaN rates with
    ``degenerate_column=False`` — explicitly unknown, never zero.  The
    last row is the overall aggregate.
    """
    trials = pd.read_csv(Path(results_dir) / "trials.csv")
    has_column = "degenerate_prediction" in trials.columns
    success = trials[trials["status"] == "success"]
    records = []
    for (algorithm, scenario), group in success.groupby(["algorithm", "scenario"], sort=True):
        n_success = int(len(group))
        n_degenerate = (
            int(group["degenerate_prediction"].fillna(False).astype(bool).sum())
            if has_column
            else np.nan
        )
        records.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "n_success": n_success,
                "n_degenerate": n_degenerate,
                "degenerate_rate": (n_degenerate / n_success)
                if (has_column and n_success)
                else np.nan,
                "degenerate_column": has_column,
            }
        )
    total = int(len(success))
    overall_degenerate = (
        int(success["degenerate_prediction"].fillna(False).astype(bool).sum())
        if has_column
        else np.nan
    )
    records.append(
        {
            "algorithm": "overall",
            "scenario": "overall",
            "n_success": total,
            "n_degenerate": overall_degenerate,
            "degenerate_rate": (overall_degenerate / total) if (has_column and total) else np.nan,
            "degenerate_column": has_column,
        }
    )
    return pd.DataFrame(records)


def compare_degenerate_condition(baseline_dir, candidate_dir):
    """§5 condition 3: degenerate prediction rate must not increase.

    ``passes`` is True only when both sides are known and the candidate
    rate is at most the baseline rate; a baseline without the degenerate
    column (pre-step-4 artifact) yields ``passes=None`` and a reason
    pointing at a companion run of the frozen baseline config.  Promotion
    additionally requires the candidate rate to be exactly zero (§3.3:
    degenerate trials are not valid classification-quality evidence) —
    that stricter call is made in findings, not by this function.
    """
    base = degenerate_rates(baseline_dir)
    cand = degenerate_rates(candidate_dir)
    base_overall = base[base["scenario"] == "overall"].iloc[0]
    cand_overall = cand[cand["scenario"] == "overall"].iloc[0]
    candidate_rate = (
        float(cand_overall["degenerate_rate"]) if cand_overall["degenerate_column"] else None
    )
    if not base_overall["degenerate_column"]:
        return {
            "baseline_rate": None,
            "candidate_rate": candidate_rate,
            "passes": None,
            "reason": (
                "baseline degenerate column missing; requires a companion run "
                "of the frozen baseline config"
            ),
        }
    baseline_rate = float(base_overall["degenerate_rate"])
    return {
        "baseline_rate": baseline_rate,
        "candidate_rate": candidate_rate,
        "passes": bool(candidate_rate <= baseline_rate),
        "reason": "",
    }
