"""Traditional PU benchmark runner: grid orchestration, trial state machine, artifacts."""

# Dataset matrices follow sklearn's conventional X/y names.
# ruff: noqa: N803, N806

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks._common import canonical_hash, git_worktree_dirty
from benchmarks.traditional_pu.data import (
    PNU_RATIOS,
    is_ill_conditioned,
    make_pnu_data,
    make_sar_linear_data,
    make_scar_data,
    pnu_counts,
)
from benchmarks.traditional_pu.leakage_audit import (
    LeakageAuditError,
    check_trial_columns,
    guard_fit_labels,
)
from benchmarks.traditional_pu.statistics import summarize
from pu_toolbox.estimators.classic.elkan_noto import ElkanNotoClassifier
from pu_toolbox.estimators.classic.llsvm import LLSVMClassifier
from pu_toolbox.estimators.risk.kldce import KLDCEClassifier
from pu_toolbox.estimators.risk.ldce import LDCEClassifier
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.estimators.risk.pnu import PNUClassifier
from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.metrics.classification import (
    average_precision,
    balanced_accuracy,
    brier_score,
    calibration_bucket_stats,
    expected_calibration_error,
    pu_accuracy,
    pu_auc_roc,
    pu_estimated_precision,
    pu_f1,
    pu_negative_rate,
    pu_recall,
    pu_zero_one_risk,
)
from pu_toolbox.workflows._evaluation import extract_proba, extract_scores

PROTOCOL = "traditional_pu_baseline"
METRIC_COLUMNS = [
    "pu_zero_one_risk",
    "pu_recall",
    "pu_estimated_precision",
    "pu_auc_roc",
    "pu_accuracy",
    "pu_f1",
    "pu_negative_rate",
    "average_precision",
    "balanced_accuracy",
    "brier_score",
    "expected_calibration_error",
]
# Optimization plan §3.1/§3.3: per-trial prediction-side stats.  Kept
# outside METRIC_COLUMNS (flags, not contract metrics — summarize() and the
# non-finite check ignore them), but every row still carries them (NaN on
# non-success paths) so the trials column set stays stable.
_PREDICTION_COLUMNS = ("pred_positive_rate", "degenerate_prediction")

# Factory call signature is uniform:
#   ESTIMATOR_FACTORY[name](config["methods"][name], seed=seed, prior=prior, meta=meta)
ESTIMATOR_FACTORY = {
    "elkan_noto": lambda params, *, seed, prior, meta: ElkanNotoClassifier(
        **params, random_state=seed
    ),
    "upu": lambda params, *, seed, prior, meta: UPUClassifier(
        class_prior=prior, **params, random_state=seed
    ),
    "nnpu": lambda params, *, seed, prior, meta: NonNegativePUClassifier(
        class_prior=prior, **params, random_state=seed
    ),
    "pnu": lambda params, *, seed, prior, meta: PNUClassifier(
        class_prior=prior, **params, random_state=seed
    ),
    "ldce": lambda params, *, seed, prior, meta: LDCEClassifier(
        flip_probability=meta["real_h"], **params, random_state=seed
    ),
    "kldce": lambda params, *, seed, prior, meta: KLDCEClassifier(
        flip_probability=meta["real_h"], **params, random_state=seed
    ),
    "llsvm": lambda params, *, seed, prior, meta: LLSVMClassifier(**params, random_state=seed),
}

_PROBA_BASED_METRICS = ("brier_score", "expected_calibration_error")
_PRIOR_BASED_METRICS = ("pu_zero_one_risk", "pu_estimated_precision")
# Binary-PU-label metrics; unavailable in PNU tri-label cells (contract §2.2).
_PU_BINARY_METRICS = (*_PRIOR_BASED_METRICS, "pu_recall", "pu_negative_rate")

# Contract §4 "必须保存的诊断": per-algorithm diagnostic columns.  Every row
# carries the full column set (NaN where the attribute does not exist), so
# trials.csv shape is stable across configs.
_DIAG_COLUMNS = (
    "diag_converged",
    "diag_n_iter",
    "diag_c_estimate",
    "diag_cond_number",
    "diag_correction_fraction",
    "diag_n_epochs",
    "diag_n_acs_iter",
)


def _ldce_cond_number(estimator) -> float | None:
    """Condition number of the regularised centroid covariance (LDCE Eq. 10)."""
    covariance = getattr(estimator, "centroid_covariance_", None)
    if covariance is None:
        return None
    return float(np.linalg.cond(covariance))


def _nnpu_correction_fraction(estimator) -> float | None:
    """Last-epoch nnPU correction fraction (fraction of corrected steps)."""
    history = getattr(estimator, "history_", None)
    values = history.get("correction_fraction") if history else None
    return float(values[-1]) if values else None


def _llsvm_n_epochs(estimator) -> int | None:
    """LLSVM stopping epoch from get_pu_metadata (loss-history length)."""
    metadata = getattr(estimator, "get_pu_metadata", lambda: {})()
    return metadata.get("n_epochs")


# Contract §4 diagnostics per algorithm; upu/pnu expose no diagnostic
# attributes, so their rows keep NaN in every diag_* column.
_DIAG_EXTRACTORS: dict[str, dict[str, Callable[[object], float | int | bool | None]]] = {
    # ElkanNotoClassifier stores the c-estimate (Pr[y=1 | s=1] propensity)
    # as ``propensity_`` after fit; there is no ``c_`` attribute.
    "elkan_noto": {"diag_c_estimate": lambda e: getattr(e, "propensity_", None)},
    "ldce": {
        "diag_converged": lambda e: getattr(e, "converged_", None),
        "diag_n_iter": lambda e: getattr(e, "n_iter_", None),
        "diag_cond_number": _ldce_cond_number,
    },
    "kldce": {
        "diag_converged": lambda e: getattr(e, "converged_", None),
        "diag_n_acs_iter": lambda e: getattr(e, "n_acs_iter_", None),
    },
    "nnpu": {"diag_correction_fraction": _nnpu_correction_fraction},
    "llsvm": {"diag_n_epochs": _llsvm_n_epochs},
}


def _extract_diagnostics(estimator, method_name: str) -> dict:
    """Contract §4: per-trial diagnostic columns, NaN for absent attributes."""
    extractors = _DIAG_EXTRACTORS.get(method_name, {})
    return {col: extractors.get(col, lambda e: None)(estimator) for col in _DIAG_COLUMNS}


def load_config(path: str | Path) -> dict:
    """Validate and load a runner config (schema_version/protocol/seeds/methods)."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version: {cfg.get('schema_version')!r}")
    if cfg.get("protocol") != PROTOCOL:
        raise ValueError(f"expected protocol={PROTOCOL!r}, got {cfg.get('protocol')!r}")
    resolved_seeds: dict[str, list] = {}
    for key in ("seed_set_development", "seed_set_confirmation"):
        seeds = cfg.get(key) or cfg.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"{key!s} (or seeds) must be a non-empty integer list")
        if not all(isinstance(s, int) and s >= 0 for s in seeds):
            raise ValueError(f"{key} contains invalid seed: {seeds!r}")
        resolved_seeds[key] = seeds
    overlap = set(resolved_seeds["seed_set_development"]) & set(
        resolved_seeds["seed_set_confirmation"]
    )
    if overlap:
        raise ValueError(
            "seed_set_development and seed_set_confirmation must not overlap; "
            f"common seeds: {sorted(overlap)}"
        )
    methods = cfg.get("methods", {})
    if not methods:
        raise ValueError("methods must not be empty")
    for name, params in methods.items():
        if name not in ESTIMATOR_FACTORY:
            raise ValueError(f"unknown method {name!r} in config")
        if not isinstance(params, dict):
            raise ValueError(f"method {name!r} parameters must be a JSON object")
    if "pnu" in methods and not (cfg.get("data") or {}).get("pn_ratios"):
        raise ValueError("data.pn_ratios must be a non-empty list when methods includes 'pnu'")
    pn_ratios = (cfg.get("data") or {}).get("pn_ratios")
    if pn_ratios is not None:
        unknown = sorted(set(pn_ratios) - set(PNU_RATIOS))
        if unknown:
            raise ValueError(f"data.pn_ratios contains unknown ratios: {unknown}")
    return cfg


def _iter_scenario_specs(config: dict):
    """Yield (algorithm, scenario, scenario_spec) cells per contract §2 grid."""
    data = config["data"]
    priors = data.get("class_priors")
    if priors is None:  # eager default would KeyError: fall back explicitly
        priors = [float(data["class_prior"])]
    scales = (("small", data["n_samples_small"]), ("mid", data["n_samples_mid"]))
    for method in config["methods"]:
        # SCAR main grid: the six binary PU methods only; PNU runs its own
        # tri-label cells (contract §2.2) and must never see binary-PU labels.
        if method == "pnu":
            continue
        for pi in priors:
            for scale, n in scales:
                yield (
                    method,
                    f"scar-pi{pi}-scale{scale}",
                    {
                        "scenario": f"scar-pi{pi}-scale{scale}",
                        "kind": "scar",
                        "class_prior": pi,
                        "scale": scale,
                        "n_samples": n,
                    },
                )
        # linear SAR diagnostic line (never ranked, flagged in scenario name);
        # SAR data has a propensity mechanism, so the flip-model methods
        # LDCE/KLDCE (contract §2.3) are excluded — their h semantics is
        # flipped-label, not propensity.
        if method in ("ldce", "kldce"):
            continue
        for pi in priors:
            for scale, n in scales:
                yield (
                    method,
                    f"sar-pi{pi}-scale{scale}",
                    {
                        "scenario": f"sar-pi{pi}-scale{scale}",
                        "kind": "sar",
                        "class_prior": pi,
                        "scale": scale,
                        "n_samples": n,
                    },
                )
    # PNU protocol: cells only in configs whose methods include "pnu".
    # The protocol itself needs no pi (contract §2.2), but the estimator
    # requires one — thread the config data prior (first class_priors entry,
    # else the scalar class_prior) into every PNU cell.
    if "pnu" in config["methods"]:
        pnu_prior = float(priors[0]) if priors else float(data["class_prior"])
        for ratio_label in data.get("pn_ratios", []):
            for scale, n in scales:
                yield (
                    "pnu",
                    f"pnu-{ratio_label}-scale{scale}",
                    {
                        "scenario": f"pnu-{ratio_label}-scale{scale}",
                        "kind": "pnu",
                        "ratio": ratio_label,
                        "scale": scale,
                        "n_samples": n,
                        "class_prior": pnu_prior,
                    },
                )


def _run_one_trial(method_name: str, scenario_spec: dict, seed: int, config: dict) -> dict:
    """One trial → one result row (contract §6 status machine).

    Fit uses the observable PU labels only; ``y_true`` is used exclusively
    for the final supervised-oracle evaluation columns (no parameter,
    threshold or early-stopping selection ever touches it).
    """
    start = time.monotonic()
    row: dict = {
        "algorithm": method_name,
        "scenario": scenario_spec["scenario"],
        "seed": seed,
    }
    try:
        status, reason, metrics = _trial_body(row, method_name, scenario_spec, seed, config)
    except TimeoutError as exc:
        status, reason, metrics = "timeout", repr(exc), empty_metrics()
    except LeakageAuditError:
        # A leakage hit must block the run (design §4), not degrade into a
        # "failed" row — re-raise so it propagates through run_trials to the CLI.
        raise
    except Exception as exc:  # noqa: BLE001 - isolation per trial
        status, reason, metrics = "failed", repr(exc)[:500], empty_metrics()
    row["elapsed_seconds"] = float(time.monotonic() - start)
    row.setdefault("warning_count", 0)
    return {**row, "status": status, "failure_reason": reason, **metrics}


def _trial_body(row: dict, method_name: str, scenario_spec: dict, seed: int, config: dict):
    """Inner trial logic; multiple returns allowed (a single exit is not needed here)."""
    row["params"] = json.dumps(config["methods"][method_name], sort_keys=True)
    data_cfg = config["data"]
    n_features = data_cfg["n_features"]
    separation = data_cfg["separation"]
    kind = scenario_spec["kind"]
    class_prior = scenario_spec.get("class_prior")
    if kind == "scar":
        # label_frequency is SCAR/SAR-only; PNU cells (contract §2.2) never
        # carry it in config, so read it lazily inside these branches.
        label_frequency = data_cfg["label_frequency"]
        # Ill-conditioning depends only on (pi, h) — known from config, before
        # any data work: exclude the unit early (contract §2.3) and record the
        # reason verbatim so ill-conditioned cells never look like generic
        # failures.
        if is_ill_conditioned(class_prior, 1.0 - label_frequency):
            row.update(
                {
                    "mechanism": "scar",
                    "class_prior": class_prior,
                    "label_frequency": label_frequency,
                    "real_h": 1.0 - label_frequency,
                    "pi_h_well_conditioned": False,
                }
            )
            return "failed", "ill_conditioned_1_minus_2pih", empty_metrics()
        X, y_pu, y_true, meta = make_scar_data(
            n_samples=scenario_spec["n_samples"],
            n_features=n_features,
            class_prior=class_prior,
            separation=separation,
            label_frequency=label_frequency,
            random_state=seed,
        )
        y_fit = y_pu
    elif kind == "sar":
        label_frequency = data_cfg["label_frequency"]
        X, y_pu, y_true, meta = make_sar_linear_data(
            n_samples=scenario_spec["n_samples"],
            n_features=n_features,
            class_prior=class_prior,
            separation=separation,
            label_frequency=label_frequency,
            strength=data_cfg["sar_strength"],
            random_state=seed,
        )
        y_fit = y_pu
    else:  # pnu
        n_p, n_n, n_u = pnu_counts(scenario_spec["ratio"], scenario_spec["n_samples"])
        X, y_pnu, y_true = make_pnu_data(
            n_p=n_p,
            n_n=n_n,
            n_u=n_u,
            n_features=n_features,
            separation=separation,
            random_state=seed,
        )
        row["n_p"] = n_p
        row["n_n"] = n_n
        row["n_u"] = n_u
        y_fit = y_pnu
        # make_pnu_data's ground truth is {+1, -1}; oracle metrics expect
        # {0, 1} — normalize once here (PNU protocol, contract §4 metrics).
        y_true = np.where(y_true == 1, 1, 0)
        meta = {"real_h": float("nan"), "pi_h_well_conditioned": True}
    row.update(meta)
    if kind != "pnu" and not meta["pi_h_well_conditioned"]:
        return "failed", "ill_conditioned_1_minus_2pih", empty_metrics()
    estimator = ESTIMATOR_FACTORY[method_name](
        config["methods"][method_name], seed=seed, prior=class_prior, meta=meta
    )
    # Leakage gate (design §3.3): must stay directly above the fit call it
    # guards — if fit ever changes arguments, this guard moves with it.
    guard_fit_labels(y_fit, y_true)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y_fit, class_prior=class_prior)
    row["warning_count"] = int(len(caught))
    row.update(_extract_diagnostics(estimator, method_name))
    if getattr(estimator, "converged_", None) is not None and not estimator.converged_:
        return "nonconverged", "explicit non-convergence", empty_metrics()
    pred = estimator.predict(X)
    positive_rate = float(np.mean(pred == 1))
    # Optimization plan §3.3: all-positive / all-negative predictions are
    # degenerate — kept for diagnostics, excluded from promotion evidence.
    row["pred_positive_rate"] = positive_rate
    row["degenerate_prediction"] = bool(positive_rate in (0.0, 1.0))
    scores = extract_scores(estimator, X)
    proba = extract_proba(estimator, X)
    metrics: dict = {}
    for name in METRIC_COLUMNS:
        # Missing prerequisites are recorded as NaN + reason (contract §1):
        # an unavailable metric must never fail the trial nor degrade silently.
        if name in _PROBA_BASED_METRICS and proba is None:
            metrics[name] = np.nan
            metrics[f"{name}_unavailable_reason"] = "requires predict_proba"
            if name == "expected_calibration_error":
                metrics["ece_buckets_api"] = np.nan
            continue
        if name in _PRIOR_BASED_METRICS and class_prior is None:
            metrics[name] = np.nan
            metrics[f"{name}_unavailable_reason"] = "requires class prior"
            continue
        if name in _PU_BINARY_METRICS and kind == "pnu":
            metrics[name] = np.nan
            metrics[f"{name}_unavailable_reason"] = "requires binary PU labels (PNU tri-label cell)"
            continue
        try:
            metrics[name] = _metric_value(name, y_fit, pred, scores, y_true, proba, class_prior)
            if name == "expected_calibration_error":
                # Contract §3: persist the full per-bin calibration
                # diagnostics alongside the scalar ECE (loads-able JSON).
                buckets = calibration_bucket_stats(y_true, proba, n_bins=10)
                metrics["ece_buckets_api"] = json.dumps(buckets, sort_keys=True)
        except ValueError:
            if name not in _PROBA_BASED_METRICS:
                # A ValueError from any other metric (e.g. pu_auc_roc on
                # degenerate labels) is a genuine trial failure, not an
                # availability gap: propagate to _run_one_trial's
                # failure-isolation backstop (contract §6).
                raise
            # ElkanNotoClassifier.predict_proba is documented to exceed [0, 1].
            metrics[name] = np.nan
            metrics[f"{name}_unavailable_reason"] = "invalid probability output (not in [0,1])"
            if name == "expected_calibration_error":
                metrics["ece_buckets_api"] = np.nan
    row.update(metrics)
    non_finite = [
        name
        for name in METRIC_COLUMNS
        if f"{name}_unavailable_reason" not in metrics and not np.isfinite(metrics[name])
    ]
    if non_finite:
        # any non-finite metric in a success path marks nan_inf (values kept)
        return "nan_inf", f"non-finite metric: {non_finite[0]}", metrics
    return "success", "", metrics


def _metric_value(name, y_fit, pred, scores, y_true, proba, prior):
    """Dispatch same metric family as workflows._evaluation.compute_metric."""
    if name == "pu_zero_one_risk":
        value = pu_zero_one_risk(y_fit, scores, prior)
    elif name == "pu_recall":
        value = pu_recall(y_fit, pred)
    elif name == "pu_estimated_precision":
        value = pu_estimated_precision(y_fit, pred, prior)
    elif name == "pu_negative_rate":
        value = pu_negative_rate(y_fit, pred)
    elif name == "pu_auc_roc":
        value = pu_auc_roc(y_true, scores)
    elif name == "pu_accuracy":
        value = pu_accuracy(y_true, pred)
    elif name == "pu_f1":
        value = pu_f1(y_true, pred)
    elif name == "average_precision":
        value = average_precision(y_true, scores)
    elif name == "balanced_accuracy":
        value = balanced_accuracy(y_true, pred)
    elif name == "brier_score":
        value = brier_score(y_true, proba)
    elif name == "expected_calibration_error":
        value = expected_calibration_error(y_true, proba)
    else:
        raise AssertionError(name)
    return float(value)


def empty_metrics() -> dict:
    metrics = {name: np.nan for name in METRIC_COLUMNS}
    metrics.update({name: np.nan for name in _PREDICTION_COLUMNS})
    return metrics


def run_trials(
    config: dict,
    *,
    results_dir: str | Path,
    seed_set: str = "development",
    resume: bool = True,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the grid; skip cells already persisted when resume=True.

    Each trial runs in its own worker thread with a timeout guard
    (ThreadPoolExecutor).  A timed-out future is recorded as
    ``status="timeout"`` and its (still-running) thread is abandoned —
    worker threads cannot be killed on CPython, so the row result is
    discarded after timeout.  Trials are appended to ``trials.csv`` in
    ``results_dir`` so later invocations can resume.

    Returns ``(trials, summary)``; the summary is computed with
    ``statistics.summarize`` (success rows only feed metric means).
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeoutError

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = (
        config["seed_set_development"]
        if seed_set == "development"
        else config["seed_set_confirmation"]
    )
    trials_file = results_dir / "trials.csv"
    fingerprint_file = results_dir / ".config_sha256"
    # Same digest as run_manifest.json's config_sha256, so the sidecar and
    # the manifest agree on one config identity per results dir.
    config_hash = canonical_hash(config)
    previous_rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    if resume and trials_file.exists():
        if (
            fingerprint_file.exists()
            and fingerprint_file.read_text(encoding="utf-8").strip() != config_hash
        ):
            raise ValueError(
                "resume config mismatch (results dir was created with a different config)"
            )
        previous = pd.read_csv(trials_file)
        # Leakage gate (design §2.1): a pre-gate trials.csv carrying raw
        # label columns must never be resumed into a promotable result set.
        label_columns = check_trial_columns(previous.columns)
        if label_columns:
            raise LeakageAuditError(
                f"trials.csv contains raw label columns {label_columns!r}; "
                "the run is blocked (design §2.1)"
            )
        previous_rows = previous.to_dict("records")
        done = {tuple(r) for r in previous[["algorithm", "scenario", "seed"]].to_numpy()}
    timeouts = config.get("timeouts", {})
    cells = list(_iter_scenario_specs(config))
    total = len(cells) * len(seeds)
    rows: list[dict] = []
    executed = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        for method, scenario, spec in cells:
            for seed in seeds:
                key = (method, scenario, seed)
                if key in done:
                    continue
                executed += 1
                timeout_s = float(timeouts.get(method, 600.0))
                future = executor.submit(_run_one_trial, method, spec, seed, config)
                try:
                    row = future.result(timeout=timeout_s)
                except FutureTimeoutError:
                    row = {
                        "algorithm": method,
                        "scenario": scenario,
                        "seed": seed,
                        "status": "timeout",
                        "failure_reason": f"exceeded {timeout_s}s",
                        "elapsed_seconds": timeout_s,
                        "warning_count": 0,
                        **empty_metrics(),
                    }
                # Leakage gate (design §2.1): block before the row is
                # appended or persisted, so trials.csv is never contaminated.
                label_columns = check_trial_columns(row)
                if label_columns:
                    raise LeakageAuditError(
                        f"trial row for ({method}, {scenario}, seed={seed}) "
                        f"carries raw label columns {label_columns!r}; "
                        "the run is blocked (design §2.1)"
                    )
                rows.append(row)
                # Incremental persist: an interrupted grid (ctrl+c, host kill)
                # leaves every completed row on disk so a later resume
                # continues from the interruption point instead of rerunning.
                pd.DataFrame(previous_rows + rows).to_csv(trials_file, index=False)
                fingerprint_file.write_text(config_hash + "\n", encoding="utf-8")
                if progress:
                    print(
                        f"[{executed}/{total}] {method} {scenario} seed={seed}: {row['status']}",
                        file=sys.stderr,
                        flush=True,
                    )
    all_rows = previous_rows + rows
    trials = pd.DataFrame(all_rows)
    trials.to_csv(trials_file, index=False)
    fingerprint_file.write_text(config_hash + "\n", encoding="utf-8")
    summary = summarize(trials, METRIC_COLUMNS)
    return trials, summary


def write_artifacts(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict,
    results_dir: str | Path,
    *,
    seed_set: str | None = None,
    leakage_audit: dict | None = None,
) -> None:
    """Persist the four artifacts plus report.md (contract §5 directory layout).

    The manifest always records a ``data_leakage_audit`` section (design
    §3.4): the preflight report is embedded verbatim when provided; without
    one the artifact set is marked ``audit_only`` (no audit proof, must not
    back performance claims).  A report whose ``config_sha256`` does not
    match *config* is stale and rejected.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(results_dir / "trials.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    if leakage_audit is None:
        audit_section = {
            "status": "audit_only",
            "reason": (
                "write_artifacts called without a leakage preflight report; "
                "artifact set lacks audit proof and must not back performance claims"
            ),
        }
    else:
        if leakage_audit.get("config_sha256") != canonical_hash(config):
            raise ValueError(
                "stale leakage audit report: report config_sha256 "
                f"{leakage_audit.get('config_sha256')!r} does not match this config"
            )
        audit_section = leakage_audit
    project_root = Path(__file__).resolve().parents[2]
    resolved = {**config, "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "paper_claim": False,
        "config_sha256": canonical_hash(config),
        "git_commit": _git_head(project_root),
        "git_worktree_dirty": git_worktree_dirty(project_root),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed_set": seed_set,
        "seeds": sorted(int(s) for s in trials["seed"].unique()),
        "n_trials": int(len(trials)),
        "environment": {
            "python": sys.version.split()[0],
            "dependencies": _fingerprint_dependencies(),
        },
        "limitations": config.get("limitations", []),
        "data_leakage_audit": audit_section,
    }
    (results_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    (results_dir / "report.md").write_text(
        _render_report(trials, summary, manifest), encoding="utf-8"
    )


def _render_report(trials, summary, manifest) -> str:
    lines = ["# Traditional PU Baseline Report", ""]
    lines.append(f"- protocol: {manifest['protocol']} (paper_claim=False)")
    lines.append(f"- config sha256: {manifest['config_sha256'][:12]}")
    lines.append(f"- n_trials: {manifest['n_trials']}")
    lines.append("- distribution_shift or oracle selection: not used (contract §2)")
    lines.append("")
    lines.append("## Summary (success rows only; success_rate includes all)")
    lines.append(_df_to_markdown(summary))
    lines.append("")
    return "\n".join(lines)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a pipe table; tabulate is not a project dependency."""
    if df.empty:
        return "_empty_"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, record in df.iterrows():
        cells = [str(record[c]) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fingerprint_dependencies() -> dict[str, str | None]:
    """Record dependency versions for provenance (missing packages → null)."""
    return {name: _package_version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")}


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_head(project_root: Path) -> str | None:
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
