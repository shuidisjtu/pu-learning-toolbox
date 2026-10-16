#!/usr/bin/env python3
"""Preplanned LBE branch/convergence diagnostics, not formal recipe selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
from _lbe_diagnostic_reference import SCARFixedPropensityDiagnostic
from run_lbe_comparison import ComparisonTrainer, build_plan, load_split
from sklearn.metrics import roc_auc_score

from pu_toolbox import LBEClassifier
from pu_toolbox.experiment.resources import runtime_environment
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA
from pu_toolbox.utils.serialization import canonical_hash, file_hash, json_safe


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def diagnostic_plan(root, dataset, seeds, tokens):
    paired = build_plan(root, dataset=dataset, seeds=seeds, c_tokens=tokens)
    return {
        "schema_version": "lbe-branch-diagnostics-1",
        "basis_protocol_sha256": paired["basis_protocol_sha256"],
        "paired_input_plan": {
            "dataset": paired["dataset"],
            "split_root": paired["split_root"],
            "pairs": [
                {k: v for k, v in pair.items() if k != "parameters"} for pair in paired["pairs"]
            ],
        },
        "variants": {"linear_em20": 20, "linear_em200": 200, "scar_fixed_propensity": None},
        "variant_parameters": {
            "linear_em20": LBEClassifier(n_em_iter=20).get_params(),
            "linear_em200": LBEClassifier(n_em_iter=200).get_params(),
            "scar_fixed_propensity": SCARFixedPropensityDiagnostic().get_params(),
        },
        "formal_admission": False,
        "test_used_to_choose_variants_or_parameters": False,
        "scope": "All variants preplanned; no winner selected or old recipe changed",
        "scar_reference": "Different assumption: c=observed train rate/known population pi; "
        "not original LBE, not SAR, not a capacity-only ablation",
    }


def training_diagnostics(model, features, truth, observed):
    """Truth is post-fit diagnostic only; it never chooses a branch or recipe."""
    result = {
        "n_iter": model.n_iter_,
        "train_class_auc": roc_auc_score(truth, model.decision_function(features)),
    }
    if isinstance(model, LBEClassifier):
        p = model.predict_proba(features)[:, 1]
        c = model.propensity_model_.predict_proba(features)[:, 1]
        product = np.clip(p * c, 1e-15, 1 - 1e-15)
        swapped = np.clip(c * p, 1e-15, 1 - 1e-15)

        def log_likelihood(values):
            return float(np.sum(observed * np.log(values) + (1 - observed) * np.log1p(-values)))

        result.update(
            {
                "train_propensity_auc": roc_auc_score(truth, c),
                "train_observed_probability_auc": roc_auc_score(truth, product),
                "branch_exchange_likelihood_delta": log_likelihood(product)
                - log_likelihood(swapped),
                "converged": model.converged_,
                "likelihood_history": model.observed_log_likelihood_history_,
                "posterior_delta_history": model.posterior_max_delta_history_,
                "m_step_iterations": model.m_step_iterations_,
                "decomposition": model.decomposition_diagnostics_,
            }
        )
    else:
        result.update(
            {
                "estimated_fixed_propensity": model.propensity_,
                "optimizer_success": model.optimizer_success_,
                "optimizer_message": model.optimizer_message_,
                "gradient_max_abs": model.gradient_max_abs_,
            }
        )
    return result


def execute(plan, output):
    output = Path(output)
    root = Path(plan["paired_input_plan"]["split_root"])
    resolved = output.resolve()
    if resolved == root or resolved in root.parents or root in resolved.parents:
        raise ValueError("output must not overlap source splits")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(
        output / "environment.json",
        {
            **runtime_environment(),
            "training_device": "cpu",
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "script_sha256": file_hash(Path(__file__)),
            "reference_sha256": file_hash(Path(__file__).with_name("_lbe_diagnostic_reference.py")),
            "lock_sha256": file_hash("uv.lock"),
            "worktree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True)
            ),
        },
    )
    rows = []
    paired = plan["paired_input_plan"]
    for pair in paired["pairs"]:
        parts, reference, prior = load_split(root, paired["dataset"], pair["seed"])
        if reference != pair["split_ref"] or prior != pair["population_prior"]:
            raise ValueError("source identities changed after planning")
        for variant, iterations in plan["variants"].items():
            cls = LBEClassifier if iterations is not None else SCARFixedPropensityDiagnostic
            model = cls(**plan["variant_parameters"][variant])
            trainer = ComparisonTrainer()
            directory = output / variant / f"c_{pair['c_token']}" / f"seed_{pair['seed']}"
            directory.mkdir(parents=True, exist_ok=False)
            runner = ExperimentRunner(
                seed=pair["seed"],
                class_prior=prior,
                protocols=[ProtocolPA(), ProtocolOA()],
                manifest_path=str(directory / "manifest.json"),
                config={
                    "trainer": trainer,
                    "c": float(pair["c_token"]),
                    "os_or_ts": "os",
                    "split_ref": reference,
                },
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = runner.fit(model, *parts)
            actual = {
                role: result.manifest["generation"][role]["label_view_sha256"]
                for role in ("train", "pu_val")
            }
            if actual != pair["label_view_sha256"] or result.failures:
                raise ValueError("marking mismatch or retry: diagnostic run not clean")
            fitted = result.selected_models["OA"]
            observed, _ = runner.generator.generate(
                parts[0].X, parts[0].labels, float(pair["c_token"]), pair["seed"]
            )
            diagnostics = training_diagnostics(fitted, parts[0].X, parts[0].labels, observed)
            row = {
                "seed": pair["seed"],
                "c": pair["c_token"],
                "variant": variant,
                "plan_sha256": canonical_hash(plan),
                "test": result.test_metrics,
                "training": diagnostics,
                "warning_count": len(caught),
            }
            write_json(directory / "diagnostics.json", row)
            rows.append(row)
            print(
                f"completed {len(rows)}/{len(paired['pairs']) * 3}: {variant} "
                f"c={pair['c_token']} seed={pair['seed']}",
                flush=True,
            )
    expected = {(p["seed"], p["c_token"], v) for p in paired["pairs"] for v in plan["variants"]}
    if {(r["seed"], r["c"], r["variant"]) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("missing or duplicated diagnostic rows")
    summary = {"runs": len(rows), "formal_admission": False, "rows": rows, "metrics": []}
    for token in sorted({r["c"] for r in rows}):
        for variant in plan["variants"]:
            group = [r for r in rows if r["c"] == token and r["variant"] == variant]
            for protocol in ("PA", "OA"):
                for metric in ("accuracy", "auc"):
                    values = [r["test"][protocol][metric] for r in group]
                    summary["metrics"].append(
                        {
                            "c": token,
                            "variant": variant,
                            "protocol": protocol,
                            "metric": metric,
                            "n": len(values),
                            "mean": float(np.mean(values)),
                            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        }
                    )
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_root", type=Path)
    parser.add_argument("--dataset", default="spambase")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--c", default="0.1,0.3,0.5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = diagnostic_plan(
        args.split_root, args.dataset, [int(s) for s in args.seeds.split(",")], args.c.split(",")
    )
    if args.dry_run:
        print(json.dumps(json_safe(plan), ensure_ascii=False, allow_nan=False))
    else:
        execute(plan, args.out_dir)


if __name__ == "__main__":
    main()
