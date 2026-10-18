#!/usr/bin/env python3
"""Temporary paired linear/neural LBE experiment on existing four-way splits.

Exploratory only: no frozen protocol changes or formal Survey admission.
Default terminal selection gives both methods one model; optional em_rounds
lets neural LBE use its EM checkpoints and explicitly records that asymmetry.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from _source_provenance import source_provenance

from pu_toolbox import LBEClassifier, NeuralLBEClassifier
from pu_toolbox.core.training_views import ROLES
from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer
from pu_toolbox.experiment.protocols import Trainer
from pu_toolbox.experiment.resources import runtime_environment
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import FitTrainer, ProtocolOA, ProtocolPA, SCARGenerator
from pu_toolbox.experiment.survey_protocol import digest, load_protocol
from pu_toolbox.utils.serialization import canonical_hash, file_hash, json_safe

DATASETS = ("spambase", "imdb", "twenty_newsgroups", "connect_4")
METHODS = ("lbe", "lbe_neural")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_split(root, dataset, seed):
    """Load exactly the requested seed, never relabel split_0 as another seed."""
    directory = Path(root) / f"split_{seed}"
    path = directory / "split_manifest.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != dataset or metadata.get("seed") != seed:
        raise ValueError("split manifest dataset/seed does not match request")
    prior = metadata.get("class_prior", {}).get("population")
    if isinstance(prior, bool) or not isinstance(prior, int | float) or not 0 < prior < 1:
        raise ValueError("split must record a population prior in (0, 1); never infer it from test")
    parts, identities = [], {"split_manifest.json": file_hash(path)}
    for role in ROLES:
        source = directory / f"{role}.npz"
        identities[source.name] = file_hash(source)
        with np.load(source, allow_pickle=False) as arrays:
            features, labels, indices = (arrays[key].copy() for key in ("X", "y", "indices"))
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("comparison needs finite 2-D features; image protocols are separate")
        if len(features) != len(labels) or len(labels) != len(indices):
            raise ValueError("features/labels/indices row counts disagree")
        if len(labels) != metadata["role_sizes"][role]:
            raise ValueError("role size differs from split manifest")
        if indices.tolist() != metadata["indices"][role]:
            raise ValueError("role indices differ from split manifest")
        parts.append(DatasetPart(features, labels, "clean", indices, for_selection=role != "test"))
    if len({part.X.shape[1] for part in parts}) != 1:
        raise ValueError("feature dimensions differ between roles")
    validate_bundle(DatasetBundle(*parts))
    reference = {
        "dataset": dataset,
        "seed": seed,
        "class_prior": metadata["class_prior"],
        "indices_sha256": canonical_hash(
            {role: part.indices.tolist() for role, part in zip(ROLES, parts, strict=True)}
        ),
        "role_sizes": metadata["role_sizes"],
        "files_sha256": identities,
        "preprocessing": metadata.get("preprocessing"),
    }
    return tuple(parts), reference, float(prior)


def _parameters(smoke, seed, device):
    linear = {**load_protocol()["method_profiles"]["lbe"]["params"], "C": 1.0}
    neural = NeuralLBEClassifier(random_state=seed, device=device).get_params(deep=False)
    if smoke:
        linear["n_em_iter"] = 2
        neural.update(pretrain_steps=2, n_em_iter=2, m_steps=2)
    return {"lbe": linear, "lbe_neural": neural}


def build_plan(
    root, *, dataset, seeds, c_tokens, device="cpu", selection_scope="terminal", smoke=False
):
    """Freeze inputs, default budgets and expected label identities before fits."""
    if dataset not in DATASETS or selection_scope not in ("terminal", "em_rounds"):
        raise ValueError("unknown dataset or selection scope")
    if not seeds or any(type(seed) is not int or seed not in range(5) for seed in seeds):
        raise ValueError("seeds must be a nonempty subset of 0..4")
    if len(set(seeds)) != len(seeds) or not c_tokens or len(set(c_tokens)) != len(c_tokens):
        raise ValueError("duplicate seed or c token")
    if any(token not in load_protocol()["c_tokens"]["scar"] for token in c_tokens):
        raise ValueError("SCAR c tokens must come from the frozen 0.1/0.3/0.5 grid")
    if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
        raise ValueError("device must be cpu, cuda or cuda:<index>")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise ValueError("requested CUDA is unavailable; no silent CPU fallback")
    records = []
    for seed in seeds:
        parts, reference, prior = load_split(root, dataset, seed)
        for token in c_tokens:
            generation = {}
            for role, part in zip(ROLES[:2], parts[:2], strict=True):
                labels, metadata = SCARGenerator().generate(part.X, part.labels, float(token), seed)
                if set(np.unique(labels)) != {0, 1}:
                    raise ValueError("PU roles need both P and U before either model trains")
                if metadata["n_labeled"] != metadata["n_labeled_requested"]:
                    raise ValueError("tiny-sample marking clamp would deviate from the protocol")
                generation[role] = metadata["label_view_sha256"]
            records.append(
                {
                    "seed": seed,
                    "c_token": token,
                    "population_prior": prior,
                    "split_ref": reference,
                    "label_view_sha256": generation,
                    "parameters": _parameters(smoke, seed, device),
                }
            )
    if len({row["population_prior"] for row in records}) != 1:
        raise ValueError("population prior differs across seeds")
    return {
        "schema_version": "lbe-paired-exploration-plan-1",
        "dataset": dataset,
        "split_root": str(Path(root).resolve()),
        "methods": list(METHODS),
        "basis_protocol_sha256": digest(load_protocol()),
        "formal_admission": False,
        "smoke": smoke,
        "selection_scope": selection_scope,
        "device": device,
        "mechanism": "scar",
        "run_view": "os",
        "candidates_per_method": 1,
        "selection_protocols": ["PA", "OA"],
        "test_used_for_selection": False,
        "budget_matched": False,
        "training_prior_semantics": {
            "lbe": "initial posterior only",
            "lbe_neural": "validated but ignored",
        },
        "limitations": [
            "not a registered Survey row or a paper-number reproduction",
            "linear logistic solvers and neural Adam have different compute/regularization budgets",
            "linear refits StandardScalers on weighted train copies; neural uses input features",
            "terminal: one model each; em_rounds: neural has extra checkpoint selection chances",
            "no hyperparameter tuning and no test-based scope/model/direction changes",
        ],
        "pairs": records,
    }


class ComparisonTrainer(Trainer):
    """Capture fitted diagnostics without changing the production estimators."""

    def __init__(self, *, checkpoints=False, checkpoint_dir=None):
        self.delegate = (
            EpochCheckpointTrainer(checkpoint_dir=checkpoint_dir) if checkpoints else FitTrainer()
        )
        self.diagnostics = {}

    def fit(self, estimator, x, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        trajectory = self.delegate.fit(
            estimator, x, y, class_prior=class_prior, val_pu=val_pu, os_or_ts=os_or_ts
        )
        self.diagnostics = {
            key: getattr(estimator, key, None)
            for key in (
                "n_iter_",
                "converged_",
                "optimizer_steps_",
                "pretraining_optimizer_steps_",
                "em_optimizer_steps_",
            )
        }
        return trajectory


def predict_saved_selection(saved, features):
    """Replay the VAL-side threshold, not the estimator's default 0.5 cutoff."""
    model, selection = saved["model"], saved["selection"]
    if selection["threshold"] is None:
        return model.predict(features)
    scores = model.decision_function(features)
    metrics = selection["metrics"]
    if metrics.get("val_score_min") is not None and metrics.get("val_score_scale"):
        scores = (scores - metrics["val_score_min"]) / metrics["val_score_scale"]
    return (scores >= selection["threshold"]).astype(int)


def run_pair(plan, pair, out_dir):
    """Require identical source/label identities; PA/OA never select on test."""
    parts, reference, prior = load_split(plan["split_root"], plan["dataset"], pair["seed"])
    if reference != pair["split_ref"] or prior != pair["population_prior"]:
        raise ValueError("split changed after planning")
    results = []
    for method in METHODS:
        directory = Path(out_dir) / method / f"c_{pair['c_token']}" / f"seed_{pair['seed']}"
        directory.mkdir(parents=True, exist_ok=False)
        trainer = ComparisonTrainer(
            checkpoints=method == "lbe_neural" and plan["selection_scope"] == "em_rounds",
            checkpoint_dir=directory / "checkpoints",
        )
        model = (LBEClassifier if method == "lbe" else NeuralLBEClassifier)(
            **pair["parameters"][method]
        )
        runner = ExperimentRunner(
            seed=pair["seed"],
            class_prior=prior,
            generator=SCARGenerator(),
            protocols=[ProtocolPA(), ProtocolOA()],
            manifest_path=str(directory / "manifest.json"),
            config={
                "method": method,
                "c": float(pair["c_token"]),
                "os_or_ts": "os",
                "trainer": trainer,
                "split_ref": reference,
                "candidates": [{}],
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = runner.fit(model, *parts)
        manifest = result.manifest
        observed = {role: manifest["generation"][role]["label_view_sha256"] for role in ROLES[:2]}
        if observed != pair["label_view_sha256"]:
            raise ValueError("actual marking differs from preplanned shared labels")
        selected_files = {}
        for protocol, selected in result.selected_models.items():
            selected_path = directory / f"selected_{protocol}.pkl"
            with selected_path.open("xb") as stream:
                pickle.dump(
                    {"model": selected, "selection": manifest["selection"][protocol]},
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            selected_files[selected_path.name] = file_hash(selected_path)
        diagnostics = {
            "method": method,
            "seed": pair["seed"],
            "c_token": pair["c_token"],
            "plan_sha256": canonical_hash(plan),
            "fit": trainer.diagnostics,
            "warnings": [
                {"category": type(item.message).__name__, "message": str(item.message)}
                for item in caught
            ],
            "manifest_sha256": file_hash(directory / "manifest.json"),
            "selected_models_sha256": selected_files,
            "pickle_security": "only load these own-generated trusted artifacts",
            "prediction_rule": "predict_saved_selection; reuse stored VAL affine/threshold",
            "formal_admission": False,
        }
        _write_json(directory / "fit_diagnostics.json", diagnostics)
        if manifest["failures"]:
            raise ValueError(
                "candidate retry occurred; inspect manifest before treating the pair as complete"
            )
        results.append(
            {
                "method": method,
                "seed": pair["seed"],
                "c_token": pair["c_token"],
                "test_results": manifest["test_results"],
                "selection": manifest["selection"],
                "label_view_sha256": observed,
                "diagnostics": diagnostics,
                "elapsed_seconds": manifest["elapsed"],
                "peak_gpu_memory_bytes": manifest["resources"]["peak_gpu_memory_bytes"],
                "manifest": str((directory / "manifest.json").resolve()),
            }
        )
        print(
            f"{plan['dataset']} {method} c={pair['c_token']} seed={pair['seed']} "
            f"{manifest['test_results']}",
            flush=True,
        )
    return results


def _stats(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("summary requires finite metrics, not invented zeros")
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def summarize(plan, records):
    """Paired seed differences, not differences of unpaired aggregate means."""
    expected = {
        (pair["c_token"], pair["seed"], method) for pair in plan["pairs"] for method in METHODS
    }
    indexed = {(row["c_token"], row["seed"], row["method"]): row for row in records}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("missing, duplicated or unplanned runs cannot produce a complete summary")
    rows = []
    for token in dict.fromkeys(pair["c_token"] for pair in plan["pairs"]):
        seeds = [pair["seed"] for pair in plan["pairs"] if pair["c_token"] == token]
        for seed in seeds:
            left, right = (indexed[token, seed, method] for method in METHODS)
            if left["label_view_sha256"] != right["label_view_sha256"]:
                raise ValueError("paired runs have different marking")
        for protocol in ("PA", "OA"):
            for metric in ("accuracy", "auc"):
                values = {
                    method: [
                        indexed[token, seed, method]["test_results"][protocol][metric]
                        for seed in seeds
                    ]
                    for method in METHODS
                }
                if any(value is None for data in values.values() for value in data):
                    raise ValueError(
                        "metric unavailable; cannot report a complete paired comparison"
                    )
                deltas = [
                    right - left
                    for left, right in zip(values["lbe"], values["lbe_neural"], strict=True)
                ]
                rows.append(
                    {
                        "c_token": token,
                        "protocol": protocol,
                        "metric": metric,
                        "lbe": _stats(values["lbe"]),
                        "lbe_neural": _stats(values["lbe_neural"]),
                        "paired_neural_minus_linear": _stats(deltas),
                        "paired_deltas_by_seed": dict(zip(map(str, seeds), deltas, strict=True)),
                    }
                )
    return {
        "schema_version": "lbe-paired-exploration-summary-1",
        "dataset": plan["dataset"],
        "formal_admission": False,
        "smoke": plan["smoke"],
        "budget_matched": False,
        "selection_scope": plan["selection_scope"],
        "metric_unit": "fraction",
        "rows": rows,
        "costs": [
            {
                key: row[key]
                for key in ("method", "c_token", "seed", "elapsed_seconds", "peak_gpu_memory_bytes")
            }
            for row in records
        ],
    }


def _write_summary(directory, summary):
    _write_json(directory / "summary.json", summary)
    with (directory / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "c",
                "protocol",
                "metric",
                "n",
                "linear_mean",
                "linear_sample_sd",
                "neural_mean",
                "neural_sample_sd",
                "paired_delta_mean",
                "paired_delta_sample_sd",
            ]
        )
        for row in summary["rows"]:
            left, right, difference = (
                row[key] for key in ("lbe", "lbe_neural", "paired_neural_minus_linear")
            )
            writer.writerow(
                [
                    row["c_token"],
                    row["protocol"],
                    row["metric"],
                    left["n"],
                    left["mean"],
                    left["std"],
                    right["mean"],
                    right["std"],
                    difference["mean"],
                    difference["std"],
                ]
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splits", type=Path, help="dataset root containing split_0..split_4")
    parser.add_argument("--dataset", choices=DATASETS, default="spambase")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="new directory; never overwrite/resume"
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--c", default="0.1,0.3,0.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--selection-scope", choices=("terminal", "em_rounds"), default="terminal")
    parser.add_argument(
        "--smoke", action="store_true", help="shortened budget, not performance evidence"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print plan only; no output directories or fits"
    )
    args = parser.parse_args(argv)
    created = False
    try:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        import torch

        torch.set_num_threads(args.threads)
        plan = build_plan(
            args.splits,
            dataset=args.dataset,
            seeds=[int(item) for item in args.seeds.split(",")],
            c_tokens=args.c.split(","),
            device=args.device,
            selection_scope=args.selection_scope,
            smoke=args.smoke,
        )
        plan["cpu_threads"] = args.threads
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if (
            args.out_dir.resolve() == args.splits.resolve()
            or args.out_dir.resolve() in args.splits.resolve().parents
            or args.splits.resolve() in args.out_dir.resolve().parents
        ):
            raise ValueError("output must be outside and disjoint from the split root")
        args.out_dir.mkdir(parents=True, exist_ok=False)
        created = True
        _write_json(args.out_dir / "plan.json", plan)
        provenance = source_provenance(
            __file__,
            {
                "lbe": inspect.getfile(LBEClassifier),
                "lbe_neural": inspect.getfile(NeuralLBEClassifier),
            },
        )
        _write_json(
            args.out_dir / "environment.json",
            {
                "code_commit": provenance["script_checkout_commit"],
                "worktree_dirty": provenance["script_checkout_dirty"],
                "source_provenance": provenance,
                "environment": runtime_environment(),
                "torch_cpu_threads": torch.get_num_threads(),
                "script_sha256": file_hash(Path(__file__)),
                "dependency_lock_sha256": file_hash(
                    Path(__file__).resolve().parents[1] / "uv.lock"
                ),
                "formal_admission": False,
            },
        )
        records = []
        for pair in plan["pairs"]:
            records.extend(run_pair(plan, pair, args.out_dir))
            _write_json(
                args.out_dir / f"pair_c_{pair['c_token']}_seed_{pair['seed']}.json", records[-2:]
            )
        _write_summary(args.out_dir, summarize(plan, records))
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        if created:
            _write_json(args.out_dir / "failure.json", {"error": str(error), "complete": False})
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
