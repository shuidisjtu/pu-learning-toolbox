# ruff: noqa: N806
"""Dry-run or execute one explicit GenPU SCAR/TS engineering recipe.

Consumes user-prepared four-role NPZ files. No data split, GPU fallback,
hyperparameter sweep, formal admission or protocol signature is performed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from pu_toolbox.core.training_views import ROLES
from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
from pu_toolbox.experiment.genpu_recipe import GenPURecipeTrainer, build_genpu_recipe
from pu_toolbox.experiment.image import fit_survey_image_preprocessing, transform_survey_images
from pu_toolbox.experiment.resources import process_lifetime_peak_rss_bytes, runtime_environment
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA, SCARGenerator
from pu_toolbox.utils.serialization import file_hash, strict_canonical_hash

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("spambase", "connect_4", "mnist", "fashion_mnist", "cifar10")
SOURCES = (
    "scripts/run_genpu_recipe.py",
    "pu_toolbox/experiment/genpu_recipe.py",
    "pu_toolbox/estimators/deep/gen_pu.py",
    "pu_toolbox/experiment/image.py",
    "pu_toolbox/estimators/deep/vision.py",
    "pu_toolbox/experiment/checkpoints.py",
    "pu_toolbox/experiment/runner.py",
    "pu_toolbox/experiment/strategies.py",
    "pu_toolbox/experiment/method_ledger.json",
    "uv.lock",
)


def _load(directory, dataset):
    """Verify role membership before train-only preprocessing; never load pickles."""
    parts = []
    for role in ROLES:
        with np.load(Path(directory) / f"{role}.npz", allow_pickle=False) as archive:
            X, y, indices = (archive[key] for key in ("X", "y", "indices"))
        if len(X) == 0 or y.shape != (len(X),) or indices.shape != (len(X),):
            raise ValueError(f"{role} must have aligned nonempty X/y/indices")
        if not np.isfinite(X).all():
            raise ValueError(f"{role} features must be finite")
        parts.append(DatasetPart(X, y, "clean", indices, role != "test"))
    bundle = DatasetBundle(*parts)
    validate_bundle(bundle)
    image = None
    if dataset in ("mnist", "fashion_mnist", "cifar10"):
        _, image = fit_survey_image_preprocessing(
            bundle.train.X, dataset=dataset, train_augmentation="none"
        )
        parts = [
            DatasetPart(
                transform_survey_images(part.X, image, role=role),
                part.labels,
                "clean",
                part.indices,
                part.for_selection,
            )
            for role, part in zip(ROLES, parts, strict=True)
        ]
        bundle = DatasetBundle(*parts)
    elif any(part.X.ndim != 2 for part in parts):
        raise ValueError("tabular recipe requires two-dimensional features")
    return bundle, image


def build_plan(
    directory,
    *,
    dataset,
    prior,
    prior_source,
    seed,
    device,
    gan_epochs,
    pn_epochs,
    batch_size,
    checkpoint_bytes,
    c_token,
):
    """Pin one explicit candidate, four input files and actual PU label draws."""
    if dataset not in DATASETS or c_token not in ("0.1", "0.3", "0.5"):
        raise ValueError("unsupported dataset or SCAR c token")
    if not isinstance(prior_source, str) or not prior_source.strip():
        raise ValueError(
            "population prior requires an explicit source; never infer from test labels"
        )
    if type(checkpoint_bytes) is not int or checkpoint_bytes < 1:
        raise ValueError("checkpoint_bytes must be a positive integer")
    directory = Path(directory).resolve()
    before = {role: file_hash(directory / f"{role}.npz") for role in ROLES}
    bundle, image = _load(directory, dataset)
    _, recipe = build_genpu_recipe(
        prior=prior,
        seed=seed,
        device=device,
        gan_epochs=gan_epochs,
        pn_epochs=pn_epochs,
        batch_size=batch_size,
        image=image,
    )
    views = {
        role: SCARGenerator().generate(
            getattr(bundle, role).X, getattr(bundle, role).labels, seed=seed, c=float(c_token)
        )[1]["label_view_sha256"]
        for role in ("train", "pu_val")
    }
    after = {role: file_hash(directory / f"{role}.npz") for role in ROLES}
    if before != after:
        raise ValueError("input files changed during planning")
    plan = {
        "schema_version": "genpu-engineering-plan-v1",
        "dataset": dataset,
        "input_directory": str(directory),
        "input_files_sha256": after,
        "role_sizes": {role: len(getattr(bundle, role).X) for role in ROLES},
        "recipe": recipe,
        "prior_source": prior_source,
        "prior_source_verified": False,
        "c_token": c_token,
        "label_view_sha256": views,
        "checkpoint_bytes_per_component": checkpoint_bytes,
        "source_files_sha256": {name: file_hash(ROOT / name) for name in SOURCES},
        "formal_admission": False,
        "formal_budget_approved": False,
    }
    plan["plan_sha256"] = strict_canonical_hash(plan)
    return plan


def _write(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def execute(plan, output):
    """Revalidate then train once, independently select PA/OA and replay both."""
    expected = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != strict_canonical_hash(expected):
        raise ValueError("GenPU plan changed after planning")
    p = plan["recipe"]["parameters"]
    actual = build_plan(
        plan["input_directory"],
        dataset=plan["dataset"],
        prior=p["class_prior"],
        prior_source=plan["prior_source"],
        seed=p["random_state"],
        device=p["device"],
        gan_epochs=p["max_epochs"],
        pn_epochs=p["classifier_epochs"],
        batch_size=p["batch_size"],
        checkpoint_bytes=plan["checkpoint_bytes_per_component"],
        c_token=plan["c_token"],
    )
    if actual != plan:
        raise ValueError("GenPU inputs, recipe or source identity changed after planning")
    source, output = Path(plan["input_directory"]), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("output must not overlap the input product")
    bundle, image = _load(source, plan["dataset"])
    if {role: file_hash(source / f"{role}.npz") for role in ROLES} != plan["input_files_sha256"]:
        raise ValueError("GenPU inputs changed immediately before fitting")
    model, recipe = build_genpu_recipe(
        prior=p["class_prior"],
        seed=p["random_state"],
        device=p["device"],
        gan_epochs=p["max_epochs"],
        pn_epochs=p["classifier_epochs"],
        batch_size=p["batch_size"],
        image=image,
    )
    if recipe != plan["recipe"]:
        raise ValueError("GenPU rebuilt recipe differs from plan")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", plan)
    environment = {
        **runtime_environment(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "source_files_sha256": plan["source_files_sha256"],
        "device_requested": p["device"],
        "frozen_lock_environment_verified": False,
    }
    _write(output / "environment.json", environment)
    runner = ExperimentRunner(
        seed=p["random_state"],
        class_prior=p["class_prior"],
        protocols=[ProtocolPA(), ProtocolOA()],
        manifest_path=str(output / "manifest.json"),
        config={
            "c": float(plan["c_token"]),
            "os_or_ts": "ts",
            "trainer": GenPURecipeTrainer(recipe),
            "candidates": [{}],
            "split_ref": {
                "dataset": plan["dataset"],
                "input_files_sha256": plan["input_files_sha256"],
            },
            "checkpoint_bytes_per_component": plan["checkpoint_bytes_per_component"],
            "reclaim_unselected_checkpoints": True,
        },
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = runner.fit(model, bundle.train, bundle.pu_val, bundle.clean_val, bundle.test)
    observed = {
        role: result.manifest["generation"][role]["label_view_sha256"]
        for role in ("train", "pu_val")
    }
    if observed != plan["label_view_sha256"]:
        raise ValueError("GenPU executed PU marking differs from plan")
    for protocol, fitted in result.selected_models.items():
        path = output / f"selected_{protocol}.pkl"
        with path.open("xb") as stream:
            pickle.dump(fitted, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with path.open("rb") as stream:
            replay = pickle.load(stream)
        # Only our own newly written trusted predictor; never external pickle.
        if (
            not np.array_equal(
                fitted.decision_function(bundle.test.X), replay.decision_function(bundle.test.X)
            )
            or not np.array_equal(fitted.predict(bundle.test.X), replay.predict(bundle.test.X))
            or float(np.mean(replay.predict(bundle.test.X) == bundle.test.labels))
            != result.test_metrics[protocol]["accuracy"]
        ):
            raise ValueError(f"GenPU selected {protocol} model failed replay")
    candidate = result.manifest["candidate_runs"][0]
    checkpoints = candidate["epoch_checkpoints"]
    retained = {row["path"] for row in checkpoints if not row["reclaimed"]}
    training = candidate["training_provenance"]
    summary = {
        "schema_version": "genpu-engineering-result-v1",
        "formal_admission": False,
        "formal_budget_approved": False,
        "paper_numerical_reproduction": False,
        "recipe_sha256": recipe["recipe_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "dataset": plan["dataset"],
        "role_sizes": plan["role_sizes"],
        "test_metrics": result.test_metrics,
        "training_provenance": training,
        "selection": result.manifest["selection"],
        "resources": result.manifest["resources"],
        "checkpoint_peak_payload_bytes": sum(training["checkpoint_bytes"]),
        "retained_checkpoint_bytes": sum(Path(path).stat().st_size for path in retained),
        "retained_checkpoint_count": len(retained),
        "reclaimed_checkpoint_count": sum(row["reclaimed"] for row in checkpoints),
        "process_lifetime_peak_rss_bytes": process_lifetime_peak_rss_bytes(),
        "elapsed_seconds_including_selection_replay": time.perf_counter() - started,
        "warning_count": len(caught),
        "retry_count": len(result.failures),
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ],
        "selected_models_replayed": True,
        "accepted_engineering_run": not caught and not result.failures,
        "files_sha256": {
            name: file_hash(output / name)
            for name in (
                "plan.json",
                "manifest.json",
                "environment.json",
                "selected_PA.pkl",
                "selected_OA.pkl",
            )
        },
    }
    _write(output / "summary.json", summary)
    if not summary["accepted_engineering_run"]:
        raise ValueError("GenPU warnings/retries require review; evidence retained, not accepted")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--class-prior", type=float, required=True)
    parser.add_argument("--prior-source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--gan-epochs", type=int, required=True)
    parser.add_argument("--pn-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--checkpoint-bytes-per-component", type=int, required=True)
    parser.add_argument("--c", choices=("0.1", "0.3", "0.5"), required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute and args.out_dir is None:
            raise ValueError("--execute requires a new --out-dir")
        plan = build_plan(
            args.data_dir,
            dataset=args.dataset,
            prior=args.class_prior,
            prior_source=args.prior_source,
            seed=args.seed,
            device=args.device,
            gan_epochs=args.gan_epochs,
            pn_epochs=args.pn_epochs,
            batch_size=args.batch_size,
            checkpoint_bytes=args.checkpoint_bytes_per_component,
            c_token=args.c,
        )
        print(
            json.dumps(
                execute(plan, args.out_dir) if args.execute else plan,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
