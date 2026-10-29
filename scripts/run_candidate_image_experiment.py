# ruff: noqa: N806
"""Plan or execute candidate image checkpoints, not a formal Survey row."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

from pu_toolbox.core.training_views import ROLES
from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier
from pu_toolbox.experiment.candidate_image_runner import CandidateImageCheckpointTrainer
from pu_toolbox.experiment.image import fit_survey_image_preprocessing, transform_survey_images
from pu_toolbox.experiment.resources import process_lifetime_peak_rss_bytes, runtime_environment
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA, SCARGenerator
from pu_toolbox.utils.serialization import file_hash, strict_canonical_hash

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "pulda": "risk/pulda.py",
    "gradpu": "deep/grad_pu.py",
    "pan": "deep/pan.py",
    "vpu": "risk/vpu.py",
}
DATASETS = ("cifar10", "mnist", "fashion_mnist", "adni")
ADNI_RECEIPT = "docs/research/pu_survey/data/adni_substitute_receipt_20261012.json"


def data_scope(dataset):
    """Describe the slot honestly, without authenticating arbitrary caller files."""
    scope = dict(dataset_slot=dataset, raw_source_verified=False)
    if dataset == "adni":
        scope.update(
            dataset_identity="Alzheimer MRI substitute for ADNI slot; not clinical ADNI",
            clinical_adni=False,
            subject_level_isolation_verified=False,
            license_status="unknown; do not infer permission to redistribute",
            project_source_receipt=ADNI_RECEIPT,
            verification_scope=(
                "project substitution policy only; caller NPZ files are not authenticated "
                "by this receipt"
            ),
        )
    return scope


def _load(directory, dataset):
    parts = []
    for role in ROLES:
        with np.load(Path(directory) / f"{role}.npz", allow_pickle=False) as data:
            X, labels, indices = (data[key] for key in ("X", "y", "indices"))
        if X.ndim != 4 or labels.shape != (len(X),) or indices.shape != (len(X),):
            raise ValueError(f"{role} requires aligned NCHW images, labels and indices")
        parts.append(DatasetPart(X, labels, "clean", indices, role != "test"))
    bundle = DatasetBundle(*parts)
    validate_bundle(bundle)
    _, image = fit_survey_image_preprocessing(
        bundle.train.X, dataset=dataset, train_augmentation="none"
    )
    prepared = [
        DatasetPart(
            transform_survey_images(part.X, image, role=role),
            part.labels,
            "clean",
            part.indices,
            part.for_selection,
        )
        for role, part in zip(ROLES, parts, strict=True)
    ]
    return DatasetBundle(*prepared), image


def _objects(method, image, *, prior, seed, parameters, checkpoint_bytes):
    if method not in METHODS or not isinstance(parameters, dict):
        raise ValueError("explicit method and parameter object are required")
    parameters = json.loads(json.dumps(parameters, allow_nan=False))
    if set(parameters).intersection(
        {"class_prior", "encoder", "train_transform", "model", "discriminator", "random_state"}
    ):
        raise ValueError("image/seed/prior bindings cannot be overridden")
    budget_keys = (
        ("warmup_epochs", "pu_epochs", "positive_batch_size", "unlabeled_batch_size")
        if method == "pulda"
        else ("max_epochs", "batch_size", "lr_schedule")
    )
    if method == "pan":
        budget_keys = ("max_epochs", "batch_size", "learning_rate", "adversarial_weight")
    if method == "vpu":
        budget_keys = (
            "hidden_dim",
            "depth",
            "max_epochs",
            "batch_size",
            "learning_rate",
            "regularization_weight",
            "mixup_alpha",
            "optimizer_schedule",
            "iterations_per_epoch",
            "data_device_policy",
        )
    if not all(key in parameters for key in (*budget_keys, "device")):
        raise ValueError(f"explicit budget/device required: {budget_keys}")
    if parameters["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be explicit cpu or cuda")
    if parameters["device"] == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("requested CUDA unavailable; no CPU fallback")
    model, setup = build_candidate_image_classifier(
        method,
        image,
        seed=seed,
        class_prior=prior if method == "pulda" else None,
        device=parameters["device"],
    )
    default_epochs = model.checkpoint_epoch_count if method == "pulda" else model.max_epochs
    model.set_params(**parameters)
    trainer = CandidateImageCheckpointTrainer(
        setup, checkpoint_bytes_per_component=checkpoint_bytes
    )
    return model, setup, trainer, default_epochs


def build_plan(
    directory, *, method, dataset, prior, prior_source, seed, parameters, checkpoint_bytes, c_token
):
    if dataset not in DATASETS or c_token not in ("0.1", "0.3", "0.5"):
        raise ValueError("unsupported dataset or SCAR c token")
    if (
        isinstance(prior, bool)
        or not isinstance(prior, int | float)
        or not np.isfinite(prior)
        or not 0 < prior < 1
    ):
        raise ValueError("selection requires explicit finite population prior in (0,1)")
    if not isinstance(prior_source, str) or not prior_source.strip():
        raise ValueError("explicit population prior source required; do not infer from test")
    directory = Path(directory).resolve()
    before = {role: file_hash(directory / f"{role}.npz") for role in ROLES}
    bundle, image = _load(directory, dataset)
    model, setup, trainer, default_epochs = _objects(
        method,
        image,
        prior=prior,
        seed=seed,
        parameters=parameters,
        checkpoint_bytes=checkpoint_bytes,
    )
    trainer.validate_experiment(
        model,
        bundle,
        {
            "os_or_ts": "os" if method == "pan" else "ts",
            "checkpoint_bytes_per_component": checkpoint_bytes,
        },
        [ProtocolPA(), ProtocolOA()],
    )
    generation = {}
    for role in ("train", "pu_val"):
        part = getattr(bundle, role)
        observed, meta = SCARGenerator().generate(part.X, part.labels, float(c_token), seed)
        if set(np.unique(observed)) != {0, 1} or meta["n_labeled"] != meta["n_labeled_requested"]:
            raise ValueError("SCAR draw needs unclamped observed P and U in each role")
        generation[role] = meta["label_view_sha256"]
    if {role: file_hash(directory / f"{role}.npz") for role in ROLES} != before:
        raise ValueError("image inputs changed during planning")
    references = (
        "scripts/run_candidate_image_experiment.py",
        "pu_toolbox/experiment/candidate_image_runner.py",
        "pu_toolbox/experiment/candidate_image.py",
        f"pu_toolbox/estimators/{METHODS[method]}",
        "pu_toolbox/experiment/runner.py",
        "pu_toolbox/experiment/checkpoints.py",
        "pu_toolbox/experiment/image.py",
        "pu_toolbox/estimators/deep/vision.py",
        "pu_toolbox/experiment/strategies.py",
        "pu_toolbox/experiment/method_ledger.json",
        "uv.lock",
    )
    if dataset == "adni":
        references += (ADNI_RECEIPT,)
    if method == "pan":
        references += ("docs/research/pu_survey/data/pan_source_review_20261018.json",)
    if method == "vpu":
        references += ("docs/research/pu_survey/data/vpu_schedule_source_20261015.json",)
    resolved = {
        key: value
        for key, value in model.get_params(deep=False).items()
        if key not in {"encoder", "train_transform", "model", "discriminator"}
    }
    plan = {
        "schema_version": "candidate-image-experiment-plan-v1",
        "method": method,
        "training_view": "os" if method == "pan" else "ts",
        "dataset": dataset,
        "data_scope": data_scope(dataset),
        "input_directory": str(directory),
        "input_files_sha256": before,
        "population_prior": float(prior),
        "prior_source": prior_source,
        "prior_source_verified": False,
        "seed": seed,
        "c_token": c_token,
        "parameters_requested": parameters,
        "parameters_resolved": resolved,
        "image_setup": setup,
        "comparison_group": setup["comparison_group"],
        "label_view_sha256": generation,
        "checkpoint_bytes_per_component": checkpoint_bytes,
        "checkpoint_epoch_bound": trainer.checkpoint_epoch_bound(),
        "source_default_snapshot_count": default_epochs,
        "source_files_sha256": {name: file_hash(ROOT / name) for name in references},
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
    if plan["plan_sha256"] != strict_canonical_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("image plan changed after planning")
    actual = build_plan(
        plan["input_directory"],
        method=plan["method"],
        dataset=plan["dataset"],
        prior=plan["population_prior"],
        prior_source=plan["prior_source"],
        seed=plan["seed"],
        parameters=plan["parameters_requested"],
        checkpoint_bytes=plan["checkpoint_bytes_per_component"],
        c_token=plan["c_token"],
    )
    if actual != plan:
        raise ValueError("image input/code/recipe identity changed after planning")
    source, output = Path(plan["input_directory"]), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("output must not overlap input")
    bundle, image = _load(source, plan["dataset"])
    if {role: file_hash(source / f"{role}.npz") for role in ROLES} != plan["input_files_sha256"]:
        raise ValueError("image input changed immediately before fitting")
    model, setup, trainer, _ = _objects(
        plan["method"],
        image,
        prior=plan["population_prior"],
        seed=plan["seed"],
        parameters=plan["parameters_requested"],
        checkpoint_bytes=plan["checkpoint_bytes_per_component"],
    )
    if setup != plan["image_setup"]:
        raise ValueError("image setup changed immediately before fitting")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", plan)
    _write(
        output / "environment.json",
        {
            **runtime_environment(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
            ).strip(),
            "worktree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=ROOT)
            ),
            "source_files_sha256": plan["source_files_sha256"],
            "frozen_lock_environment_verified": False,
        },
    )
    runner = ExperimentRunner(
        seed=plan["seed"],
        class_prior=plan["population_prior"],
        protocols=[ProtocolPA(), ProtocolOA()],
        manifest_path=str(output / "manifest.json"),
        config={
            "c": float(plan["c_token"]),
            "os_or_ts": plan["training_view"],
            "trainer": trainer,
            "candidates": [{}],
            "architecture": "cnn",
            "checkpoint_bytes_per_component": plan["checkpoint_bytes_per_component"],
            "reclaim_unselected_checkpoints": True,
            "split_ref": {
                "dataset": plan["dataset"],
                "data_scope": plan["data_scope"],
                "input_files_sha256": plan["input_files_sha256"],
            },
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = runner.fit(model, bundle.train, bundle.pu_val, bundle.clean_val, bundle.test)
    if {
        role: result.manifest["generation"][role]["label_view_sha256"]
        for role in ("train", "pu_val")
    } != plan["label_view_sha256"]:
        raise ValueError("executed image PU draw differs from plan")
    for name, selected in result.selected_models.items():
        path = output / f"selected_{name}.pkl"
        with path.open("xb") as stream:
            pickle.dump(selected, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with path.open("rb") as stream:
            restored = pickle.load(stream)  # Only this run's own new trusted predictor.
        if (
            not np.array_equal(
                selected.decision_function(bundle.test.X), restored.decision_function(bundle.test.X)
            )
            or not np.array_equal(selected.predict(bundle.test.X), restored.predict(bundle.test.X))
            or float(np.mean(restored.predict(bundle.test.X) == bundle.test.labels))
            != result.test_metrics[name]["accuracy"]
        ):
            raise ValueError(f"selected {name} predictor failed replay")
    candidate = result.manifest["candidate_runs"][0]
    provenance = candidate["training_provenance"]
    rows = candidate["epoch_checkpoints"]
    retained = {row["path"] for row in rows if not row["reclaimed"]}
    largest = max(provenance["checkpoint_bytes"])
    summary = {
        "schema_version": "candidate-image-experiment-result-v1",
        "method": plan["method"],
        "dataset": plan["dataset"],
        "data_scope": plan["data_scope"],
        "training_view": plan["training_view"],
        "comparison_group": plan["comparison_group"],
        "formal_admission": False,
        "formal_budget_approved": False,
        "formal_resource_profile_complete": False,
        "paper_numerical_reproduction": False,
        "plan_sha256": plan["plan_sha256"],
        "training_provenance": provenance,
        "role_sizes": {role: len(getattr(bundle, role).X) for role in ROLES},
        "selection": result.manifest["selection"],
        "test_metrics": result.test_metrics,
        "resources": result.manifest["resources"],
        "checkpoint_peak_payload_bytes": sum(provenance["checkpoint_bytes"]),
        "retained_checkpoint_bytes": sum(Path(path).stat().st_size for path in retained),
        "retained_checkpoint_count": len(retained),
        "reclaimed_checkpoint_count": sum(row["reclaimed"] for row in rows),
        "source_default_snapshot_projection": {
            "epochs": plan["source_default_snapshot_count"],
            "observed_max_file_bytes": largest,
            "single_attempt_payload_bytes": largest * plan["source_default_snapshot_count"],
            "two_attempt_declared_bound_bytes": plan["checkpoint_bytes_per_component"]
            * plan["source_default_snapshot_count"]
            * 2,
            "scope": "same_architecture_weights_only_not_a_full_training_memory_bound",
        },
        "process_lifetime_peak_rss_bytes": process_lifetime_peak_rss_bytes(),
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
                "environment.json",
                "manifest.json",
                "selected_PA.pkl",
                "selected_OA.pkl",
            )
        },
    }
    if plan["method"] in {"pan", "vpu"}:
        summary["source_default_snapshot_projection"]["origin"] = (
            "toolbox_constructor_not_author_or_frozen_recipe"
        )
    _write(output / "summary.json", summary)
    if not summary["accepted_engineering_run"]:
        raise ValueError("image warnings/retries require review; evidence retained, not accepted")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--class-prior", type=float, required=True)
    parser.add_argument("--prior-source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--checkpoint-bytes-per-component", type=int, required=True)
    parser.add_argument("--c", required=True, choices=("0.1", "0.3", "0.5"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.execute and args.out_dir is None:
            raise ValueError("--execute requires a new --out-dir")
        plan = build_plan(
            args.data_dir,
            method=args.method,
            dataset=args.dataset,
            prior=args.class_prior,
            prior_source=args.prior_source,
            seed=args.seed,
            parameters=json.loads(args.parameters.read_text()),
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
