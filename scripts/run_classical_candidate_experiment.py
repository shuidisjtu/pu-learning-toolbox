"""Plan or execute CPU PUET/RP candidates without neural epochs or formal admission."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np

from pu_toolbox.core.training_views import ROLES
from pu_toolbox.estimators.classic.rank_pruning import RankPruningClassifier
from pu_toolbox.estimators.risk.puet import PUExtraTreesClassifier
from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
from pu_toolbox.experiment.classical_candidate import (
    ClassicalCandidateTrainer,
    ClassicalSelectionPredictor,
)
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA, SCARGenerator
from pu_toolbox.utils.serialization import array_hash, file_hash, strict_canonical_hash

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"puet": "risk/puet.py", "rp": "classic/rank_pruning.py"}
PARAMETERS = {
    "puet": {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "max_candidates",
        "bootstrap",
    },
    "rp": {"n_cv_folds", "frac_pos2neg", "min_retained"},
}


def _load(directory):
    parts = []
    for role in ROLES:
        with np.load(directory / f"{role}.npz", allow_pickle=False) as data:
            features, labels, indices = (data[key] for key in ("X", "y", "indices"))
        if (
            features.ndim != 2
            or not len(features)
            or features.shape[1] < 1
            or features.dtype.kind not in "iuf"
            or not np.isfinite(features).all()
            or labels.shape != (len(features),)
            or indices.shape != labels.shape
            or indices.dtype.kind not in "iu"
        ):
            raise ValueError(f"{role} requires finite 2-D features, aligned labels and integer IDs")
        parts.append(DatasetPart(features, labels, "clean", indices, role != "test"))
    if len({part.X.shape[1] for part in parts}) != 1:
        raise ValueError("four roles must share the same feature count")
    bundle = DatasetBundle(*parts)
    validate_bundle(bundle)
    return bundle


def _objects(method, parameters, prior, seed, view):
    if (
        method not in METHODS
        or not isinstance(parameters, dict)
        or set(parameters) != PARAMETERS[method]
    ):
        raise ValueError(
            "explicit complete native parameter object required; unknown keys forbidden"
        )
    parameters = json.loads(json.dumps(parameters, allow_nan=False))
    if view not in {"os", "ts"} or (method == "rp" and view != "os"):
        raise ValueError("RP supports native OS only; PUET requires explicit os/ts")
    integers = (
        ("n_estimators", "min_samples_leaf", "max_candidates")
        if method == "puet"
        else ("min_retained",)
    )
    for key in integers:
        if type(parameters[key]) is not int or parameters[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if method == "puet":
        depth, features = parameters["max_depth"], parameters["max_features"]
        if depth is not None and (type(depth) is not int or depth < 1):
            raise ValueError("max_depth must be None or a positive integer")
        if features not in ("sqrt", "all") and (type(features) is not int or features < 1):
            raise ValueError("max_features must be sqrt/all or a positive integer")
        if type(parameters["bootstrap"]) is not bool:
            raise ValueError("bootstrap must be boolean")
        return PUExtraTreesClassifier(prior, random_state=seed, **parameters)
    folds, fraction = parameters["n_cv_folds"], parameters["frac_pos2neg"]
    if type(folds) is not int or folds < 2:
        raise ValueError("n_cv_folds must be an integer >= 2")
    if fraction is not None and (
        isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0 <= fraction < 1
    ):
        raise ValueError("frac_pos2neg must be None or a finite fraction in [0,1)")
    return RankPruningClassifier(random_state=seed, **parameters)


def build_plan(directory, *, method, dataset, prior, prior_source, seed, view, parameters, c_token):
    """Pin inputs/code/environment and count non-epoch work without fitting."""
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("explicit caller dataset identity required")
    if not isinstance(prior_source, str) or not prior_source.strip():
        raise ValueError("explicit population prior source required; never infer from test")
    if (
        isinstance(prior, bool)
        or not isinstance(prior, int | float)
        or not np.isfinite(prior)
        or not 0 < prior < 1
    ):
        raise ValueError("population prior must be finite and in (0,1)")
    if type(seed) is not int or not 0 <= seed < 2**32 or c_token not in ("0.1", "0.3", "0.5"):
        raise ValueError("explicit uint32 seed and canonical SCAR c token required")
    model = _objects(method, parameters, prior, seed, view)
    directory = Path(directory).resolve()
    inputs = {role: file_hash(directory / f"{role}.npz") for role in ROLES}
    bundle = _load(directory)
    generation, observed = {}, {}
    for role in ("train", "pu_val"):
        part = getattr(bundle, role)
        labels, meta = SCARGenerator().generate(part.X, part.labels, float(c_token), seed)
        if set(np.unique(labels)) != {0, 1} or meta["n_labeled"] != meta["n_labeled_requested"]:
            raise ValueError("SCAR draw requires unclamped observed P and U in both roles")
        generation[role] = meta["label_view_sha256"]
        observed[role] = {
            "P": int(np.count_nonzero(labels == 1)),
            "U": int(np.count_nonzero(labels == 0)),
        }
    if method == "rp" and min(observed["train"].values()) < model.n_cv_folds:
        raise ValueError("each train PU role requires at least n_cv_folds rows")
    work = model.n_estimators if method == "puet" else model.n_cv_folds + 1
    references = (
        "scripts/run_classical_candidate_experiment.py",
        "pu_toolbox/experiment/classical_candidate.py",
        f"pu_toolbox/estimators/{METHODS[method]}",
        "pu_toolbox/experiment/runner.py",
        "pu_toolbox/experiment/bundle.py",
        "pu_toolbox/experiment/strategies.py",
        "pu_toolbox/experiment/protocols.py",
        "pu_toolbox/core/training_views.py",
        "pu_toolbox/experiment/method_ledger.json",
        "uv.lock",
    )
    if inputs != {role: file_hash(directory / f"{role}.npz") for role in ROLES}:
        raise ValueError("inputs changed during planning")
    plan = dict(
        schema_version="classical-candidate-experiment-plan-1",
        method=method,
        dataset=dataset,
        dataset_identity_verified=False,
        prior_source_verified=False,
        feature_preprocessing="caller_supplied_not_fitted_or_authenticated_by_this_script",
        input_directory=str(directory),
        input_files_sha256=inputs,
        role_sizes={role: len(getattr(bundle, role).X) for role in ROLES},
        population_prior=float(prior),
        prior_source=prior_source,
        seed=seed,
        training_view=view,
        c_token=c_token,
        parameters=parameters,
        label_view_sha256=generation,
        observed_role_sizes=observed,
        execution_device="cpu",
        neural_epoch_count=0,
        budget_unit="tree_build" if method == "puet" else "cross_fits_and_final_refit",
        work_units_per_attempt=work,
        two_attempt_work_upper_bound=2 * work,
        rp_base_estimator="LogisticRegression(max_iter=1000, random_state=seed); sklearn defaults"
        if method == "rp"
        else None,
        training_class_prior_used=method == "puet",
        formal_admission=False,
        formal_budget_approved=False,
        formal_resource_profile_complete=False,
        environment_identity={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
        },
        source_files_sha256={name: file_hash(ROOT / name) for name in references},
    )
    plan["plan_sha256"] = strict_canonical_hash(plan)
    return plan


def _write(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def execute(plan, output):
    """Replay a pinned plan and preserve all actual attempt cost receipts."""
    if plan["plan_sha256"] != strict_canonical_hash(
        {k: v for k, v in plan.items() if k != "plan_sha256"}
    ):
        raise ValueError("classical plan changed after planning")
    actual = build_plan(
        plan["input_directory"],
        method=plan["method"],
        dataset=plan["dataset"],
        prior=plan["population_prior"],
        prior_source=plan["prior_source"],
        seed=plan["seed"],
        view=plan["training_view"],
        parameters=plan["parameters"],
        c_token=plan["c_token"],
    )
    if actual != plan:
        raise ValueError("input/code/recipe/environment identity changed after planning")
    source, output = Path(plan["input_directory"]), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("output must not overlap any input ancestor")
    bundle = _load(source)
    if any(file_hash(source / f"{role}.npz") != plan["input_files_sha256"][role] for role in ROLES):
        raise ValueError("inputs changed immediately before fitting")
    arrays = {role: array_hash(getattr(bundle, role).X) for role in ROLES}
    model = _objects(
        plan["method"],
        plan["parameters"],
        plan["population_prior"],
        plan["seed"],
        plan["training_view"],
    )
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", plan)
    _write(
        output / "environment.json",
        {
            "environment_identity": plan["environment_identity"],
            "execution_device": "cpu",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "worktree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            ),
            "frozen_lock_environment_verified": False,
        },
    )
    trainer = ClassicalCandidateTrainer(receipt_dir=output / "attempts")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ExperimentRunner(
                seed=plan["seed"],
                class_prior=plan["population_prior"],
                protocols=[ProtocolPA(), ProtocolOA()],
                manifest_path=str(output / "manifest.json"),
                config={
                    "trainer": trainer,
                    "method": plan["method"],
                    "c": float(plan["c_token"]),
                    "os_or_ts": plan["training_view"],
                    "candidates": [{}],
                    "split_ref": {
                        "dataset": plan["dataset"],
                        "input_files_sha256": plan["input_files_sha256"],
                    },
                },
            ).fit(model, *(getattr(bundle, role) for role in ROLES))
        if any(
            array_hash(getattr(bundle, role).X) != arrays[role]
            or file_hash(source / f"{role}.npz") != plan["input_files_sha256"][role]
            for role in ROLES
        ):
            raise ValueError("input changed during execution")
        if any(
            result.manifest["generation"][role]["label_view_sha256"]
            != plan["label_view_sha256"][role]
            for role in ("train", "pu_val")
        ):
            raise ValueError("actual PU draw differs from plan")
        receipt = result.manifest["candidate_runs"][0]["training_provenance"]
        if (
            receipt["work_completed"] != plan["work_units_per_attempt"]
            or receipt["neural_epoch_count"] != 0
        ):
            raise ValueError("completed non-epoch work differs from plan")
        for name, selected in result.selected_models.items():
            predictor = ClassicalSelectionPredictor(selected, result.manifest["selection"][name])
            path = output / f"selected_{name}.pkl"
            with path.open("xb") as stream:
                pickle.dump(predictor, stream, protocol=pickle.HIGHEST_PROTOCOL)
            with path.open("rb") as stream:
                restored = pickle.load(stream)  # Only this run's new trusted artifact.
            if (
                not np.array_equal(
                    restored.decision_function(bundle.test.X),
                    selected.decision_function(bundle.test.X),
                )
                or not np.array_equal(
                    restored.predict(bundle.test.X), predictor.predict(bundle.test.X)
                )
                or float(np.mean(restored.predict(bundle.test.X) == bundle.test.labels))
                != result.test_metrics[name]["accuracy"]
            ):
                raise ValueError("selected predictor failed replay")
        summary = dict(
            schema_version="classical-candidate-experiment-result-1",
            method=plan["method"],
            dataset=plan["dataset"],
            plan_sha256=plan["plan_sha256"],
            training_view=plan["training_view"],
            execution_device="cpu",
            formal_admission=False,
            formal_budget_approved=False,
            formal_resource_profile_complete=False,
            paper_numerical_reproduction=False,
            neural_epoch_count=0,
            selected_models_replayed=True,
            selected_prediction_rule="recorded_VAL_affine_and_threshold_not_native_cutoff",
            warning_count=len(caught),
            retry_count=len(result.failures),
            warnings=[{"category": w.category.__name__, "message": str(w.message)} for w in caught],
            selection=result.manifest["selection"],
            test_metrics=result.test_metrics,
            training_provenance=result.manifest["candidate_runs"][0]["training_provenance"],
            resources=result.manifest["resources"],
            attempt_receipts_sha256={
                str(Path(p).relative_to(output)): file_hash(p) for p in trainer.receipt_paths
            },
            accepted_engineering_run=not caught and not result.failures,
        )
        _write(output / "summary.json", summary)
        if not summary["accepted_engineering_run"]:
            raise ValueError("warnings/retries require review; evidence retained, not accepted")
        return summary
    except BaseException as exc:
        _write(
            output / "failure.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "accepted_engineering_run": False,
                "formal_admission": False,
                "attempt_receipts": [
                    str(Path(p).relative_to(output)) for p in trainer.receipt_paths
                ],
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--class-prior", type=float, required=True)
    parser.add_argument("--prior-source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--view", required=True, choices=("os", "ts"))
    parser.add_argument("--parameters", type=Path, required=True)
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
            view=args.view,
            parameters=json.loads(args.parameters.read_text()),
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
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
