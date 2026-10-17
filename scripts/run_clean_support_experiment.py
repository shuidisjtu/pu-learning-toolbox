#!/usr/bin/env python3
"""One explicit CPU/MLP clean-support experiment, not a formal Survey recipe.

Default dry-run pins product, method parameters and SCAR label identities.
--execute writes a new result directory and a selected OA model. Only explicit
extra-supervised exploratory OA is allowed; no automatic budget or GPU choice.
"""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

from pu_toolbox.experiment.clean_support_adapters import (
    LaGAMCleanSupportTrainer,
    SelfPUCleanSupportTrainer,
)
from pu_toolbox.experiment.pulns_adapter import PULNSExtraSupportTrainer
from pu_toolbox.experiment.resources import runtime_environment
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, SCARGenerator
from pu_toolbox.experiment.support_data import load_support_dataset
from pu_toolbox.utils.serialization import file_hash, strict_canonical_hash

METHODS = ("self_pu", "lagam", "pulns")
_BUDGETS = {
    "self_pu": (
        "max_epochs",
        "warmup_epochs",
        "self_paced_start",
        "self_paced_end",
        "distill_start",
    ),
    "lagam": ("max_epochs", "warmup_epochs", "num_clusters"),
    "pulns": ("pretrain_epochs", "episodes", "classifier_epochs"),
}


def _objects(method, parameters, prior, seed, support, metadata):
    if method not in METHODS or not isinstance(parameters, dict):
        raise ValueError("choose a supported method and an explicit parameter object")
    parameters = json.loads(json.dumps(parameters, allow_nan=False))
    if parameters.get("device") != "cpu":
        raise ValueError("this entry point requires explicit device=cpu; no GPU fallback")
    if any(key in parameters for key in ("class_prior", "random_state", "encoder", "backbone")):
        raise ValueError("prior/seed are artifact-bound; injected architectures are not admitted")
    required = (*_BUDGETS[method], "batch_size", "hidden_dim")
    if any(key not in parameters for key in required):
        raise ValueError(f"explicit compute budget required: {required}")
    for key in required:
        minimum = (
            0
            if key in ("warmup_epochs", "self_paced_start", "self_paced_end", "distill_start")
            else 1
        )
        if type(parameters[key]) is not int or parameters[key] < minimum:
            raise ValueError(f"invalid integer budget: {key}")
    if method == "self_pu":
        from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier

        model = SelfPUClassifier(class_prior=prior, random_state=seed, **parameters)
        model._validate_parameters(prior)
        adapter = SelfPUCleanSupportTrainer
    elif method == "lagam":
        from pu_toolbox.estimators.deep.lagam import LaGAMClassifier

        model = LaGAMClassifier(random_state=seed, **parameters)
        adapter = LaGAMCleanSupportTrainer
    else:
        from pu_toolbox.estimators.deep.pulns import PULNSClassifier

        model = PULNSClassifier(random_state=seed, **parameters)
        adapter = PULNSExtraSupportTrainer
    options = dict(metadata["support_adapter_options"])
    source_audit = metadata["support_source_audit"]
    signature = inspect.signature(adapter).parameters
    accepted = set(source_audit).intersection(signature)
    if accepted and accepted != set(source_audit):
        raise ValueError("partial support-source audit contract; refusing to drop required fields")
    options.update({key: source_audit[key] for key in accepted})
    return model, adapter(support, **options)


def build_plan(directory, *, method, c_token, parameters):
    """Resolve one terminal OA candidate; never fit, tune or inspect test scores."""
    if c_token not in ("0.1", "0.3", "0.5"):
        raise ValueError("c must be the canonical SCAR token 0.1/0.3/0.5")
    bundle, support, metadata = load_support_dataset(directory)
    if bundle.train.X.ndim != 2:
        raise ValueError(
            "CPU/MLP runner requires tabular products; CNN needs a separate specification"
        )
    prior, seed = metadata["class_prior"]["population"], metadata["seed"]
    model, trainer = _objects(method, parameters, prior, seed, support, metadata)
    config = {"c": float(c_token), "os_or_ts": "os", "split_ref": {"dataset": metadata["dataset"]}}
    trainer.validate_experiment(model, bundle, config, [ProtocolOA()])
    generation = {}
    for role in ("train", "pu_val"):
        part = getattr(bundle, role)
        observed, context = SCARGenerator().generate(part.X, part.labels, float(c_token), seed)
        if (
            set(np.unique(observed)) != {0, 1}
            or context["n_labeled"] != context["n_labeled_requested"]
        ):
            raise ValueError("PU role lacks P/U or would clamp the requested marking count")
        generation[role] = context["label_view_sha256"]
    root = Path(__file__).resolve().parents[1]
    sources = (
        f"pu_toolbox/estimators/deep/{method}.py",
        "pu_toolbox/experiment/clean_support_adapters.py",
        "pu_toolbox/experiment/pulns_adapter.py",
        "pu_toolbox/experiment/support_data.py",
        "pu_toolbox/experiment/runner.py",
        "pu_toolbox/experiment/strategies.py",
        "pu_toolbox/experiment/method_ledger.json",
        "scripts/run_clean_support_experiment.py",
        "uv.lock",
    )
    return {
        "schema_version": "clean-support-cpu-oa-plan-v1",
        "method": method,
        "artifact_dir": str(Path(directory).resolve()),
        "product_receipt_sha256": metadata["receipt_sha256"],
        "parameters_requested": parameters,
        "parameters_effective": model.get_params(deep=False),
        "seed": seed,
        "c_token": c_token,
        "population_prior": prior,
        "label_view_sha256": generation,
        "extra_supervision": trainer.supervision_context(),
        "support_product_audit": {
            **metadata["support_source_audit"],
            "role_pool_effect": {
                name: {
                    "original_count": count,
                    "remaining_count": metadata["role_sizes"].get(name, 0),
                    "reduced_count": count - metadata["role_sizes"].get(name, 0),
                }
                for name, count in metadata["support_source_audit"]["original_role_counts"].items()
            },
            "native_adapter_source_audit": "support_origin" in trainer.supervision_context(),
            "source_verification_scope": "caller_assertion_unverified",
        },
        "source_files_sha256": {name: file_hash(root / name) for name in sources},
        "formal_admission": False,
        "budget_approval": "not_granted_by_runner",
        "comparison_group": "extra_clean_support",
        "protocols": ["OA"],
        "candidate_count": 1,
        "selection_scope": "terminal",
        "test_used_for_selection": False,
        "training_device": "cpu",
        "limitation": (
            "independent exploratory MLP; not source replay, GPU or formal budget evidence"
        ),
    }


def _write_json(path, payload):
    strict_canonical_hash(payload)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def predict_saved_selection(saved, features):
    """Apply the OA validation affine/threshold unchanged to replay predictions."""
    model, selection = saved["model"], saved["selection"]
    if selection["threshold"] is None:
        return model.predict(features)
    scores = model.decision_function(features)
    metrics = selection["metrics"]
    if metrics.get("val_score_min") is not None and metrics.get("val_score_scale"):
        scores = (scores - metrics["val_score_min"]) / metrics["val_score_scale"]
    return (scores >= selection["threshold"]).astype(int)


def execute(plan, output):
    """Recheck the frozen plan; save only the one selected OA model and receipt."""
    actual = build_plan(
        plan["artifact_dir"],
        method=plan["method"],
        c_token=plan["c_token"],
        parameters=plan["parameters_requested"],
    )
    if strict_canonical_hash(actual) != strict_canonical_hash(plan):
        raise ValueError("support product, code or parameters changed after planning")
    directory = Path(output).resolve()
    source = Path(plan["artifact_dir"]).resolve()
    if directory == source or source in directory.parents or directory in source.parents:
        raise ValueError("result output must not overlap support product")
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "plan.json", plan)
    _write_json(
        directory / "environment.json",
        {
            **runtime_environment(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "worktree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True)
            ),
            "training_device": "cpu",
            "source_files_sha256": plan["source_files_sha256"],
        },
    )
    bundle, support, metadata = load_support_dataset(source)
    if metadata["receipt_sha256"] != plan["product_receipt_sha256"]:
        raise ValueError("support product changed immediately before fitting")
    model, trainer = _objects(
        plan["method"],
        plan["parameters_requested"],
        plan["population_prior"],
        plan["seed"],
        support,
        metadata,
    )
    runner = ExperimentRunner(
        seed=plan["seed"],
        class_prior=plan["population_prior"],
        protocols=[ProtocolOA()],
        manifest_path=str(directory / "manifest.json"),
        config={
            "c": float(plan["c_token"]),
            "os_or_ts": "os",
            "trainer": trainer,
            "split_ref": {
                "dataset": metadata["dataset"],
                "support_receipt_sha256": metadata["receipt_sha256"],
            },
            "candidates": [{}],
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = runner.fit(model, bundle.train, bundle.pu_val, bundle.clean_val, bundle.test)
    generation = {
        role: result.manifest["generation"][role]["label_view_sha256"]
        for role in ("train", "pu_val")
    }
    if generation != plan["label_view_sha256"]:
        raise ValueError("executed SCAR marking differs from plan")
    saved = {"model": result.selected_models["OA"], "selection": result.manifest["selection"]["OA"]}
    model_path = directory / "selected_OA.pkl"
    with model_path.open("xb") as stream:
        pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
    # Only our just-generated trusted model is loaded; never arbitrary external pickles.
    with model_path.open("rb") as stream:
        restored = pickle.load(stream)
    expected = predict_saved_selection(saved, bundle.test.X)
    predicted = predict_saved_selection(restored, bundle.test.X)
    scores_equal = np.array_equal(
        saved["model"].decision_function(bundle.test.X),
        restored["model"].decision_function(bundle.test.X),
    )
    if (
        not scores_equal
        or not np.array_equal(expected, predicted)
        or float(np.mean(predicted == bundle.test.labels)) != result.test_metrics["OA"]["accuracy"]
    ):
        raise ValueError("selected OA model does not replay its recorded test accuracy")
    summary = {
        "method": plan["method"],
        "formal_admission": False,
        "comparison_group": "extra_clean_support",
        "plan_sha256": strict_canonical_hash(plan),
        "product_receipt_sha256": metadata["receipt_sha256"],
        "test": result.test_metrics,
        "selection": result.manifest["selection"],
        "extra_supervision": result.manifest["extra_supervision"],
        "support_product_audit": plan["support_product_audit"],
        "training_provenance": result.manifest["candidate_runs"][0]["training_provenance"],
        "warning_count": len(caught),
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ],
        "retry_count": len(result.failures),
        "selected_model_replay_verified": True,
        "pickle_security": (
            "load only own-generated trusted models; replay uses OA affine/threshold"
        ),
        "accepted_engineering_run": not caught and not result.failures,
        "files_sha256": {
            name: file_hash(directory / name)
            for name in (
                "plan.json",
                "environment.json",
                "manifest.json",
                "selected_OA.pkl",
            )
        },
    }
    _write_json(directory / "summary.json", summary)
    if not summary["accepted_engineering_run"]:
        raise ValueError("warnings/retries require review; artifacts retained, not accepted")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--c", choices=["0.1", "0.3", "0.5"], required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute and args.out_dir is None:
            raise ValueError("--execute requires a new --out-dir")
        parameters = json.loads(args.parameters.read_text(encoding="utf-8"))
        plan = build_plan(
            args.artifact_dir, method=args.method, c_token=args.c, parameters=parameters
        )
        payload = execute(plan, args.out_dir) if args.execute else plan
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
