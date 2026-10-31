"""Read-only candidate checkpoint-space planning, without torch or training.

An optional caller-selected checkpoint supplies a file-size observation only;
its architecture, format and experiment provenance are NOT validated or loaded.
This is never a full-resource profile, formal recipe or run authorization.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "pulda": ("pu_toolbox/estimators/risk/pulda.py", "PULDAClassifier", "resnet18"),
    "gradpu": ("pu_toolbox/estimators/deep/grad_pu.py", "GradPUClassifier", "cnn13_no_bn"),
    "pan": ("pu_toolbox/estimators/deep/pan.py", "PANClassifier", "resnet18"),
    "vpu": ("pu_toolbox/estimators/risk/vpu.py", "VPUClassifier", "resnet18"),
    "p3mix": ("pu_toolbox/estimators/deep/p3mix.py", "P3MixClassifier", "caller_injected_encoder"),
}


def constructor_facts(path, class_name):
    """Read literal constructor defaults and property returns without importing torch."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    args = init.args
    pairs = (
        list(zip(args.args[-len(args.defaults) :], args.defaults, strict=True))
        if args.defaults
        else []
    )
    pairs += list(zip(args.kwonlyargs, args.kw_defaults, strict=True))
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in pairs if value is not None}
    properties = {
        node.name: ast.unparse(node.body[-1].value).replace("self.", "")
        for node in klass.body
        if isinstance(node, ast.FunctionDef)
        and node.decorator_list
        and any(isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list)
        and isinstance(node.body[-1], ast.Return)
    }
    return defaults, properties


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(
    method,
    *,
    target_dir,
    n_positive,
    n_unlabeled,
    candidates=1,
    retained_attempts=2,
    concurrent_runs=1,
    safety_margin_percent=20,
    sample_checkpoint=None,
    root=ROOT,
    variant=None,
    batching_policy=None,
):
    """Describe current default budgets and a conservative weights-only estimate.

    The free-space observation is instantaneous, not a reservation. Two retained
    attempts represent the runner's one same-seed retry. Reclaiming survivors
    after selection is not credited against the training-time estimate.
    """
    if method not in METHODS:
        raise ValueError("unsupported candidate method")
    if method != "p3mix" and (variant is not None or batching_policy is not None):
        raise ValueError("variant and batching_policy apply only to p3mix")
    for name, value in (
        ("n_positive", n_positive),
        ("n_unlabeled", n_unlabeled),
        ("candidates", candidates),
        ("retained_attempts", retained_attempts),
        ("concurrent_runs", concurrent_runs),
    ):
        _positive_integer(value, name)
    if retained_attempts > 2:
        raise ValueError("retained_attempts cannot exceed the runner's two attempts")
    if type(safety_margin_percent) is not int or safety_margin_percent < 0:
        raise ValueError("safety_margin_percent must be a nonnegative integer")
    directory = Path(target_dir).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("target_dir must be an existing directory; nothing is created")

    source, class_name, backbone = METHODS[method]
    defaults, properties = constructor_facts(root / source, class_name)
    if method == "p3mix":
        variant = defaults["variant"] if variant is None else variant
        batching_policy = (
            defaults["batching_policy"] if batching_policy is None else batching_policy
        )
        if variant not in ("base", "e", "c"):
            raise ValueError("P3Mix variant must be base, e or c")
        if batching_policy not in ("role_balanced", "merged_shuffle"):
            raise ValueError("P3Mix batching_policy must be role_balanced or merged_shuffle")
        phases = {"max_epochs": defaults["max_epochs"]}
        snapshots = defaults["max_epochs"]
        batch = defaults["batch_size"]
        steps = (
            (n_positive + n_unlabeled + batch - 1) // batch
            if batching_policy == "merged_shuffle"
            else max((n_positive + batch - 1) // batch, (n_unlabeled + batch - 1) // batch)
        )
        updates = snapshots * steps
    elif method == "pulda":
        if properties.get("checkpoint_epoch_count") != "warmup_epochs + pu_epochs":
            raise ValueError("PULDA checkpoint-count contract changed; update the planner")
        phases = {key: defaults[key] for key in ("warmup_epochs", "pu_epochs")}
        snapshots = sum(phases.values())
        updates = snapshots * (
            (n_unlabeled + defaults["unlabeled_batch_size"] - 1) // defaults["unlabeled_batch_size"]
        )
    elif method == "vpu":
        phases = {"max_epochs": defaults["max_epochs"]}
        snapshots = defaults["max_epochs"]
        steps = defaults["iterations_per_epoch"] or max(
            (n_positive + n_unlabeled + defaults["batch_size"] - 1) // defaults["batch_size"],
            (n_positive + defaults["batch_size"] - 1) // defaults["batch_size"],
        )
        updates = snapshots * steps
    elif method == "pan":
        phases = {"max_epochs": defaults["max_epochs"]}
        snapshots = defaults["max_epochs"]
        updates = (
            2 * snapshots * ((n_unlabeled + defaults["batch_size"] - 1) // defaults["batch_size"])
        )
    else:
        phases = {"max_epochs": defaults["max_epochs"]}
        snapshots = defaults["max_epochs"]
        batch_size = defaults["batch_size"]
        updates = snapshots * max(
            (n_positive + batch_size - 1) // batch_size,
            (n_unlabeled + batch_size - 1) // batch_size,
        )
    _positive_integer(snapshots, "default_snapshot_count")

    sample = None
    estimate = None
    if sample_checkpoint is not None:
        path = Path(sample_checkpoint).resolve(strict=True)
        if not path.is_file():
            raise ValueError("sample_checkpoint must be an existing regular file")
        before = path.stat()
        _positive_integer(before.st_size, "sample_checkpoint size")
        sha256 = _file_digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("sample_checkpoint changed while being read")
        sample = {
            "path": str(path),
            "bytes": before.st_size,
            "sha256": sha256,
            "evidence_scope": "caller_selected_file_size_only",
            "format_architecture_and_training_provenance_verified": False,
        }
        base = before.st_size * snapshots * candidates * retained_attempts * concurrent_runs
        estimate = (base * (100 + safety_margin_percent) + 99) // 100

    capacity = shutil.disk_usage(directory)
    references = (
        "scripts/prepare_candidate_image_resource_plan.py",
        source,
        "pu_toolbox/experiment/candidate_image.py",
        "pu_toolbox/experiment/checkpoints.py",
    )
    result = {
        "schema_version": "candidate-image-resource-plan-1",
        "status": "planning_only_not_full_resource_profile",
        "method": method,
        "backbone": backbone,
        "comparison_group": "method-private/no-BN CNN" if method == "gradpu" else "end-to-end CNN",
        "defaults": phases,
        "default_snapshot_count": snapshots,
        "default_optimizer_updates_per_fit": updates,
        "training_rows": {"P": n_positive, "native_U": n_unlabeled},
        "candidates": candidates,
        "retained_attempts": retained_attempts,
        "concurrent_runs": concurrent_runs,
        "safety_margin_percent": safety_margin_percent,
        "target_directory": str(directory),
        "available_bytes_at_check": capacity.free,
        "sample_checkpoint": sample,
        "weights_only_estimate_bytes": estimate,
        "estimated_weights_fit_available_space": None
        if estimate is None
        else estimate <= capacity.free,
        "capacity_is_reserved": False,
        "post_selection_reclaim_reduces_estimate": False,
        "excluded_costs": [
            "architecture_templates_and_live_optimizer",
            "host_and_GPU_peak_memory",
            "data_and_feature_caches",
            "logs_and_other_processes_or_runs",
            "extra_components_not_in_current_single_model_path",
        ],
        "source_files_sha256": {ref: _file_digest(root / ref) for ref in references},
        "training_started": False,
        "cuda_initialized": False,
        "checkpoint_or_dataset_loaded": False,
        "formal_resource_profile_complete": False,
        "formal_admission": False,
        "approved_representation_ref": None,
        "ready_for_formal_run": False,
    }
    if method == "pan":
        result.update(
            training_view="os",
            calibration_applied=False,
            training_network_instances=2,
            inference_snapshot_components=["classifier_C"],
            checkpoint_contains_discriminator=False,
            default_budget_origin="toolbox_constructor_not_author_or_frozen_recipe",
        )
    if method == "vpu":
        result.update(
            training_view="ts",
            calibration_applied=True,
            training_marginal_rows=n_positive + n_unlabeled,
            normalization_forward_batches_per_epoch=(
                n_positive + n_unlabeled + defaults["batch_size"] - 1
            )
            // defaults["batch_size"],
            normalization_cost_is_additional_to_optimizer_updates=True,
            default_budget_origin="toolbox_constructor_not_author_or_frozen_recipe",
        )
    if method == "p3mix":

        def exposure(rows):
            full, remainder = divmod(rows, batch)
            lengths = [batch] * full + ([remainder] if remainder else [])
            cycles, prefix = divmod(steps, len(lengths))
            return cycles * rows + sum(lengths[:prefix])

        result.update(
            variant=variant,
            batching_policy=batching_policy,
            training_view="os",
            calibration_applied=False,
            training_network_instances=2 if variant == "e" else 1,
            optimizer_instances=1,
            checkpoint_contains_teacher=False,
            inference_snapshot_components=["student"],
            optimizer_steps_per_epoch=steps,
            rows_seen_per_epoch={
                "P": n_positive if batching_policy == "merged_shuffle" else exposure(n_positive),
                "U": n_unlabeled if batching_policy == "merged_shuffle" else exposure(n_unlabeled),
            },
            student_training_forward_batches_per_epoch=2 * steps,
            teacher_forward_batches_per_epoch=steps if variant == "e" else 0,
            positive_pool_refresh_forward_batches_per_epoch=(n_positive + batch - 1) // batch,
            additional_forwards_are_not_optimizer_updates=True,
            forward_counts_exclude_encoder_probe_and_external_selection=True,
            augmentation_and_backbone_require_separate_approval=True,
            default_budget_origin="toolbox_constructor_with_explicit_policy_not_frozen_recipe",
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--n-positive", type=int, required=True)
    parser.add_argument("--n-unlabeled", type=int, required=True)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--retained-attempts", type=int, default=2)
    parser.add_argument("--concurrent-runs", type=int, default=1)
    parser.add_argument("--safety-margin-percent", type=int, default=20)
    parser.add_argument("--sample-checkpoint", type=Path)
    parser.add_argument("--variant", choices=["base", "e", "c"])
    parser.add_argument("--batching-policy", choices=["role_balanced", "merged_shuffle"])
    args = parser.parse_args(argv)
    try:
        plan = build_plan(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
