"""Read-only PULDA/GradPU checkpoint-space planning, without torch or training.

An optional caller-selected checkpoint supplies a file-size observation only;
its architecture, format and experiment provenance are NOT validated or loaded.
This is never a full-resource profile, formal recipe or run authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.check_p3_admission_evidence import constructor_facts

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "pulda": ("pu_toolbox/estimators/risk/pulda.py", "PULDAClassifier", "resnet18"),
    "gradpu": ("pu_toolbox/estimators/deep/grad_pu.py", "GradPUClassifier", "cnn13_no_bn"),
}


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
):
    """Describe current default budgets and a conservative weights-only estimate.

    The free-space observation is instantaneous, not a reservation. Two retained
    attempts represent the runner's one same-seed retry. Reclaiming survivors
    after selection is not credited against the training-time estimate.
    """
    if method not in METHODS:
        raise ValueError("method must be pulda or gradpu")
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
    if method == "pulda":
        if properties.get("checkpoint_epoch_count") != "warmup_epochs + pu_epochs":
            raise ValueError("PULDA checkpoint-count contract changed; update the planner")
        phases = {key: defaults[key] for key in ("warmup_epochs", "pu_epochs")}
        snapshots = sum(phases.values())
        updates = snapshots * (
            (n_unlabeled + defaults["unlabeled_batch_size"] - 1) // defaults["unlabeled_batch_size"]
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
    return {
        "schema_version": "candidate-image-resource-plan-1",
        "status": "planning_only_not_full_resource_profile",
        "method": method,
        "backbone": backbone,
        "comparison_group": "end-to-end CNN" if method == "pulda" else "method-private/no-BN CNN",
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
    args = parser.parse_args(argv)
    try:
        plan = build_plan(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
