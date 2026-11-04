"""Full-split candidate readiness evidence, never frozen Survey admission.

Prepare validates the four-role receipt and records explicit constructor-based
budgets. Run is an opt-in, detached-host-friendly two-GPU dispatcher. Existing
outputs are never resumed or overwritten, and any failed/warned run holds the
remaining campaign. No protocol, recipe registry or previous run is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts._survey_host import lock_controller

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gradpu", "cvir", "vpu", "pan", "p3mix")
ROLES = ("train", "pu_val", "clean_val", "test")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        temp = path.with_suffix(f".{os.getpid()}.new")
        temp.write_text(text, encoding="utf-8")
        temp.replace(path)


def validate_split(directory, *, seed=0):
    """Verify supplied counts/indices and population metadata; do not infer pi."""
    import numpy as np

    from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle
    from pu_toolbox.experiment.survey_protocol import role_indices_digest

    directory = Path(directory)
    receipt_path = directory / "split_manifest.json"
    receipt_hash = sha(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt["dataset"] != "cifar10" or receipt["seed"] != seed:
        raise ValueError(f"this readiness campaign requires CIFAR-10 split seed {seed}")
    if receipt["role_sizes"] != dict(train=45000, pu_val=2500, clean_val=2500, test=10000):
        raise ValueError("full 45000/2500/2500/10000 role sizes required; no subset fallback")
    if receipt["positive_classes"] != [0, 1, 8, 9]:
        raise ValueError("unexpected positive-class mapping")
    prior = receipt["class_prior"]["population"]
    if isinstance(prior, bool) or prior != 0.4:
        raise ValueError("expected recorded population prior 0.4; never estimate from test")
    parts, hashes, indices = [], {}, {}
    for role in ROLES:
        path = directory / f"{role}.npz"
        before = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            features, labels, ids = (archive[key] for key in ("X", "y", "indices"))
        if features.shape != (receipt["role_sizes"][role], 3, 32, 32):
            raise ValueError(f"unexpected full image shape: {role}")
        if not np.isfinite(features).all() or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"invalid images/labels: {role}")
        indices[role] = ids.tolist()
        if indices[role] != receipt["indices"][role] or sha(path) != before:
            raise ValueError(f"split receipt/file changed or disagrees: {role}")
        hashes[role] = before
        parts.append(DatasetPart(features, labels, "clean", ids, role != "test"))
    validate_bundle(DatasetBundle(*parts))
    if (
        role_indices_digest(indices) != receipt["indices_sha256"]
        or sha(receipt_path) != receipt_hash
    ):
        raise ValueError("split receipt indices digest changed or disagrees")
    return dict(
        receipt_sha256=receipt_hash,
        files_sha256=hashes,
        population_prior=prior,
        role_sizes=receipt["role_sizes"],
        indices_sha256=receipt["indices_sha256"],
        raw_source_reaudited=False,
    )


def prepare(directory, output):
    from prepare_candidate_image_resource_plan import METHODS as SOURCES
    from prepare_candidate_image_resource_plan import constructor_facts

    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("campaign source must be committed and clean")
    if output.exists():
        raise ValueError("campaign output already exists")
    evidence = validate_split(directory)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    jobs = []
    for method in METHODS:
        source, klass, _ = SOURCES[method]
        defaults, _ = constructor_facts(ROOT / source, klass)
        params = {
            k: v
            for k, v in defaults.items()
            if k
            not in {
                "model",
                "encoder",
                "discriminator",
                "random_state",
                "unlabeled_positive_prior",
                "prior_source",
            }
        }
        params["device"] = "cuda"
        if method in ("vpu", "cvir"):
            params["data_device_policy"] = "cpu_stream"
        if method == "p3mix":
            params["batching_policy"] = "merged_shuffle"
        for view in (
            ("os", "ts") if method == "cvir" else ("os" if method in ("pan", "p3mix") else "ts",)
        ):
            label = f"{method}-cifar10-s0-c0.1-{view}"
            jobs.append(
                dict(label=label, method=method, view=view, parameters=params, status="pending")
            )
    campaign = dict(
        schema_version="candidate-readiness-1",
        source=str(ROOT),
        commit=commit,
        dependency_lock_sha256=sha(ROOT / "uv.lock"),
        input_directory=str(directory.resolve()),
        input_evidence=evidence,
        jobs=jobs,
        formal_admission=False,
        frozen_budget_approved=False,
        scope="full_split_full_constructor_budget_single_seed_readiness_not_paper_reproduction",
        adaptations=[
            "GradPU private CNN13/no-BN; other methods shared ResNet-18, no augmentation",
            "P3MIX explicit merged_shuffle instead of compatibility role_balanced default",
            "VPU/CVIR cpu_stream instead of auto; mathematical training budgets unchanged",
            "constructor learning rates/schedules are engineering candidates, not author recipes",
            "seed 0, SCAR c=0.1 only; does not replace the frozen five-seed full grid",
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    for job in jobs:
        write(output / f"{job['label']}.parameters.json", job["parameters"], exclusive=True)
    write(output / "campaign.json", campaign, exclusive=True)
    return campaign


def available_gpus(indices):
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"]
    devices = {}
    for line in subprocess.check_output(command, text=True, timeout=5).splitlines():
        index, uuid, free = (field.strip() for field in line.split(","))
        if int(index) in indices:
            devices[uuid] = (int(index), int(free))
    busy = set(
        subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).splitlines()
    )
    return sorted(
        index for uuid, (index, free) in devices.items() if uuid not in busy and free >= 40960
    )


def command(campaign, job, output):
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_candidate_image_experiment.py"),
        "--method",
        job["method"],
        "--data-dir",
        campaign["input_directory"],
        "--dataset",
        "cifar10",
        "--class-prior",
        str(campaign["input_evidence"]["population_prior"]),
        "--prior-source",
        "validated full CIFAR-10 split_manifest population metadata; not test estimate",
        "--seed",
        "0",
        "--parameters",
        str(output / f"{job['label']}.parameters.json"),
        "--checkpoint-bytes-per-component",
        str(256 * 1024**2),
        "--c",
        "0.1",
        "--view",
        job["view"],
        "--execute",
        "--out-dir",
        str(output / job["label"]),
    ]


def run(output, indices):
    """Fresh campaign only; no automatic recovery of possibly live workers."""
    if (
        not indices
        or len(indices) > 2
        or len(set(indices)) != len(indices)
        or any(i not in (0, 5, 6, 7) for i in indices)
    ):
        raise ValueError("use at most two distinct GPUs within the existing four-card allowance")
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    _controller_lock = lock_controller(output / "controller.lock")
    campaign = json.loads((output / "campaign.json").read_text())
    if (
        campaign["source"] != str(ROOT)
        or campaign["commit"]
        != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip()
    ):
        raise ValueError("campaign source identity changed")
    if sha(ROOT / "uv.lock") != campaign["dependency_lock_sha256"]:
        raise ValueError("dependency lock changed")
    if validate_split(Path(campaign["input_directory"])) != campaign["input_evidence"]:
        raise ValueError("campaign split changed")
    jobs, processes, streak = campaign["jobs"], {}, {}
    write(output / "status.json", dict(jobs=jobs, controller_pid=os.getpid()), exclusive=True)
    held = False
    while any(j["status"] in ("pending", "running") for j in jobs):
        for job in jobs:
            if job["status"] != "running":
                continue
            code = processes[job["label"]].poll()
            if code is None:
                continue
            path = output / job["label"] / "summary.json"
            summary = json.loads(path.read_text()) if path.exists() else {}
            good = code == 0 and summary.get("accepted_engineering_run") is True
            job.update(status="completed" if good else "failed", exit_code=code)
            held |= not good
        if held:
            for job in jobs:
                if job["status"] == "pending":
                    job["status"] = "held_after_failure_or_warning"
        query_error = None
        try:
            available = available_gpus(indices)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            available, query_error = [], str(exc)
        occupied = {j["gpu"] for j in jobs if j["status"] == "running"}
        for index in indices:
            streak[index] = (
                streak.get(index, 0) + 1 if index in available and index not in occupied else 0
            )
            if held or streak[index] < 2:
                continue
            job = next((j for j in jobs if j["status"] == "pending"), None)
            if job is None:
                break
            try:
                still_available = available_gpus(indices)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                query_error = str(exc)
                streak[index] = 0
                continue
            if index not in still_available:
                continue
            parameter_path = output / f"{job['label']}.parameters.json"
            if json.loads(parameter_path.read_text()) != job["parameters"]:
                raise ValueError("candidate parameter file changed")
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(index),
                PYTHONPATH=str(ROOT),
                NUMPY_MADVISE_HUGEPAGE="0",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
            )
            argv = command(campaign, job, output)
            with (output / f"{job['label']}.log").open("x") as log:
                process = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes[job["label"]] = process
            job.update(
                status="running",
                gpu=index,
                pid=process.pid,
                command=argv,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            streak[index] = 0
        write(
            output / "status.json",
            dict(
                jobs=jobs,
                controller_pid=os.getpid(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                query_error=query_error,
                formal_admission=False,
                gpu_allowlist=indices,
            ),
        )
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpus", default="0,5")
    args = parser.parse_args()
    if args.run:
        run(args.output.resolve(), [int(token) for token in args.gpus.split(",")])
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required when preparing")
        print(json.dumps(prepare(args.data_dir, args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
