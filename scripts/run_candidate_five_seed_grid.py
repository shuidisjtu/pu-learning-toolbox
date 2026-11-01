"""Extend accepted full-data canaries to five seeds and three SCAR rates.

Existing seed0/c0.1 fits are referenced, never overwritten or rerun. Each
method/view waits for its own accepted canary. This user-authorized observation
grid does not grant formal protocol/recipe admission. Only scheduler/preflight
code may differ from the immutable canary source.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts import run_candidate_readiness_campaign as base
from scripts._survey_host import lock_controller

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = (0, 5, 6, 7)


def read(path):
    return json.loads(Path(path).read_text())


def source_commit():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("source must be committed and clean")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def expand_jobs(parent):
    jobs = []
    for c in ("0.1", "0.3", "0.5"):
        for seed in range(5):
            if seed == 0 and c == "0.1":
                continue
            for canary in parent["jobs"]:
                method, view = canary["method"], canary["view"]
                jobs.append(
                    dict(
                        label=f"{method}-cifar10-s{seed}-c{c}-{view}",
                        method=method,
                        view=view,
                        seed=seed,
                        c=c,
                        canary=canary["label"],
                        parameters=canary["parameters"],
                        status="waiting_canary",
                    )
                )
    if len(jobs) != 84 or len({j["label"] for j in jobs}) != 84:
        raise ValueError("expected six canaries and 84 unique additional fits")
    return jobs


def prepare(parent_dir, data_root, output):
    if output.exists():
        raise ValueError("output already exists; no implicit resume")
    commit = source_commit()
    parent = read(parent_dir / "campaign.json")
    if parent["schema_version"] != "candidate-readiness-1":
        raise ValueError("expected a pinned single-seed readiness campaign")
    changed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            parent["commit"],
            "HEAD",
            "--",
            "pu_toolbox",
            "scripts/run_candidate_image_experiment.py",
            "uv.lock",
        ],
        cwd=ROOT,
        text=True,
    )
    if changed.strip() or base.sha(ROOT / "uv.lock") != parent["dependency_lock_sha256"]:
        raise ValueError("canary training source or dependency lock differs")
    inputs = {}
    for seed in range(5):
        directory = (data_root / f"split_{seed}").resolve()
        inputs[str(seed)] = dict(
            directory=str(directory), evidence=base.validate_split(directory, seed=seed)
        )
    if inputs["0"]["evidence"] != parent["input_evidence"]:
        raise ValueError("seed0 differs from the canary inputs")
    campaign = dict(
        schema_version="candidate-five-seed-grid-1",
        source=str(ROOT),
        commit=commit,
        parent_directory=str(parent_dir.resolve()),
        parent_campaign_sha256=base.sha(parent_dir / "campaign.json"),
        dependency_lock_sha256=base.sha(ROOT / "uv.lock"),
        inputs=inputs,
        jobs=expand_jobs(parent),
        inherited_canaries=[j["label"] for j in parent["jobs"]],
        total_fits_including_canaries=90,
        additional_fits=84,
        formal_admission=False,
        frozen_budget_approved=False,
        scope="user_authorized_full_split_five_seed_SCAR_observations_not_frozen_recipe_admission",
        adaptations=parent["adaptations"][:4],
    )
    output.mkdir(parents=True, exist_ok=False)
    for job in campaign["jobs"]:
        base.write(output / f"{job['label']}.parameters.json", job["parameters"], exclusive=True)
    base.write(output / "campaign.json", campaign, exclusive=True)
    return campaign


def accepted(directory, job, inputs):
    """Check identity, source, input and persisted artifacts, not just exit code."""
    summary = read(directory / "summary.json")
    plan = read(directory / "plan.json")
    if not (
        summary.get("accepted_engineering_run") is True
        and summary.get("selected_models_replayed") is True
        and summary.get("warning_count") == 0
        and summary.get("retry_count") == 0
        and plan.get("method") == job["method"]
        and plan.get("training_view") == job["view"]
        and plan.get("seed") == job["seed"]
        and plan.get("c_token") == job["c"]
        and plan.get("parameters_requested") == job["parameters"]
        and plan.get("input_files_sha256") == inputs["evidence"]["files_sha256"]
        and summary.get("plan_sha256") == plan.get("plan_sha256")
    ):
        raise ValueError("result identity or warning/replay gate failed")
    source_hashes = plan["source_files_sha256"]
    files = summary["files_sha256"]
    if (
        not source_hashes
        or not {"plan.json", "manifest.json", "selected_PA.pkl", "selected_OA.pkl"} <= files.keys()
    ):
        raise ValueError("missing artifact/source identity")
    for name, digest in source_hashes.items():
        if base.sha(ROOT / name) != digest:
            raise ValueError(f"training source mismatch: {name}")
    for name, digest in files.items():
        if base.sha(directory / name) != digest:
            raise ValueError(f"result artifact mismatch: {name}")
    return base.sha(directory / "summary.json")


def release_canaries(campaign, parent_state, validated):
    parents = {j["label"]: j for j in parent_state["jobs"]}
    for job in campaign["jobs"]:
        if job["status"] != "waiting_canary":
            continue
        label = job["canary"]
        state = parents[label]["status"]
        if state == "completed":
            try:
                if label not in validated:
                    canary = dict(job, seed=0, c="0.1")
                    validated[label] = accepted(
                        Path(campaign["parent_directory"]) / label, canary, campaign["inputs"]["0"]
                    )
                job.update(status="pending", canary_summary_sha256=validated[label])
            except (OSError, ValueError, KeyError) as error:
                job.update(status="held_canary_invalid", reason=str(error))
        elif state not in ("pending", "running"):
            job.update(status="held_canary_failed", reason=f"canary status: {state}")


def worker_command(campaign, job, output):
    command = base.command(
        dict(
            input_directory=campaign["inputs"][str(job["seed"])]["directory"],
            input_evidence={"population_prior": 0.4},
        ),
        job,
        output,
    )
    command[command.index("--seed") + 1] = str(job["seed"])
    command[command.index("--c") + 1] = job["c"]
    return command


def run(output, indices):
    if not indices or len(set(indices)) != len(indices) or any(i not in ALLOWED for i in indices):
        raise ValueError("use distinct GPUs from 0,5,6,7 only (maximum four)")
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    _controller_lock = lock_controller(output / "controller.lock")
    campaign = read(output / "campaign.json")
    parent_dir = Path(campaign["parent_directory"])
    if campaign["commit"] != source_commit() or campaign["source"] != str(ROOT):
        raise ValueError("campaign source changed")
    if base.sha(parent_dir / "campaign.json") != campaign["parent_campaign_sha256"]:
        raise ValueError("parent campaign changed")
    if base.sha(ROOT / "uv.lock") != campaign["dependency_lock_sha256"]:
        raise ValueError("dependency lock changed")
    for seed, entry in campaign["inputs"].items():
        if base.validate_split(Path(entry["directory"]), seed=int(seed)) != entry["evidence"]:
            raise ValueError("campaign input changed")
    jobs, processes, streak, validated = campaign["jobs"], {}, {}, {}
    base.write(output / "status.json", dict(jobs=jobs, controller_pid=os.getpid()), exclusive=True)
    while any(j["status"] in ("waiting_canary", "pending", "running") for j in jobs):
        parent_state = read(parent_dir / "status.json")
        release_canaries(campaign, parent_state, validated)
        for job in jobs:
            if job["status"] != "running":
                continue
            code = processes[job["label"]].poll()
            if code is None:
                continue
            try:
                if code != 0:
                    raise ValueError(f"worker exit {code}")
                job["summary_sha256"] = accepted(
                    output / job["label"], job, campaign["inputs"][str(job["seed"])]
                )
                job["status"] = "completed"
            except (OSError, ValueError, KeyError) as error:
                job.update(status="failed", reason=str(error))
                for sibling in jobs:
                    if sibling["canary"] == job["canary"] and sibling["status"] in (
                        "pending",
                        "waiting_canary",
                    ):
                        sibling.update(status="held_family_failure", reason=job["label"])
            job["exit_code"] = code
        # The original controller retains 0/5 until its entire queue is terminal.
        parent_active = any(j["status"] in ("pending", "running") for j in parent_state["jobs"])
        scope = [i for i in indices if i not in (0, 5) or not parent_active]
        error = None
        try:
            available = base.available_gpus(scope)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            available, error = [], str(exc)
        occupied = {j["gpu"] for j in jobs if j["status"] == "running"}
        for index in scope:
            streak[index] = (
                streak.get(index, 0) + 1 if index in available and index not in occupied else 0
            )
            job = next((j for j in jobs if j["status"] == "pending"), None)
            if streak[index] < 2 or job is None:
                continue
            try:
                if index not in base.available_gpus(scope):
                    continue
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                error = str(exc)
                continue
            if read(output / f"{job['label']}.parameters.json") != job["parameters"]:
                raise ValueError("job parameter file changed")
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(index),
                PYTHONPATH=str(ROOT),
                NUMPY_MADVISE_HUGEPAGE="0",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
            )
            argv = worker_command(campaign, job, output)
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
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            streak[index] = 0
        base.write(
            output / "status.json",
            dict(
                jobs=jobs,
                controller_pid=os.getpid(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                query_error=error,
                gpu_allowlist=indices,
                formal_admission=False,
            ),
        )
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpus", default="0,5,6,7")
    args = parser.parse_args()
    if args.run:
        run(args.output.resolve(), [int(i) for i in args.gpus.split(",")])
    else:
        if args.parent is None or args.data_root is None:
            parser.error("preparation requires --parent and --data-root")
        result = prepare(args.parent.resolve(), args.data_root.resolve(), args.output.resolve())
        print(json.dumps(dict(additional_fits=result["additional_fits"], formal_admission=False)))


if __name__ == "__main__":
    main()
