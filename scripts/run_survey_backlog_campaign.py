"""Pinned full-grid backlog queue, with family canaries and at most four GPUs.

Uses the versioned Survey entrypoint, never the exploratory image CLI. No
subsampling, parameter tuning, historical rebinding or automatic failed retries.
Run the controller inside independent tmux; closing VS Code is then harmless.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts import run_candidate_readiness_campaign as service

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_GPUS = (0, 5, 6, 7)
REVISIONS = {
    "survey-v3.1": ("survey-v3.0", "survey_recipe_registry_v1_1.json"),
    "survey-v3.2": ("survey-v3.1", "survey_recipe_registry_v1_2.json"),
}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def clean_commit():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("backlog source must be committed and clean")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def expand_jobs(protocol, previous, locations, support_root):
    """Expand precisely the new cells; no completed legacy unit is rerun."""
    from pu_toolbox.experiment.survey_profile_identity import is_clean_support_profile
    from pu_toolbox.experiment.survey_staged_budget import checkpoint_epochs

    def identity(row):
        return row["dataset"], row["method"], row["training_path"]

    old = {identity(r) for r in previous["execution_units"] if r["runnable"]}
    jobs = []
    for row in protocol["execution_units"]:
        if not row["runnable"] or identity(row) in old:
            continue
        dataset, method, path = identity(row)
        profile = protocol["method_profiles"][method]
        support = is_clean_support_profile(protocol, method)
        directory = support_root / dataset if support else Path(locations[dataset])
        for mechanism, tokens in protocol["c_tokens"].items():
            if mechanism not in profile.get("labeling_mechanisms", protocol["c_tokens"]):
                continue
            family = f"{method}-{dataset}-{path}-{mechanism}"
            for seed in protocol["seeds"]:
                for token in tokens:
                    canary = seed == protocol["seeds"][0] and token == tokens[0]
                    jobs.append(
                        dict(
                            label=f"{family}-s{seed}-c{token}",
                            family=family,
                            dataset=dataset,
                            method=method,
                            path=path,
                            mechanism=mechanism,
                            seed=seed,
                            c=token,
                            canary=canary,
                            status="pending" if canary else "waiting_canary",
                            data_dir=str((directory / f"split_{seed}").resolve()),
                            support=support,
                            lane="cpu" if method in {"kldce", "puet", "rp"} else "gpu",
                            checkpoint_epochs=checkpoint_epochs(protocol["budgets"][row["budget"]]),
                        )
                    )
    if len({j["label"] for j in jobs}) != len(jobs):
        raise ValueError("backlog contains duplicate fits")
    return jobs


def input_evidence(directory, *, dataset, seed, support):
    """Pin full role files and generation metadata without changing any data."""
    from pu_toolbox.experiment.support_data import load_support_dataset

    receipt = directory / ("support_manifest.json" if support else "split_manifest.json")
    metadata = read(receipt)
    if metadata.get("dataset") != dataset or metadata.get("seed") != seed:
        raise ValueError("backlog split identity mismatch")
    if support:
        load_support_dataset(directory)
    prior = metadata["class_prior"]["population"]
    if isinstance(prior, bool) or not 0 < prior < 1:
        raise ValueError("split must declare a valid population prior")
    files = [
        receipt,
        *(directory / f"{role}.npz" for role in ("train", "pu_val", "clean_val", "test")),
    ]
    if support:
        files.append(directory / "support.npz")
    return dict(
        population_prior=prior,
        files={
            str(p): dict(
                sha256=service.sha(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns
            )
            for p in files
        },
    )


def verify_inputs(evidence, *, full_hash=False):
    for name, expected in evidence["files"].items():
        path = Path(name)
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"input artifact changed: {path}")
        if full_hash and service.sha(path) != expected["sha256"]:
            raise ValueError(f"input artifact hash changed: {path}")


def prepare(output, previous_queue, support_root, *, revision="survey-v3.1"):
    from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_protocol_path
    from pu_toolbox.experiment.survey_recipe_registry import canonical_digest, load_registry

    if output.exists():
        raise ValueError("output exists; never overwrite a campaign")
    commit = clean_commit()
    previous_revision, registry_file = REVISIONS[revision]
    previous = load_protocol(resolve_protocol_path(previous_revision))
    protocol = load_protocol(resolve_protocol_path(revision))
    registry = load_registry(ROOT / "pu_toolbox/experiment" / registry_file)
    locations = {}
    for job in read(previous_queue)["jobs"]:
        if not job.get("support_data"):
            key, value = job["dataset"], job["data_root"]
            if key in locations and locations[key] != value:
                raise ValueError("previous queue has ambiguous data roots")
            locations[key] = value
    jobs = expand_jobs(protocol, previous, locations, support_root)
    inputs = {}
    for job in jobs:
        name = job["data_dir"]
        if name not in inputs:
            print(
                "VERIFY_INPUT", job["dataset"], job["seed"], "support", job["support"], flush=True
            )
            inputs[name] = input_evidence(
                Path(name), dataset=job["dataset"], seed=job["seed"], support=job["support"]
            )
    payload = dict(
        schema_version="survey-backlog-campaign-1",
        source=str(ROOT),
        commit=commit,
        dependency_lock_sha256=service.sha(ROOT / "uv.lock"),
        protocol_version=revision,
        protocol_sha256=digest(protocol),
        registry_sha256=canonical_digest(registry),
        inputs=inputs,
        jobs=jobs,
        total_fits=len(jobs),
        gpu_allowlist=[] if revision == "survey-v3.2" else list(ALLOWED_GPUS),
        max_gpu_jobs=0 if revision == "survey-v3.2" else 4,
        max_cpu_jobs=1,
        formal_admission=False,
        recipe_owner_confirmed=True,
        previous_queue=str(previous_queue),
        previous_results_rebound=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    service.write(output / "campaign.json", payload, exclusive=True)
    return payload


def worker(output, label):
    """One full-config fit with a checked batch packet, never a silent CPU fallback."""
    from pu_toolbox.experiment import batch_evidence
    from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_protocol_path
    from pu_toolbox.experiment.survey_recipe_registry import canonical_digest, load_registry

    campaign = read(output / "campaign.json")
    if clean_commit() != campaign["commit"] or campaign["source"] != str(ROOT):
        raise ValueError("worker source changed")
    job = next(j for j in campaign["jobs"] if j["label"] == label)
    evidence = campaign["inputs"][job["data_dir"]]
    verify_inputs(evidence)
    target = output / "runs" / label
    if target.exists():
        raise ValueError("worker output already exists; no implicit rerun")
    revision = campaign.get("protocol_version", "survey-v3.1")
    protocol = load_protocol(resolve_protocol_path(revision))
    registry_path = ROOT / "pu_toolbox/experiment" / REVISIONS[revision][1]
    registry = load_registry(registry_path)
    if (
        digest(protocol) != campaign["protocol_sha256"]
        or canonical_digest(registry) != campaign["registry_sha256"]
    ):
        raise ValueError("worker protocol/registry changed")
    device = "cpu"
    if job["lane"] == "gpu":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("requested CUDA unavailable; no CPU fallback")
        device = "cuda"
    packet = batch_evidence.build_evidence(
        unit=dict(
            dataset=job["dataset"],
            method=job["method"],
            training_path=job["path"],
            mechanism=job["mechanism"],
        ),
        seeds=[job["seed"]],
        c_tokens=[job["c"]],
        protocol_version=protocol["protocol_version"],
        protocol_sha256=digest(protocol),
        recipe_registry=dict(
            version=registry["recipe_registry_version"], sha256=canonical_digest(registry)
        ),
    )
    if packet["code_commit"] != campaign["commit"] or packet["code_commit_dirty"]:
        raise ValueError("batch source identity differs")
    receipt = batch_evidence.write(output / batch_evidence.DIRECTORY, packet)
    argv = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_survey_experiment.py"),
        job["data_dir"],
        "--protocol",
        revision,
        "--method",
        job["method"],
        "--dataset",
        job["dataset"],
        "--training-path",
        job["path"],
        "--labeling-mechanism",
        job["mechanism"],
        "--seeds",
        str(job["seed"]),
        "--c",
        job["c"],
        "--class-prior",
        str(evidence["population_prior"]),
        "--recipe-registry",
        str(registry_path),
        "--recipe-binding-schema-version",
        "survey-recipe-binding-2",
        "--device",
        device,
        "--out-dir",
        str(target),
        "--batch-evidence",
        str(receipt),
        "--adapter-cache",
        str(output / "adapter-cache" / job["dataset"] / str(job["seed"])),
        "--reclaim-unselected-checkpoints",
    ]
    if job["support"]:
        argv.extend(
            [
                "--support-dir",
                job["data_dir"],
                "--split-ref",
                str(Path(job["data_dir"]) / "support_manifest.json"),
            ]
        )
    print("BACKLOG_FIT", label, "FULL_RECIPE", campaign["protocol_sha256"], flush=True)
    os.execv(sys.executable, argv)


def accept(output, job, campaign):
    """A worker exit is not completion: validate identities, budget and weights."""
    paths = list((output / "runs" / job["label"]).rglob("manifest.json"))
    if len(paths) != 1:
        raise ValueError("expected exactly one manifest")
    manifest = read(paths[0])
    row = manifest["execution_unit"]
    if (
        manifest.get("failures")
        or manifest["protocol_sha256"] != campaign["protocol_sha256"]
        or manifest["recipe_registry_sha256"] != campaign["registry_sha256"]
        or manifest["seed"] != job["seed"]
        or manifest["c_requested_token"] != job["c"]
        or (row["dataset"], row["method"], row["training_path"])
        != (job["dataset"], job["method"], job["path"])
    ):
        raise ValueError("result identity/failure gate failed")
    if any(
        manifest["generation"][role]["mechanism"] != job["mechanism"]
        for role in ("train", "pu_val")
    ):
        raise ValueError("result labeling mechanism differs from queued fit")
    expected = (
        {"PA+clean-support" if job["support"] else "PA", "OA"}
        if job["mechanism"] == "scar"
        else {"OA"}
    )
    if set(manifest["test_results"]) != expected:
        raise ValueError("selection protocols incomplete")
    runs = manifest["candidate_runs"]
    if len(runs) != 1 or runs[0]["status"] != "succeeded" or runs[0]["attempts"] != 1:
        raise ValueError("candidate or retry gate failed")
    if campaign.get("protocol_version") == "survey-v3.2":
        validate_classical_image_receipt(manifest, job)
    checkpoints = runs[0].get("epoch_checkpoints", [])
    if (
        job["checkpoint_epochs"]
        and len({c["epoch_position"] for c in checkpoints}) != job["checkpoint_epochs"]
    ):
        raise ValueError("incomplete epoch budget")
    for selection in manifest["selection"].values():
        checkpoint = selection.get("checkpoint")
        if checkpoint and service.sha(Path(checkpoint["path"])) != checkpoint["sha256"]:
            raise ValueError("selected checkpoint missing or changed")
    return dict(manifest=str(paths[0]), manifest_sha256=service.sha(paths[0]))


def validate_classical_image_receipt(manifest, job):
    """Accept the feature path only with frozen representation and real work counts."""
    if job["method"] not in {"puet", "rp"} or job["path"] != "cnn_feature_adapter":
        raise ValueError("v3.2 campaign is limited to PUET/RP image adapter additions")
    representation = manifest["representation"]
    adapter = representation["adapter"]
    image = representation["image_preprocessing"]
    if (
        adapter.get("encoder_training") != "none; seeded_random_frozen"
        or adapter.get("feature_dimension") != 512
        or adapter["feature_sha256"] != representation["feature_sha256"]
        or adapter["encoder_state_sha256"] != image["encoder_state_sha256"]
        or image["initialization_seed"] != job["seed"]
        or image["backbone"]["weights"] is not None
    ):
        raise ValueError("classical image representation receipt differs")
    receipt = manifest["candidate_runs"][0]["training_provenance"]
    if (
        receipt["status"] != "completed"
        or receipt["validation_used_for_training"]
        or receipt["neural_epoch_count"] != 0
        or receipt["work_started"] != receipt["work_completed"]
    ):
        raise ValueError("classical image training receipt is incomplete")
    if job["method"] == "puet":
        if receipt["trees_requested"] != 100 or receipt["trees_completed"] != 100:
            raise ValueError("PUET image budget must complete all 100 trees")
    elif receipt["cross_fits_completed"] != 3 or receipt["final_refits_completed"] != 1:
        raise ValueError("RP image budget must complete three folds and final refit")


def release(jobs):
    canaries = {j["family"]: j for j in jobs if j["canary"]}
    failed = {j["family"] for j in jobs if j["status"] == "failed"}
    for job in jobs:
        if job["status"] not in {"waiting_canary", "pending"}:
            continue
        if job["family"] in failed:
            job.update(
                status="held_family_failure", reason="inspect family failure; no automatic retries"
            )
        elif job["status"] == "waiting_canary" and canaries[job["family"]]["status"] == "completed":
            job["status"] = "pending"


def memory_available():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) * 1024


def run(output):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    lock = (output / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign = read(output / "campaign.json")
    allowed_gpus = tuple(campaign["gpu_allowlist"])
    if any(gpu not in ALLOWED_GPUS for gpu in allowed_gpus):
        raise ValueError("campaign cannot expand the shared four-GPU limit")
    if campaign["source"] != str(ROOT) or clean_commit() != campaign["commit"]:
        raise ValueError("campaign source changed")
    if service.sha(ROOT / "uv.lock") != campaign["dependency_lock_sha256"]:
        raise ValueError("dependency lock changed")
    for evidence in campaign["inputs"].values():
        verify_inputs(evidence, full_hash=True)
    if (output / "status.json").exists():
        raise ValueError(
            "existing status requires explicit reconciliation, not an automatic restart"
        )
    jobs, processes, streak = campaign["jobs"], {}, {}
    while any(j["status"] in {"waiting_canary", "pending", "running"} for j in jobs):
        for job in jobs:
            if job["status"] != "running":
                continue
            code = processes[job["label"]].poll()
            if code is None:
                continue
            try:
                if code:
                    raise ValueError(f"worker exit {code}")
                job.update(accept(output, job, campaign), status="completed")
            except (OSError, ValueError, KeyError) as error:
                job.update(status="failed", reason=str(error))
            job.update(exit_code=code, finished_at=now())
        release(jobs)
        query_error = None
        try:
            available = service.available_gpus(allowed_gpus) if allowed_gpus else []
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            available, query_error = [], str(error)
        occupied = {j.get("gpu") for j in jobs if j["status"] == "running"}
        for gpu in (*allowed_gpus, None):
            streak[gpu] = (
                streak.get(gpu, 0) + 1
                if gpu not in occupied and (gpu is None or gpu in available)
                else 0
            )
            if streak[gpu] < 2:
                continue
            lane = "cpu" if gpu is None else "gpu"
            pending = [j for j in jobs if j["status"] == "pending" and j["lane"] == lane]
            pending.sort(
                key=lambda j: (
                    j["method"] == "kldce",
                    not j["canary"],
                    j["mechanism"] != "scar",
                    j["dataset"] != "cifar10" if lane == "gpu" else False,
                    j["path"] != "native_cnn",
                    j["method"] != "rp",
                    j["label"],
                )
            )
            if not pending:
                continue
            job = pending[0]
            # Dense kernel payload is not peak RSS. Leave ample headroom and
            # never run two kernel fits concurrently on this shared machine.
            if memory_available() < (650 if job["method"] == "kldce" else 64) * 1024**3:
                query_error = f"waiting for RAM headroom: {job['label']}"
                continue
            if gpu is not None:
                try:
                    if gpu not in service.available_gpus(allowed_gpus):
                        continue
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    query_error = str(error)
                    continue
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                PYTHONPATH=str(ROOT),
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                NUMPY_MADVISE_HUGEPAGE="0",
            )
            argv = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--output",
                str(output),
                "--worker",
                job["label"],
            ]
            with (output / f"{job['label']}.log").open("x") as log:
                process = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes[job["label"]] = process
            job.update(status="running", pid=process.pid, gpu=gpu, started_at=now())
            streak[gpu] = 0
            print("START", job["label"], "GPU", gpu, flush=True)
        service.write(
            output / "status.json",
            dict(
                jobs=jobs,
                controller_pid=os.getpid(),
                updated_at=now(),
                query_error=query_error,
                gpu_allowlist=list(allowed_gpus),
                formal_admission=False,
            ),
        )
        time.sleep(10)


def monitor(output):
    while True:
        state = read(output / "status.json")
        jobs = state["jobs"]
        print("\033[2J\033[H", end="")
        print(
            "Survey v3.1 backlog |",
            state["updated_at"],
            "|",
            dict(Counter(j["status"] for j in jobs)),
        )
        print(
            f"{'GPU':<5} {'METHOD':<12} {'DATASET':<19} {'MECHANISM':<12} "
            f"{'SEED/C':<9} {'SNAPSHOTS':<14} STATUS"
        )
        for job in jobs:
            if job["status"] != "running":
                continue
            snapshots = len(list((output / "runs" / job["label"]).rglob("epoch_*.pt")))
            expected = job["checkpoint_epochs"]
            if job["method"] == "self_pu_full_meta" and expected:
                expected *= 2
            progress = (
                f"{snapshots}/{expected}"
                if expected
                else {"puet": "100 trees", "rp": "3 folds+refit"}.get(job["method"], "solver")
            )
            print(
                f"{str(job['gpu']):<5} {job['method']:<12} {job['dataset']:<19} "
                f"{job['mechanism']:<12} {str(job['seed']) + '/' + job['c']:<9} "
                f"{progress:<14} running PID={job['pid']}"
            )
        if state.get("query_error"):
            print(state["query_error"])
        print(
            f"Full recipe; GPU allowlist={state['gpu_allowlist']}; "
            "classical column is budget, not live iteration count; failures held.",
            flush=True,
        )
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-queue", type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--revision", choices=tuple(REVISIONS), default="survey-v3.1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker")
    mode.add_argument("--monitor", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        worker(output, args.worker)
    elif args.run:
        run(output)
    elif args.monitor:
        monitor(output)
    else:
        if args.previous_queue is None or args.support_root is None:
            parser.error("preparation requires previous-queue and support-root")
        print(
            json.dumps(
                {
                    "total_fits": prepare(
                        output, args.previous_queue, args.support_root, revision=args.revision
                    )["total_fits"]
                }
            )
        )


if __name__ == "__main__":
    main()
