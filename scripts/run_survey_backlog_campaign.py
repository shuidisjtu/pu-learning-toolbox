"""Pinned full-grid backlog queue, with family canaries and at most four GPUs.

Uses the versioned Survey entrypoint, never the exploratory image CLI. No
subsampling, parameter tuning, historical rebinding or automatic failed retries.
Run the controller inside independent tmux; closing VS Code is then harmless.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts import run_candidate_readiness_campaign as service
from scripts._survey_host import lock_controller

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
    verify_reconciliation(campaign, job)
    verify_followup(campaign)
    if job["status"] == "blocked_preflight":
        raise ValueError("mathematically/resource-blocked fit cannot be launched")
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
    memory = campaign.get("gradpu_activation_memory", "device")
    if memory != "device":
        if memory != "cpu_offload" or job["method"] != "gradpu":
            raise ValueError("invalid backlog activation memory policy")
        argv.extend(["--gradpu-activation-memory", memory])
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
    if campaign.get("gradpu_activation_memory") == "cpu_offload":
        memory = runs[0].get("training_provenance", {}).get("activation_memory", {})
        if (
            memory.get("policy") != "cpu_offload"
            or memory.get("effective_batch_changed") is not False
            or memory.get("precision_changed") is not False
        ):
            raise ValueError("missing verified GradPU offload execution receipt")
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


def verify_reconciliation(campaign, job=None):
    """Replacement jobs retain parent identities and never duplicate live/completed fits."""
    correction = campaign.get("reconciliation")
    if correction is None:
        return
    parent = Path(correction["parent"])
    if service.sha(parent / "campaign.json") != correction["parent_campaign_sha256"]:
        raise ValueError("replacement parent campaign changed")
    original = read(parent / "campaign.json")
    statuses = {j["label"]: j for j in read(parent / "status.json")["jobs"]}
    originals = {j["label"]: j for j in original["jobs"]}
    if (
        correction["additional_scientific_fits"] != 0
        or campaign["protocol_sha256"] != original["protocol_sha256"]
        or campaign["registry_sha256"] != original["registry_sha256"]
        or sorted(correction["replaces_labels"]) != sorted(j["label"] for j in campaign["jobs"])
    ):
        raise ValueError("replacement scope or recipe differs")
    for row in [job] if job is not None else campaign["jobs"]:
        before = originals[row["label"]]
        if row["method"] != "kldce" or statuses[row["label"]]["status"] not in {
            "failed",
            "held_family_failure",
        }:
            raise ValueError("replacement requires failed/held KLDCE parent work")
        for field in ("dataset", "method", "path", "mechanism", "seed", "c", "data_dir", "lane"):
            if row[field] != before[field]:
                raise ValueError(f"replacement identity changed: {field}")


def verify_followup(campaign):
    """Revalidate the accepted GradPU canary and held parent rows before launch."""
    followup = campaign.get("followup")
    if followup is None:
        return
    recovery = Path(followup["recovery"])
    parent = Path(followup["parent"])
    if (
        campaign.get("gradpu_activation_memory") != "cpu_offload"
        or followup["additional_scientific_fits"] != 0
        or service.sha(recovery / "campaign.json") != followup["recovery_campaign_sha256"]
        or service.sha(parent / "campaign.json") != followup["parent_campaign_sha256"]
    ):
        raise ValueError("followup source or offload policy changed")
    fixed, state = read(recovery / "campaign.json"), read(recovery / "status.json")
    if len(state["jobs"]) != 1 or state["jobs"][0]["status"] != "completed":
        raise ValueError("full recovery canary is not completed")
    if accept(recovery, state["jobs"][0], fixed) != followup["accepted_canary"]:
        raise ValueError("accepted recovery artifacts changed")
    if (
        campaign["protocol_sha256"] != fixed["protocol_sha256"]
        or campaign["registry_sha256"] != fixed["registry_sha256"]
        or campaign["gpu_allowlist"] != fixed["gpu_allowlist"]
        or sorted(followup["replaces_labels"]) != sorted(j["label"] for j in campaign["jobs"])
    ):
        raise ValueError("followup recipe, GPU scope or replacement list changed")
    originals = {j["label"]: j for j in read(parent / "campaign.json")["jobs"]}
    statuses = {j["label"]: j["status"] for j in read(parent / "status.json")["jobs"]}
    for row in campaign["jobs"]:
        if row["method"] != "gradpu" or statuses[row["label"]] != "held_family_failure":
            raise ValueError("followup parent job is not held GradPU work")
        for field in (
            "dataset",
            "method",
            "path",
            "mechanism",
            "seed",
            "c",
            "data_dir",
            "lane",
            "family",
        ):
            if row[field] != originals[row["label"]][field]:
                raise ValueError(f"followup identity changed: {field}")


def process_identity(pid):
    """Linux process identity, including start ticks to reject reused PIDs."""
    path = Path(f"/proc/{pid}")
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return path.stat().st_uid, fields[19], (path / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None


class AdoptedWorker:
    """Observe a pre-existing child without signalling or claiming its exit code."""

    def __init__(self, output, job, source):
        self.pid = job["pid"]
        self.identity = process_identity(self.pid)
        if self.identity is not None:
            uid, _, argv = self.identity
            target = str(output / "runs" / job["label"]).encode()
            worker = str(source / "scripts/run_survey_experiment.py").encode()
            launcher = str(source / "scripts/run_survey_backlog_campaign.py").encode()
            matches = worker in argv and target in argv
            matches |= (
                launcher in argv and str(output).encode() in argv and job["label"].encode() in argv
            )
            if uid != os.getuid() or not matches:
                raise ValueError("cannot adopt a foreign or reused worker PID")

    def poll(self):
        current = process_identity(self.pid)
        # The worker launcher execs the training entrypoint with the same PID.
        if self.identity is not None and current is not None and current[:2] == self.identity[:2]:
            return None
        # The controller must still validate all final artifacts with accept().
        return 0


def hold_selectors(job):
    """Exact selectors only: an optional mechanism must not pause other protocols."""
    base = f"{job['method']}:{job['dataset']}"
    return {base, f"{base}:{job['mechanism']}"} if job.get("mechanism") else {base}


def resume_jobs(output, campaign, state, source, holds):
    """Reconcile runtime state only; never rewrite frozen campaign identities."""
    import copy

    original = {job["label"]: job for job in campaign["jobs"]}
    valid_holds = {selector for job in original.values() for selector in hold_selectors(job)}
    if not set(holds) <= valid_holds:
        raise ValueError("unknown method:dataset[:mechanism] hold")
    jobs = copy.deepcopy(state["jobs"])
    if len(jobs) != len(original) or {j["label"] for j in jobs} != set(original):
        raise ValueError("runtime state does not cover the frozen campaign")
    processes = {}
    for job in jobs:
        for key in (
            "method",
            "dataset",
            "path",
            "mechanism",
            "seed",
            "c",
            "data_dir",
            "family",
            "lane",
            "checkpoint_epochs",
            "canary",
        ):
            if job.get(key) != original[job["label"]].get(key):
                raise ValueError("runtime job identity changed")
        held = bool(hold_selectors(job).intersection(holds))
        if held and job["status"] == "running":
            raise ValueError("stop held workers explicitly before resuming their controller")
        if held and job["status"] in {"pending", "waiting_canary", "held_family_failure"}:
            job.update(
                status="paused_by_user",
                reason="user requested transfer/pause; no automatic dispatch",
            )
        if job["status"] == "running":
            processes[job["label"]] = AdoptedWorker(output, job, source)
    return jobs, processes


def dispatch_gpus(indices, shared_families=()):
    """Opt-in sharing is admission only, not a reservation or performance guarantee."""
    exclusive = set(service.available_gpus(indices)) if indices else set()
    shared = set()
    if shared_families and indices:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for row in rows.splitlines():
            index, utilization, free = (int(value.strip()) for value in row.split(","))
            if index in indices and 0 <= utilization <= 10 and free >= 40960:
                shared.add(index)
    return exclusive, shared - exclusive


def sharing_families(campaign, state, requested):
    """Require an explicit profiled family and a completed family canary."""
    completed = {
        j["family"] for j in state["jobs"] if j.get("canary") and j["status"] == "completed"
    }
    known = {j["family"] for j in campaign["jobs"] if j["lane"] == "gpu"}
    if not set(requested) <= known & completed:
        raise ValueError("GPU sharing requires a known GPU family with completed canary")
    return tuple(sorted(set(requested)))


def overlap_gpus(indices):
    """Memory admission for an extra fit beside this queue's own live worker."""
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    available = set()
    for row in rows.splitlines():
        index, free = (int(value.strip()) for value in row.split(","))
        if index in indices and free >= 32768:
            available.add(index)
    return available


def fits_overlap(job, active):
    """At most two fits, with at least one native-2d fit; never two image fits."""
    return len(active) == 1 and (job["path"] == "native_2d" or active[0]["path"] == "native_2d")


def run(output, *, resume=False, holds=(), shared_families=(), gpu_slots=1):
    if gpu_slots not in (1, 2) or (gpu_slots == 2 and not shared_families):
        raise ValueError("GPU slots must be 1 or 2; overlap requires profiled families")
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    _controller_lock = lock_controller(output / "controller.lock")
    campaign = read(output / "campaign.json")
    verify_reconciliation(campaign)
    verify_followup(campaign)
    if campaign.get("reconciliation"):
        # A separate lock does not interrupt the live parent controller. It
        # serializes all replacement campaigns derived from this parent.
        _replacement_lock = lock_controller(
            Path(campaign["reconciliation"]["parent"]) / "kldce-recovery.lock"
        )
    if campaign.get("followup"):
        _followup_lock = lock_controller(
            Path(campaign["followup"]["parent"]) / "gradpu-followup.lock"
        )
    allowed_gpus = tuple(campaign["gpu_allowlist"])
    if any(gpu not in ALLOWED_GPUS for gpu in allowed_gpus):
        raise ValueError("campaign cannot expand the shared four-GPU limit")
    worker_source = Path(campaign["source"])
    controller_commit = clean_commit()
    if not resume and (worker_source != ROOT or controller_commit != campaign["commit"]):
        raise ValueError("campaign source changed")
    if resume:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=worker_source).strip():
            raise ValueError("pinned worker source is dirty")
        if (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=worker_source, text=True
            ).strip()
            != campaign["commit"]
        ):
            raise ValueError("pinned worker commit changed")
    if service.sha(worker_source / "uv.lock") != campaign["dependency_lock_sha256"]:
        raise ValueError("dependency lock changed")
    for evidence in campaign["inputs"].values():
        verify_inputs(evidence, full_hash=True)
    if (output / "status.json").exists() and not resume:
        raise ValueError(
            "existing status requires explicit reconciliation, not an automatic restart"
        )
    if holds and not resume:
        raise ValueError("holds require explicit resume")
    if resume:
        state = read(output / "status.json")
        holds = tuple(sorted(set(state.get("user_holds", ())) | set(holds)))
        jobs, processes = resume_jobs(output, campaign, state, worker_source, holds)
        service.write(output / f"status.before-resume-{os.getpid()}.json", state, exclusive=True)
    else:
        jobs, processes = campaign["jobs"], {}
    shared_families = sharing_families(campaign, dict(jobs=jobs), shared_families)
    streak = {}
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
            adopted = isinstance(processes[job["label"]], AdoptedWorker)
            job.update(exit_code=None if adopted else code, finished_at=now())
            if adopted:
                job["completion_evidence"] = "artifact acceptance; original exit code unavailable"
        release(jobs)
        query_error = None
        try:
            exclusive, shared = dispatch_gpus(allowed_gpus, shared_families)
            available = exclusive | shared
            overlap = overlap_gpus(allowed_gpus) if gpu_slots == 2 else set()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            available, exclusive, shared, query_error = set(), set(), set(), str(error)
            overlap = set()
        for gpu in (*allowed_gpus, None):
            active = [j for j in jobs if j["status"] == "running" and j.get("gpu") == gpu]
            overlapping = gpu is not None and len(active) == 1 and gpu_slots == 2
            ready = (not active and (gpu is None or gpu in available)) or (
                overlapping and gpu in overlap
            )
            streak[gpu] = streak.get(gpu, 0) + 1 if ready else 0
            if streak[gpu] < 2:
                continue
            lane = "cpu" if gpu is None else "gpu"
            pending = [j for j in jobs if j["status"] == "pending" and j["lane"] == lane]
            if gpu in shared or overlapping:
                pending = [j for j in pending if j["family"] in shared_families]
            if overlapping:
                pending = [j for j in pending if fits_overlap(j, active)]
            pending.sort(
                key=lambda j: (
                    j["method"] == "kldce",
                    not j["canary"],
                    j["mechanism"] != "scar",
                    j["dataset"] != "cifar10" if lane == "gpu" else False,
                    j["path"] != "native_cnn",
                    j["method"] != "rp",
                    j.get("preflight", {}).get("named_payload_bytes", 0),
                    j["label"],
                )
            )
            if not pending:
                continue
            job = pending[0]
            # Dense kernel payload is not peak RSS. Leave ample headroom and
            # never run two kernel fits concurrently on this shared machine.
            ram_gib = (
                650
                if job["method"] == "kldce"
                else (192 if campaign.get("gradpu_activation_memory") == "cpu_offload" else 64)
            )
            if memory_available() < ram_gib * 1024**3:
                query_error = f"waiting for RAM headroom: {job['label']}"
                continue
            if gpu is not None:
                try:
                    fresh_exclusive, fresh_shared = dispatch_gpus(allowed_gpus, shared_families)
                    if overlapping:
                        if gpu not in overlap_gpus(allowed_gpus):
                            continue
                    elif gpu not in fresh_exclusive and not (
                        gpu in fresh_shared and job["family"] in shared_families
                    ):
                        continue
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    query_error = str(error)
                    continue
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                PYTHONPATH=str(worker_source),
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                NUMPY_MADVISE_HUGEPAGE="0",
            )
            argv = [
                sys.executable,
                "-u",
                str(worker_source / "scripts/run_survey_backlog_campaign.py"),
                "--output",
                str(output),
                "--worker",
                job["label"],
            ]
            with (output / f"{job['label']}.log").open("x") as log:
                process = subprocess.Popen(
                    argv,
                    cwd=worker_source,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes[job["label"]] = process
            job.update(
                status="running",
                pid=process.pid,
                gpu=gpu,
                started_at=now(),
                shared_gpu_admission=gpu is not None and gpu not in fresh_exclusive,
                gpu_overlap_admission=overlapping,
            )
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
                controller_source=str(ROOT),
                controller_commit=controller_commit,
                user_holds=list(holds),
                shared_gpu_families=list(shared_families),
                gpu_job_slots=gpu_slots,
            ),
        )
        time.sleep(10)


def monitor(output):
    from scripts.monitor_survey_campaign import monitor as display

    display([output])


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
    parser.add_argument(
        "--resume", action="store_true", help="explicitly adopt existing runtime state"
    )
    parser.add_argument("--hold", action="append", default=[], metavar="METHOD:DATASET[:MECHANISM]")
    parser.add_argument(
        "--share-low-memory-family",
        action="append",
        default=[],
        metavar="FAMILY",
        help="operator-profiled family eligible for co-tenancy on low-load GPUs with 40 GiB free",
    )
    parser.add_argument("--gpu-slots", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if (
        args.resume or args.hold or args.share_low_memory_family or args.gpu_slots != 1
    ) and not args.run:
        parser.error("resume/hold/sharing require --run")
    output = args.output.resolve()
    if args.worker:
        worker(output, args.worker)
    elif args.run:
        run(
            output,
            resume=args.resume,
            holds=args.hold,
            shared_families=args.share_low_memory_family,
            gpu_slots=args.gpu_slots,
        )
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
