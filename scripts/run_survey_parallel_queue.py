#!/usr/bin/env python3
"""Linux host controller: adopt live Survey workers and shard untouched groups.

Training stays in the original pinned checkout through the legacy worker.
Only the controller is replaced; a worker is never killed, paused or migrated.
Closing an editor is safe when this service is hosted in independent tmux.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pu_toolbox.experiment.queue_scheduling import (
    allowed_devices,
    choose_pending,
    dispatch_priority,
    dispatch_slots,
    parse_gpu_indices,
    split_pending_jobs,
    validate_fit_ownership,
)
from pu_toolbox.experiment.support_campaign import (
    load_support_campaign,
    mount_campaign,
    release_canary_followups,
    repair_manifest_findings,
    write_campaign_tables,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.new")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def process_identity(pid):
    """Linux PID start time prevents a recycled PID from being adopted/killed."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "start_ticks": int(fields[19]),
            "state": fields[0],
            "wait_status": int(fields[49]) if len(fields) > 49 else None,
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def command_line(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")[:-1]
    except FileNotFoundError:
        return []


def devices():
    """Fresh ownership and memory samples, never reserve a card with dummy work."""
    rows = csv.reader(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).splitlines()
    )
    result = {
        int(r[0]): {
            "uuid": r[1].strip(),
            "util": int(r[2]),
            "free_mib": int(r[3]),
            "used_mib": int(r[4]),
            "pids": set(),
        }
        for r in rows
    }
    by_uuid = {d["uuid"]: d for d in result.values()}
    rows = csv.reader(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).splitlines()
    )
    for row in rows:
        if len(row) == 2 and row[1].strip().isdigit():
            by_uuid[row[0].strip()]["pids"].add(int(row[1]))
    return result


def successful_artifacts(job):
    """Reconcile identities, binding and metrics, not just a worker's exit."""
    from pu_toolbox.experiment.pilot_plan import manifest_identity
    from pu_toolbox.experiment.survey_recipe_binding import matches_recipe_binding

    expected = {tuple(k) for k in job["expected_keys"]}
    found = set()
    for path in Path(job["root"]).rglob("manifest.json"):
        try:
            m = read(path)
            key = manifest_identity(m)
            if (
                key is None
                or key in found
                or key not in expected
                or m.get("failures")
                or not m.get("test_results")
                or not m.get("selection")
                or m.get("protocol_sha256") != job["protocol_sha256"]
                or m.get("run_view") != job["expected_view"]
                or not m.get("batch")
                or not matches_recipe_binding(m, job["expected_recipe"])
            ):
                return False
            found.add(key)
            if job.get("campaign_id"):
                scripts = str(Path(__file__).resolve().parent)
                if scripts not in sys.path:
                    sys.path.insert(0, scripts)
                from audit_survey_batches import _checkpoint_reassessment_problems

                findings = repair_manifest_findings(
                    job, m, checkpoint_check=_checkpoint_reassessment_problems
                )
                job["quality_findings"] = findings
                if findings:
                    return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return found == expected


def terminal_marker(job):
    """Adopted non-child workers have no waitpid exit code; state this honestly."""
    try:
        with Path(job["log"]).open("rb") as stream:
            stream.seek(max(0, Path(job["log"]).stat().st_size - 4096))
            return b"Versioned pilot evidence only: check manifest formal_blockers" in stream.read()
    except OSError:
        return False


def load_legacy(path):
    spec = importlib.util.spec_from_file_location("legacy_survey_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconcile_workers(state, gpu_devices):
    """Discover owned workers after controller exit, including unsaved spawns."""
    jobs = state["jobs"]
    by_root = {j["root"]: j for j in jobs}
    seen = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            pid = int(proc.name)
            argv = command_line(pid)
            job = None
            if "--out-dir" in argv:
                job = by_root.get(argv[argv.index("--out-dir") + 1])
            elif "--worker" in argv:
                request_path = Path(argv[argv.index("--worker") + 1])
                if request_path.is_relative_to(state["queue_root"]):
                    job = by_root.get(read(request_path)["root"])
            if job is None:
                continue
            if job["root"] in seen and seen[job["root"]] != pid:
                raise ValueError("multiple live workers own one output")
            seen[job["root"]] = pid
            if job["status"] not in ("pending", "running"):
                raise ValueError("live worker disagrees with terminal state")
            if job["status"] == "pending":
                env = dict(
                    pair.split(b"=", 1)
                    for pair in proc.joinpath("environ").read_bytes().split(bytes([0]))
                    if b"=" in pair
                )
                uuid = env.get(b"CUDA_VISIBLE_DEVICES", b"").decode()
                gpu = next(
                    (
                        {"index": i, "uuid": d["uuid"]}
                        for i, d in gpu_devices.items()
                        if d["uuid"] == uuid
                    ),
                    {},
                )
                job.update(
                    status="running",
                    pid=pid,
                    gpu=gpu,
                    started_at=now(),
                    log=str(Path(state["queue_root"]) / (job["label"] + ".log")),
                    recovered_spawn_window=True,
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
    for job in jobs:
        if job["status"] == "pending" and (
            any(Path(job["root"]).rglob("manifest.json")) or any(Path(job["root"]).rglob("*.pt"))
        ):
            job["log"] = str(Path(state["queue_root"]) / (job["label"] + ".log"))
            success = successful_artifacts(job) and terminal_marker(job)
            job.update(
                status="completed" if success else "failed",
                ended_at=now(),
                exit_code=None,
                exit_verification="orphan_artifact_reconciliation",
            )
        if job["status"] == "running":
            identity = process_identity(job["pid"])
            if identity:
                if identity["state"] != "Z" and seen.get(job["root"]) != job["pid"]:
                    raise ValueError("saved worker PID is no longer the expected process")
                job["process_start_ticks"] = identity["start_ticks"]
            job["adopted"] = True
    return jobs


def takeover(queue, legacy_path, *, allow_orphans=False):
    """Terminate only a verified controller; never signal a training worker.

    tmux may immediately continue its stopped foreground pane, so SIGSTOP is
    not a reliable snapshot lock. Exit the controller, then reconcile ALL live
    owned workers and partial outputs before admitting any new fit.
    """
    state_path = queue / "queue_status.json"
    state = read(state_path)
    old_pid = state["scheduler"]["supervisor_pid"]
    identity = process_identity(old_pid)
    argv = command_line(old_pid)
    if identity is not None and identity["state"] != "Z":
        legacy_verified = "--run" in argv and str(legacy_path) in argv
        parallel_verified = (
            any(Path(arg).name == "run_survey_parallel_queue.py" for arg in argv)
            and "--queue" in argv
            and "--legacy-worker" in argv
            and Path(argv[argv.index("--queue") + 1]).resolve() == queue
            and Path(argv[argv.index("--legacy-worker") + 1]).resolve() == legacy_path
            and state["scheduler"].get("hosting") == "independent_tmux"
        )
        if allow_orphans or not (legacy_verified or parallel_verified):
            raise ValueError("refuse takeover: legacy controller identity is not verified")
        validate_fit_ownership(state["jobs"])
        # Only this exact PID receives a signal. The children were created
        # with start_new_session=True and survive their controller's exit.
        os.kill(old_pid, signal.SIGTERM)
        for _ in range(100):
            current = process_identity(old_pid)
            if current is None or current["state"] == "Z":
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("legacy controller did not exit; do not dispatch")
    elif not allow_orphans:
        raise ValueError("refuse takeover: legacy controller identity is not verified")
    state = read(state_path)
    state["queue_root"] = str(queue)
    reconcile_workers(state, devices())
    validate_fit_ownership(state["jobs"])
    jobs, parents = split_pending_jobs(state["jobs"], queue)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = queue / f"pre_parallel_handoff_{stamp}.json"
    if backup.exists():
        raise ValueError("handoff backup already exists")
    save(backup, state)
    save(
        queue / "parallel_split_plan.json",
        {
            "created_at": now(),
            "parents": parents,
            "old_controller": {"pid": old_pid, "identity": identity},
            "backup": str(backup),
            "preserved_fit_count": sum(j["expected_runs"] for j in jobs),
        },
    )
    return jobs


def spawn(job, gpu, queue, legacy_path):
    root = Path(job["root"])
    if root.exists() and any(root.iterdir()):
        raise ValueError("refusing to overwrite an existing fit output")
    logs = queue / "parallel_logs"
    logs.mkdir(exist_ok=True)
    request = logs / (job["label"] + ".request.json")
    save(request, job)
    log = logs / (job["label"] + ".log")
    env = dict(
        os.environ,
        PYTHONPATH=job["source"],
        CUDA_VISIBLE_DEVICES="" if gpu is None else gpu["uuid"],
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMPY_MADVISE_HUGEPAGE="0",
    )
    with log.open("x") as stream:
        worker_path = (
            Path(__file__).with_name("run_survey_queue_fit.py")
            if job.get("campaign_id")
            else legacy_path
        )
        process = subprocess.Popen(
            [sys.executable, "-u", str(worker_path), "--worker", str(request)],
            cwd=job["source"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    identity = process_identity(process.pid)
    job.update(
        status="running",
        pid=process.pid,
        gpu=gpu or {},
        log=str(log),
        started_at=now(),
        process_start_ticks=identity["start_ticks"],
        adopted=False,
    )
    return process


def poll_job(job, process=None):
    """None means still running; return (success, exit code, verification)."""
    if process is not None:
        code = process.poll()
        if code is None:
            return None
        return code == 0 and successful_artifacts(job), code, "waitpid_and_artifacts"
    identity = process_identity(job["pid"])
    if (
        identity
        and identity["start_ticks"] == job.get("process_start_ticks")
        and identity["state"] != "Z"
    ):
        return None
    if identity and identity["start_ticks"] == job.get("process_start_ticks"):
        code = os.waitstatus_to_exitcode(identity["wait_status"])
        return code == 0 and successful_artifacts(job), code, "proc_exit_status_and_artifacts"
    return successful_artifacts(job) and terminal_marker(job), None, "terminal_marker_and_artifacts"


def run(
    queue,
    legacy_path,
    *,
    recover=False,
    repair_plan=None,
    session_name="pu-survey-parallel-20261028",
    gpu_indices=None,
    packed_gpu_indices=(),
):
    if packed_gpu_indices and (
        gpu_indices is None or not set(packed_gpu_indices) <= set(gpu_indices)
    ):
        raise ValueError("packed GPU trial must be a subset of the explicit GPU allowlist")
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    log = (queue / "parallel_supervisor.log").open("a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    print(now(), "parallel service preflight", os.getpid(), flush=True)
    upgrade_lock = (queue / "controller_upgrade.lock").open("a")
    fcntl.flock(upgrade_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    legacy = load_legacy(legacy_path)
    if legacy.OUTPUT.resolve() != queue:
        raise ValueError("legacy worker output does not match queue root")
    for source in {j["source"] for j in read(queue / "queue_status.json")["jobs"]}:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=source).strip():
            raise ValueError("training checkout is dirty; refusing queue takeover")
    campaign = load_support_campaign(repair_plan) if repair_plan is not None else None
    if campaign:
        mount_campaign(read(queue / "queue_status.json")["jobs"], *campaign)
    # Policy code may evolve independently; training code and all frozen files
    # remain pinned to the original checkout recorded in every worker request.
    jobs = takeover(queue, legacy_path, allow_orphans=recover)
    handoff_lock = (queue / "parallel_controller.lock").open("a")
    fcntl.flock(handoff_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = (queue / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    helpers = legacy.helper()
    if campaign:
        metadata, additions = campaign
        jobs = mount_campaign(jobs, metadata, additions)
        campaign_root = Path(metadata["root"])
        campaign_root.mkdir(parents=True)
        save(campaign_root / "campaign.json", metadata)
    service_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    processes = {}
    idle_counts, low_counts = {}, {}
    packed_counts = {}
    last_tables = 0
    upgrade_released = False
    print(
        now(), "adopted workers", [j["pid"] for j in jobs if j["status"] == "running"], flush=True
    )
    while any(j["status"] in ("pending", "running") for j in jobs):
        for job in jobs:
            if job["status"] != "running":
                continue
            result = poll_job(job, processes.get(job["pid"]))
            if result is None:
                continue
            success, code, verification = result
            job.update(
                status="completed" if success else "failed",
                ended_at=now(),
                exit_code=code,
                exit_verification=verification,
            )
            if job.get("campaign_id"):
                job["gate_passed"] = success
            processes.pop(job["pid"], None)
            print(now(), "finished", job["label"], job["status"], verification, flush=True)
        release_canary_followups(jobs)
        active = [j for j in jobs if j["status"] == "running"]
        paused = {
            j.get("gpu", {}).get("index")
            for j in jobs
            if j["status"] == "failed" and j.get("dispatch_mode") == "packed_compute_trial"
        }
        packed_allowed = set(packed_gpu_indices) - paused
        query_error = None
        try:
            snapshot = devices()
            own_pids = {j["pid"] for j in active}
            for index, d in snapshot.items():
                idle_counts[index] = idle_counts.get(index, 0) + 1 if not d["pids"] else 0
                low = not d["pids"] - own_pids and d["util"] <= 35
                low_counts[index] = low_counts.get(index, 0) + 1 if low else 0
                packed_ok = bool(d["pids"]) and not d["pids"] - own_pids and d["util"] <= 95
                packed_counts[index] = packed_counts.get(index, 0) + 1 if packed_ok else 0
            slots = dispatch_slots(
                allowed_devices(snapshot, gpu_indices),
                active,
                idle_counts=idle_counts,
                low_counts=low_counts,
                packed_indices=packed_allowed,
                packed_counts=packed_counts,
            )
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            snapshot, slots, query_error = {}, [], str(exc)
            idle_counts.clear()
            low_counts.clear()
            packed_counts.clear()
        cpu_count = sum(j["lane"] == "cpu" for j in active)
        for index, mode in [(None, "cpu")] * max(0, 2 - cpu_count) + slots:
            pending = sorted(
                (
                    j
                    for j in jobs
                    if j["status"] == "pending" and j["lane"] == ("cpu" if index is None else "gpu")
                ),
                key=dispatch_priority,
            )
            if not pending:
                continue
            job = choose_pending(pending, [j for j in jobs if j["status"] == "running"], mode)
            if (
                mode in {"exclusive", "bounded_concurrency_trial"}
                and index in packed_allowed
                and snapshot[index]["free_mib"] >= 32768
            ):
                # Start the trial card with a measured native model rather
                # than endlessly cycling short adapters that cannot anchor it.
                job = choose_pending(pending, active, "packed_compute_trial") or job
            if job is None:
                continue
            # Recheck immediately before launch, not only at the beginning
            # of a potentially slow loop. A shared host still has a race.
            if index is not None:
                try:
                    fresh = devices()
                    current_active = [j for j in jobs if j["status"] == "running"]
                    if (index, mode) not in dispatch_slots(
                        allowed_devices(fresh, gpu_indices),
                        current_active,
                        idle_counts=idle_counts,
                        low_counts=low_counts,
                        packed_indices=packed_allowed,
                        packed_counts=packed_counts,
                    ):
                        continue
                except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                    continue
            gpu = None if index is None else {"index": index, "uuid": snapshot[index]["uuid"]}
            try:
                process = spawn(job, gpu, queue, legacy_path)
            except (OSError, ValueError) as exc:
                job.update(status="blocked", reason=f"launch_preflight: {exc}")
                print(now(), "launch blocked", job["label"], str(exc), flush=True)
                continue
            job["dispatch_mode"] = mode
            processes[process.pid] = process
            low_counts[index] = 0
            packed_counts[index] = 0
            print(
                now(), "started", job["label"], "gpu", index, mode, "pid", process.pid, flush=True
            )
        scheduler = {
            "supervisor_pid": os.getpid(),
            "service_code_commit": service_commit,
            "hosting": "independent_tmux",
            "tmux_session": session_name,
            "check_interval_seconds": 5,
            "granularity": "single_fit_except_adopted_batches",
            "max_jobs_per_gpu": 3 if packed_gpu_indices else 2,
            "max_gpu_training_workers_per_packed_card": 2,
            "packed_gpu_indices": list(packed_gpu_indices),
            "packed_gpu_indices_paused": sorted(i for i in paused if i is not None),
            "gpu_indices": list(gpu_indices) if gpu_indices is not None else None,
            "gpu_card_limit": len(gpu_indices) if gpu_indices is not None else None,
            "concurrency_policy": "bounded_overlap_optional_two_compute_trial_foreign_process_veto",
            "allocation_policy": {"NUMPY_MADVISE_HUGEPAGE": "0"},
            "last_query_error": query_error,
            "frozen_lock_environment_verified": False,
            "correction_campaigns": sorted(
                {j["campaign_id"] for j in jobs if j.get("campaign_id")}
            ),
        }
        save(
            queue / "queue_status.json", {"updated_at": now(), "jobs": jobs, "scheduler": scheduler}
        )
        if not upgrade_released:
            upgrade_lock.close()
            upgrade_released = True
        if time.monotonic() - last_tables > 120:
            helpers.tables(queue, [j for j in jobs if not j.get("campaign_id")])
            for root in {j["campaign_root"] for j in jobs if j.get("campaign_id")}:
                write_campaign_tables(
                    Path(root), [j for j in jobs if j.get("campaign_root") == root]
                )
            last_tables = time.monotonic()
        time.sleep(5)
    helpers.tables(queue, [j for j in jobs if not j.get("campaign_id")])
    for root in {j["campaign_root"] for j in jobs if j.get("campaign_id")}:
        write_campaign_tables(Path(root), [j for j in jobs if j.get("campaign_root") == root])
    print(now(), "released jobs terminal; blocked fits remain explicitly blocked", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--legacy-worker", type=Path, required=True)
    parser.add_argument("--session-name", default="pu-survey-parallel-20261028")
    parser.add_argument(
        "--packed-gpu-indices",
        type=parse_gpu_indices,
        default=(),
        help="opt-in subset for two GPU training workers and at most one CPU tail",
    )
    parser.add_argument(
        "--gpu-indices",
        type=parse_gpu_indices,
        help="physical GPU allowlist; busy allowed cards never spill to other GPUs",
    )
    parser.add_argument(
        "--add-support-repair",
        type=Path,
        help="mount a verified D53 plan with per-unit full-budget canary gates",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--takeover", action="store_true")
    mode.add_argument(
        "--recover", action="store_true", help="adopt orphans only if old controller exited"
    )
    args = parser.parse_args()
    run(
        args.queue.resolve(),
        args.legacy_worker.resolve(),
        recover=args.recover,
        repair_plan=args.add_support_repair,
        session_name=args.session_name,
        gpu_indices=args.gpu_indices,
        packed_gpu_indices=args.packed_gpu_indices,
    )


if __name__ == "__main__":
    main()
