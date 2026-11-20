"""Local, restartable scheduler for the supplemental ResNet-18 paired study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from pu_toolbox.utils.serialization import file_hash, strict_canonical_hash

STUDY = "paired-resnet18-scar-c03-v1"
SOURCE = Path(__file__).resolve().parents[1]
METHODS = (
    "robust_pu",
    "split_pu",
    "holistic_pu",
    "self_pu",
    "self_pu_full_meta",
    "dist_pu",
)
DATASETS = ("cifar10", "fashion_mnist")
ARMS = ("frozen", "trainable")
METRICS = ("accuracy", "auc", "f1", "recall", "precision")
MAX_ATTEMPTS = 2
LAUNCH_GRACE_SECONDS = 45
RETRY_BACKOFF_SECONDS = 120
MIN_DISK_BYTES = 30 * 1024**3


@dataclass(frozen=True)
class Job:
    """A single fit; PA and OA selections do not multiply hardware jobs."""

    dataset: str
    method: str
    seed: int
    arm: str

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.method}/seed{self.seed}/{self.arm}"

    def identity(self) -> dict:
        return dict(dataset=self.dataset, method=self.method, seed=self.seed, arm=self.arm)


@dataclass(frozen=True)
class GPU:
    """Read-only host snapshot keyed by physical GPU UUID."""

    index: int
    uuid: str
    model: str
    used_mib: int
    utilization: int
    compute_pids: tuple[int, ...]

    @property
    def idle(self) -> bool:
        return self.used_mib < 500 and self.utilization <= 10 and not self.compute_pids


def jobs() -> tuple[Job, ...]:
    """Prioritize seed-zero pairs before seeded siblings in the same dataset."""
    return tuple(
        Job(dataset, method, seed, arm)
        for dataset in DATASETS
        for seed in range(5)
        for method in METHODS
        for arm in ARMS
    )


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    pending = path.with_suffix(".tmp")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    pending = path.with_suffix(".tmp")
    with pending.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


@contextmanager
def _lock(path: Path) -> Iterator[BinaryIO]:
    # The host launcher is Linux-only; plan and report logic stay portable.
    import fcntl

    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def _locked(path: Path) -> bool:
    try:
        with _lock(path):
            return False
    except BlockingIOError:
        return True


def _output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=SOURCE, text=True, timeout=10).strip()


def source_commit() -> str:
    """Reject dirty or changed code before dispatching any governed fit."""
    if _output(["git", "status", "--porcelain"]):
        raise ValueError("campaign requires a clean, committed source checkout")
    return _output(["git", "rev-parse", "HEAD"])


def gpu_snapshot() -> list[GPU]:
    """Fail closed on query/parse errors, rather than infer GPU availability."""
    command = ["nvidia-smi", "--format=csv,noheader,nounits"]
    rows = _output(
        [
            *command,
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        ]
    )
    processes = _output([*command, "--query-compute-apps=gpu_uuid,pid"])
    owners: dict[str, list[int]] = {}
    for uuid, pid in csv.reader(
        line for line in processes.splitlines() if line and line != "No running processes found"
    ):
        owners.setdefault(uuid.strip(), []).append(int(pid))
    devices = [
        GPU(
            int(index),
            uuid.strip(),
            model.strip(),
            int(used),
            int(util),
            tuple(owners.pop(uuid.strip(), ())),
        )
        for index, uuid, model, used, util in csv.reader(rows.splitlines())
    ]
    if owners or not devices or len({gpu.uuid for gpu in devices}) != len(devices):
        raise ValueError("inconsistent physical GPU/process snapshot")
    return devices


def _digest(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(byte in "0123456789abcdef" for byte in value)
    )


def _numeric(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def verify_receipt(job: Job, out: Path, commit: str, gpu_model: str) -> dict:
    """Bind one full-budget receipt to its manifest, scores and local checkpoints."""
    receipt = _read(out / "accepted.json")
    expected = {
        **job.identity(),
        "schema_version": 1,
        "study": STUDY,
        "mode": "full",
        "source_commit": commit,
    }
    if type(receipt.get("seed")) is not int or any(
        receipt.get(k) != value for k, value in expected.items()
    ):
        raise ValueError("wrong experiment identity or source commit")
    if not _digest(receipt.get("pair_identity")):
        raise ValueError("missing paired scientific identity")
    filename = out / "manifest.json"
    if file_hash(filename) != receipt.get("manifest_sha256"):
        raise ValueError("manifest digest mismatch")
    manifest = _read(filename)
    if manifest.get("failures") != [] or manifest.get("seed") != job.seed:
        raise ValueError("manifest has a different seed or failed candidates")
    pair = manifest["paired_study"]
    if (
        strict_canonical_hash(pair) != receipt["pair_identity"]
        or any(
            pair.get(k) != expected[k]
            for k in ("study", "dataset", "method", "seed", "mode", "source_commit")
        )
        or pair.get("c") != 0.3
        or pair.get("sampling") != "os"
        or manifest.get("paired_arm") != job.arm
        or manifest.get("checkpoint_replay_verified") is not True
    ):
        raise ValueError("manifest has wrong paired design or unverified checkpoint replay")
    env = manifest["environment"]
    devices = env["gpu_devices"]
    if (
        env != receipt.get("environment")
        or strict_canonical_hash(env) != receipt.get("environment_identity")
        or env.get("cuda_available") is not True
        or len(devices) != 1
        or devices[0]["name"] != gpu_model
    ):
        raise ValueError("manifest has wrong or unverified CUDA environment")
    pa = "PA+clean-support" if job.method == "self_pu_full_meta" else "PA"
    metrics = receipt["metrics"]
    if set(metrics) != {pa, "OA"} or set(manifest["selection"]) != set(metrics):
        raise ValueError("missing or mixed PA/OA selections")
    for protocol, values in metrics.items():
        for metric in METRICS:
            value = values[metric]
            if value != manifest["test_results"][protocol].get(metric):
                raise ValueError(f"receipt does not match manifest: {protocol}/{metric}")
            if value is None and metric != "accuracy":
                continue
            if not _numeric(value) or not 0 <= value <= 1:
                raise ValueError(f"non-finite or out-of-range metric: {protocol}/{metric}")
    if not _numeric(receipt["elapsed_seconds"]) or receipt["elapsed_seconds"] <= 0:
        raise ValueError("elapsed_seconds must be a positive finite number")
    for field in ("peak_gpu_bytes", "peak_rss_bytes"):
        if type(receipt[field]) is not int or receipt[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    checkpoints = receipt.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("selected checkpoint receipts are required")
    paths = {}
    for checkpoint in checkpoints:
        path = Path(checkpoint["path"]).resolve()
        if not path.is_relative_to(out.resolve()) or path in paths:
            raise ValueError("checkpoint missing from private run directory or repeated")
        if not _digest(checkpoint["sha256"]) or file_hash(path) != checkpoint["sha256"]:
            raise ValueError(f"selected checkpoint digest mismatch: {path}")
        paths[path] = checkpoint["sha256"]
    expected_checkpoints = {}
    for selection in manifest["selection"].values():
        checkpoint = selection["checkpoint"]
        if checkpoint.get("reclaimed") or checkpoint.get("persistent") is not True:
            raise ValueError("selected checkpoint was not retained")
        expected_checkpoints[Path(checkpoint["path"]).resolve()] = checkpoint["sha256"]
    if expected_checkpoints != paths:
        raise ValueError("checkpoint receipts do not cover distinct PA/OA selected states")
    return receipt


def _accepted(state: dict, dataset: str, method: str, seed: int) -> tuple[dict, dict] | None:
    pair = tuple(state[Job(dataset, method, seed, arm).key] for arm in ARMS)
    if not all(row["status"] == "completed" for row in pair):
        return None
    frozen, trainable = (row["accepted"] for row in pair)
    if frozen["pair_identity"] != trainable["pair_identity"]:
        return None
    if frozen.get("environment_identity") != trainable.get("environment_identity"):
        return None
    return frozen, trainable


def family_failure(state: dict, dataset: str, method: str) -> str | None:
    """Only failures of the same method/dataset (or its CIFAR wave) hold siblings."""
    relevant = [
        (job, state[job.key])
        for job in jobs()
        if job.method == method and (job.dataset == dataset or job.dataset == "cifar10")
    ]
    if failed := next((job.key for job, row in relevant if row["status"] == "failed"), None):
        return f"failed fit: {failed}"
    for phase in DATASETS if dataset == "fashion_mnist" else ("cifar10",):
        for seed in range(5):
            rows = tuple(state[Job(phase, method, seed, arm).key] for arm in ARMS)
            if (
                all(row["status"] == "completed" for row in rows)
                and _accepted(state, phase, method, seed) is None
            ):
                return f"unmatched paired receipt: {phase}/{method}/seed{seed}"
    return None


def waiting_reason(job: Job, state: dict, enabled: set[str], now: float) -> str | None:
    """Determine dispatch eligibility without consulting held-out scores."""
    row = state[job.key]
    if job.method not in enabled:
        return "method not enabled by the operator"
    if reason := family_failure(state, job.dataset, job.method):
        return reason
    if job.dataset == "fashion_mnist":
        if not all(_accepted(state, "cifar10", job.method, seed) for seed in range(5)):
            return "CIFAR five paired seeds not accepted"
        if not all(
            family_failure(state, "cifar10", method)
            or all(_accepted(state, "cifar10", method, seed) for seed in range(5))
            for method in enabled
        ):
            return "waiting for enabled CIFAR families to complete or be isolated"
    if job.seed and _accepted(state, job.dataset, job.method, 0) is None:
        return "seed-zero paired canary not accepted"
    if now < row.get("not_before", 0):
        return "same-seed retry cooldown"
    return None


def _workspace_args(job: Job, campaign: dict, output: Path) -> list[str]:
    return [
        campaign["python"],
        "-m",
        "scripts.run_survey_paired_resnet",
        "--data-root",
        campaign["data_root"],
        "--dataset",
        job.dataset,
        "--method",
        job.method,
        "--seed",
        str(job.seed),
        "--arm",
        job.arm,
        "--device",
        "cuda",
        "--output",
        str(output),
    ]


def run_attempt(spec_path: Path) -> int:
    """Own one worker until an atomic exit receipt can be persisted."""
    spec = _read(spec_path)
    root = spec_path.parent
    with _lock(root / "worker.lock") as lease:
        if (root / "exit.json").exists():
            return 1
        status = {"exit_code": 1}
        try:
            if source_commit() != spec["source_commit"]:
                raise ValueError("source changed after dispatch")
            active = _read(Path(spec["queue_state"]))[spec["job_key"]]
            if active["status"] != "running" or active["attempts"][-1]["root"] != str(root):
                raise ValueError("attempt no longer owns the fit")
            device = next((g for g in gpu_snapshot() if g.uuid == spec["gpu_uuid"]), None)
            if device is None or device.model != spec["gpu_model"] or not device.idle:
                # No fit was started: scheduling contention does not consume a retry.
                status["deferred"] = True
                raise ValueError("GPU is no longer demonstrably idle")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": device.uuid,
                "PYTHONPATH": str(SOURCE),
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            }
            # A child surviving its wrapper still owns the lease. A restarted
            # scheduler must not start a duplicate on another idle card.
            process = subprocess.run(spec["argv"], cwd=SOURCE, env=env, pass_fds=(lease.fileno(),))
            status["exit_code"] = process.returncode
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            status["error"] = str(exc)
        _atomic_json(root / "exit.json", status)
        return status["exit_code"]


def _reconcile(job: Job, row: dict, campaign: dict, now: float) -> None:
    attempt = row["attempts"][-1]
    root = Path(attempt["root"])
    path = root / "exit.json"
    if root.is_dir() and _locked(root / "worker.lock"):
        return
    if not path.exists() and now - attempt["launched_at"] < LAUNCH_GRACE_SECONDS:
        return
    result = (
        _read(path)
        if path.exists()
        else {
            "exit_code": 1,
            "error": "worker disappeared without an exit receipt",
        }
    )
    attempt.update(result)
    row.pop("not_before", None)
    if result.get("deferred"):
        row.update(status="pending", not_before=now + 5)
    elif result.get("exit_code") == 0:
        try:
            receipt = verify_receipt(
                job, root / "result", campaign["source_commit"], campaign["gpu_model"]
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            row.update(status="failed", failure=f"artifact verification: {exc}")
        else:
            row.update(status="completed", accepted=receipt)
    elif sum(not item.get("deferred", False) for item in row["attempts"]) < MAX_ATTEMPTS:
        row.update(status="pending", not_before=now + RETRY_BACKOFF_SECONDS)
    else:
        row.update(status="failed", failure=result.get("error", "worker failed twice"))


def _launch(job: Job, row: dict, gpu: GPU, campaign: dict, state_path: Path, state: dict) -> None:
    root = state_path.parent / "attempts" / job.key / f"{len(row['attempts']) + 1:03d}"
    launch = {
        "argv": _workspace_args(job, campaign, root / "result"),
        "gpu_uuid": gpu.uuid,
        "gpu_model": gpu.model,
        "source_commit": campaign["source_commit"],
        "queue_state": str(state_path),
        "job_key": job.key,
    }
    row["attempts"].append({"root": str(root), "gpu": gpu.uuid, "launched_at": time.time()})
    row["status"] = "running"
    # The reserved attempt is durable before launch; a crash on either side
    # cannot reuse its directory or produce two writers for that fit.
    _atomic_json(state_path, state)
    root.mkdir(parents=True, exist_ok=False)
    spec_path = root / "launch.json"
    _atomic_json(spec_path, launch)
    try:
        with (root / "console.log").open("xb") as log:
            subprocess.Popen(
                [
                    campaign["python"],
                    str(Path(__file__).resolve()),
                    "--attempt-spec",
                    str(spec_path),
                ],
                cwd=SOURCE,
                env={**os.environ, "PYTHONPATH": str(SOURCE)},
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        _atomic_json(root / "exit.json", {"exit_code": 1, "error": str(exc)})


def write_summary(root: Path, state: dict) -> dict:
    """Compute paired effects from exact accepted pairs, not standalone best seeds."""
    raw_fields = [
        "dataset",
        "method",
        "seed",
        "selection",
        "metric",
        "frozen",
        "trainable",
        "delta",
    ]
    combined_fields = [
        "dataset",
        "method",
        "selection",
        "metric",
        "n",
        "frozen_mean",
        "frozen_sd",
        "trainable_mean",
        "trainable_sd",
        "delta_mean",
        "delta_sd",
        "wins",
        "ties",
        "losses",
    ]
    raw, combined, resources = [], [], []
    n_pairs = 0
    for dataset in DATASETS:
        for method in METHODS:
            pairs = [
                (seed, pair)
                for seed in range(5)
                if (pair := _accepted(state, dataset, method, seed)) is not None
            ]
            n_pairs += len(pairs)
            for seed, pair in pairs:
                for arm, receipt in zip(ARMS, pair, strict=True):
                    resources.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "arm": arm,
                            **{
                                key: receipt[key]
                                for key in (
                                    "elapsed_seconds",
                                    "peak_gpu_bytes",
                                    "peak_rss_bytes",
                                )
                            },
                            "peak_checkpoint_payload_bytes": receipt.get(
                                "peak_checkpoint_payload_bytes"
                            ),
                        }
                    )
            if not pairs:
                continue
            for selection in pairs[0][1][0]["metrics"]:
                for metric in METRICS:
                    values = []
                    for seed, (frozen, trainable) in pairs:
                        f = frozen["metrics"][selection][metric]
                        t = trainable["metrics"][selection][metric]
                        delta = t - f if t is not None and f is not None else None
                        raw.append(
                            dict(
                                dataset=dataset,
                                method=method,
                                seed=seed,
                                selection=selection,
                                metric=metric,
                                frozen=f,
                                trainable=t,
                                delta=delta,
                            )
                        )
                        if delta is not None:
                            values.append((f, t, delta))
                    environments = {pair[0].get("environment_identity") for _, pair in pairs}
                    if len(values) == 5 and len(environments) == 1:
                        f, t, d = zip(*values, strict=True)
                        combined.append(
                            dict(
                                dataset=dataset,
                                method=method,
                                selection=selection,
                                metric=metric,
                                n=5,
                                frozen_mean=statistics.mean(f),
                                frozen_sd=statistics.stdev(f),
                                trainable_mean=statistics.mean(t),
                                trainable_sd=statistics.stdev(t),
                                delta_mean=statistics.mean(d),
                                delta_sd=statistics.stdev(d),
                                wins=sum(v > 0 for v in d),
                                ties=sum(v == 0 for v in d),
                                losses=sum(v < 0 for v in d),
                            )
                        )
    _write_csv(root / "paired_per_seed.csv", raw, raw_fields)
    _write_csv(root / "paired_five_seed.csv", combined, combined_fields)
    _write_csv(
        root / "paired_resources.csv",
        resources,
        [
            "dataset",
            "method",
            "seed",
            "arm",
            "elapsed_seconds",
            "peak_gpu_bytes",
            "peak_rss_bytes",
            "peak_checkpoint_payload_bytes",
        ],
    )
    status = dict(Counter(row["status"] for row in state.values()))
    summary = {
        "study": STUDY,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_count": len(state),
        "paired_seed_count": n_pairs,
        "complete": n_pairs * 2 == len(state),
        "counts": status,
        "formal_admission": False,
        "active": {
            key: row["attempts"][-1] for key, row in state.items() if row["status"] == "running"
        },
        "held_or_failed": {
            key: row.get("failure", row.get("reason"))
            for key, row in state.items()
            if row["status"] in {"failed", "held"}
        },
        "five_seed_metric_rows": len(combined),
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def _read_enabled(args: argparse.Namespace, commit: str) -> set[str]:
    enabled = set(args.methods.split(",")) if args.methods else set()
    if args.ready_file is not None and args.ready_file.exists():
        gate = _read(args.ready_file)
        if gate.get("source_commit") != commit or not isinstance(gate.get("methods"), list):
            raise ValueError("method readiness file does not match the committed campaign source")
        enabled.update(gate["methods"])
    if enabled - set(METHODS):
        raise ValueError(f"unknown enabled methods: {enabled - set(METHODS)}")
    return enabled


def campaign_loop(args: argparse.Namespace, campaign: dict) -> int:
    """Advance only eligible fits while giving already-running jobs their lifetime."""
    root = args.root
    state_path = root / "queue_state.json"
    immutable = root / "queue_manifest.json"
    if immutable.exists() and _read(immutable) != campaign:
        raise ValueError("campaign settings/source changed; use a separate result root")
    if not immutable.exists():
        _atomic_json(immutable, campaign)
    state = (
        _read(state_path)
        if state_path.exists()
        else {job.key: {"status": "pending", "attempts": []} for job in jobs()}
    )
    if set(state) != {job.key for job in jobs()}:
        raise ValueError("state does not contain the campaign's full fit grid")
    changed = True
    while True:
        now = time.time()
        for job in jobs():
            row = state[job.key]
            if row["status"] == "running":
                _reconcile(job, row, campaign, now)
                changed = True
        try:
            enabled = _read_enabled(args, campaign["source_commit"])
        except (OSError, ValueError, TypeError) as exc:
            print(f"method gate refused dispatch: {exc}", flush=True)
            enabled = set()
        pending = []
        for job in jobs():
            row = state[job.key]
            if row["status"] in {"pending", "held"}:
                reason = waiting_reason(job, state, enabled, now)
                fundamental_hold = (
                    job.method not in enabled
                    or family_failure(state, job.dataset, job.method) is not None
                )
                if reason != row.get("reason") or (row["status"] == "held") != fundamental_hold:
                    changed = True
                row.update(status="held" if fundamental_hold else "pending", reason=reason)
                if reason is None:
                    pending.append(job)
        running = [row["attempts"][-1] for row in state.values() if row["status"] == "running"]
        owned = {attempt["gpu"] for attempt in running}
        free_slots = args.max_parallel - len(running)
        if pending and free_slots > 0 and not (root / "PAUSE").exists():
            try:
                if source_commit() != campaign["source_commit"]:
                    raise ValueError("source commit changed")
                free_disk = shutil.disk_usage(root).free
                if free_disk < MIN_DISK_BYTES * (len(running) + 1):
                    raise ValueError(f"insufficient disk reserve: {free_disk} bytes")
                devices = [
                    gpu
                    for gpu in gpu_snapshot()
                    if (
                        gpu.index in args.gpus
                        and gpu.model == args.gpu_model
                        and gpu.idle
                        and gpu.uuid not in owned
                    )
                ][: min(free_slots, free_disk // MIN_DISK_BYTES - len(running))]
                for job, gpu in zip(pending, devices, strict=False):
                    _launch(job, state[job.key], gpu, campaign, state_path, state)
                    changed = True
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                print(f"dispatch held: {exc}", flush=True)
        if changed:
            _atomic_json(state_path, state)
            report = write_summary(root, state)
            print(json.dumps(report["counts"], sort_keys=True), flush=True)
            changed = False
        active = any(row["status"] in {"pending", "running"} for row in state.values())
        if not active:
            return 0 if write_summary(root, state)["complete"] else 2
        time.sleep(args.poll_seconds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--gpus", default="1,2", help="approved physical GPU indices")
    parser.add_argument("--gpu-model", default="NVIDIA RTX A6000")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--methods", default="", help="comma-separated approved methods")
    parser.add_argument("--ready-file", type=Path, help="JSON with source_commit and methods")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--attempt-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.attempt_spec is not None:
        return run_attempt(args.attempt_spec)
    if args.root is None or args.data_root is None or not args.data_root.is_dir():
        parser.error("--root and an existing --data-root are required")
    try:
        args.gpus = tuple(dict.fromkeys(int(gpu) for gpu in args.gpus.split(",")))
    except ValueError:
        parser.error("--gpus must contain comma-separated physical indices")
    if any(gpu < 0 for gpu in args.gpus):
        parser.error("--gpus cannot contain negative physical indices")
    if not 1 <= args.max_parallel <= min(4, len(args.gpus)) or args.poll_seconds < 2:
        parser.error("max-parallel must fit the approved card pool (cap 4); poll at least 2 s")
    if args.dry_run:
        print(json.dumps([job.identity() for job in jobs()], indent=2))
        return 0
    if sys.platform != "linux":
        parser.error("live campaign scheduling requires a Linux GPU host")
    if not (args.methods or args.ready_file):
        parser.error("enable smoke-validated methods explicitly with --methods or --ready-file")
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    campaign = {
        "study": STUDY,
        "source_path": str(SOURCE),
        "source_commit": source_commit(),
        "python": sys.executable,
        "data_root": str(args.data_root.resolve()),
        "gpu_pool": list(args.gpus),
        "gpu_model": args.gpu_model,
        "max_parallel": args.max_parallel,
        "jobs": [job.identity() for job in jobs()],
    }
    with _lock(args.root / "controller.lock"):
        return campaign_loop(args, campaign)


if __name__ == "__main__":
    raise SystemExit(main())
