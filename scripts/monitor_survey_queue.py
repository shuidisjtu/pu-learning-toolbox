"""Compact read-only terminal dashboard. Never launches or stops training."""

import argparse
import csv
import json
import math
import re
import select
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/data2/user/zihenglin/pu-learning-toolbox/results/survey-tasks-20261027")
NAMES = {
    "fashion_mnist": "F-MNIST",
    "twenty_newsgroups": "20News",
    "connect_4": "Connect4",
    "cifar10": "CIFAR10",
    "mnist": "MNIST",
    "adni": "ADNI-sub",
    "spambase": "Spambase",
    "imdb": "IMDB",
}
METHODS = {
    "self_pu_full_meta": "SelfPU-meta",
    "self_pu": "SelfPU-abl",
    "pusb_kernel": "PUSB-kernel",
    "pn_oracle": "PN-oracle",
    "robust_pu": "Robust-PU",
    "holistic_pu": "Holistic-PU",
    "split_pu": "Split-PU",
    "dist_pu": "Dist-PU",
}
PATHS = {"native_cnn": "CNN", "cnn_feature_adapter": "FEAT", "native_2d": "2D"}
MECHS = {"scar": "SCAR", "sar_lbe_a": "SAR-A", "sar_lbe_b": "SAR-B", None: "PN"}
EPOCHS = {
    "pn_oracle": 200,
    "nnpu": 200,
    "dist_pu": 200,
    "self_pu": 200,
    "self_pu_full_meta": 200,
    "pulda": 120,
    "robust_pu": 30,
    "split_pu": 40,
    "holistic_pu": 130,
    "lagam": 20,
    "pulns": 20,
    "genpu": 100,
}
VIEWS = ("running", "pending", "completed", "failed")


def read_states():
    states = []
    for directory in sorted(BASE.glob("gpu-batch-*")):
        path = directory / "queue_status.json"
        if path.exists():
            state = json.loads(path.read_text())
            state["directory"] = directory
            states.append(state)
    return states


def visible_jobs(states):
    jobs = [j for s in states for j in s["jobs"]]
    retries = {j["retry_of"] for j in jobs if j.get("retry_of")}
    fixed = {
        (j["method"], j.get("dataset", "cifar10"), j["path"])
        for j in jobs
        if j.get("preflight_evidence")
    }
    retries.update(
        j["root"]
        for j in jobs
        if j["status"] == "failed"
        and (j["method"], j.get("dataset", "cifar10"), j["path"]) in fixed
    )
    historical = [j for j in jobs if j["status"] == "failed" and j["root"] in retries]
    return [j for j in jobs if j not in historical], len(historical)


def gpu_status():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=4,
        check=True,
    )
    devices = {
        int(r[0]): dict(uuid=r[1].strip(), memory=float(r[2]), util=int(r[3]))
        for r in csv.reader(result.stdout.splitlines())
    }
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=4,
        check=True,
    )
    owners = {}
    for row in csv.reader(processes.stdout.splitlines()):
        if len(row) == 2 and row[1].strip().isdigit():
            owners.setdefault(row[0].strip(), set()).add(int(row[1]))
    for device in devices.values():
        device["pids"] = owners.get(device["uuid"], set())
    return devices


def progress(job):
    """Display only the active fit, never a retained checkpoint of a completed fit."""
    root = Path(job["root"])
    finished = set()
    for path in root.rglob("manifest.json"):
        try:
            m = json.loads(path.read_text())
            if m.get("selection") and not m.get("failures"):
                finished.add((m["seed"], m.get("c_requested_token")))
        except (OSError, ValueError, KeyError):
            pass
    facts = dict(
        done=f"{len(finished)}/{job.get('expected_runs', 15)}", seed="-", c="-", epoch="-", age="-"
    )
    if job["status"] != "running":
        return facts
    remaining = [k for k in job.get("expected_keys", []) if (k[5], k[4]) not in finished]
    if remaining:
        facts["seed"], facts["c"] = str(remaining[0][5]), str(remaining[0][4])
    latest = None
    for path in root.rglob("*.pt"):
        run_dir = next((p for p in path.parents if p.name.startswith("seed_")), None)
        if run_dir is None or (run_dir / "manifest.json").exists():
            continue
        try:
            modified = path.stat().st_mtime
            if latest is None or modified > latest[0]:
                latest = modified, path
        except OSError:
            continue
    if latest is None:
        facts["epoch"] = "GAN/prePN" if job["method"] == "genpu" else "load/prep"
        return facts
    modified, path = latest
    for part in path.parts:
        if part.startswith("seed_"):
            facts["seed"] = part[5:]
        if part.startswith("c_"):
            facts["c"] = part[2:]
    match = re.search(r"epoch_(\d+)_", path.name)
    if match:
        prefix = "PN" if job["method"] == "genpu" else ""
        facts["epoch"] = f"{prefix}{int(match[1])}/{EPOCHS.get(job['method'], '?')}"
    age = max(0, int(time.time() - modified))
    facts["age"] = f"{age}s" if age < 60 else f"{age // 60}m"
    return facts


def table(columns, rows):
    def line(values):
        return "  ".join(
            str(v)[:width].ljust(width) for (_, width), v in zip(columns, values, strict=False)
        ).rstrip()

    return [
        line([key for key, _ in columns]),
        line(["-" * n for _, n in columns]),
        *[line([row.get(key, "-") for key, _ in columns]) for row in rows],
    ]


def render(view="running", page=0, width=None, height=None, states=None, devices=None):
    terminal = shutil.get_terminal_size((120, 31))
    width = max(30, (width or terminal.columns) - 1)
    height = max(12, (height or terminal.lines) - 1)
    lines = [f"PU SURVEY | {datetime.now().astimezone():%m-%d %H:%M:%S} | 3s refresh"]
    try:
        states = read_states() if states is None else states
        jobs, historic = visible_jobs(states)
        current = max(states, key=lambda s: s["updated_at"])
        counts = {key: sum(j["status"] == key for j in current["jobs"]) for key in VIEWS}
        delay = max(
            0,
            int(
                (
                    datetime.now(timezone.utc) - datetime.fromisoformat(current["updated_at"])
                ).total_seconds()
            ),
        )
        lines.append(
            f"Latest queue {current.get('directory', Path('queue')).name}: "
            f"{counts['running']} RUN | {counts['pending']} WAIT | "
            f"{counts['completed']} DONE | {counts['failed']} FAIL | "
            + str(sum(j["status"] == "blocked" for j in current["jobs"]))
            + " BLOCK"
        )
        lines.append(
            f"Heartbeat {delay}s ago"
            + (" [STALE: inspect scheduler]" if delay > 60 else "")
            + f" | {historic} superseded failures hidden"
        )
    except (OSError, ValueError, KeyError) as error:
        return "\n".join(line[:width] for line in [*lines, str(error)])
    try:
        devices = gpu_status() if devices is None else devices
        gpu_error = None
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        devices = {}
        gpu_error = str(error)
    selected = [j for j in jobs if j["status"] == view]
    if view == "running":
        selected.sort(key=lambda j: (j.get("gpu", {}).get("index", 99), j["method"]))
    elif view == "completed":
        selected.sort(key=lambda j: j.get("ended_at", ""), reverse=True)
    capacity = max(1, height - 12)
    pages = max(1, math.ceil(len(selected) / capacity))
    page = max(0, page) % pages
    selected = selected[page * capacity : (page + 1) * capacity]
    lines.extend(
        [
            f"VIEW {view.upper()} | page {page + 1}/{pages} | "
            "1:run 2:wait 3:done 4:fail | n/p:page",
            "",
        ]
    )
    columns = [
        ("DEV", 4),
        ("DATA", 9),
        ("METHOD", 11),
        ("MECH", 5),
        ("DONE", 6),
        ("S", 1),
        ("c", 4),
        ("PHASE", 9),
    ]
    if width < 70:
        columns = [(key, size) for key, size in columns if key not in ("S", "c")]
    if width < 60:
        columns = [(key, size) for key, size in columns if key != "MECH"]
    if width < 50:
        columns = [(key, size) for key, size in columns if key != "PHASE"]
    if width >= 85:
        columns[3:3] = [("PATH", 4)]
    if width >= 95:
        columns.extend([("U%", 3), ("GiB", 5)])
    if width >= 110:
        columns.extend([("AGE", 5), ("PID", 7)])
    rows = []
    own_pids = {j.get("pid") for j in jobs if j["status"] == "running"}
    for job in selected:
        info = progress(job)
        index = job.get("gpu", {}).get("index")
        device = devices.get(index, {})
        shared = bool(device.get("pids", set()) - own_pids)
        dev = "CPU" if index is None else str(index) + ("*" if shared else "")
        rows.append(
            {
                "DEV": dev if view == "running" else "-",
                "DATA": NAMES.get(job.get("dataset", "cifar10"), job.get("dataset", "?")),
                "METHOD": METHODS.get(job["method"], job["method"]),
                "PATH": PATHS.get(job["path"], job["path"]),
                "MECH": MECHS.get(job.get("mechanism"), job.get("mechanism"))
                if not job["oracle"]
                else "PN",
                "DONE": info["done"],
                "S": info["seed"],
                "c": info["c"],
                "PHASE": info["epoch"],
                "U%": device.get("util", "-"),
                "GiB": f"{device['memory'] / 1024:.1f}" if device else "-",
                "AGE": info["age"],
                "PID": job.get("pid", "-") if view == "running" else "-",
            }
        )
    lines.extend(table(columns, rows))
    if not rows:
        lines.append("(none)")
    lines.append("")
    if devices:
        lines.append(
            "GPU load: "
            + " | ".join(f"{i}:{d['util']}%/{d['memory'] / 1024:.1f}G" for i, d in devices.items())
        )
    elif gpu_error:
        lines.append("GPU query unavailable; queue still shown.")
    if view == "running":
        pending = [j for j in jobs if j["status"] == "pending"]
        lines.append(
            "NEXT: "
            + "; ".join(
                f"{NAMES.get(j['dataset'], j['dataset'])}/"
                f"{METHODS.get(j['method'], j['method'])}/"
                f"{MECHS.get(j.get('mechanism'), 'PN')}"
                for j in pending[:2]
            )
        )
    lines.append("* = shared GPU; U%/GiB are whole-device, not this task alone.")
    lines.append("DONE=fits; S=seed; PHASE=checkpoint/inference; GAN/prePN != waiting.")
    own_counts = {}
    for job in jobs:
        if job["status"] == "running" and job.get("gpu"):
            i = job["gpu"]["index"]
            own_counts[i] = own_counts.get(i, 0) + 1
    shared_own = ",".join(f"GPU{i}={n}" for i, n in sorted(own_counts.items()) if n > 1)
    lines.append(
        ("Own concurrent workers: " + shared_own + " | " if shared_own else "")
        + "Ctrl+B,D detach; q stops monitor only."
    )
    return "\n".join(line[:width] for line in lines[:height])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--view", choices=VIEWS, default="running")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    view = args.view
    page = max(0, args.page - 1)
    terminal_state = None
    if not args.once and sys.stdin.isatty():
        import termios
        import tty

        terminal_state = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            print(
                ("" if args.once else "\033[H\033[2J")
                + render(view, page, args.width, args.height),
                flush=True,
            )
            if args.once:
                break
            if terminal_state is None:
                time.sleep(3)
                continue
            ready, _, _ = select.select([sys.stdin], [], [], 3)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key == "q":
                break
            if key in "1234":
                view = VIEWS[int(key) - 1]
                page = 0
            elif key == "n":
                page += 1
            elif key == "p":
                page = max(0, page - 1)
    except KeyboardInterrupt:
        pass
    finally:
        if terminal_state is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_state)


if __name__ == "__main__":
    main()
