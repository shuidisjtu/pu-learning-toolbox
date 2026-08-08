"""Run PUSB Table 2 shards with bounded CPU parallelism and automatic aggregation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmarks.assigned_methods.pusb_table2_aggregate import aggregate_shards

THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _completed(shard_dir: Path, shard_count: int, shard_index: int) -> bool:
    manifest_path = shard_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("status") == "completed" and manifest.get("execution_scope") == {
        "shard_count": shard_count,
        "shard_index": shard_index,
    }


def _shard_command(
    *,
    config: Path,
    data_root: Path,
    shard_dir: Path,
    shard_count: int,
    shard_index: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.assigned_methods.pusb_table2_benchmark",
        "--config",
        str(config),
        "--data-root",
        str(data_root),
        "--output",
        str(shard_dir),
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
    ]
    if (shard_dir / "resolved_config.json").is_file():
        command.append("--resume")
    return command


def _run_shard(
    *,
    config: Path,
    data_root: Path,
    shard_root: Path,
    shard_count: int,
    shard_index: int,
) -> tuple[int, int]:
    shard_dir = shard_root / f"shard-{shard_index:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if _completed(shard_dir, shard_count, shard_index):
        return shard_index, 0
    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_LIMIT_VARIABLES})
    command = _shard_command(
        config=config,
        data_root=data_root,
        shard_dir=shard_dir,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    with (shard_dir / "run.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return shard_index, result.returncode


def run_parallel(
    *,
    config: str | Path,
    data_root: str | Path,
    shard_root: str | Path,
    plan_path: str | Path,
    aggregate_output: str | Path,
    shard_count: int,
    workers: int,
    retries: int,
) -> None:
    """Run or resume every shard and aggregate only after exact completion."""
    if shard_count <= 0 or workers <= 0 or retries < 0:
        raise ValueError("shard_count/workers must be positive and retries non-negative")
    config = Path(config).resolve()
    data_root = Path(data_root).resolve()
    shard_root = Path(shard_root).resolve()
    pending = set(range(shard_count))
    for attempt in range(retries + 1):
        failures = set()
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {
                executor.submit(
                    _run_shard,
                    config=config,
                    data_root=data_root,
                    shard_root=shard_root,
                    shard_count=shard_count,
                    shard_index=index,
                ): index
                for index in pending
            }
            for future in as_completed(futures):
                index, returncode = future.result()
                if returncode:
                    failures.add(index)
        if not failures:
            break
        pending = failures
        if attempt == retries:
            failed = ", ".join(str(index) for index in sorted(failures))
            raise RuntimeError(f"PUSB Table 2 shards failed after retries: {failed}")
    aggregate_shards(
        shard_root,
        plan_path=plan_path,
        output_dir=aggregate_output,
        shard_count=shard_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=45)
    parser.add_argument("--workers", type=int, default=45)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    run_parallel(
        config=args.config,
        data_root=args.data_root,
        shard_root=args.shard_root,
        plan_path=args.plan,
        aggregate_output=args.aggregate_output,
        shard_count=args.shard_count,
        workers=args.workers,
        retries=args.retries,
    )
    print(f"Completed and aggregated {args.shard_count} PUSB Table 2 shards")


if __name__ == "__main__":
    main()
