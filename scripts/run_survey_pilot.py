"""Drive the whole pilot: work out what is left, then run it in batches.

``run_survey_experiment.py`` executes one bound unit and stops the moment a run
fails, which is the right shape for a single unit and the wrong one for a
pilot of several hundred runs: one bad run would abandon everything queued
behind it.  This driver sits above it and never decides what to run from a
directory layout -- it asks :mod:`pu_toolbox.experiment.pilot_plan` what the
protocol asks for, what is already on disk, and what is therefore left.

It does not invent protocol values.  The population class prior is a hard gate
for most methods, and protocol §3.1 defines it as data-generation metadata --
the positive rate of the complete pool before the stratified split, not
something to back-infer from a subset -- so it is read from the splits that
recorded it, and `--class-prior` only overrides.  A dataset with neither stops
the run before the first batch rather than at the hundredth.  The same goes for
the device: the unit script defaults to CPU, and a pilot that silently ran its
image rows on CPU would take far longer to discover than to prevent.

``--dry-run`` prints the plan and the checkpoint storage the whole pilot
implies, which is where the disk budget comes from::

    uv run python scripts/run_survey_pilot.py --dry-run
    uv run python scripts/run_survey_pilot.py --class-prior spambase=0.39,imdb=0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pu_toolbox.experiment.pilot_plan import (
    Batch,
    PilotRun,
    batch_command,
    batches,
    estimate_checkpoint_bytes,
    pending_runs,
    planned_runs,
    population_priors,
    split_digests,
)
from pu_toolbox.experiment.survey_protocol import load_protocol

_SCRIPTS = Path(__file__).resolve().parent
UNIT_SCRIPT = _SCRIPTS / "run_survey_experiment.py"

#: Feature dimensions the MLP backbones size against.  Image rows name a
#: ResNet and need none.  These are the prepared splits' widths: Spambase's 57
#: columns, SBERT's 384.
DEFAULT_INPUT_DIMS = {"spambase": 57, "imdb": 384}

_GIB = 1024**3


def _parse_pairs(raw: str, *, cast) -> dict:
    parsed: dict = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        if not name or not value:
            raise ValueError(f"expected name=value, got {item!r}")
        parsed[name.strip()] = cast(value)
    return parsed


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--splits", default="data/splits", help="prepared split products root")
    parser.add_argument("--results", default="results/survey", help="where run trees live")
    parser.add_argument("--protocol", default="survey-v1", help="protocol version or path")
    parser.add_argument(
        "--input-dims",
        default=",".join(f"{name}={value}" for name, value in DEFAULT_INPUT_DIMS.items()),
        help="feature dimensions for the MLP-backed rows",
    )
    parser.add_argument(
        "--class-prior",
        default="",
        help="population class prior per dataset, e.g. spambase=0.39,imdb=0.5",
    )
    parser.add_argument(
        "--allow-prior-override",
        action="store_true",
        help="accept a --class-prior that contradicts the prior a split recorded",
    )
    parser.add_argument("--device", default=None, help="passed through to the unit script")
    parser.add_argument("--adapter-cache", default=None, help="passed through to the unit script")
    parser.add_argument(
        "--extraction-batch-size", default=None, type=int, help="passed through to the unit script"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the plan and the disk it needs, then stop"
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue with later batches after one fails (default: stop)",
    )
    return parser.parse_args(argv)


def _unit_out_root(results_root: Path, batch: Batch) -> Path:
    return results_root / batch.dataset / batch.method / batch.training_path


def _counts(runs: tuple[PilotRun, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.dataset] = counts.get(run.dataset, 0) + 1
    return counts


def _summarise(runs: tuple[PilotRun, ...]) -> str:
    return (
        ", ".join(f"{count} {name}" for name, count in sorted(_counts(runs).items()) if count)
        or "none"
    )


def _print_disk(protocol: dict, runs: tuple[PilotRun, ...], dims: dict[str, int]) -> None:
    """Checkpoint storage for ``runs``.

    Estimated over the whole planned set, not only the pending part: nothing
    deletes checkpoints, so the runs already on disk are still occupying it and
    a figure that excluded them would understate what the host must hold.
    """
    try:
        estimate = estimate_checkpoint_bytes(protocol, input_dims=dims, runs=runs)
    except ValueError as exc:
        print(f"  disk:      cannot be estimated: {exc}", file=sys.stderr)
        return
    writing = sum(
        entry["runs"] for entry in estimate["per_unit_bytes"].values() if entry["bytes_per_run"]
    )
    print(
        f"  checkpoints: {estimate['total_bytes'] / _GIB:.1f} GiB for the whole pilot, "
        f"from {writing} of {estimate['runs']} run(s) that write any, "
        f"{estimate['candidates_per_run']} candidate(s) each"
    )
    print(
        f"    peak per run: {estimate['peak_bytes_per_run'] / _GIB:.2f} GiB "
        f"({estimate['peak_unit']})"
    )
    print(
        f"    the pre-run guard asks for {estimate['guard_peak_bytes_per_run'] / _GIB:.2f} GiB "
        f"free at the largest run start, reserving "
        f"{estimate['attempts_reserved']} attempt(s) per candidate"
    )
    for dataset, size in estimate["per_dataset_bytes"].items():
        print(f"    {dataset}: {size / _GIB:.1f} GiB")


def _print_plan(
    protocol: dict,
    pending: tuple[PilotRun, ...],
    done: tuple[PilotRun, ...],
    dims: dict[str, int],
) -> None:
    print(f"planned: {len(pending) + len(done)} run(s)")
    print(f"  completed: {len(done)} ({_summarise(done)})")
    print(f"  pending:   {len(pending)} ({_summarise(pending)})")
    print(f"  batches:   {len(batches(pending))} still to run")
    _print_disk(protocol, planned_runs(protocol), dims)


def _resolve_priors(
    recorded: dict[str, float], given: dict[str, float], *, allow_override: bool
) -> dict[str, float]:
    """What the splits declare, with the command line able to correct it.

    The prior is data-generation metadata the protocol fixes per dataset, so an
    override is not a tuning knob.  It is legitimate when a split never
    recorded one, and at least deliberate when it contradicts one -- so a
    contradiction stops the run unless ``--allow-prior-override`` says the
    disagreement is meant.  Warning and carrying on would let a mistyped
    ``imdb=0.05`` train thirty-five runs on the wrong prior, and the
    aggregation gates do not compare the prior, so nothing downstream would
    catch it either.
    """
    for dataset, value in sorted(given.items()):
        if dataset in recorded and recorded[dataset] != value:
            if not allow_override:
                raise ValueError(
                    f"--class-prior gives {dataset}={value}, but its splits record "
                    f"{recorded[dataset]}. The protocol defines one constant per dataset. "
                    "Pass --allow-prior-override if the disagreement is intended."
                )
            print(
                f"warning: overriding {dataset}'s recorded prior {recorded[dataset]} with {value}",
                file=sys.stderr,
            )
    return {**recorded, **given}


def _missing_priors(
    protocol: dict, runs: tuple[PilotRun, ...], priors: dict[str, float]
) -> list[str]:
    """Datasets whose methods need a population class prior that was not given.

    Checked across the whole planned set once, before anything runs: the unit
    script enforces this per run, so finding out batch by batch would mean
    paying for every run that happened to come first.
    """
    from pu_toolbox.registry import get_metadata, register_all_builtin_methods

    register_all_builtin_methods()
    missing = {
        run.dataset
        for run in runs
        if not run.is_oracle
        and get_metadata(run.method).requires_class_prior
        and run.dataset not in priors
    }
    return sorted(missing)


def _run_batch(batch: Batch, args: argparse.Namespace, priors: dict[str, float]) -> bool:
    argv = batch_command(
        batch,
        splits_root=args.splits,
        out_root=_unit_out_root(Path(args.results), batch),
        protocol=args.protocol,
        class_prior=None if batch.method == "pn_oracle" else priors.get(batch.dataset),
        device=args.device,
        adapter_cache=args.adapter_cache,
        extraction_batch_size=args.extraction_batch_size,
    )
    label = f"{batch.dataset}/{batch.method}/{batch.training_path} ({batch.mechanism or 'oracle'})"
    print(f"== {label}: {len(batch.covers)} run(s), seeds {list(batch.seeds)}")
    completed = subprocess.run(  # noqa: S603 - fixed script, arguments built here
        [sys.executable, str(UNIT_SCRIPT), *argv], check=False
    )
    if completed.returncode == 0:
        return True
    print(
        f"error: {label} exited {completed.returncode}; its remaining runs stay pending",
        file=sys.stderr,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        dims = _parse_pairs(args.input_dims, cast=int)
        given_priors = _parse_pairs(args.class_prior, cast=float)
        # Read once: a run only counts as done if it ran on the splits on disk
        # now, and the prior is read from those same splits.
        splits = split_digests(args.splits)
        priors = _resolve_priors(
            population_priors(args.splits),
            given_priors,
            allow_override=args.allow_prior_override,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    protocol = load_protocol()
    planned = planned_runs(protocol)

    if args.dry_run:
        pending, done = pending_runs(protocol, args.results, splits=splits)
        _print_plan(protocol, pending, done, dims)
        if priors:
            print(f"  class prior: {', '.join(f'{k}={v}' for k, v in sorted(priors.items()))}")
        missing = _missing_priors(protocol, planned, priors)
        if missing:
            print(f"  note: no class prior for {missing}; those runs cannot start")
        return 0

    missing = _missing_priors(protocol, planned, priors)
    if missing:
        print(
            f"error: methods in {missing} need the population class prior, and neither "
            "their splits nor --class-prior supply it. Prepare the splits with a "
            "generator that records it, or pass --class-prior dataset=value.",
            file=sys.stderr,
        )
        return 1

    pending, _ = pending_runs(protocol, args.results, splits=splits)
    failed = False
    for batch in batches(pending):
        if failed and not args.keep_going:
            break
        failed = not _run_batch(batch, args, priors) or failed

    # Reported from the records rather than from the loop's own bookkeeping: a
    # batch can succeed and still leave runs behind, and the operator needs the
    # count that the manifests agree with.
    still_pending, now_done = pending_runs(protocol, args.results, splits=splits)
    print(
        f"completed {len(now_done)} of {len(now_done) + len(still_pending)} run(s); "
        f"{len(still_pending)} still pending"
    )
    return 1 if still_pending else 0


if __name__ == "__main__":
    sys.exit(main())
