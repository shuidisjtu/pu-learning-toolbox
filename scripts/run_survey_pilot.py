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

``--datasets`` narrows the matrix to the datasets it names, which is how one
pilot is split across hosts: each host is handed its own list.  Nothing
outside enforces that two lists partition the matrix instead of overlapping,
so ``--plan-json`` writes the expanded plan out and the files can be checked
against each other.  The restriction is applied to the matrix before anything
reads it, so the plan, the storage a host is sized against, the resume check
and the batches all describe the same shard::

    uv run python scripts/run_survey_pilot.py --dry-run --datasets imdb,spambase \\
        --plan-json shard-b.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _parse_datasets(raw: str | None) -> tuple[str, ...] | None:
    """The datasets a request names, or ``None`` when it names none at all.

    ``None`` means the whole matrix, which is what the flag's absence asks for.
    A flag that is *present* but names nothing is a different request and is
    refused by the caller: splitting a pilot across two hosts means each host
    is given a list, and a list that quietly planned nothing would have both
    hosts report success over the same hole.
    """
    if raw is None:
        return None
    names = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not names:
        raise ValueError(
            "--datasets names no dataset; pass a comma-separated list such as "
            "'cifar10' or 'imdb,spambase'."
        )
    return names


def _dataset_names(protocol: dict[str, Any]) -> tuple[str, ...]:
    """Every dataset the matrix plans runs for.

    Read off the same ``runnable`` flag :func:`planned_runs` honours: a dataset
    whose units are all excluded cannot be requested into existence, and naming
    it has to say so rather than plan an empty shard.
    """
    return tuple(
        sorted(
            {
                unit["dataset"]
                for unit in protocol["execution_units"]
                if unit.get("runnable") is True
            }
        )
    )


def _select_datasets(protocol: dict[str, Any], requested: tuple[str, ...] | None) -> dict[str, Any]:
    """The matrix reduced to ``requested``, or unchanged when that is ``None``.

    The restriction is applied to the protocol rather than to the plan so that
    every reader narrows together: the plan itself, the view resolution, the
    checkpoint estimate a host is sized against, the resume check and the
    batches.  A filter that reached only the report would print one shard and
    then train the whole matrix.
    """
    if requested is None:
        return protocol
    known = _dataset_names(protocol)
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; the matrix plans {list(known)}.")
    wanted = set(requested)
    return {
        **protocol,
        "execution_units": [
            unit for unit in protocol["execution_units"] if unit["dataset"] in wanted
        ],
    }


def _planned_datasets(planned: tuple[PilotRun, ...]) -> list[str]:
    """The datasets a plan covers, sorted: what a scoped report has to name."""
    return sorted({run.dataset for run in planned})


def _write_plan_snapshot(
    path: str,
    *,
    protocol: dict[str, Any],
    planned: tuple[PilotRun, ...],
    pending: tuple[PilotRun, ...],
    completed: tuple[PilotRun, ...],
    views: Mapping[tuple[Any, ...], str],
) -> None:
    """Write what this request covers, in a form two hosts can be checked against.

    The runs recorded are the *planned* set rather than the pending one: two
    hosts partition the matrix, and what each is accountable for does not shrink
    as it finishes.  The view travels with each run because the resume check
    holds a run to the view it is going to use, so a shard is only reproducible
    together with the ledger state that resolved it.
    """
    payload = {
        "protocol_version": protocol["protocol_version"],
        "datasets": _planned_datasets(planned),
        "totals": {
            "planned": len(planned),
            "completed": len(completed),
            "pending": len(pending),
        },
        "runs": [
            {
                "dataset": run.dataset,
                "method": run.method,
                "training_path": run.training_path,
                "mechanism": run.mechanism,
                "c_token": run.c_token,
                "seed": run.seed,
                "view": views[run.key],
            }
            for run in planned
        ],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--splits", default="data/splits", help="prepared split products root")
    parser.add_argument("--results", default="results/survey", help="where run trees live")
    parser.add_argument("--protocol", default="survey-v1", help="protocol version or path")
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "comma-separated datasets to run (default: every dataset in the matrix). "
            "Splitting a pilot across hosts gives each host its own list; an unknown "
            "name is refused rather than planned empty, because an empty plan exits 0 "
            "and the other host would be believed to have covered it"
        ),
    )
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
    parser.add_argument(
        "--os-or-ts",
        choices=("os", "ts"),
        default=None,
        help=(
            "passed through to the unit script (default: each method's ledger view). "
            "'ts' applies the TS-OS calibration and is refused -- before any batch "
            "starts -- for a method that cannot carry it, the oracle included. The "
            "resume check holds every run to its own view whether that value was "
            "explicit or derived, so results recorded under the other view stay "
            "pending. This is a whole-matrix request: a plan containing any OS-native "
            "method or the oracle cannot be run under an explicit 'ts'"
        ),
    )
    parser.add_argument("--adapter-cache", default=None, help="passed through to the unit script")
    parser.add_argument(
        "--extraction-batch-size", default=None, type=int, help="passed through to the unit script"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the plan and the disk it needs, then stop"
    )
    parser.add_argument(
        "--plan-json",
        default=None,
        help=(
            "path to write the expanded plan to (requires --dry-run): every run this "
            "request covers, so two hosts' shards can be checked for overlap instead "
            "of being asserted disjoint"
        ),
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


def _print_disk(estimate: dict) -> None:
    """Checkpoint storage the whole planned set implies.

    The estimate is taken over everything planned, not only the pending part:
    nothing deletes checkpoints, so the runs already on disk still occupy it and
    a figure that excluded them would understate what the host must hold.
    """
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
    # Which storage model produced those figures.  The adapter rows save a
    # trainable head rather than the ResNet they read features from, and saying
    # so here is what keeps the reading of the total from being mistaken for a
    # full network per row.
    by_profile: dict[str, int] = {}
    for entry in estimate["per_unit_bytes"].values():
        if entry["bytes_per_run"]:
            by_profile[entry["profile"]] = by_profile.get(entry["profile"], 0) + 1
    if by_profile:
        listed = ", ".join(
            f"{count} unit(s) {name}"
            for name, count in sorted(by_profile.items(), key=lambda item: (-item[1], item[0]))
        )
        print(f"    profiles:  {listed}")
    print(
        "    checkpoint storage only; data, logs, manifests, scratch files and the "
        "host's own headroom are not included"
    )


def _print_plan(
    pending: tuple[PilotRun, ...],
    done: tuple[PilotRun, ...],
    estimate: dict,
) -> None:
    print(f"planned: {len(pending) + len(done)} run(s)")
    print(f"  completed: {len(done)} ({_summarise(done)})")
    print(f"  pending:   {len(pending)} ({_summarise(pending)})")
    print(f"  batches:   {len(batches(pending))} still to run")
    _print_disk(estimate)


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


def expected_run_views(
    runs: tuple[PilotRun, ...],
    requested: str | None,
) -> dict[tuple[Any, ...], str]:
    """The view every planned run will execute under, keyed by run identity.

    The driver used to hold one global expectation, and to drop it entirely when
    ``--os-or-ts`` was absent -- which left the default path comparing runs by
    identity alone.  A method's default view is not one value, though: a method
    native to TS and wired for calibration runs calibrated, one declared native
    to TS but not yet wired falls back to OS, and the oracle is never calibrated.
    Resolving per run is what lets a resumed unit be held to the view it will
    actually run with.

    The resolution is not reimplemented here.  ``resolve_training_view`` is shared
    with the unit script, so a driver and a runner cannot disagree about what
    "default" means -- and an explicit request is refused for a method that
    cannot carry it before any batch starts, rather than at the batch that
    happens to hit that method.

    A method the protocol marks runnable but the registry cannot supply a class
    for is a driver error, not a view: it is refused here, named, rather than
    silently resolved as OS.
    """
    from pu_toolbox.core.exceptions import RegistryError
    from pu_toolbox.experiment.method_ledger import load_ledger
    from pu_toolbox.experiment.training_views import resolve_training_view
    from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

    register_all_builtin_methods()
    ledger = load_ledger()
    #: A run's view is a property of its method and the request, not of its seed
    #: or c token, so the same resolution serves every unit of one method.
    resolved: dict[tuple[str, bool], str] = {}
    views: dict[tuple[Any, ...], str] = {}
    for run in runs:
        cache_key = (run.method, run.is_oracle)
        if cache_key not in resolved:
            estimator_class = None
            if not run.is_oracle:
                try:
                    estimator_class = get_algorithm(run.method)
                except RegistryError as exc:
                    raise ValueError(
                        f"the protocol marks {run.dataset}/{run.method}/"
                        f"{run.training_path} runnable, but the registry holds no "
                        f"estimator class for {run.method!r}, so the view that run "
                        "would use cannot be resolved."
                    ) from exc
            resolved[cache_key] = resolve_training_view(
                ledger,
                run.method,
                requested,
                is_oracle=run.is_oracle,
                estimator_class=estimator_class,
            )
        views[run.key] = f"{resolved[cache_key]}-compatible"
    return views


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
        os_or_ts=args.os_or_ts,
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
    if args.plan_json and not args.dry_run:
        print(
            "error: --plan-json records the plan, and --dry-run is what produces one; "
            "writing it during a batch run would leave a file nothing reads.",
            file=sys.stderr,
        )
        return 1
    try:
        requested = _parse_datasets(args.datasets)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
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
    try:
        protocol = _select_datasets(protocol, requested)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    planned = planned_runs(protocol)
    # Resolved once, before anything runs: a request the plan cannot satisfy is
    # refused here rather than at whichever batch happens to hit that method.
    try:
        views = expected_run_views(planned, args.os_or_ts)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # Both the dry run and the batch loop report this figure, and a row it
    # cannot size is a configuration error rather than a batch.  Refusing here
    # costs nothing, while meeting it at whichever batch happens to reach that
    # unit would mean paying for every batch queued ahead of it.
    try:
        estimate = estimate_checkpoint_bytes(protocol, input_dims=dims, runs=planned)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        pending, done = pending_runs(protocol, args.results, splits=splits, expected_views=views)
        _print_plan(pending, done, estimate)
        if args.plan_json:
            _write_plan_snapshot(
                args.plan_json,
                protocol=protocol,
                planned=planned,
                pending=pending,
                completed=done,
                views=views,
            )
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

    pending, _ = pending_runs(protocol, args.results, splits=splits, expected_views=views)
    failed = False
    for batch in batches(pending):
        if failed and not args.keep_going:
            break
        failed = not _run_batch(batch, args, priors) or failed

    # Reported from the records rather than from the loop's own bookkeeping: a
    # batch can succeed and still leave runs behind, and the operator needs the
    # count that the manifests agree with.
    still_pending, now_done = pending_runs(
        protocol, args.results, splits=splits, expected_views=views
    )
    # A subset request reports a subset's count, so the count has to name the
    # subset: two hosts each reading "completed 215 of 215" would call the
    # matrix finished while half of it has not started.
    scope = "" if requested is None else f" in {', '.join(_planned_datasets(planned))}"
    print(
        f"completed {len(now_done)} of {len(now_done) + len(still_pending)} run(s){scope}; "
        f"{len(still_pending)} still pending"
    )
    return 1 if still_pending else 0


if __name__ == "__main__":
    sys.exit(main())
