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
reads it -- the protocol the request names is loaded and narrowed first, and the
splits are read afterwards and only for the datasets that survived -- so the
plan, the storage a host is sized against, the resume check and the batches all
describe the same shard, and a shard never fails over data it does not run::

    uv run python scripts/run_survey_pilot.py --dry-run --datasets imdb,spambase \\
        --plan-json shard-b.json

The dataset list is not the only axis a host can be handed.  ``--methods`` and
``--training-paths`` narrow the same matrix along its other two, and the three
intersect: CIFAR-10's adapter rows, its native CNN row and its two-student rows
are one dataset and three different shards, and no dataset list separates
them::

    uv run python scripts/run_survey_pilot.py --dry-run --datasets cifar10 \\
        --methods self_pu,dist_pu --training-paths cnn_feature_adapter

A filter that names something the matrix does not have, something it has never
run, or a combination that intersects nothing is refused rather than narrowed
to nothing, for the reason the dataset list is: an empty plan exits 0 and the
host meant to cover that slice would be believed to have covered it.  What the
request named travels into the plan snapshot and onto the console, so 70 runs
inside a dataset of 215 cannot be read as the whole dataset.
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
    PilotScope,
    batch_command,
    batches,
    estimate_checkpoint_bytes,
    pending_runs,
    planned_runs,
    population_priors,
    select_execution_units,
    split_digests,
)
from pu_toolbox.experiment.survey_protocol import (
    digest,
    load_protocol,
    resolve_protocol_path,
)

_SCRIPTS = Path(__file__).resolve().parent
UNIT_SCRIPT = _SCRIPTS / "run_survey_experiment.py"

#: Feature dimensions the MLP backbones size against.  Image rows name a
#: ResNet and need none.  These are the prepared splits' widths: Spambase's 57
#: columns, SBERT's 384.
DEFAULT_INPUT_DIMS = {"spambase": 57, "imdb": 384}

_GIB = 1024**3

#: The snapshot's own format version.  Moved only when a field is removed,
#: renamed or retyped: adding one is a backward-compatible change and leaving
#: the version alone is what keeps it worth reading.  It is not the survey
#: protocol's version -- the plan and the matrix are different artifacts.
SNAPSHOT_SCHEMA_VERSION = "1.0"


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


def _parse_names(raw: str | None, *, flag: str, example: str) -> tuple[str, ...] | None:
    """The values one filter flag names, or ``None`` when the flag is absent.

    ``None`` means the whole matrix along that axis, which is what the flag's
    absence asks for.  A flag that is *present* but names nothing is a
    different request and is refused by the caller: splitting a pilot across
    two hosts means each host is given a list, and a list that quietly planned
    nothing would have both hosts report success over the same hole.

    Sorted and deduplicated, because an axis names a *set*: which order it was
    typed in is not part of the request, and a snapshot that recorded it would
    have two hosts writing different files for the same shard -- the one thing
    those files exist to rule out.
    """
    if raw is None:
        return None
    names = tuple(sorted({item.strip() for item in raw.split(",") if item.strip()}))
    if not names:
        raise ValueError(f"{flag} names nothing; pass a comma-separated list such as {example}.")
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


def _planned_datasets(planned: tuple[PilotRun, ...]) -> list[str]:
    """The datasets a plan covers, sorted: what a scoped report has to name."""
    return sorted({run.dataset for run in planned})


def _planned_units(planned: tuple[PilotRun, ...]) -> list[dict[str, str]]:
    """The runnable units a plan covers, in matrix order.

    Not the narrowed protocol's rows: that list still carries the matrix's
    exclusions, and a unit that will never run does not belong in the record of
    what a request covers.  ``planned`` is already in matrix order, so a unit's
    first appearance is that order.
    """
    units: dict[tuple[str, str, str], None] = {}
    for run in planned:
        units.setdefault((run.dataset, run.method, run.training_path), None)
    return [
        {"dataset": dataset, "method": method, "training_path": training_path}
        for dataset, method, training_path in units
    ]


def _print_scope(scope: PilotScope) -> None:
    """Name the axes a request narrowed by, once the datasets stop saying it.

    A dataset-only request needs nothing here: its plans have reported ``in
    <datasets>`` since the flag existed, and every run of those datasets is in
    scope.  Once methods or training paths narrow *within* a dataset that
    report stops being enough -- ``completed 70 of 70 run(s) in cifar10`` reads
    as CIFAR-10 finished, and 145 of its runs were never asked for.
    """
    if scope.methods is None and scope.training_paths is None:
        return
    named = [
        f"{name}={','.join(sorted(values))}"
        for name, values in (
            ("datasets", scope.datasets),
            ("methods", scope.methods),
            ("training_paths", scope.training_paths),
        )
        if values is not None
    ]
    print(f"scope: {'; '.join(named)}")


def _narrowed_by_request(scope: PilotScope) -> bool:
    """Whether a request named anything at all, i.e. whether its plan is a subset."""
    return any(
        values is not None for values in (scope.datasets, scope.methods, scope.training_paths)
    )


def _write_plan_snapshot(
    path: str,
    *,
    protocol: dict[str, Any],
    source_protocol_sha256: str,
    scope: PilotScope,
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

    ``source_protocol_sha256`` is the digest of the *whole* frozen matrix, not
    of the slice this request runs, and it is written by the same function the
    unit script hashes the same file with -- so a snapshot can be held against
    any manifest of its batch rather than merely against another snapshot.
    ``selection`` records the axes the request spelled, which is what makes two
    hosts' files comparable as requests and not only as run lists; ``datasets``
    below stays what it always was, the datasets the *plan* covers.

    Every field here is additive.  A reader that only knows the older keys
    still finds them, with the names and meanings it knew them by; that is why
    ``snapshot_schema_version`` does not move for this change.
    """
    payload = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "protocol_version": protocol["protocol_version"],
        "source_protocol_sha256": source_protocol_sha256,
        "datasets": _planned_datasets(planned),
        #: Sorted rather than as typed, so two hosts spelling one request
        #: differently still write the same bytes -- the same reason the plan is
        #: recorded in matrix order and the datasets are sorted.
        "selection": {
            "datasets": None if scope.datasets is None else sorted(scope.datasets),
            "methods": None if scope.methods is None else sorted(scope.methods),
            "training_paths": (
                None if scope.training_paths is None else sorted(scope.training_paths)
            ),
        },
        "execution_units": _planned_units(planned),
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
        "--methods",
        default=None,
        help=(
            "comma-separated methods to run (default: every runnable method in scope). "
            "Intersects with --datasets and --training-paths; an unknown name, one the "
            "matrix never runs, or a combination that intersects nothing is refused "
            "rather than planned empty, for the reason --datasets gives"
        ),
    )
    parser.add_argument(
        "--training-paths",
        default=None,
        help=(
            "comma-separated training paths to run (default: every path in scope). "
            "This is the axis that separates one dataset's adapter rows from its "
            "native-backbone row"
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
        "--reclaim-unselected-checkpoints",
        action="store_true",
        help=(
            "passed through to the unit script: delete the per-epoch weights that "
            "neither PA nor OA selected, once selection and test scoring are done"
        ),
    )
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
        reclaim_unselected_checkpoints=args.reclaim_unselected_checkpoints,
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
        scope = PilotScope(
            datasets=_parse_names(
                args.datasets, flag="--datasets", example="'cifar10' or 'imdb,spambase'"
            ),
            methods=_parse_names(
                args.methods, flag="--methods", example="'nnpu' or 'self_pu,dist_pu'"
            ),
            training_paths=_parse_names(
                args.training_paths, flag="--training-paths", example="'native_2d'"
            ),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # Resolved exactly as the unit script resolves it, so the matrix this
    # request is planned from is the one its runs are given: a driver that
    # planned the shipped matrix and passed a custom path to every batch would
    # size, resume and validate one pilot and train another.
    try:
        source_protocol = load_protocol(resolve_protocol_path(args.protocol))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    #: The digest of the matrix as shipped, which is the identity every unit
    #: manifest is bound to: the unit script loads that same file by the path
    #: below and hashes it with this same function.  Narrowing therefore happens
    #: on a copy, after this read -- a request that shrank its own protocol
    #: would have its runs writing a digest no reader of the plan could match.
    source_protocol_sha256 = digest(source_protocol)
    #: Narrowed here, before anything reads the matrix, for the reason
    #: ``--datasets`` was: the plan, the storage a host is sized against, the
    #: resume check and the batches have to describe one shard.  A filter that
    #: reached only the report would print one shard and train the matrix.
    try:
        protocol = select_execution_units(source_protocol, scope)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    #: The datasets this request runs, read off the matrix rather than off the
    #: flag: a request that named its matrix by path is narrowed by that matrix
    #: whether or not it also passed ``--datasets``, and the two reads below have
    #: to follow whatever narrowed them.
    effective = _dataset_names(protocol)
    try:
        dims = _parse_pairs(args.input_dims, cast=int)
        given_priors = _parse_pairs(args.class_prior, cast=float)
        # Read once, after the filter: a run only counts as done if it ran on
        # the splits on disk now, and the prior comes from those same splits.
        # Both reads stop at the shard's own datasets -- a host holds the splits
        # it was handed, and a defect in one this request does not name must not
        # stop it before its first batch.
        splits = split_digests(args.splits, datasets=effective)
        priors = _resolve_priors(
            population_priors(args.splits, datasets=effective),
            given_priors,
            allow_override=args.allow_prior_override,
        )
    except (OSError, ValueError) as exc:
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
        _print_scope(scope)
        if args.plan_json:
            # Refused rather than traced back: an unwritable path is an operator
            # mistake -- a typo, or a directory that was never made -- and this
            # is the same error-and-exit shape every other gate here uses.
            try:
                _write_plan_snapshot(
                    args.plan_json,
                    protocol=protocol,
                    source_protocol_sha256=source_protocol_sha256,
                    scope=scope,
                    planned=planned,
                    pending=pending,
                    completed=done,
                    views=views,
                )
            except OSError as exc:
                print(
                    f"error: cannot write the plan snapshot to {args.plan_json}: {exc}",
                    file=sys.stderr,
                )
                return 1
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
    subset = (
        "" if not _narrowed_by_request(scope) else f" in {', '.join(_planned_datasets(planned))}"
    )
    print(
        f"completed {len(now_done)} of {len(now_done) + len(still_pending)} run(s){subset}; "
        f"{len(still_pending)} still pending"
    )
    _print_scope(scope)
    return 1 if still_pending else 0


if __name__ == "__main__":
    sys.exit(main())
