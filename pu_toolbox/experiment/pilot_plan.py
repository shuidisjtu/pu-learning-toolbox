"""Which runs the pilot is made of, which of them are done, and what they need.

A pilot run is named by six things -- dataset, method, training path, labeling
mechanism, ``c`` token and seed -- and the protocol fixes the set of them.  The
single-unit entry point in ``scripts/run_survey_experiment.py`` executes one
such name; this module is what a batch driver needs to know before and after
it does: the full set, the part already on disk, and the checkpoint storage the
remaining part will want.

*Done* is read from the manifests rather than predicted from the directory
layout.  Two reasons, and the second is the important one.  A run directory
exists whether the run succeeded or was refused at preflight, so presence is
not completion -- the manifest's ``execution_mode`` distinguishes them.  And a
driver that computes where a run *should* have landed will disagree with what
the run script actually did the moment either one changes, silently, in the
direction that skips work: it would report a run as already done and never
look at it again.  Reading the records cannot drift from the thing that wrote
them.

A manifest is also evidence about one particular split, so a run counts as done
only when the split digest it recorded is still the one on disk.  Rebuilding the
splits -- as P1.2/P1.4 did -- otherwise leaves the pilot reporting complete
while holding results that belong to no split it still has.

The failure direction is deliberate throughout: a manifest this module cannot
fully identify counts as *not done*, so an unreadable record, or one whose
split has been replaced, costs a repeated run rather than a hole in the pilot.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .resources import (
    DEFAULT_CHECKPOINT_ATTEMPTS,
    checkpoint_disk_requirement,
)
from .survey_protocol import unit_checkpoint_bytes, unit_checkpoint_profile
from .training_views import validated_run_view

#: The method every oracle unit runs under.  The run script substitutes it for
#: ``--method`` when ``--oracle`` is given, and an oracle run carries no
#: labeling mechanism or ``c``: it is defined by a clean label view.
PN_ORACLE = "pn_oracle"

#: What the runner writes when a run completed.  Anything else -- a preflight
#: refusal writes ``rejected_versioned_pilot``, a crash writes nothing -- means
#: present is not the same as done.
COMPLETED_MODE = "versioned_pilot"

#: An oracle's declared components, matching ``BasePUClassifier``'s default:
#: the oracle is a single supervised network, not a two-student ensemble.
_ORACLE_COMPONENTS: tuple[str, ...] = ("model",)


@dataclass(frozen=True)
class PilotRun:
    """One run's identity, as the protocol names it."""

    dataset: str
    method: str
    training_path: str
    mechanism: str | None
    c_token: str | None
    seed: int

    @property
    def is_oracle(self) -> bool:
        return self.method == PN_ORACLE

    @property
    def key(self) -> tuple[str, str, str, str | None, str | None, int]:
        return (
            self.dataset,
            self.method,
            self.training_path,
            self.mechanism,
            self.c_token,
            self.seed,
        )

    @property
    def group(self) -> tuple[str, str, str, str | None]:
        """The runs that one invocation of the unit script can cover together."""
        return (self.dataset, self.method, self.training_path, self.mechanism)


def planned_runs(protocol: dict[str, Any]) -> tuple[PilotRun, ...]:
    """Every run the protocol asks for, in a stable order.

    Only units marked runnable contribute: the matrix carries its exclusions
    explicitly so that a reader can tell "not planned" from "not yet built",
    and expanding them here would quietly run something the protocol declined
    to bind.
    """
    seeds = [int(seed) for seed in protocol["seeds"]]
    runs: list[PilotRun] = []
    for unit in protocol["execution_units"]:
        if unit.get("runnable") is not True:
            continue
        oracle = unit["method"] == PN_ORACLE
        for seed in seeds:
            if oracle:
                runs.append(
                    PilotRun(
                        unit["dataset"], unit["method"], unit["training_path"], None, None, seed
                    )
                )
                continue
            for mechanism, tokens in protocol["c_tokens"].items():
                for token in tokens:
                    runs.append(
                        PilotRun(
                            unit["dataset"],
                            unit["method"],
                            unit["training_path"],
                            mechanism,
                            token,
                            seed,
                        )
                    )
    return tuple(runs)


@dataclass(frozen=True)
class _Axis:
    """One way a request narrows the matrix, and the words a refusal uses.

    ``scope`` is the attribute it is written as, ``row`` the key it reads on an
    execution unit, and the two nouns are singular and plural labels.  Kept in
    one record so a diagnostic cannot name one axis while listing another's
    values.
    """

    scope: str
    row: str
    noun: str
    plural: str


#: The axes, in the order a refusal considers them.  This order breaks a tie
#: when more than one axis could be blamed -- see :func:`_refuse_empty_selection`.
_AXES: tuple[_Axis, ...] = (
    _Axis("datasets", "dataset", "dataset", "datasets"),
    _Axis("methods", "method", "method", "methods"),
    _Axis("training_paths", "training_path", "training_path", "training_paths"),
)


@dataclass(frozen=True)
class PilotScope:
    """Which slice of the frozen matrix a request asks for.

    Each axis is an allowlist, or ``None`` when the request does not name it at
    all.  "No filter" and "a filter that names nothing" are different requests,
    and the second is refused rather than narrowed to nothing: an empty plan
    exits 0, and the host that was meant to cover that slice would be believed
    to have covered it.
    """

    datasets: tuple[str, ...] | None = None
    methods: tuple[str, ...] | None = None
    training_paths: tuple[str, ...] | None = None


def _scope_values(scope: PilotScope) -> dict[str, tuple[str, ...] | None]:
    """Each axis as the request spells it, repeats dropped.

    Refusing an axis that names nothing here, and not only in the argument
    parser, is what keeps the meaning of an empty sequence from being decided
    somewhere else later: ``None`` already means "do not filter", so an empty
    one has no meaning left except "match nothing".
    """
    values: dict[str, tuple[str, ...] | None] = {}
    for axis in _AXES:
        named = getattr(scope, axis.scope)
        if named is None:
            values[axis.scope] = None
            continue
        names = tuple(dict.fromkeys(named))
        if not names:
            raise ValueError(
                f"{axis.scope} names nothing; pass a non-empty sequence, or None for "
                "an axis that does not filter."
            )
        values[axis.scope] = names
    return values


def _refuse_unknown_names(
    rows: list[dict[str, Any]], requested: dict[str, tuple[str, ...] | None]
) -> None:
    """A name the matrix does not have is a typo, and says so with the options.

    Checked against *every* row rather than the runnable ones, so that "the
    matrix has no such name" and "the matrix has the name and never runs it"
    stay two different answers -- see :func:`_refuse_names_with_no_runnable_row`.
    """
    for axis in _AXES:
        names = requested[axis.scope]
        if names is None:
            continue
        declared = sorted({row[axis.row] for row in rows})
        unknown = [name for name in names if name not in set(declared)]
        if unknown:
            raise ValueError(f"unknown {axis.noun}(s) {unknown}; the matrix declares {declared}.")


def _refuse_names_with_no_runnable_row(
    rows: list[dict[str, Any]], requested: dict[str, tuple[str, ...] | None]
) -> None:
    """A name the matrix holds and never runs is not an unknown name.

    ``kldce`` sits in the matrix at every dataset with ``runnable: false``, so
    reporting it as unknown would send an operator looking for a spelling
    mistake that is not there.  A name can also be runnable elsewhere and
    barren here, which is the intersection's business rather than this check's
    -- this one only asks whether *any* row runs it.
    """
    runnable = [row for row in rows if row.get("runnable") is True]
    for axis in _AXES:
        names = requested[axis.scope]
        if names is None:
            continue
        runs = {row[axis.row] for row in runnable}
        barren = [name for name in names if name not in runs]
        if barren:
            raise ValueError(
                f"{axis.noun}(s) {barren} exist in the matrix, but no execution unit "
                "is runnable for them."
            )


def _refuse_names_that_cover_nothing(
    rows: list[dict[str, Any]], requested: dict[str, tuple[str, ...] | None]
) -> None:
    """Refuse a request that names a value the *other* axes leave nothing for.

    Every name here is in the matrix and some are even runnable, so the useful
    report is which axis has no runnable value left once the other axes are
    applied, and which values it does have.  The axis reported is the one with
    the most alternatives remaining, because that is the axis whose other
    values would work: blaming an axis that is itself starved would name a list
    that is empty either way.  A tie keeps :data:`_AXES` order.

    This runs whether or not the plan came out empty, because the empty plan is
    only the loudest case of the same thing.  ``--methods nnpu,self_pu`` beside
    ``--training-paths native_cnn`` plans the ``nnpu`` runs and silently drops
    ``self_pu`` -- and then reports ``self_pu`` as in scope, so the host that
    was to cover it never runs one and no count is short.  A name that covers
    nothing is refused whenever it is named, not only when every name does.
    """
    runnable = [row for row in rows if row.get("runnable") is True]
    blamed: tuple[_Axis, list[str], list[str], list[tuple[_Axis, tuple[str, ...]]]] | None = None
    for axis in _AXES:
        names = requested[axis.scope]
        if names is None:
            continue
        others = [
            (other, requested[other.scope])
            for other in _AXES
            if other is not axis and requested[other.scope] is not None
        ]
        available = sorted(
            {
                row[axis.row]
                for row in runnable
                if all(row[other.row] in set(values) for other, values in others)
            }
        )
        unmatched = [name for name in names if name not in set(available)]
        if not unmatched:
            continue
        if blamed is None or len(available) > len(blamed[2]):
            blamed = (axis, unmatched, available, others)
    if blamed is None:
        return
    axis, unmatched, available, others = blamed
    under = ", ".join(f"{other.plural}={list(values)}" for other, values in others)
    clause = f" under {under}" if under else ""
    raise ValueError(
        f"requested {axis.noun}(s) {unmatched} have no runnable execution unit{clause}; "
        f"available matching {axis.plural}: {available}."
    )


def select_execution_units(protocol: dict[str, Any], scope: PilotScope) -> dict[str, Any]:
    """The matrix narrowed to ``scope``, keeping the rows the matrix excluded.

    A pure intersection per axis: rows outside the request are dropped, rows
    inside it are returned in the matrix's own order, and the *unrunnable* ones
    come along.  That last part is deliberate.  The matrix states its
    exclusions and why (``non_runnable_reason``), and the only thing that has
    ever kept those rows from running is :func:`planned_runs` skipping them --
    so narrowing them away here would be this function re-deciding a matrix
    question, and would leave ``--datasets`` returning a different protocol
    than the driver returns today for the same request.

    Narrowing the protocol rather than the plan is what makes the plan, the
    storage estimate, the resume check and the batches describe one shard: a
    filter applied where the plan is printed would print one batch and train
    the matrix.

    Refusals are per axis and distinct, because "there is no such name", "the
    matrix has it and never runs it" and "the other axes leave it nothing" need
    three different fixes.  Anything that would plan nothing raises rather than
    returning an empty matrix: the caller reports the count, and a shard that
    planned nothing exits 0 while another host is believed to have covered it.
    The same refusal covers the quieter half of that case -- a value that covers
    nothing while its neighbours cover something -- because the count would not
    be short either way, and the printed scope would claim the value was run.
    """
    rows = list(protocol["execution_units"])
    requested = _scope_values(scope)
    _refuse_unknown_names(rows, requested)
    _refuse_names_with_no_runnable_row(rows, requested)
    allowed = {
        axis.row: None if requested[axis.scope] is None else set(requested[axis.scope])
        for axis in _AXES
    }
    narrowed = {
        **protocol,
        "execution_units": [
            row
            for row in rows
            if all(names is None or row[key] in names for key, names in allowed.items())
        ],
    }
    #: Asked of the plan, not of the rows: a row that never runs cannot make a
    #: name covered.  With every axis named this already rejects an empty
    #: intersection, so the guard below is the shape an unnamed axis leaves --
    #: defence for a scope that filters nothing, which is never empty today.
    _refuse_names_that_cover_nothing(rows, requested)
    if not planned_runs(narrowed):
        raise ValueError("the request selects no runnable execution unit.")
    return narrowed


def manifest_identity(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    """The run a manifest records, or ``None`` when it records no completed one.

    Everything is read from the manifest itself, including the mechanism the
    generator reported rather than the one the command line asked for: a run
    whose generator disagreed with its request is not a run that can be called
    done under either name.

    ``execution_mode`` is necessary and not sufficient.  A run whose candidates
    were *all* excluded -- every attempt failed -- also writes the completed
    mode, with an empty ``selection`` beside its ``failures``, and then raises.
    Reading the mode alone would file that as done and never retry it, leaving
    a hole in the pilot that no later pass would fill.  A selection is what a
    finished run has and a failed one does not.
    """
    if payload.get("execution_mode") != COMPLETED_MODE:
        return None
    selection = payload.get("selection")
    if not isinstance(selection, dict) or not selection:
        return None
    unit = payload.get("execution_unit")
    seed = payload.get("seed")
    if not isinstance(unit, dict) or not isinstance(seed, int):
        return None
    method = unit.get("method")
    dataset = unit.get("dataset")
    training_path = unit.get("training_path")
    if (
        not isinstance(method, str)
        or not isinstance(dataset, str)
        or not isinstance(training_path, str)
    ):
        return None
    if method == PN_ORACLE:
        return (dataset, method, training_path, None, None, seed)
    generation = payload.get("generation")
    train_meta = generation.get("train") if isinstance(generation, dict) else None
    mechanism = train_meta.get("mechanism") if isinstance(train_meta, dict) else None
    token = payload.get("c_requested_token")
    if not isinstance(mechanism, str) or not isinstance(token, str):
        # Written after the run succeeded, so its absence marks a record this
        # driver cannot place.  Unknown is treated as not done.
        return None
    return (dataset, method, training_path, mechanism, token, seed)


def _dataset_allowlist(datasets: Iterable[str] | None) -> set[str] | None:
    """The datasets a read covers, or ``None`` for every one of them.

    A bare string is refused rather than iterated: ``"spambase"`` is a sequence
    of eight characters, and a set of them matches no dataset at all, so a
    caller who meant one dataset would be handed an empty read -- a wrong answer
    that reads exactly like a fact.  Naming *nothing* is a different request and
    is allowed: an empty collection asks for no datasets, not for all of them.
    """
    if datasets is None:
        return None
    if isinstance(datasets, str):
        raise ValueError(f"datasets takes a collection of names, not the string {datasets!r}")
    return set(datasets)


def _split_manifests(splits_root: str | Path, allowed: set[str] | None) -> list[Path]:
    """The split manifests a read covers, in a stable order.

    Scoping the *walk*, rather than reading everything and discarding the rest,
    is what makes "a shard reads its own splits" true of the I/O and not only of
    the result.  The directory a manifest sits in names the dataset it belongs
    to -- the convention ``split_archive`` already packs and verifies under --
    so a manifest can only be read by the shard whose directory holds it.
    What a manifest *declares* is checked by the callers as well, so one that
    names another dataset is skipped rather than attributed across datasets.
    """
    root = Path(splits_root)
    if allowed is None:
        return sorted(root.glob("*/split_*/split_manifest.json"))
    return sorted(
        manifest
        for dataset in sorted(allowed)
        for manifest in (root / dataset).glob("split_*/split_manifest.json")
    )


def split_digests(
    splits_root: str | Path, *, datasets: Iterable[str] | None = None
) -> dict[tuple[str, int], str]:
    """The indices digest each prepared split currently declares.

    A run is only evidence about the data it ran on.  When the splits are
    rebuilt -- as P1.2/P1.4 did on 2026-09-19 -- every manifest written against
    the old ones stops describing anything the pilot still has, and counting it
    as done would leave the pilot reporting complete while holding results
    that belong to no split on disk.

    ``datasets`` limits both the walk and the result to those datasets, which is
    what a shard asks for: it is sized, resumed and gated against its own
    splits, and nothing under a dataset it never names is this read's business.
    """
    allowed = _dataset_allowlist(datasets)
    found: dict[tuple[str, int], str] = {}
    for path in _split_manifests(splits_root, allowed):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if allowed is not None and payload.get("dataset") not in allowed:
            continue
        dataset, seed, digest = (
            payload.get("dataset"),
            payload.get("seed"),
            payload.get("indices_sha256"),
        )
        if isinstance(dataset, str) and isinstance(seed, int) and isinstance(digest, str):
            found[(dataset, seed)] = digest
    return found


def population_priors(
    splits_root: str | Path, *, datasets: Iterable[str] | None = None
) -> dict[str, float]:
    """The population class prior each dataset's splits declare.

    Protocol §3.1 makes this a constant shared by every seed of a dataset, so
    seeds disagreeing is a defect rather than a choice, and it is reported as
    one instead of being resolved by whichever split happened to sort first.

    A dataset whose splits record none is simply absent: the value was added to
    the manifest after the first artifacts were built, and a caller can still
    supply one for those.

    ``datasets`` limits both the read and the defect it reports.  A shard reads
    the priors of the datasets it runs, so a disagreement inside a dataset it
    does not name cannot stop it -- the shards of one pilot are prepared and
    held separately, and a host should not fail over data no run of it touches.
    """
    allowed = _dataset_allowlist(datasets)
    found: dict[str, float] = {}
    sources: dict[str, str] = {}
    for path in _split_manifests(splits_root, allowed):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if allowed is not None and payload.get("dataset") not in allowed:
            continue
        block = payload.get("class_prior")
        population = block.get("population") if isinstance(block, dict) else None
        dataset = payload.get("dataset")
        if not isinstance(dataset, str) or not isinstance(population, (int, float)):
            continue
        if dataset in found and found[dataset] != float(population):
            raise ValueError(
                f"{dataset} declares two population priors: {found[dataset]} in "
                f"{sources[dataset]} and {float(population)} in {path}. The protocol "
                "defines it as one constant per dataset."
            )
        found[dataset] = float(population)
        sources[dataset] = str(path)
    return found


def manifest_resume_key(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    """The run *and view* a manifest records, or ``None`` when it records neither.

    ``manifest_identity`` names a protocol unit; this names one execution of it.
    A resume scan keys on the pair, because a unit may legitimately have been run
    under both views and each manifest is evidence only for the view it used.
    """
    identity = manifest_identity(payload)
    if identity is None:
        return None
    return (*identity, validated_run_view(payload))


def completed_runs(
    results_root: str | Path,
    *,
    splits: dict[tuple[str, int], str] | None = None,
) -> dict[tuple[Any, ...], Path]:
    """Every completed run under ``results_root``, keyed by ``(identity, view)``.

    With ``splits`` given, a run whose manifest records a different split digest
    than the one on disk is not completed: it ran on data this pilot no longer
    has.  A split the mapping does not cover counts as not done too -- the
    driver re-running a run is recoverable, a hole in the matrix is not.

    The key carries the view the run recorded, so the OS and TS manifests of one
    unit stay separate entries instead of collapsing to whichever path sorted
    first.  Which of them satisfies a request is the caller's question, and it
    is answered against the view the run is going to use.

    A manifest whose view cannot be validated counts as not done, like any other
    record this module cannot fully identify: one unusable artifact is not
    evidence of completion, and stopping the scan over it would block a resume
    over a file unrelated to the planned matrix.
    """
    done: dict[tuple[Any, ...], Path] = {}
    for path in sorted(Path(results_root).rglob("manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        identity = manifest_identity(payload)
        if identity is None:
            continue
        if splits is not None and not _ran_on_current_split(payload, identity, splits):
            continue
        try:
            view = validated_run_view(payload)
        except ValueError:
            continue
        done.setdefault((*identity, view), path)
    return done


def _ran_on_current_split(
    payload: dict[str, Any], identity: tuple[Any, ...], splits: dict[tuple[str, int], str]
) -> bool:
    dataset, _, _, _, _, seed = identity
    representation = payload.get("representation")
    recorded = representation.get("split_sha256") if isinstance(representation, dict) else None
    return isinstance(recorded, str) and recorded == splits.get((dataset, seed))


def pending_runs(
    protocol: dict[str, Any],
    results_root: str | Path,
    *,
    expected_views: Mapping[tuple[Any, ...], str],
    splits: dict[tuple[str, int], str] | None = None,
) -> tuple[tuple[PilotRun, ...], tuple[PilotRun, ...]]:
    """``(pending, completed)`` for the protocol's runs under ``results_root``.

    Each planned run is looked up under the view it is going to run with, not
    under its identity alone: an OS result does not satisfy a unit this pilot
    will run calibrated, so changing the view cannot silently skip work.  That
    view is resolved by the caller, which reads the same ledger the unit script
    reads -- this module never guesses it.

    ``expected_views`` must cover every planned run.  A missing entry is a
    caller bug, and falling back to an identity-only lookup would reinstate
    exactly the blind comparison this signature exists to remove.

    The completed half is returned alongside so a caller can report what it
    skipped; a driver that resumes silently is indistinguishable from one that
    ran nothing.
    """
    planned = planned_runs(protocol)
    missing = [run.key for run in planned if run.key not in expected_views]
    if missing:
        raise ValueError(
            f"expected_views does not cover {len(missing)} planned run(s); "
            f"first missing: {missing[0]}"
        )
    done = completed_runs(results_root, splits=splits)
    pending = tuple(run for run in planned if (*run.key, expected_views[run.key]) not in done)
    completed = tuple(run for run in planned if (*run.key, expected_views[run.key]) in done)
    return pending, completed


@dataclass(frozen=True)
class Batch:
    """One invocation of the unit script, and the runs that invocation covers.

    The script executes the cross product of ``seeds`` and ``tokens``, with no
    per-cell completion check of its own.  A batch is therefore only honest if
    that cross product is exactly the set it was built from -- see
    :func:`batches`, which guarantees it rather than assuming it.
    """

    dataset: str
    method: str
    training_path: str
    mechanism: str | None
    seeds: tuple[int, ...]
    tokens: tuple[str, ...]

    @property
    def unit(self) -> tuple[str, str, str, str | None]:
        return (self.dataset, self.method, self.training_path, self.mechanism)

    @property
    def covers(self) -> tuple[tuple[Any, ...], ...]:
        """The run identities this invocation will execute."""
        if self.method == PN_ORACLE:
            return tuple(
                (self.dataset, self.method, self.training_path, None, None, seed)
                for seed in self.seeds
            )
        return tuple(
            (self.dataset, self.method, self.training_path, self.mechanism, token, seed)
            for seed in self.seeds
            for token in self.tokens
        )


def batches(runs: Iterable[PilotRun]) -> tuple[Batch, ...]:
    """Group pending runs into invocations that cover exactly those runs.

    Whole units coalesce into one invocation, but only when the pending runs
    are the unit's complete seed-by-token grid.  A partly finished unit is
    split per seed instead: one seed's tokens are a cross product the script
    can run exactly, while passing the union of several seeds' tokens would
    re-run the cells that are already on disk.  Re-running is not free -- it
    trains again and writes a second ``checkpoints/attempt-*`` directory that
    nothing ever reuses.
    """
    collected: dict[tuple[str, str, str, str | None], list[PilotRun]] = {}
    for run in runs:
        collected.setdefault(run.group, []).append(run)

    result: list[Batch] = []
    for (dataset, method, training_path, mechanism), group in collected.items():
        seeds = sorted({run.seed for run in group})
        tokens = sorted({str(run.c_token) for run in group if run.c_token is not None}, key=float)
        whole = len(group) == len(seeds) * max(len(tokens), 1)
        if whole:
            result.append(
                Batch(dataset, method, training_path, mechanism, tuple(seeds), tuple(tokens))
            )
            continue
        for seed in seeds:
            seed_tokens = sorted(
                {str(run.c_token) for run in group if run.seed == seed and run.c_token is not None},
                key=float,
            )
            result.append(
                Batch(dataset, method, training_path, mechanism, (seed,), tuple(seed_tokens))
            )
    return tuple(sorted(result, key=lambda batch: str(batch.unit) + str(batch.seeds)))


def batch_command(
    batch: Batch,
    *,
    splits_root: str | Path,
    out_root: str | Path,
    protocol: str = "survey-v1",
    class_prior: float | None = None,
    device: str | None = None,
    adapter_cache: str | None = None,
    extraction_batch_size: int | None = None,
    os_or_ts: str | None = None,
    reclaim_unselected_checkpoints: bool = False,
) -> list[str]:
    """The unit script's arguments for one batch.

    ``--out-dir`` is always given rather than left to the script's default, so
    the driver decides where results land instead of inferring it afterwards.
    The optional arguments are passed through only when the caller supplies
    them: the unit script's own defaults stay in force otherwise, and inventing
    a device or a class prior here is exactly the sort of quiet divergence the
    driver exists to avoid.
    """
    argv = [
        str(Path(splits_root) / batch.dataset),
        "--dataset",
        batch.dataset,
        "--training-path",
        batch.training_path,
        "--protocol",
        protocol,
        "--out-dir",
        str(out_root),
        "--seeds",
        ",".join(str(seed) for seed in batch.seeds),
    ]
    if batch.method == PN_ORACLE:
        # An oracle is defined by a clean label view: no method, no mechanism,
        # no c.  The script refuses --oracle beside --method or --class-prior.
        argv.append("--oracle")
    else:
        argv += [
            "--method",
            batch.method,
            "--labeling-mechanism",
            str(batch.mechanism),
            "--c",
            ",".join(batch.tokens),
        ]
        if class_prior is not None:
            argv += ["--class-prior", repr(class_prior)]
    if device is not None:
        argv += ["--device", device]
    if adapter_cache is not None:
        argv += ["--adapter-cache", adapter_cache]
    if extraction_batch_size is not None:
        argv += ["--extraction-batch-size", str(extraction_batch_size)]
    if os_or_ts is not None:
        # Omitted unless asked for: the unit script's ledger-derived default
        # stays in force, so the driver never invents a view.
        argv += ["--os-or-ts", os_or_ts]
    if reclaim_unselected_checkpoints:
        # Opt-in only.  The flag is absent by default so the unit script keeps
        # every checkpoint, which is what the library and probe paths expect.
        argv.append("--reclaim-unselected-checkpoints")
    return argv


def _declared_components(method: str) -> tuple[str, ...]:
    """How many per-epoch checkpoints the method keeps, from its own class.

    Declared on the class rather than discovered by fitting, which is what
    makes a pre-run budget possible at all.  An unregistered or oracle method
    reports the base default, matching what the runner would see.
    """
    if method == PN_ORACLE:
        return _ORACLE_COMPONENTS
    from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

    register_all_builtin_methods()
    declared = getattr(get_algorithm(method), "epoch_components", _ORACLE_COMPONENTS)
    return tuple(declared)


def _run_checkpoint_bytes(
    protocol: dict[str, Any],
    unit: dict[str, Any],
    method: str,
    input_dims: dict[str, int],
    candidates: int,
    *,
    attempts: int,
) -> int:
    """Storage one run of this unit writes, or 0 when it writes none.

    ``attempts`` is not a fudge factor: each attempt persists into its own
    ``checkpoints/attempt-*`` directory, so a retried run really does occupy
    twice as much.  A run that succeeds first time occupies one attempt's
    worth, which is the quantity that accumulates across the pilot.
    """
    epochs = protocol["budgets"][unit["budget"]].get("epochs")
    if not epochs:
        return 0  # closed-form and kernel units keep no per-epoch state
    component = unit_checkpoint_bytes(
        protocol, unit, input_dim=_input_dim(input_dims, unit, method)
    )
    if component is None:
        return 0
    return checkpoint_disk_requirement(
        bytes_per_component=component,
        epochs=int(epochs),
        components=len(_declared_components(method)),
        candidates=candidates,
        attempts=attempts,
    )


def estimate_checkpoint_bytes(
    protocol: dict[str, Any],
    *,
    input_dims: dict[str, int],
    runs: Iterable[PilotRun] | None = None,
    candidates: int | None = None,
) -> dict[str, Any]:
    """Checkpoint storage the given runs need, by dataset and in total.

    Three figures, because the storage that accumulates and the storage the
    pre-run guard insists on are not the same number, and reading either one
    as the other is wrong in a different direction.

    ``total_bytes`` and the ``per_*`` breakdowns are what the pilot actually
    occupies: one attempt per run, because a run that succeeds first time
    writes one ``checkpoints/attempt-*`` directory.  Nothing deletes them and
    offline selection needs them afterwards, so this is the figure a host
    holding the whole pilot is sized against.  ``peak_bytes_per_run`` is the
    largest single run's share of it -- enough if checkpoints are reclaimed as
    runs finish.

    ``guard_*`` is what ``_enforce_disk_preflight`` checks free space against.
    It reserves ``DEFAULT_CHECKPOINT_ATTEMPTS`` attempts per candidate, so it
    is higher than what a run normally writes; a run that does retry writes
    the second attempt too, which is why the reserve exists.

    Uses the same two functions the guard uses, so a deficit this predicts is
    the deficit that guard would refuse the run for, and the candidate count
    comes from the protocol's pool unless overridden.  Which storage model
    applies is decided per row -- by training path, backbone and model family
    together -- so a row that saves a trainable head is not priced as the ResNet
    it reads features from.  Each unit's figure carries the profile that decided
    it, and the profile is what says which way that figure bounds: the MLP
    formula is a lower bound on what a component costs, while the two image
    constants are conservative upper bounds rounded up from real serialisations.

    Cost is per *unit*: mechanism, ``c`` and seed change how many times a unit
    runs, never how much one run weighs.
    """
    selected = planned_runs(protocol) if runs is None else tuple(runs)
    candidate_count = len(protocol["candidate_pool"]) if candidates is None else int(candidates)
    per_unit: dict[str, dict[str, Any]] = {}
    per_dataset: dict[str, int] = {}
    guard_total = guard_peak = 0
    for run in selected:
        label = _unit_label(run)
        if label not in per_unit:
            unit = _unit_of(protocol, run)
            dimension = _input_dim(input_dims, unit, run.method)
            per_unit[label] = {
                "runs": 0,
                # Named next to the figure so a report can say which storage
                # model produced it rather than leaving the reader to infer it.
                "profile": unit_checkpoint_profile(protocol, unit, input_dim=dimension),
                "bytes_per_run": _run_checkpoint_bytes(
                    protocol, unit, run.method, input_dims, candidate_count, attempts=1
                ),
                "guard_bytes_per_run": _run_checkpoint_bytes(
                    protocol,
                    unit,
                    run.method,
                    input_dims,
                    candidate_count,
                    attempts=DEFAULT_CHECKPOINT_ATTEMPTS,
                ),
            }
        entry = per_unit[label]
        entry["runs"] += 1
        per_dataset[run.dataset] = per_dataset.get(run.dataset, 0) + entry["bytes_per_run"]
        guard_total += entry["guard_bytes_per_run"]
        guard_peak = max(guard_peak, entry["guard_bytes_per_run"])
    peak_label, peak_entry = max(
        per_unit.items(),
        key=lambda item: item[1]["bytes_per_run"],
        default=("", {"bytes_per_run": 0}),
    )
    return {
        "runs": len(selected),
        "units": len(per_unit),
        "candidates_per_run": candidate_count,
        "attempts_reserved": DEFAULT_CHECKPOINT_ATTEMPTS,
        "per_dataset_bytes": dict(sorted(per_dataset.items())),
        "per_unit_bytes": dict(sorted(per_unit.items())),
        "peak_bytes_per_run": peak_entry["bytes_per_run"],
        "guard_total_bytes": guard_total,
        "guard_peak_bytes_per_run": guard_peak,
        "peak_unit": peak_label,
        "total_bytes": sum(per_dataset.values()),
    }


def _unit_of(protocol: dict[str, Any], run: PilotRun) -> dict[str, Any]:
    for unit in protocol["execution_units"]:
        if (
            unit["dataset"] == run.dataset
            and unit["method"] == run.method
            and unit["training_path"] == run.training_path
        ):
            return unit
    raise ValueError(f"no execution unit for {run.key}")


def _input_dim(input_dims: dict[str, int], unit: dict[str, Any], method: str) -> int:
    """The dimension an MLP backbone sizes against; unused by image rows.

    Image units name a ResNet and size from a protocol constant, so the value
    returned for them is never read -- asking the caller to supply one anyway
    would demand a number nothing consumes.
    """
    if str(unit["backbone"]).startswith("resnet18"):
        return 0
    if unit["dataset"] not in input_dims:
        raise ValueError(
            f"{unit['dataset']}/{method} sizes its {unit['backbone']} backbone from a "
            f"feature dimension, which was not given; supply one for {unit['dataset']!r} "
            f"(known: {sorted(input_dims)})"
        )
    return int(input_dims[unit["dataset"]])


def _unit_label(run: PilotRun) -> str:
    return f"{run.dataset}/{run.method}/{run.training_path}"
