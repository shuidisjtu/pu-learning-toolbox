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


def split_digests(splits_root: str | Path) -> dict[tuple[str, int], str]:
    """The indices digest each prepared split currently declares.

    A run is only evidence about the data it ran on.  When the splits are
    rebuilt -- as P1.2/P1.4 did on 2026-09-19 -- every manifest written against
    the old ones stops describing anything the pilot still has, and counting it
    as done would leave the pilot reporting complete while holding results
    that belong to no split on disk.
    """
    found: dict[tuple[str, int], str] = {}
    for path in sorted(Path(splits_root).glob("*/split_*/split_manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        dataset, seed, digest = (
            payload.get("dataset"),
            payload.get("seed"),
            payload.get("indices_sha256"),
        )
        if isinstance(dataset, str) and isinstance(seed, int) and isinstance(digest, str):
            found[(dataset, seed)] = digest
    return found


def population_priors(splits_root: str | Path) -> dict[str, float]:
    """The population class prior each dataset's splits declare.

    Protocol §3.1 makes this a constant shared by every seed of a dataset, so
    seeds disagreeing is a defect rather than a choice, and it is reported as
    one instead of being resolved by whichever split happened to sort first.

    A dataset whose splits record none is simply absent: the value was added to
    the manifest after the first artifacts were built, and a caller can still
    supply one for those.
    """
    found: dict[str, float] = {}
    sources: dict[str, str] = {}
    for path in sorted(Path(splits_root).glob("*/split_*/split_manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        block = payload.get("class_prior") if isinstance(payload, dict) else None
        population = block.get("population") if isinstance(block, dict) else None
        dataset = payload.get("dataset") if isinstance(payload, dict) else None
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
    the deficit that guard would refuse the run for.  The per-component figure
    is a lower bound -- it ignores filesystem overhead and any candidate that
    changes the network size -- and the candidate count comes from the
    protocol's pool unless overridden.  Which storage model applies is decided
    per row -- by training path, backbone and model family together -- so a row
    that saves a trainable head is not priced as the ResNet it reads features
    from.  Each unit's figure carries the profile that decided it.

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
