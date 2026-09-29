# tests/unit/experiment/test_survey_pilot_batch_contract.py

# ruff: noqa: N803, N806, F811, S101

"""The five batches P2.1 runs, against the matrix as frozen.

P2.1 splits the pilot into batches by dataset *and* by method and training
path, because one dataset is not one shard: CIFAR-10's adapter rows, its native
CNN row and its two-student rows are three different costs, and the cheapest
one validates the chain before the expensive one is paid for.

This file is the counterweight to that freedom.  A scope is free to select
anything, so the partition it is meant to trace is asserted here instead:
five requests, their counts, that they do not overlap, and that together they
are the whole runnable matrix -- 645 runs and no others.  The two units the
matrix excludes are pinned in the same place, because "not selected" and "not
runnable" would otherwise look the same from a batch that happens to omit them.
"""

import pytest

from pu_toolbox.experiment.pilot_plan import (
    PilotScope,
    batches,
    planned_runs,
    select_execution_units,
)
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit

#: The six runnable PU methods plus the oracle: what the partition plan calls
#: "the full matrix" for a tabular dataset.
ALL_METHODS = ("upu", "nnpu", "dist_pu", "pusb_kernel", "lbe", "self_pu", "pn_oracle")

#: The batches, as the partition plan names them, with their expected run counts.
BATCHES: dict[str, tuple[PilotScope, int]] = {
    "B1": (
        PilotScope(datasets=("spambase",), methods=ALL_METHODS, training_paths=("native_2d",)),
        215,
    ),
    "B2": (
        PilotScope(datasets=("imdb",), methods=ALL_METHODS, training_paths=("native_2d",)),
        215,
    ),
    "B3a": (
        PilotScope(
            datasets=("cifar10",),
            methods=("upu", "lbe", "pusb_kernel", "pn_oracle"),
            training_paths=("cnn_feature_adapter",),
        ),
        110,
    ),
    "B3b": (
        PilotScope(
            datasets=("cifar10",),
            methods=("self_pu", "dist_pu"),
            training_paths=("cnn_feature_adapter",),
        ),
        70,
    ),
    "B4": (
        PilotScope(datasets=("cifar10",), methods=("nnpu",), training_paths=("native_cnn",)),
        35,
    ),
}


def _protocol() -> dict:
    return load_protocol()


def _keys(name: str) -> set[tuple]:
    scope, _ = BATCHES[name]
    return {run.key for run in planned_runs(select_execution_units(_protocol(), scope))}


def _units(protocol: dict) -> set[tuple[str, str, str]]:
    return {(run.dataset, run.method, run.training_path) for run in planned_runs(protocol)}


def test_basic_each_batch_is_the_slice_the_partition_plan_names():
    expected = {name: count for name, (_, count) in BATCHES.items()}

    counted = {
        name: len(planned_runs(select_execution_units(_protocol(), scope)))
        for name, (scope, _) in BATCHES.items()
    }

    assert counted == expected


def test_basic_the_five_batches_are_disjoint_and_cover_the_matrix():
    """Every run is in exactly one batch: the union is the matrix, and no run twice.

    Overlap re-trains runs and bills for them twice; a gap leaves a hole no
    later pass fills, because every batch reports itself complete.
    """
    whole = {run.key for run in planned_runs(_protocol())}
    union: set[tuple] = set()

    for name in BATCHES:
        keys = _keys(name)
        assert not (union & keys), name
        union |= keys

    assert union == whole
    assert len(whole) == 645


def test_basic_every_runnable_unit_belongs_to_exactly_one_batch():
    """The granularity the batches are scheduled at, not only the run count.

    Batches are handed to hosts whole, so a unit split across two of them would
    be two invocations of one training path -- and the second would re-run
    whatever the first had already written.
    """
    covered: list[tuple[str, str, str]] = []
    for name in BATCHES:
        scope, _ = BATCHES[name]
        covered += list(_units(select_execution_units(_protocol(), scope)))

    assert sorted(covered) == sorted(_units(_protocol()))
    assert len(covered) == len(set(covered)) == 21


def test_basic_the_adapter_oracle_belongs_to_the_classical_batch_only():
    """CIFAR-10 has one runnable oracle and it is an adapter unit.

    The native CNN oracle is ``runnable: false``, so the batch that runs the
    native CNN row has no oracle at all -- a gap the protocol states rather
    than a result of how the batches were drawn.
    """
    oracle = ("cifar10", "pn_oracle", "cnn_feature_adapter")

    holders = [
        name
        for name in BATCHES
        if oracle
        in {
            (run.dataset, run.method, run.training_path)
            for run in planned_runs(select_execution_units(_protocol(), BATCHES[name][0]))
        }
    ]

    assert holders == ["B3a"]


def test_edge_the_units_the_matrix_excludes_enter_no_batch():
    """``kldce`` and the native CNN oracle are absent by the matrix's decision.

    Asserted against the matrix as well, so the batches' silence is read as
    agreeing with the protocol rather than as the protocol allowing it.
    """
    protocol = _protocol()
    excluded = {
        (row["dataset"], row["method"], row["training_path"])
        for row in protocol["execution_units"]
        if row["runnable"] is False
    }
    assert excluded == {
        ("spambase", "kldce", "native_2d"),
        ("imdb", "kldce", "native_2d"),
        ("cifar10", "kldce", "cnn_feature_adapter"),
        ("cifar10", "pn_oracle", "native_cnn"),
    }

    for name in BATCHES:
        units = {
            (run.dataset, run.method, run.training_path)
            for run in planned_runs(select_execution_units(protocol, BATCHES[name][0]))
        }
        assert not (units & excluded), name


def test_param_the_native_oracle_cannot_be_requested():
    """Every name is legal and the slice still runs nothing, so it is refused.

    This is the request the batch plan most invites: B4 is native CNN and
    ``pn_oracle`` is the oracle everywhere else, so asking for both reads as
    reasonable and plans nothing at all.
    """
    scope = PilotScope(
        datasets=("cifar10",),
        methods=("pn_oracle",),
        training_paths=("native_cnn",),
    )

    with pytest.raises(ValueError) as caught:
        select_execution_units(_protocol(), scope)

    assert "available matching methods: ['nnpu']" in str(caught.value)


def test_param_a_method_the_matrix_never_runs_cannot_be_requested():
    with pytest.raises(ValueError, match="no execution unit is runnable"):
        select_execution_units(_protocol(), PilotScope(methods=("kldce",)))


def test_determ_each_batch_covers_exactly_the_runs_its_scope_planned():
    """A batch is only honest if its own cross product is the set it was built from.

    The unit script executes ``seeds × tokens`` with no per-cell completion
    check, so a batch covering one run too few leaves a hole and one covering
    one too many trains a run twice.  Asserted on the frozen matrix's five
    scopes, where the grouping is not hypothetical.
    """
    for name, (scope, count) in BATCHES.items():
        planned = planned_runs(select_execution_units(_protocol(), scope))
        covered = [key for batch in batches(planned) for key in batch.covers]

        assert len(planned) == count, name
        assert sorted(covered) == sorted(run.key for run in planned), name
