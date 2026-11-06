# tests/unit/experiment/test_survey_pilot_execution_scope.py

# ruff: noqa: N803, N806, F811, S101

"""Scope selection preserves the v3 matrix and rejects uncovered requests.

Scopes intersect dataset, method and training-path allowlists. Blocked rows
remain visible in a selected matrix, but never contribute planned runs.
"""

import copy

import pytest

from pu_toolbox.experiment.pilot_plan import (
    PilotScope,
    planned_runs,
    select_execution_units,
)
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit


def _protocol() -> dict:
    return load_protocol()


def _units(protocol: dict) -> list[tuple[str, str, str]]:
    """Every row the narrowed protocol still holds, runnable or not, in order."""
    return [
        (row["dataset"], row["method"], row["training_path"]) for row in protocol["execution_units"]
    ]


def _runs(protocol: dict) -> int:
    return len(planned_runs(protocol))


# --- an axis that is given is an allowlist, never an empty one ---------------


def test_param_an_axis_that_names_nothing_is_refused():
    """``None`` is how an axis asks for no filtering; empty is a different request.

    The snapshot records an unfiltered axis as ``null``, so an empty sequence
    cannot mean the same thing -- and the only meaning left for it is "match
    nothing", which plans an empty shard and exits 0.
    """
    for scope in (
        PilotScope(datasets=()),
        PilotScope(methods=()),
        PilotScope(training_paths=()),
    ):
        with pytest.raises(ValueError, match="names nothing"):
            select_execution_units(_protocol(), scope)


def test_param_a_named_value_that_covers_nothing_is_refused():
    """Every name is in the matrix, and one of them still runs nothing here.

    ``self_pu`` is runnable at CIFAR-10 -- on the adapter path.  Named beside
    ``native_cnn`` it is therefore neither a typo nor a name the matrix never
    runs, and the intersection is not empty: ``nnpu`` runs.  Left alone the
    request plans 35 runs while its own record says ``self_pu`` is in scope,
    and a host told it covers ``self_pu`` here would never run one -- the same
    silent gap the dataset list was refused for, one level down.
    """
    scope = PilotScope(
        datasets=("cifar10",),
        methods=("nnpu", "self_pu"),
        training_paths=("native_cnn",),
    )

    with pytest.raises(ValueError) as caught:
        select_execution_units(_protocol(), scope)

    message = str(caught.value)
    assert "requested method(s) ['self_pu']" in message
    assert (
        "available matching methods: ['cvir', 'genpu', 'gradpu', 'lagam', 'nnpu', "
        "'p3mix', 'pan', 'pn_oracle', 'pulda', 'pulns', 'vpu']" in message
    )


# --- what a name has to be, and the three ways it can be wrong ---------------


def test_param_an_unknown_dataset_is_refused_and_the_matrix_is_listed():
    with pytest.raises(ValueError) as caught:
        select_execution_units(_protocol(), PilotScope(datasets=("cifra10",)))

    message = str(caught.value)
    assert "unknown dataset(s) ['cifra10']" in message
    assert "cifar10" in message  # the names it would have accepted
    assert "imdb" in message


def test_param_an_unknown_method_is_refused_and_the_matrix_is_listed():
    with pytest.raises(ValueError) as caught:
        select_execution_units(_protocol(), PilotScope(methods=("kldcce",)))

    message = str(caught.value)
    assert "unknown method(s) ['kldcce']" in message
    assert "kldce" in message
    assert "self_pu" in message


def test_param_an_unknown_training_path_is_refused_and_the_matrix_is_listed():
    with pytest.raises(ValueError) as caught:
        select_execution_units(_protocol(), PilotScope(training_paths=("native_3d",)))

    message = str(caught.value)
    assert "unknown training_path(s) ['native_3d']" in message
    assert "native_2d" in message
    assert "cnn_feature_adapter" in message


def test_param_a_method_with_no_runnable_row_is_refused_by_its_own_message():
    """A caller-disabled method remains known, with a distinct refusal."""
    protocol = _protocol()
    for row in protocol["execution_units"]:
        if row["method"] == "kldce":
            row["runnable"] = False
            row["non_runnable_reason"] = "disabled in this caller-selected matrix"
    with pytest.raises(ValueError) as caught:
        select_execution_units(protocol, PilotScope(methods=("kldce",)))

    message = str(caught.value)
    assert "unknown" not in message
    assert "kldce" in message
    assert "no execution unit is runnable" in message


# --- a slice the matrix does not have is refused, not planned empty ----------


def test_edge_native_cnn_oracle_scope_does_not_include_the_adapter_oracle():
    """Each runnable oracle path has one c-independent run per seed."""
    protocol = _protocol()
    scope = PilotScope(
        datasets=("cifar10",),
        methods=("pn_oracle",),
        training_paths=("native_cnn",),
    )

    narrowed = select_execution_units(protocol, scope)
    runs = planned_runs(narrowed)

    assert _units(narrowed) == [("cifar10", "pn_oracle", "native_cnn")]
    assert {run.key for run in runs} == {
        ("cifar10", "pn_oracle", "native_cnn", None, None, seed) for seed in protocol["seeds"]
    }
    assert len(runs) == 5


def test_edge_a_dataset_only_scope_keeps_the_rows_the_matrix_excluded():
    """Exclusions narrow nothing; they are carried, not re-decided.

    Renaming or dropping the non-runnable rows here would be the driver
    quietly rewriting the matrix it was handed, and the exclusion records are
    the protocol's own statement about why they are not run.
    """
    protocol = _protocol()
    for row in protocol["execution_units"]:
        if row["method"] == "kldce":
            row["runnable"] = False
            row["non_runnable_reason"] = "disabled in this caller-selected matrix"

    narrowed = select_execution_units(protocol, PilotScope(datasets=("spambase",)))
    rows = narrowed["execution_units"]

    assert rows == [row for row in protocol["execution_units"] if row["dataset"] == "spambase"]
    excluded = [row for row in rows if not row["runnable"]]
    assert excluded and all(row["non_runnable_reason"] for row in excluded)
    # Carried, and never run -- the two halves of one invariant: no excluded row
    # contributes a run, and every row that can run does.
    assert {(run.dataset, run.method, run.training_path) for run in planned_runs(narrowed)} == {
        (row["dataset"], row["method"], row["training_path"])
        for row in narrowed["execution_units"]
        if row["runnable"]
    }
    # Eighteen PU profiles cover seven cells each; three support profiles cover
    # three SCAR cells each, plus one c-independent oracle, across five seeds.
    assert _runs(narrowed) == 5 * (18 * 7 + 3 * 3 + 1)


# --- the axes narrow together, and the narrowing is a pure function ----------


def test_basic_the_axes_intersect_each_other_and_values_union_within_one():
    """AND between the axes, OR inside one: the only reading that cuts a grid.

    Two datasets and two methods is four units; naming one dataset has to leave
    two, and naming both methods has to leave both -- an axis that ANDed with
    itself, or an axis that was ignored, would leave a different count.
    """
    protocol = _protocol()

    both_axes = _units(
        select_execution_units(
            protocol, PilotScope(datasets=("spambase", "imdb"), methods=("nnpu", "lbe"))
        )
    )
    one_dataset = _units(
        select_execution_units(
            protocol, PilotScope(datasets=("spambase",), methods=("nnpu", "lbe"))
        )
    )

    assert both_axes == [
        ("spambase", "nnpu", "native_2d"),
        ("spambase", "lbe", "native_2d"),
        ("imdb", "nnpu", "native_2d"),
        ("imdb", "lbe", "native_2d"),
    ]
    assert one_dataset == [
        ("spambase", "nnpu", "native_2d"),
        ("spambase", "lbe", "native_2d"),
    ]


def test_basic_a_dataset_only_scope_plans_that_dataset_s_whole_runnable_slice():
    """The dataset axis alone still means what ``--datasets`` always meant."""
    protocol = _protocol()
    narrowed = select_execution_units(protocol, PilotScope(datasets=("cifar10",)))

    # Eighteen PU units cover seven cells each, KLDCE and three support units
    # cover three SCAR cells, and both oracle paths ignore c.
    assert _runs(narrowed) == 5 * (18 * 7 + 4 * 3 + 2)
    assert {run.dataset for run in planned_runs(narrowed)} == {"cifar10"}
    assert {(run.method, run.training_path) for run in planned_runs(narrowed)} == {
        (row["method"], row["training_path"])
        for row in protocol["execution_units"]
        if row["dataset"] == "cifar10" and row["runnable"]
    }


def test_determ_the_order_of_axis_values_does_not_change_the_selection():
    """A set written in another order is the same set; the matrix sets the order.

    What the narrowed protocol holds follows the matrix, never the request, so
    two hosts naming the same slice cannot end up with different evidence.
    """
    protocol = _protocol()

    forward = select_execution_units(
        protocol, PilotScope(datasets=("spambase", "imdb"), methods=("nnpu", "lbe"))
    )
    reversed_ = select_execution_units(
        protocol, PilotScope(datasets=("imdb", "spambase"), methods=("lbe", "nnpu"))
    )

    assert _units(forward) == _units(reversed_)


def test_determ_selecting_twice_from_one_protocol_returns_the_same_slice():
    """The narrowing reads its input and does not write to it.

    A scope that mutated the protocol in place would make the second call see
    the first call's result, and the driver reads the source protocol again
    after narrowing it -- for the digest every manifest is bound to.
    """
    protocol = _protocol()
    untouched = copy.deepcopy(protocol)
    scope = PilotScope(
        datasets=("cifar10",),
        methods=("self_pu", "dist_pu"),
        training_paths=("cnn_feature_adapter",),
    )

    first = select_execution_units(protocol, scope)
    second = select_execution_units(protocol, scope)

    assert _units(first) == _units(second)
    assert protocol == untouched


def test_determ_an_unfiltered_scope_returns_the_protocol_word_for_word():
    """No axis given is not a filter that happens to match everything."""
    protocol = _protocol()

    narrowed = select_execution_units(protocol, PilotScope())

    assert narrowed == protocol
    assert narrowed["execution_units"] is not protocol["execution_units"]
