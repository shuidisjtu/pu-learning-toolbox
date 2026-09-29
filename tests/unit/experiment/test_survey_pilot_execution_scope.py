# tests/unit/experiment/test_survey_pilot_execution_scope.py

# ruff: noqa: N803, N806, F811, S101

"""Which slice of the frozen matrix a request asks for.

``--datasets`` narrowed the matrix by dataset and nothing else, so the batches
P2.1 runs could not be requested one at a time: asking for CIFAR-10 queued the
adapter rows, the native CNN row and the two-student rows together.  A scope
names one slice along three axes, and the refusals here are what keep a
mistyped or impossible slice from planning an empty shard -- an empty plan
exits 0, and the host that was meant to cover that slice would be believed to
have covered it.

Two invariants hold across all of it.  The rows a scope excludes stay in the
narrowed protocol (the matrix carries its exclusions, and ``non_runnable_reason``
is part of what it carries); what keeps them from running is
:func:`planned_runs`, which has always skipped them.  And the narrowing is a
pure intersection: no scope perturbs the protocol it was read from.
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
    assert "available matching methods: ['nnpu']" in message


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
    """A name the matrix holds but never runs is not an unknown name.

    ``kldce`` is in the matrix at every dataset and runnable at none, so
    "unknown" would send the operator looking for a spelling mistake that is
    not there.
    """
    with pytest.raises(ValueError) as caught:
        select_execution_units(_protocol(), PilotScope(methods=("kldce",)))

    message = str(caught.value)
    assert "unknown" not in message
    assert "kldce" in message
    assert "no execution unit is runnable" in message


# --- a slice the matrix does not have is refused, not planned empty ----------


def test_edge_a_valid_slice_with_no_runnable_row_is_refused_with_its_alternatives():
    """Every name is legal; the intersection still holds nothing that can run.

    The CIFAR-10 oracle is the adapter row -- the native CNN oracle is in the
    matrix and ``runnable: false`` -- so this request is spelled correctly and
    must still be refused.  The available list is what makes it actionable.
    """
    scope = PilotScope(
        datasets=("cifar10",),
        methods=("pn_oracle",),
        training_paths=("native_cnn",),
    )

    with pytest.raises(ValueError) as caught:
        select_execution_units(_protocol(), scope)

    message = str(caught.value)
    assert "requested method(s) ['pn_oracle']" in message
    assert "have no runnable execution unit" in message
    assert "datasets=['cifar10']" in message
    assert "training_paths=['native_cnn']" in message
    assert "available matching methods: ['nnpu']" in message


def test_edge_a_dataset_only_scope_keeps_the_rows_the_matrix_excluded():
    """Exclusions narrow nothing; they are carried, not re-decided.

    Renaming or dropping the non-runnable rows here would be the driver
    quietly rewriting the matrix it was handed, and the exclusion records are
    the protocol's own statement about why they are not run.
    """
    protocol = _protocol()

    narrowed = select_execution_units(protocol, PilotScope(datasets=("spambase",)))
    rows = [row for row in narrowed["execution_units"] if row["dataset"] == "spambase"]

    assert [(row["method"], row["runnable"]) for row in rows] == [
        ("upu", True),
        ("nnpu", True),
        ("kldce", False),
        ("dist_pu", True),
        ("pusb_kernel", True),
        ("lbe", True),
        ("self_pu", True),
        ("pn_oracle", True),
    ]
    # Carried, and never run -- the two halves of one invariant: no excluded row
    # contributes a run, and every row that can run does.
    assert {(run.dataset, run.method, run.training_path) for run in planned_runs(narrowed)} == {
        (row["dataset"], row["method"], row["training_path"])
        for row in narrowed["execution_units"]
        if row["runnable"]
    }
    assert _runs(narrowed) == 215


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
    narrowed = select_execution_units(_protocol(), PilotScope(datasets=("cifar10",)))

    assert _runs(narrowed) == 215
    assert {run.dataset for run in planned_runs(narrowed)} == {"cifar10"}


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
