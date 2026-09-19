# ruff: noqa: N803, N806, F811
"""Budget families, and what the aggregation gate means by comparable.

The locked matrix puts four budget families in one comparability group -- uPU
solves in closed form, LBE runs EM, PUSB runs internal CV, KLDCE alternates --
so the fairness comparison cannot rest on the budget dictionaries, which share
no key between them.  These tests pin what it rests on instead, in both
directions: families sharing nothing must compare, and a row whose epoch cap
disagrees must not.
"""

import pytest
from _aggregate_script_helpers import (  # noqa: F401
    aggregate_script,
    aggregate_tree,
    manifest,
    write_tree,
)

pytestmark = pytest.mark.unit


def test_edge_budget_families_without_epochs_or_batch_size_still_aggregate(
    aggregate_script, tmp_path
):
    """Four of the seven budget families cap no epochs and define no batch size.

    A closed-form solve has neither, so the gate's positive-int fields have to
    come from somewhere other than the budget -- found by running the entry
    point against real artifacts, where synthetic manifests had always
    supplied both.
    """
    root = write_tree(
        tmp_path,
        {
            "a": manifest(method="upu", budget="closed_form", epochs=None, batch_size=None),
            "b": manifest(method="kldce", budget="closed_form", epochs=None, batch_size=None),
        },
    )
    report = aggregate_tree(aggregate_script, root)
    assert report["groups"][0]["comparability_group"] == "spambase/native_2d/classical"
    assert report["groups"][0]["units"][0]["methods"] == ["kldce", "upu"]


def test_edge_descriptive_batch_size_still_aggregates(aggregate_script, tmp_path):
    """The fullbatch family describes its batch size instead of numbering it.

    ``budgets.fullbatch.batch_size`` is the string "full_train; estimator
    batch_size is unused".  It is truthy, so a ``value or 1`` fallback passes it
    straight to a gate that requires positive integers -- the same fallback that
    has to cover the families defining no batch size at all.
    """
    root = write_tree(
        tmp_path,
        {
            "a": manifest(
                method="dist_pu",
                budget="fullbatch",
                epochs=200,
                batch_size="full_train; estimator batch_size is unused",
            )
        },
    )
    report = aggregate_tree(aggregate_script, root)

    assert report["groups"][0]["comparability_group"] == "spambase/native_2d/fullbatch"
    assert report["groups"][0]["methods"] == ["dist_pu"]


def test_edge_classical_group_ranks_four_budget_families_together(aggregate_script, tmp_path):
    """The classical group's key names a method class, not a budget family.

    uPU solves in closed form, LBE runs EM, PUSB-kernel runs internal CV and
    KLDCE alternates: four budgets the protocol lists side by side, and the
    group key is what keeps *datasets* apart rather than families.  Comparing
    the raw budget dicts refuses exactly those rows; the fields the fairness
    gates actually consume -- epoch cap and batch-size candidate set -- agree.
    """
    root = write_tree(
        tmp_path,
        {
            "upu": manifest(method="upu", budget="closed_form", epochs=None, batch_size=None),
            "lbe": manifest(method="lbe", budget="em", epochs=None, batch_size=None),
            "kldce": manifest(method="kldce", budget="alternating", epochs=None, batch_size=None),
            "pusb": manifest(
                method="pusb_kernel", budget="kernel_cv", epochs=None, batch_size=None
            ),
        },
    )
    report = aggregate_tree(aggregate_script, root)

    group = report["groups"][0]
    assert group["comparability_group"] == "spambase/native_2d/classical"
    assert group["methods"] == ["kldce", "lbe", "pusb_kernel", "upu"]
    assert group["units"][0]["methods"] == ["kldce", "lbe", "pusb_kernel", "upu"]
    assert report["formal_ready"] is True


def test_param_mixing_a_minibatch_row_into_the_classical_group_is_refused(
    aggregate_script, tmp_path
):
    """Widening the budget comparison must not turn the group key into free text.

    The classical group holds families sharing no budget key at all, so the
    comparison rests on the epoch cap.  A mini-batch row dropped into that group
    disagrees on it, which is the mismatch the gate still has to catch.
    """
    root = write_tree(
        tmp_path,
        {
            "upu": manifest(
                method="upu",
                budget="closed_form",
                epochs=None,
                batch_size=None,
                group="spambase/native_2d/classical",
            ),
            "nnpu": manifest(
                method="nnpu",
                budget="minibatch",
                epochs=200,
                batch_size=256,
                group="spambase/native_2d/classical",
            ),
        },
    )
    with pytest.raises(ValueError, match="budget"):
        aggregate_tree(aggregate_script, root)
