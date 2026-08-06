# ruff: noqa: N802, N803, N806, E501

"""Tests for the list-methods / list-priors subcommands."""

from __future__ import annotations

import pytest

from pu_toolbox.cli import build_parser


@pytest.mark.unit
def test_basic_list_methods_prints_table(capsys):
    """Table header and representative registered methods appear."""
    args = build_parser().parse_args(["list-methods"])
    args.func(args)
    out = capsys.readouterr().out
    assert "Name" in out and "Family" in out and "Auto-inst" in out
    for name in ("nnpu", "elkan_noto", "upu"):
        assert name in out


@pytest.mark.unit
def test_edge_ldce_not_auto_instantiable(capsys):
    """ldce requires flip_probability → Auto-inst column shows 'no'."""
    args = build_parser().parse_args(["list-methods"])
    args.func(args)
    lines = capsys.readouterr().out.splitlines()
    ldce_line = next(line for line in lines if line.strip().startswith("ldce"))
    assert ldce_line.split()[-1] == "no"


@pytest.mark.unit
def test_list_methods_rows_are_sorted(capsys):
    args = build_parser().parse_args(["list-methods"])
    args.func(args)
    lines = [line for line in capsys.readouterr().out.splitlines()[2:] if line.strip()]
    names = [line.split()[0] for line in lines]
    assert names == sorted(names)


@pytest.mark.unit
def test_basic_list_priors_lists_estimators(capsys):
    args = build_parser().parse_args(["list-priors"])
    args.func(args)
    out = capsys.readouterr().out
    for name in ("recpe", "pen_l1", "km1", "km2"):
        assert name in out


@pytest.mark.unit
def test_param_list_priors_includes_aliases(capsys):
    """Registry aliases accepted by --prior-estimator appear in the listing."""
    args = build_parser().parse_args(["list-priors"])
    args.func(args)
    out = capsys.readouterr().out
    for alias in ("cpe", "pe", "re_cpe", "rethinking_cpe"):
        assert alias in out, f"{alias!r} accepted by --prior-estimator but not listed"


@pytest.mark.unit
def test_deterministic_list_methods_output_stable(capsys):
    """Two invocations produce byte-identical output (deterministic table)."""
    args = build_parser().parse_args(["list-methods"])
    args.func(args)
    first = capsys.readouterr().out
    args = build_parser().parse_args(["list-methods"])
    args.func(args)
    second = capsys.readouterr().out
    assert first == second


@pytest.mark.unit
def test_edge_list_priors_names_all_accepted_by_pipeline(capsys):
    """Every listed name must construct a PUPipeline prior estimator.

    Regression guard: _PRIOR_NAMES used to hard-code registry aliases
    (recpe/pen_l1); if the registry drops an alias, list-priors would
    list a name PUPipeline rejects.  The listing must never outrun the
    accepted set.
    """
    from pu_toolbox.workflows import PUPipeline

    args = build_parser().parse_args(["list-priors"])
    args.func(args)
    names = [line.strip() for line in capsys.readouterr().out.splitlines()[1:] if line.strip()]
    assert names, "list-priors printed nothing"
    for name in names:
        PUPipeline(prior_estimator=name)  # unknown names raise at construction
