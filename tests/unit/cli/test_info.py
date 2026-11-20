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
    assert "Required prior inputs" in out
    assert "training_table_prior" in next(
        line for line in out.splitlines() if line.startswith("pusb_kernel ")
    )
    assert "unlabeled_positive_prior" in next(
        line for line in out.splitlines() if line.startswith("cvir ")
    )
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
    """Registry aliases appear in the listing."""
    args = build_parser().parse_args(["list-priors"])
    args.func(args)
    out = capsys.readouterr().out
    for alias in ("cpe", "pe", "re_cpe", "rethinking_cpe"):
        assert alias in out, f"{alias!r} not listed"
    assert "pe [deprecated]" in out


@pytest.mark.unit
def test_param_list_priors_groups_aliases_with_canonical(capsys):
    """Aliases appear on the same line as their canonical estimator name."""
    args = build_parser().parse_args(["list-priors"])
    args.func(args)
    lines = [line for line in capsys.readouterr().out.splitlines()[1:] if line.strip()]
    by_line = {line.split()[0]: line for line in lines}
    assert "cpe" in by_line["class_prior_estimation"]
    assert "pen_l1" in by_line["class_prior_estimation"]
    assert "re_cpe" in by_line["recpe"]
    # km1/km2 are standalone names, not registry aliases.
    assert "aliases" not in by_line["km1"]
    assert "aliases" not in by_line["km2"]


@pytest.mark.unit
def test_list_methods_shows_capability_columns(capsys):
    """Input and Arch columns reflect the capability declarations."""
    args = build_parser().parse_args(["list-methods"])
    args.func(args)
    out = capsys.readouterr().out
    assert "Input" in out and "Arch" in out
    lines = [line for line in out.splitlines()[2:] if line.strip()]
    by_name = {line.split()[0]: line.split() for line in lines}
    assert "2,4" in by_name["infomax_pu"] and "mlp,cnn" in by_name["infomax_pu"]
    assert "2,4" in by_name["nnpu"] and "mlp,cnn" in by_name["nnpu"]
    assert "tabular" in by_name["elkan_noto"]


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
