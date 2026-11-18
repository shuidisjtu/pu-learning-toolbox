"""Regression tests for retirement of the recommendation command."""

import pytest

from pu_toolbox.cli import build_parser

pytestmark = pytest.mark.unit


def test_edge_recommend_is_not_registered():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["recommend"])
