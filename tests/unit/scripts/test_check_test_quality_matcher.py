"""Tests for the coverage-category name matcher of the test-quality gate.

The matcher used to be a bare substring search, so ``edge`` matched inside
``ledger`` and ``all_`` inside ``small_`` -- crediting files with boundary
categories they never asserted.  A keyword now matches only at a token
boundary: the start of the name, or just after an underscore.

The rejected cases are the two collision mechanisms measured in the field
(``.superpowers/sdd/probe_category_matcher.py``) plus their nearest
neighbours; the sweep keeps every keyword in ``CATEGORY_KEYWORDS``
reachable, so the tightening cannot silently disable one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_test_quality as gate  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "category"),
    [
        # the naming convention names the category itself
        ("test_basic_fit_and_predict", "basic"),
        ("test_param_invalid_bad_value", "param"),
        ("test_edge_glob_token_skipped", "edge"),
        ("test_determ_same_seed_identical", "determ"),
        # keyword sitting mid-name, after an underscore
        ("test_w_zero_returns_m_hat", "edge"),
        ("test_none_is_not_routed", "edge"),
        ("test_x_generates_a_seed", "determ"),
        # A trailing underscore on the keyword means it matches a whole
        # token: ``all_`` credits ``all_zeros`` and a bare ``all`` token.
        ("test_all_zeros_returns_empty", "edge"),
        ("test_all_candidates_were_filtered", "edge"),
        ("test_edge_all_filtered_returns_empty", "edge"),
        # basic keywords that are not category prefixes
        ("test_class_prior_is_estimated", "basic"),
    ],
)
def test_positive_names_keep_their_category(name, category):
    """A name that genuinely carries the keyword still scores the category."""
    assert category in gate._classify_name(name)


@pytest.mark.parametrize(
    ("name", "category"),
    [
        # 'edge' inside 'ledger' is not a token boundary (field case)
        ("test_load_ledger_reads_methods_mapping", "edge"),
        ("test_shipped_ledger_normalises_for_every_method", "edge"),
        # 'all' inside 'small' / 'smaller' is not a token boundary (field case)
        ("test_max_iter_too_small_raises", "edge"),
        ("test_max_iter_too_smaller_raises", "edge"),
        # `all_` names a whole token, so an 'all'-prefixed word is not a hit
        # (field case: this name used to be read as a permissive boundary)
        ("test_mixed_views_inside_one_dataset_are_allowed", "edge"),
        ("test_allocated_to_the_last_fold", "edge"),
        ("test_overall_rate_is_reported", "edge"),
        # 'score' inside 'underscore' is not a token boundary
        ("test_underscore_notation_is_preserved", "basic"),
        # 'seed' inside 'unseeded' is not a token boundary
        ("test_unseeded_fit_returns_same_model", "determ"),
    ],
)
def test_invalid_substrings_are_rejected(name, category):
    """A keyword embedded inside a longer word must not score."""
    assert category not in gate._classify_name(name)


def test_boundary_position_decides_a_match():
    """A keyword scores only where it opens a token.

    The first two names carry ``zero`` / ``edge`` at a token start; the third
    carries ``edge`` inside ``ledger``, where there is no boundary.
    """
    assert "edge" in gate._classify_name("test_zero_is_at_the_boundary")
    assert "edge" in gate._classify_name("test_edge_case")
    assert "edge" not in gate._classify_name("test_ledger_case")


def test_trailing_underscore_keyword_needs_the_whole_token():
    """``all_`` matches the token ``all``, not any word starting with it."""
    assert "edge" in gate._classify_name("test_all_upper_bounds")
    assert "edge" in gate._classify_name("test_keep_all")
    assert "edge" not in gate._classify_name("test_allocated_fold")
    assert "edge" not in gate._classify_name("test_allowed_views")
    assert "edge" not in gate._classify_name("test_overall_summary")


def test_every_declared_keyword_is_still_reachable():
    """Tightening the matcher must not drop a keyword from every name.

    Derived from the gate's own keyword table (the naming convention's
    authority), not from the matcher: each keyword must score the category
    when it opens the token that follows the ``test_`` prefix.  This is
    what catches a fix that over-tightens (e.g. requiring the keyword to
    equal the whole token, which would orphan ``all_``).
    """
    unreachable = {}
    for category, keywords in gate.CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if category not in gate._classify_name(f"test_{keyword}x"):
                unreachable.setdefault(category, []).append(keyword)
    assert unreachable == {}


def test_repeated_classification_is_consistent():
    """Classification is a pure function of the name."""
    name = "test_edge_max_iter_too_small_raises"
    assert gate._classify_name(name) == gate._classify_name(name)
