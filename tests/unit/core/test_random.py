"""Unit tests for the seed-normalisation helper.

Scope note (batch E5): ``check_random_state`` serves the ``preprocessing`` and
``experiment`` layers, whose ``random_state`` parameters are declared
``int | np.random.RandomState | None`` and where passing an already-built
``RandomState`` downstream is a load-bearing idiom -- it continues the stream.
Estimators declare ``random_state: int | None`` and construct
``np.random.RandomState(self.random_state)`` inline on purpose; see
``docs/dev/single_source_map.md`` (随机源) for why the two layers keep two
contracts.  Do not "fix" the estimators to call this helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.random import check_random_state

pytestmark = pytest.mark.unit


def test_basic_int_seed_returns_random_state():
    rng = check_random_state(42)
    assert isinstance(rng, np.random.RandomState)


def test_basic_none_seed_is_accepted():
    rng = check_random_state(None)
    assert isinstance(rng, np.random.RandomState)


def test_param_non_seed_type_raises_type_error():
    for bad in ("x", 0.5, np.random.default_rng(0), [42], np.array(42)):
        with pytest.raises(TypeError, match="must be int, RandomState, or None"):
            check_random_state(bad)


def test_param_bool_is_accepted_as_int():
    # bool is an int subclass; both this helper and the inline construction
    # accept it, so it is not part of the two-way behaviour difference.
    rng = check_random_state(True)
    assert isinstance(rng, np.random.RandomState)


def test_edge_instance_is_returned_unchanged():
    given = np.random.RandomState(42)
    assert check_random_state(given) is given


def test_edge_array_like_int_is_rejected_here_but_accepted_inline():
    # The two-way difference: np.random.RandomState(v) accepts int array-likes,
    # this helper does not.  Pinned so the difference stays deliberate.
    assert isinstance(np.random.RandomState(np.array(42)), np.random.RandomState)
    assert isinstance(np.random.RandomState([42]), np.random.RandomState)
    for array_like in (np.array(42), [42], np.array([42])):
        with pytest.raises(TypeError):
            check_random_state(array_like)


def test_determ_same_int_seed_gives_identical_draws():
    first = check_random_state(7).randint(0, 1000, 8)
    second = check_random_state(7).randint(0, 1000, 8)
    assert np.array_equal(first, second)


def test_determ_instance_passed_downstream_continues_the_stream():
    # Mirrors the load-bearing idiom in preprocessing/pu_labeling.py:400-408:
    # a caller builds an rng, hands it down, and the callee keeps drawing from
    # the SAME stream rather than restarting it.
    caller_rng = np.random.RandomState(0)
    handed_down = check_random_state(caller_rng)
    drawn_downstream = handed_down.randint(0, 1000, 4)
    # The caller's own next draws differ: the stream advanced, it did not restart.
    assert not np.array_equal(drawn_downstream, caller_rng.randint(0, 1000, 4))
    # And the draws are exactly "the first four of seed 0", i.e. the same stream.
    continued = np.random.RandomState(0)
    assert np.array_equal(drawn_downstream, continued.randint(0, 1000, 4))
