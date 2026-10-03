# tests/unit/utils/test_json_scalars.py

"""``json_scalars`` is the strict half of the manifest serialisation contract.

The lists it returns are hashed into survey artifacts (``indices_sha256``,
``split_sha256``, the adapter cache key), so a value it lets through becomes
part of an artifact's identity and a value it drops changes one.  These tests
pin that contract rather than the implementation: the accepted set, the exact
refusal messages, and the fact that the caller's array is never written to.
"""

import numpy as np
import pytest

from pu_toolbox.utils.serialization import json_scalars

pytestmark = pytest.mark.unit


def test_basic_converts_numpy_arrays_to_plain_json_scalars():
    """Every element leaves as a Python scalar, never a numpy one."""
    integers = json_scalars(np.array([0, 1, 2]), name="indices")
    assert integers == [0, 1, 2]
    assert [type(value) for value in integers] == [int, int, int]

    assert json_scalars(np.array([1.5, -2.0]), name="indices") == [1.5, -2.0]
    assert json_scalars(np.array(["a", "b"]), name="indices") == ["a", "b"]
    assert json_scalars(np.array([True, False]), name="indices") == [True, False]


def test_param_refusals_carry_the_callers_name_verbatim():
    """The two messages keep the exact wording the two predecessors used."""
    boxed = np.empty(1, dtype=object)
    boxed[0] = (2, 3)
    with pytest.raises(ValueError) as non_scalar:
        json_scalars(boxed, name="bundle indices")
    assert str(non_scalar.value) == "bundle indices must contain JSON scalar values."

    for not_finite in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError) as non_finite:
            json_scalars(np.array([not_finite]), name="train indices")
        assert str(non_finite.value) == "floating-point train indices must be finite."


def test_edge_object_dtype_is_unboxed_and_empty_and_nested_are_handled():
    """``tolist`` leaves object arrays alone, so the numpy branch must run."""
    boxed = np.empty(3, dtype=object)
    boxed[:] = [np.int64(4), np.float64(5.5), "six"]
    scalars = json_scalars(boxed, name="indices")
    assert scalars == [4, 5.5, "six"]
    assert [type(value) for value in scalars] == [int, float, str]

    mixed = np.empty(2, dtype=object)
    mixed[:] = [None, True]
    assert json_scalars(mixed, name="indices") == [None, True]

    assert json_scalars(np.array([], dtype=np.int64), name="indices") == []

    with pytest.raises(ValueError, match="must contain JSON scalar values"):
        json_scalars(np.array([[1, 2], [3, 4]]), name="indices")


def test_determ_is_pure_and_leaves_the_callers_array_untouched():
    """Two calls agree, and neither writes to the array it was handed."""
    values = np.array([3, 1, 4])
    original = values.copy()

    first = json_scalars(values, name="indices")

    assert json_scalars(values, name="indices") == first
    assert np.array_equal(values, original)
