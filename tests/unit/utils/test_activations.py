# tests/unit/utils/test_activations.py

# ruff: noqa: N803, S101

"""``sigmoid_stable`` must stay finite whatever dtype its caller passes.

The survey splits carry float32 features, and float32 ``exp`` saturates some
620 decimal orders earlier than float64 does: a clip bound picked for float64
(``exp(500)`` is a finite 1.4e217 there) still overflows to ``inf`` on the
float32 path, where numpy reports it and the sigmoid is only saved by
``1/(1+inf)`` happening to equal the right limit.
"""

import warnings

import numpy as np
import pytest

from pu_toolbox.utils.activations import sigmoid_stable

pytestmark = pytest.mark.unit

#: Far beyond any logit a fitted model produces, and past float32's exp range.
EXTREME = 1e4


class TestBasic:
    def test_basic_matches_the_direct_formula_on_moderate_input(self):
        z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert sigmoid_stable(z) == pytest.approx(1.0 / (1.0 + np.exp(-z)))

    def test_basic_is_symmetric_about_zero(self):
        z = np.array([0.3, 1.7, 12.0])
        assert sigmoid_stable(z) + sigmoid_stable(-z) == pytest.approx(np.ones(3))


class TestDtype:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_param_extremes_do_not_warn(self, dtype):
        """The regression: on float32, ``exp(+500)`` is ``inf`` and numpy says so."""
        z = np.array([-EXTREME, EXTREME], dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sigmoid_stable(z)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_param_extremes_saturate_to_the_bounds(self, dtype):
        out = sigmoid_stable(np.array([-EXTREME, EXTREME], dtype=dtype))
        assert out[0] == 0.0
        assert out[1] == 1.0

    def test_basic_keeps_the_input_dtype(self):
        assert sigmoid_stable(np.zeros(3, dtype=np.float32)).dtype == np.float32

    def test_edge_integer_input_returns_floats(self):
        """An integer array must not be written back through an integer buffer."""
        out = sigmoid_stable(np.array([-1, 0, 1]))
        assert out.dtype.kind == "f"
        assert out[0] == pytest.approx(0.2689414213699951)


class TestDeterminism:
    def test_determ_is_elementwise(self):
        """A value's result must not depend on which batch it travelled in."""
        values = np.array([-3.0, -0.1, 0.0, 0.1, 3.0])
        batch = sigmoid_stable(values)
        for index, value in enumerate(values):
            assert sigmoid_stable(np.array([value]))[0] == pytest.approx(batch[index])

    def test_determ_does_not_jump_at_the_branch_point(self):
        """Both evaluation branches meet at 0; a mismatch shows up as a step."""
        left = sigmoid_stable(np.array([-1e-8]))[0]
        at_zero = sigmoid_stable(np.array([0.0]))[0]
        right = sigmoid_stable(np.array([1e-8]))[0]
        assert left < at_zero < right
        assert right - left < 1e-7
