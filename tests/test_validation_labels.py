# ruff: noqa: N806, E501

"""Label-meaning validation: ``validate_label_semantics`` and ``validate_true_binary_labels``.

Split from ``test_validation.py`` along the question each answers.  That module
asks whether a PU/PnU dataset is *shaped* correctly (dimensions, class presence,
array type); this one asks whether the labels *mean* what the caller claims --
an estimator declaring the wrong semantics, or a "true label" vector that is not
binary.  A caller can satisfy either contract while violating the other.
"""

import numpy as np
import pytest

from pu_toolbox.core.exceptions import ValidationError
from pu_toolbox.core.validation import (
    validate_label_semantics,
    validate_true_binary_labels,
)


@pytest.mark.unit
def test_label_semantics_checks_meaning_not_numeric_values():
    class PUModel:
        label_semantics = "pu"

    class PNModel:
        label_semantics = "pn"

    validate_label_semantics(PUModel(), "pu")
    validate_label_semantics(PNModel(), "pn")
    validate_label_semantics(object(), "pu")  # third-party PU default
    with pytest.raises(ValueError, match="requires 'pn'"):
        validate_label_semantics(PUModel(), "pn")
    with pytest.raises(ValueError, match="requires 'pu'"):
        validate_label_semantics(PNModel(), "pu")
    with pytest.raises(ValueError, match="invalid label_semantics"):
        validate_label_semantics(type("Bad", (), {"label_semantics": []})(), "pu")
    with pytest.raises(ValueError, match="expected label semantics"):
        validate_label_semantics(PUModel(), "unknown")


@pytest.mark.unit
class TestValidateTrueBinaryLabels:
    """Unit tests for validate_true_binary_labels — basic, errors, edge."""

    # ── Basic / happy path ────────────────────────────────────────────
    def test_basic_accepts_binary_labels(self):
        """int, float, list, and single-class inputs with values in {0, 1} pass.

        Repeated validation is deterministic and side-effect free: the
        input array is never mutated.
        """
        validate_true_binary_labels(np.array([1, 0, 1, 0]))
        validate_true_binary_labels(np.array([0.0, 1.0]))
        validate_true_binary_labels([1, 0, 0])
        validate_true_binary_labels(np.array([0.0, 1.0], dtype=np.float32))
        validate_true_binary_labels(np.zeros(5, dtype=int))
        validate_true_binary_labels(np.ones(3, dtype=int))
        y_true = np.array([1, 0, 1, 0, 1])
        validate_true_binary_labels(y_true)
        validate_true_binary_labels(y_true)
        np.testing.assert_array_equal(y_true, np.array([1, 0, 1, 0, 1]))

    # ── Error cases ───────────────────────────────────────────────────
    @pytest.mark.parametrize(
        "y_true, match",
        [
            (np.array([1, 0, 2]), "only binary labels"),
            (np.array([1, 0, -1]), "got unique values"),
            (np.array([0.0, 1.0, np.nan]), "only binary labels"),
            (np.array(["0", "1"]), "only binary labels"),
        ],
    )
    def test_invalid_values_raise(self, y_true, match):
        with pytest.raises(ValueError, match=match):
            validate_true_binary_labels(y_true)

    def test_invalid_estimator_name_in_message(self):
        with pytest.raises(ValueError, match="my_labels must contain only binary labels"):
            validate_true_binary_labels(np.array([0, 2]), estimator_name="my_labels")

    def test_invalid_non_1d_raises(self):
        with pytest.raises(ValidationError, match="1-D"):
            validate_true_binary_labels(np.array([[1, 0], [0, 1]]))

    # ── Edge cases ────────────────────────────────────────────────────
    def test_edge_empty_array_passes(self):
        """An empty 1-D array has no out-of-domain values."""
        validate_true_binary_labels(np.array([]))
