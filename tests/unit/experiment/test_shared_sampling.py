"""Shared fixed-count policies preserve labels, audit fields and random streams."""

import numpy as np
import pytest

from pu_toolbox.experiment.strategies import SCARGenerator, scar_realized_rate
from pu_toolbox.preprocessing import make_fixed_count_labels, make_scar_labels
from pu_toolbox.utils.serialization import canonical_hash

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "y_true,c,expected_count",
    [
        ([1, 0, 1, 0, 1, 0, 1, 0, 1], 0.5, 2),
        ([1, 1, 1, 1, 1, 1, 1], 0.5, 4),
        ([0, 1, 0, 1], 0.01, 1),
        ([1, 0, 1, 1, 0], 1.0, 3),
        ([0, 0, 0], 0.5, 0),
        ([], 0.5, 0),
        ([1], 0.01, 1),
    ],
)
@pytest.mark.parametrize("seed", [0, 19, 31])
def test_param_basic_generator_matches_public_labels_and_audit_fields(
    y_true, c, expected_count, seed
):
    truth = np.asarray(y_true, dtype=int)
    before = truth.copy()
    expected = make_scar_labels(truth, c, random_state=seed)
    actual, metadata = SCARGenerator().generate(np.zeros((len(truth), 2)), truth, c, seed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(truth, before)
    assert actual.dtype == int
    assert int(actual.sum()) == expected_count
    assert np.all(truth[actual == 1] == 1)
    if expected_count:
        np.testing.assert_array_equal(
            actual, make_fixed_count_labels(truth, expected_count, random_state=seed)
        )
    assert metadata["n_labeled"] == expected_count
    assert metadata["n_labeled_requested"] == int(np.round(c * truth.sum()))
    assert metadata["c_realized"] == scar_realized_rate(truth, c)
    assert metadata["generation_seed"] == seed
    assert metadata["label_view_sha256"] == canonical_hash({"y_pu": actual.tolist()})


@pytest.mark.parametrize("c,indices", [(0.5, [2, 5, 8]), (1.0, [0, 2, 3, 5, 8, 9])])
def test_determ_edge_full_selection_preserves_legacy_random_stream(c, indices):
    truth = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    for entry in ("scar", "fixed", "generator"):
        rng = np.random.RandomState(19)
        if entry == "scar":
            labels = make_scar_labels(truth, c, random_state=rng)
        elif entry == "fixed":
            labels = make_fixed_count_labels(truth, len(indices), random_state=rng)
        else:
            labels, _ = SCARGenerator().generate(np.zeros((len(truth), 2)), truth, c, rng)
        np.testing.assert_array_equal(np.flatnonzero(labels), indices)
        np.testing.assert_array_equal(rng.randint(0, 10000, size=5), [834, 3973, 266, 2279, 4048])
