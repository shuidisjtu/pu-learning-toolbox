"""Shared fixed-count policies preserve labels, audit fields and random streams."""

import numpy as np
import pytest

from pu_toolbox.experiment.strategies import SCARGenerator, scar_realized_rate
from pu_toolbox.preprocessing import PULabelingConfig, SourcePartition, construct_pu_dataset
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
    actual, metadata = SCARGenerator().generate(np.zeros((len(truth), 2)), truth, c, seed)
    if len(truth):
        expected = construct_pu_dataset(
            SourcePartition(np.zeros((len(truth), 2)), truth, "fixture", "train"),
            PULabelingConfig("scar", "fixed_count", seed, c=c),
            positive_classes=(1,),
            negative_classes=(0,),
        ).observations.labels
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(truth, before)
    assert actual.dtype == int
    assert int(actual.sum()) == expected_count
    assert np.all(truth[actual == 1] == 1)
    assert metadata["n_labeled"] == expected_count
    assert metadata["n_labeled_requested"] == int(np.round(c * truth.sum()))
    assert metadata["c_realized"] == scar_realized_rate(truth, c)
    assert metadata["generation_seed"] == seed
    assert metadata["label_view_sha256"] == canonical_hash({"y_pu": actual.tolist()})


@pytest.mark.parametrize("c,indices", [(0.5, [2, 5, 8]), (1.0, [0, 2, 3, 5, 8, 9])])
def test_determ_edge_full_selection_preserves_legacy_random_stream(c, indices):
    truth = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    for entry in ("rate", "quota", "generator"):
        rng = np.random.RandomState(19)
        if entry == "generator":
            labels, _ = SCARGenerator().generate(np.zeros((len(truth), 2)), truth, c, rng)
        else:
            budget = {"c": c} if entry == "rate" else {"n_labeled": len(indices)}
            labels = construct_pu_dataset(
                SourcePartition(np.zeros((len(truth), 2)), truth, "fixture", "train"),
                PULabelingConfig("scar", "fixed_count", rng, **budget),
                positive_classes=(1,),
                negative_classes=(0,),
            ).observations.labels
        np.testing.assert_array_equal(np.flatnonzero(labels), indices)
        np.testing.assert_array_equal(rng.randint(0, 10000, size=5), [834, 3973, 266, 2279, 4048])
