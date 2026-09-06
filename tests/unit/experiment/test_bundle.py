# tests/unit/experiment/test_bundle.py
import numpy as np
import pytest

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart, validate_bundle


def part(labels, indices=None, view="clean", for_selection=True):
    n = len(labels)
    return DatasetPart(
        X=np.zeros((n, 3)),
        labels=np.asarray(labels, dtype=int),
        view=view,
        indices=np.arange(n) if indices is None else np.asarray(indices),
        for_selection=for_selection,
    )


@pytest.fixture
def ok_bundle():
    # indices are global identifiers (disjoint across parts); default
    # indices=np.arange(n) would repeat 0 in every part and trip overlap.
    return DatasetBundle(
        train=part([1, 1, 0, 0, 0], view="clean", indices=[0, 1, 2, 3, 4]),
        pu_val=part([1, 0, 0], view="clean", indices=[5, 6, 7]),
        clean_val=part([1, 0, 0, 0], view="clean", indices=[8, 9, 10, 11]),
        test=part([1, 1, 0, 0], view="clean", indices=[12, 13, 14, 15], for_selection=False),
    )


def test_valid_bundle_passes(ok_bundle):
    assert validate_bundle(ok_bundle) is None


def test_input_views_must_be_clean():
    bundle = DatasetBundle(
        train=part([1, 1, 0], view="pu"),
        pu_val=part([1, 0]),
        clean_val=part([1, 0]),
        test=part([1, 0], for_selection=False),
    )
    with pytest.raises(ValueError, match="train must use view='clean'"):
        validate_bundle(bundle)


def test_test_must_be_for_selection_false():
    bundle = DatasetBundle(
        train=part([1, 0]),
        pu_val=part([1, 0]),
        clean_val=part([1, 0]),
        test=part([1, 0]),  # for_selection default True -> illegal
    )
    with pytest.raises(ValueError, match="test must have for_selection=False"):
        validate_bundle(bundle)


def test_overlapping_indices_rejected():
    bundle = DatasetBundle(
        train=part([1, 0, 0], indices=[0, 1, 2]),
        pu_val=part([1, 0], indices=[2, 3]),  # overlaps at 2
        clean_val=part([1, 0], indices=[4, 5]),
        test=part([1, 0], indices=[6, 7], for_selection=False),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_bundle(bundle)


def test_invalid_labels_rejected():
    bundle = DatasetBundle(
        train=part([1, 2, 0]),  # label 2 invalid
        pu_val=part([1, 0]),
        clean_val=part([1, 0]),
        test=part([1, 0], for_selection=False),
    )
    with pytest.raises(ValueError):
        validate_bundle(bundle)
