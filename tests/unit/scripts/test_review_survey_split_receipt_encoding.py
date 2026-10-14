"""Locked source encoding is checked for every role, not only training."""

import importlib

import numpy as np
import pytest
from _review_split_receipt_helpers import write_split

pytestmark = pytest.mark.unit


def corrupt(base, role, transform):
    path = base / f"{role}.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["X"] = transform(arrays["X"])
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize(
    "dataset,defect",
    [
        ("mnist", "dtype"),
        ("fashion_mnist", "dtype"),
        ("cifar10", "dtype"),
        ("adni", "dtype"),
        ("mnist", "channels"),
        ("fashion_mnist", "height"),
        ("cifar10", "width"),
        ("adni", "height"),
    ],
)
def test_param_edge_image_encoding_rejected_on_test_role(tmp_path, dataset, defect):
    base, _ = write_split(tmp_path, dataset)

    def transform(values):
        if defect == "dtype":
            return values.astype(np.float32)
        if defect == "channels":
            return np.repeat(values, 2, axis=1)
        if defect == "height":
            return values[:, :, :-1, :]
        return values[:, :, :, :-1]

    corrupt(base, "test", transform)
    with pytest.raises(AssertionError):
        importlib.import_module("scripts.review_survey_split_receipt").audit_split(
            base, dataset=dataset, seed=0
        )


@pytest.mark.parametrize("role", ["train", "pu_val", "clean_val"])
def test_param_edge_image_dtype_rejected_on_development_roles(tmp_path, role):
    base, _ = write_split(tmp_path, "mnist")
    corrupt(base, role, lambda values: values.astype(np.float32))
    with pytest.raises(AssertionError):
        importlib.import_module("scripts.review_survey_split_receipt").audit_split(
            base, dataset="mnist", seed=0
        )


@pytest.mark.parametrize("dataset", ["spambase", "connect_4"])
def test_param_edge_tabular_feature_count_rejected_on_validation(tmp_path, dataset):
    base, _ = write_split(tmp_path, dataset)
    corrupt(base, "pu_val", lambda values: values[:, :-1])
    with pytest.raises(AssertionError):
        importlib.import_module("scripts.review_survey_split_receipt").audit_split(
            base, dataset=dataset, seed=0
        )
