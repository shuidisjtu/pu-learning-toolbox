"""Offline modality and source-pool semantics for receipt audits."""

import importlib
import json

import numpy as np
import pytest
from _review_split_receipt_helpers import write_split

pytestmark = pytest.mark.unit


def audit(base, dataset, seed=0):
    return importlib.import_module("scripts.review_survey_split_receipt").audit_split(
        base, dataset=dataset, seed=seed
    )


@pytest.mark.parametrize("dataset", ["mnist", "fashion_mnist", "cifar10", "imdb"])
def test_param_official_population_excludes_test(tmp_path, dataset):
    base, manifest = write_split(tmp_path, dataset)
    result = audit(base, dataset)
    assert result["population_prior"] == manifest["class_prior"]["population"] == 0.5
    assert result["roles"]["test"]["positive_rate"] == 1
    assert result["bundle_contract"] == "passed"


@pytest.mark.parametrize("dataset", ["adni", "spambase", "connect_4"])
def test_param_derived_population_includes_test(tmp_path, dataset):
    base, manifest = write_split(tmp_path, dataset)
    path = base / "test.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["y"] = np.ones(4, dtype=int)
    np.savez_compressed(path, **arrays)
    manifest["role_positive_rates"]["test"] = 1.0
    manifest["class_prior"]["population"] = 10 / 16
    (base / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert audit(base, dataset)["population_prior"] == 10 / 16


@pytest.mark.parametrize("dataset", ["mnist", "adni", "cifar10"])
def test_param_image_statistics_use_actual_channels_and_spatial_shape(tmp_path, dataset):
    base, manifest = write_split(tmp_path, dataset)
    result = audit(base, dataset)
    reference = manifest["preprocessing"]["normalization"]
    assert result["channel_mean"] == pytest.approx(reference["mean"], abs=1e-7)
    assert result["channel_std"] == pytest.approx(reference["std"], abs=1e-7)


def test_edge_tabular_constant_columns_have_zero_std(tmp_path):
    base, _ = write_split(tmp_path, "spambase", constant=True)
    result = audit(base, "spambase")
    assert result["train_std_max_error"] == 0
    assert result["train_mean_max_abs"] == 0


@pytest.mark.parametrize("defect", ["dimension", "norm", "finite"])
def test_param_text_preserves_384_normalized_finite_contract(tmp_path, defect):
    base, _ = write_split(tmp_path, "imdb")
    path = base / "train.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    if defect == "dimension":
        arrays["X"] = arrays["X"][:, :-1]
    elif defect == "norm":
        arrays["X"] *= 2
    else:
        arrays["X"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    with pytest.raises(AssertionError):
        audit(base, "imdb")
