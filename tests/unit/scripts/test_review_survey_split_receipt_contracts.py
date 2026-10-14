"""Semantic corruption fails closed even with readable split products."""

import importlib
import json

import numpy as np
import pytest
from _review_split_receipt_helpers import write_split

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "defect",
    [
        "dataset",
        "seed",
        "modality",
        "test_source",
        "size",
        "indices",
        "rate",
        "population",
        "train_prior",
        "provenance",
        "normalization",
    ],
)
def test_param_edge_manifest_contract_corruption(tmp_path, defect):
    base, manifest = write_split(tmp_path, "mnist")
    if defect in ("dataset", "modality", "test_source"):
        manifest[defect] = "wrong"
    elif defect == "seed":
        manifest["seed"] = 99
    elif defect == "size":
        manifest["role_sizes"]["train"] += 1
    elif defect == "indices":
        manifest["indices"]["train"][0] = 999
    elif defect == "rate":
        manifest["role_positive_rates"]["test"] = 0
    elif defect in ("population", "train_prior"):
        manifest["class_prior"]["train" if defect == "train_prior" else "population"] = 0
    elif defect == "provenance":
        manifest["provenance"]["status"] = "missing"
    else:
        manifest["preprocessing"]["normalization"]["mean"] = [0, 0]
    (base / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssertionError):
        importlib.import_module("scripts.review_survey_split_receipt").audit_split(
            base, dataset="mnist", seed=0
        )


def test_edge_tabular_nonstandardized_feature_is_not_constant_exception(tmp_path):
    base, _ = write_split(tmp_path, "spambase")
    path = base / "train.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["X"] *= 0.5
    np.savez_compressed(path, **arrays)
    with pytest.raises(AssertionError):
        importlib.import_module("scripts.review_survey_split_receipt").audit_split(
            base, dataset="spambase", seed=0
        )


def test_edge_bundle_overlap_rejected(tmp_path):
    base, manifest = write_split(tmp_path, "imdb")
    path = base / "test.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["indices"][0] = 0
    manifest["indices"]["test"][0] = 0
    np.savez_compressed(path, **arrays)
    (base / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        importlib.import_module("scripts.review_survey_split_receipt").audit_split(
            base, dataset="imdb", seed=0
        )
