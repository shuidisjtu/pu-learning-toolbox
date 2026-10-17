"""Train-only refitting and independently bound clean-support products."""

import json

import numpy as np
import pytest

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.datasets import prepare_survey_dataset
from pu_toolbox.experiment.support_data import (
    load_support_dataset,
    prepare_support_dataset,
    save_support_dataset,
)
from pu_toolbox.utils.serialization import strict_canonical_hash

pytestmark = pytest.mark.unit


def arrays(dataset="spambase"):
    rng = np.random.RandomState(3)
    if dataset in ("mnist", "fashion_mnist", "cifar10"):
        channels = 3 if dataset == "cifar10" else 1
        source = rng.randint(0, 256, (400, channels, 28, 28), dtype=np.uint8)
        test = rng.randint(0, 256, (80, channels, 28, 28), dtype=np.uint8)
        return source, np.arange(400) % 10, test, np.arange(80) % 10
    source = rng.normal(size=(500, 57 if dataset == "spambase" else 126))
    truth = np.tile([0, 1], 250)
    if dataset == "connect_4":
        truth = np.where(truth == 1, "win", "loss")
    return source, truth, None, None


def prepare(dataset="spambase", **overrides):
    source, truth, test, test_truth = arrays(dataset)
    options = dict(
        dataset=dataset,
        seed=0,
        support_size=40,
        support_seed=7,
        source="independent offline test raw arrays",
        group_policy="not_applicable",
        X_test=test,
        y_test=test_truth,
    )
    options.update(overrides)
    return prepare_support_dataset(source, truth, **options)


@pytest.mark.parametrize("dataset", ["spambase", "connect_4", "mnist", "fashion_mnist", "cifar10"])
def test_basic_five_datasets_roundtrip_without_frozen_split_impersonation(dataset, tmp_path):
    bundle, support, metadata = prepare(dataset)
    output = tmp_path / dataset
    receipt = save_support_dataset(bundle, support, metadata, output)
    restored, extra, actual = load_support_dataset(output)
    assert actual == receipt
    assert actual["budget_approval"] == "not_granted_by_preparer"
    assert actual["formal_admission"] is False
    assert not (output / "split_manifest.json").exists()
    assert support.for_selection is False and extra.for_selection is False
    assert actual["support_adapter_options"]["support_removed_from_train"] is True
    assert actual["support_adapter_options"]["prior_estimation_indices"] == []
    assert actual["support_source_audit"]["support_origin"] == "carved_from_original_pool"
    assert actual["support_source_audit"]["support_source_verified"] is False
    for name in ("train", "pu_val", "clean_val", "test"):
        left, right = getattr(bundle, name), getattr(restored, name)
        np.testing.assert_array_equal(left.X, right.X)
        np.testing.assert_array_equal(left.indices, right.indices)
    np.testing.assert_array_equal(support.X, extra.X)
    assert len(set(bundle.train.indices) & set(support.indices)) == 0
    assert set(bundle.train.indices) | set(support.indices) == set(
        actual["lineage"]["original_train_indices"]
    )


def test_basic_label_blind_sampling_and_remaining_train_only_statistics():
    source, truth, _, _ = arrays()
    base, _ = prepare_survey_dataset(source, truth, dataset="spambase", seed=0)
    bundle, support, receipt = prepare()
    chosen = np.sort(np.random.RandomState(7).choice(len(base.train.X), 40, replace=False))
    np.testing.assert_array_equal(support.indices, base.train.indices[chosen])
    expected = source[bundle.train.indices]
    np.testing.assert_allclose(receipt["preprocessing"]["feature_mean"], expected.mean(axis=0))
    np.testing.assert_allclose(receipt["preprocessing"]["feature_std"], expected.std(axis=0))
    np.testing.assert_allclose(bundle.train.X.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(bundle.train.X.std(axis=0), 1, atol=1e-6)
    assert receipt["class_prior"]["population"] == 0.5
    for name in ("pu_val", "clean_val", "test"):
        np.testing.assert_array_equal(getattr(bundle, name).indices, getattr(base, name).indices)


def test_determ_support_and_heldout_feature_outliers_never_change_fitted_statistics():
    source, truth, _, _ = arrays()
    bundle, support, receipt = prepare()
    changed = source.copy()
    for ids in (
        support.indices,
        bundle.pu_val.indices,
        bundle.clean_val.indices,
        bundle.test.indices,
    ):
        changed[ids] += 10000
    second, extra, other = prepare_support_dataset(
        changed,
        truth,
        dataset="spambase",
        seed=0,
        support_size=40,
        support_seed=7,
        source="perturbed heldout features",
        group_policy="not_applicable",
    )
    assert other["preprocessing"] == receipt["preprocessing"]
    np.testing.assert_array_equal(second.train.X, bundle.train.X)
    assert not np.array_equal(extra.X, support.X)
    np.testing.assert_array_equal(source, arrays()[0])


def test_determ_image_statistics_ignore_support_and_official_test_truth():
    source, truth, test, test_truth = arrays("mnist")
    first, support, receipt = prepare("mnist")
    changed = source.copy()
    changed[support.indices] = 0
    second, _, metadata = prepare_support_dataset(
        changed,
        truth,
        dataset="mnist",
        seed=0,
        support_size=40,
        support_seed=7,
        source="changed support/test only",
        group_policy="not_applicable",
        X_test=255 - test,
        y_test=(test_truth + 1) % 10,
    )
    assert receipt["preprocessing"] == metadata["preprocessing"]
    np.testing.assert_array_equal(first.train.X, second.train.X)
    assert metadata["class_prior"]["population"] == receipt["class_prior"]["population"]
    assert metadata["preprocessing"]["stored_features"] == "raw_uint8_normalization_not_applied"


@pytest.mark.parametrize(
    "options",
    [
        {"support_size": True},
        {"support_size": 1},
        {"support_size": 359},
        {"support_seed": -1},
        {"support_seed": True},
        {"seed": -1},
        {"group_policy": "grouped"},
        {"source": ""},
    ],
)
def test_param_explicit_budget_seed_and_identity_required(options):
    with pytest.raises(ValueError):
        prepare(**options)


@pytest.mark.parametrize("dataset", ["adni", "imdb", "twenty_newsgroups"])
def test_edge_unreviewed_group_or_text_policy_rejected(dataset):
    with pytest.raises(ValueError, match="reviewed"):
        prepare(dataset=dataset)


def test_edge_single_class_draw_is_refused_without_resampling():
    source, truth, _, _ = arrays()
    base, _ = prepare_survey_dataset(source, truth, dataset="spambase", seed=0)
    seed = next(
        seed
        for seed in range(100)
        if len(
            set(base.train.labels[np.random.RandomState(seed).choice(len(base.train.X), 2, False)])
        )
        == 1
    )
    with pytest.raises(ValueError, match="no label-based retry"):
        prepare(support_seed=seed, support_size=2)


@pytest.mark.parametrize(
    "kind", ["receipt", "file", "missing_role", "filename", "lineage", "seed", "approval", "pool"]
)
def test_edge_corrupted_or_inconsistent_artifacts_refused(tmp_path, kind):
    bundle, support, metadata = prepare()
    output = tmp_path / "product"
    save_support_dataset(bundle, support, metadata, output)
    path = output / "support_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if kind == "file":
        with (output / "support.npz").open("ab") as stream:
            stream.write(b"corrupted")
    else:
        if kind == "receipt":
            payload["seed"] = 99
        elif kind == "missing_role":
            del payload["files_sha256"]["support.npz"]
        elif kind == "filename":
            payload["files_sha256"]["../foreign.npz"] = "a" * 64
        elif kind == "lineage":
            payload["support_adapter_options"]["original_train_count"] += 1
        elif kind == "seed":
            payload["support_adapter_options"]["preparation_seed"] = 9
        elif kind == "pool":
            payload["support_source_audit"]["original_role_counts"]["clean_val"] += 1
        else:
            payload["budget_approval"] = "approved"
        if kind != "receipt":
            payload.pop("receipt_sha256")
            payload["receipt_sha256"] = strict_canonical_hash(payload)
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_support_dataset(output)


def test_edge_repeated_write_and_mutation_are_refused(tmp_path):
    bundle, support, receipt = prepare()
    directory = tmp_path / "product"
    save_support_dataset(bundle, support, receipt, directory)
    with pytest.raises(FileExistsError):
        save_support_dataset(bundle, support, receipt, directory)
    support.X[0, 0] += 1
    with pytest.raises(ValueError, match="disagree"):
        save_support_dataset(bundle, support, receipt, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_basic_loaded_receipt_can_be_resaved_to_a_new_directory(tmp_path):
    bundle, support, metadata = prepare()
    saved = save_support_dataset(bundle, support, metadata, tmp_path / "original")
    loaded, extra, receipt = load_support_dataset(tmp_path / "original")
    copied = save_support_dataset(loaded, extra, receipt, tmp_path / "copy")
    assert copied == saved
    assert load_support_dataset(tmp_path / "copy")[2] == saved


def test_edge_duplicate_ids_cannot_hide_behind_copied_arrays(tmp_path):
    bundle, support, receipt = prepare()
    invalid = DatasetPart(
        support.X.copy(), support.labels.copy(), "clean", bundle.train.indices[:40], False
    )
    with pytest.raises(ValueError, match="overlap"):
        save_support_dataset(
            DatasetBundle(
                *(getattr(bundle, name) for name in ("train", "pu_val", "clean_val", "test"))
            ),
            invalid,
            receipt,
            tmp_path / "invalid",
        )
