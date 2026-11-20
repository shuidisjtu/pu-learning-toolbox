"""Config-driven local reading is a format contract, not a dataset dispatcher."""

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pu_toolbox.preprocessing import (
    PULabelingConfig,
    SourcePartition,
    construct_pu_dataset,
    read_dataset,
)

pytestmark = pytest.mark.unit


def _idx_config(source_id="local-example", *, compressed=False):
    suffix = ".gz" if compressed else ""
    return {
        "source_id": source_id,
        "version": "v1",
        "format": "idx_images",
        "options": {"image_shape": [2, 3], "label_values": list(range(10))},
        "partitions": {
            pool: {
                "features": f"{pool}-images{suffix}",
                "labels": f"{pool}-labels{suffix}",
                "max_rows": 100,
            }
            for pool in ("train", "test")
        },
    }


def _npz_config():
    return {
        "source_id": "local-example",
        "version": "v1",
        "format": "npz_arrays",
        "options": {
            "image_shape": [2, 3],
            "archive_keys": ["train_x", "train_y", "test_x", "test_y"],
        },
        "partitions": {
            pool: {"file": "snapshot.npz", "features_key": f"{pool}_x", "labels_key": f"{pool}_y"}
            for pool in ("train", "test")
        },
    }


def _idx_pair(root, partition, features, labels):
    images = struct.pack(">4I", 2051, len(features), *features.shape[1:]) + features.tobytes()
    targets = struct.pack(">2I", 2049, len(labels)) + labels.tobytes()
    for kind, contents in (("features", images), ("labels", targets)):
        path = root / partition[kind]
        path.write_bytes(gzip.compress(contents) if path.suffix == ".gz" else contents)


def _sample(n=10, shape=(2, 3)):
    features = np.arange(n * int(np.prod(shape)), dtype=np.uint8).reshape(n, *shape)
    return features, np.arange(n, dtype=np.uint8)


@pytest.mark.parametrize("compressed", [True, False])
@pytest.mark.parametrize("partition", ["train", "test"])
def test_param_reads_only_the_requested_local_idx_pair(tmp_path, partition, compressed):
    config = _idx_config(compressed=compressed)
    features, labels = _sample()
    _idx_pair(tmp_path, config["partitions"][partition], features, labels)
    source = read_dataset(config, tmp_path, partition=partition)
    assert isinstance(source, SourcePartition)
    assert source.source_id == "local-example@v1" and source.partition == partition
    assert source.features.dtype == np.uint8
    assert_array_equal(source.features, features[:, None])
    assert_array_equal(source.labels, labels)
    assert source.sample_ids.tolist() == [f"local-example@v1:{partition}:{i}" for i in range(10)]


def test_basic_two_builtin_profiles_use_the_same_idx_format(tmp_path):
    root = Path(__file__).resolve().parents[3] / "pu_toolbox/preprocessing/source_configs"
    features, labels = _sample(shape=(28, 28))
    ids = set()
    shapes = []
    for name in ("mnist", "fashion_mnist"):
        config_path = root / f"{name}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["format"] == "idx_images"
        folder = tmp_path / name
        folder.mkdir()
        _idx_pair(folder, config["partitions"]["train"], features, labels)
        source = read_dataset(config_path, folder, partition="train")
        shapes.append(source.features.shape)
        assert_array_equal(source.labels, labels)
        assert ids.isdisjoint(source.sample_ids)
        ids.update(source.sample_ids)
    assert shapes == [(10, 1, 28, 28)] * 2


def test_basic_a_reader_and_custom_arrays_use_the_same_pu_contract(tmp_path):
    features, labels = _sample()
    config = _idx_config()
    _idx_pair(tmp_path, config["partitions"]["train"], features, labels)
    builtin = read_dataset(config, tmp_path, partition="train")
    custom = SourcePartition(features[:, None], labels, "caller-v1", "chosen")
    labeling = PULabelingConfig("scar", "fixed_count", 7, c=0.5)
    maps = [((0, 2, 4, 6, 8), (1, 3, 5, 7, 9)), ((3,), (0, 1, 2, 4, 5, 6, 7, 8, 9))]
    labels_by_map = []
    for pos, neg in maps:
        args = dict(positive_classes=pos, negative_classes=neg)
        product = construct_pu_dataset(builtin, labeling, **args)
        expected = construct_pu_dataset(custom, labeling, **args)
        assert_array_equal(product.observations.features, builtin.features)
        assert_array_equal(product.observations.sample_ids, builtin.sample_ids)
        assert_array_equal(product.observations.labels, expected.observations.labels)
        assert_array_equal(product.audit.true_labels, np.isin(labels, pos))
        labels_by_map.append(product.observations.labels)
    assert not np.array_equal(*labels_by_map)
    assert_array_equal(builtin.labels, labels)


def test_basic_npz_without_image_options_retains_tabular_shapes_and_labels(tmp_path):
    features = np.arange(6, dtype=np.float32).reshape(3, 2)
    labels = np.array(["low", "high", "medium"])
    np.savez(tmp_path / "table.npz", input=features, classes=labels)
    config = {
        "source_id": "caller-table",
        "version": "v1",
        "format": "npz_arrays",
        "partitions": {
            "published": {"file": "table.npz", "features_key": "input", "labels_key": "classes"}
        },
    }
    source = read_dataset(config, tmp_path, partition="published")
    assert_array_equal(source.features, features)
    assert source.features.dtype == features.dtype
    assert_array_equal(source.labels, labels)
    assert source.sample_ids[0] == "caller-table@v1:published:0"


def test_determ_row_ids_match_between_idx_and_array_snapshot(tmp_path):
    features, labels = _sample()
    idx = _idx_config(compressed=True)
    for part in idx["partitions"].values():
        _idx_pair(tmp_path, part, features, labels)
    np.savez(
        tmp_path / "snapshot.npz", train_x=features, train_y=labels, test_x=features, test_y=labels
    )
    train = read_dataset(idx, tmp_path, partition="train")
    test = read_dataset(idx, tmp_path, partition="test")
    assert set(train.sample_ids).isdisjoint(test.sample_ids)
    for from_idx in (train, test):
        from_snapshot = read_dataset(_npz_config(), tmp_path, partition=from_idx.partition)
        assert_array_equal(from_idx.sample_ids, from_snapshot.sample_ids)
        assert_array_equal(from_idx.features, from_snapshot.features)
        assert_array_equal(from_idx.labels, from_snapshot.labels)
    changed = {**idx, "version": "v2"}
    other_version = read_dataset(changed, tmp_path, partition="train")
    assert set(train.sample_ids).isdisjoint(other_version.sample_ids)


def test_edge_a_snapshot_only_loads_arrays_for_the_requested_pool(tmp_path):
    features, labels = _sample()
    np.savez(
        tmp_path / "snapshot.npz",
        train_x=features,
        train_y=labels,
        test_x=np.array([object()], dtype=object),
        test_y=labels,
    )
    train = read_dataset(_npz_config(), tmp_path, partition="train")
    assert_array_equal(train.labels, labels)
    with pytest.raises(ValueError, match="allow_pickle"):
        read_dataset(_npz_config(), tmp_path, partition="test")


def test_edge_missing_inputs_point_to_explicit_acquisition(tmp_path):
    for config in (_idx_config(), _npz_config()):
        with pytest.raises(FileNotFoundError, match="(?i)fetch_dataset|download"):
            read_dataset(config, tmp_path, partition="test")
    with pytest.raises(ValueError, match="partition"):
        read_dataset(_idx_config(), tmp_path, partition="validation")
    with pytest.raises(ValueError, match="format"):
        read_dataset({**_idx_config(), "format": "not-a-plugin"}, tmp_path, partition="train")


@pytest.mark.parametrize("relative", ["../outside", "/outside", "C:/outside", "a\\outside", "NUL"])
def test_param_root_escape_is_refused_before_reading(tmp_path, relative):
    config = _idx_config()
    config["partitions"]["train"]["features"] = relative
    with pytest.raises(ValueError, match="path"):
        read_dataset(config, tmp_path, partition="train")


@pytest.mark.parametrize(
    ("header", "data", "match"),
    [
        (struct.pack(">4I", 2049, 1, 2, 3), bytes(6), "magic"),
        (struct.pack(">4I", 2051, 1, 3, 2), bytes(6), "shape"),
        (struct.pack(">4I", 2051, 1, 2, 3), bytes(5), "payload"),
        (struct.pack(">4I", 2051, 1, 2, 3), bytes(7), "payload"),
        (struct.pack(">4I", 2051, 2**32 - 1, 2, 3), b"", "count"),
        (b"\x00", b"", "header"),
    ],
)
def test_param_malformed_idx_payloads_are_refused(tmp_path, header, data, match):
    config = _idx_config()
    part = config["partitions"]["train"]
    features, labels = _sample(1)
    _idx_pair(tmp_path, part, features, labels)
    (tmp_path / part["features"]).write_bytes(header + data)
    with pytest.raises(ValueError, match=match):
        read_dataset(config, tmp_path, partition="train")
