"""A single configured numeric-table format preserves source rows and labels."""

import json

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pu_toolbox.preprocessing import PULabelingConfig, construct_pu_dataset, read_dataset

pytestmark = pytest.mark.unit


def _config(**options):
    return {
        "source_id": "caller-table",
        "version": "raw-v1",
        "format": "delimited_numeric",
        "options": options,
        "partitions": {
            "source": {"file": "rows.csv", "feature_columns": [1, 0], "label_column": 2},
        },
    }


def test_basic_named_columns_read_custom_file_without_mapping_labels(tmp_path):
    config = _config(delimiter=";", has_header=True, column_count=4)
    config["partitions"]["source"].update(feature_columns=["second", "first"], label_column="class")
    config_path = tmp_path / "profile.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "rows.csv").write_text(
        '\ufefffirst;second;class;unused\n1.5;-2;"flag;review";A\n3;4;idle;B\n',
        encoding="utf-8",
    )
    source = read_dataset(config_path, tmp_path, partition="source")
    assert_array_equal(source.features, [[-2.0, 1.5], [4.0, 3.0]])
    assert source.features.dtype == np.float64
    assert_array_equal(source.labels, ["flag;review", "idle"])
    assert source.sample_ids.tolist() == [
        "caller-table@raw-v1:source:0",
        "caller-table@raw-v1:source:1",
    ]
    product = construct_pu_dataset(
        source,
        PULabelingConfig("scar", "fixed_count", 5, n_labeled=1),
        positive_classes=("flag;review",),
        negative_classes=("idle",),
    )
    assert_array_equal(product.observations.sample_ids, source.sample_ids)
    assert_array_equal(product.observations.labels, [1, 0])


@pytest.mark.parametrize(
    ("dtype", "raw", "expected"),
    [
        ("str", "001", "001"),
        ("int", f"{np.iinfo(int).max}.0e+00", int(np.iinfo(int).max)),
        ("float", "1.25", 1.25),
    ],
)
def test_param_headerless_tables_use_explicit_label_type(tmp_path, dtype, raw, expected):
    (tmp_path / "rows.csv").write_text(f"10,20,{raw}\n", encoding="utf-8")
    source = read_dataset(_config(label_dtype=dtype), tmp_path, partition="source")
    assert_array_equal(source.features, [[20.0, 10.0]])
    assert source.labels.tolist() == [expected]
    assert type(source.labels.tolist()[0]) is type(expected)


@pytest.mark.parametrize(
    ("text", "options", "label_column", "message"),
    [
        ("1,2,yes\n3,4\n", {}, 2, "columns"),
        ("1,2,1.5\n", {"label_dtype": "int"}, 2, "label"),
        ("1,2,yes\n", {}, 1, "overlap"),
    ],
)
def test_edge_table_contract_refuses_silent_row_or_label_changes(
    tmp_path, text, options, label_column, message
):
    (tmp_path / "rows.csv").write_text(text, encoding="utf-8")
    config = _config(**options)
    config["partitions"]["source"]["label_column"] = label_column
    with pytest.raises(ValueError, match=message):
        read_dataset(config, tmp_path, partition="source")


def test_determ_only_selected_partition_is_read_and_keeps_positional_ids(tmp_path):
    config = _config(delimiter="\t", label_dtype="int")
    config["partitions"]["official_test"] = {
        **config["partitions"]["source"],
        "file": "not_collected.tsv",
    }
    (tmp_path / "rows.csv").write_text("9\t8\t0\n7\t6\t1\n", encoding="utf-8")
    first = read_dataset(config, tmp_path, partition="source")
    second = read_dataset(config, tmp_path, partition="source")
    assert_array_equal(first.sample_ids, second.sample_ids)
    assert_array_equal(first.labels, [0, 1])
    assert_array_equal(first.features, [[8.0, 9.0], [6.0, 7.0]])
    with pytest.raises(FileNotFoundError, match="fetch_dataset"):
        read_dataset(config, tmp_path, partition="official_test")
