# ruff: noqa: S101

"""Dependency-light tests for UI data and configuration helpers."""

import inspect
import io

import numpy as np
import pandas as pd
import pytest

from pu_toolbox.ui import (
    classifier_catalog,
    load_feature_data,
    load_label_data,
    parameter_schema,
    parse_json_mapping,
)

pytestmark = pytest.mark.unit


def _csv_bytes(frame):
    return frame.to_csv(index=False).encode()


def test_basic_load_feature_and_label_csv():
    features, names = load_feature_data(
        _csv_bytes(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})),
        "X.csv",
    )
    labels = load_label_data(_csv_bytes(pd.DataFrame({"label": [1, 0]})))
    assert features.shape == (2, 2)
    assert names == ["a", "b"]
    assert labels.tolist() == [1, 0]


def test_param_load_npy_image_array():
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((2, 3, 4, 4), dtype=np.float32))
    images, _ = load_feature_data(buffer.getvalue(), "images.npy")
    assert images.shape == (2, 3, 4, 4)


def test_edge_reject_invalid_uploads_and_json():
    with pytest.raises(ValueError, match="one column"):
        load_label_data(_csv_bytes(pd.DataFrame({"a": [1], "b": [0]})))
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_mapping("[1, 2]", field_name="params")
    with pytest.raises(ValueError, match="decimals are invalid"):
        load_label_data(_csv_bytes(pd.DataFrame({"label": [1.0, 0.5]})))
    with pytest.raises(ValueError, match="invalid values"):
        load_label_data(_csv_bytes(pd.DataFrame({"label": [1, -1]})))


def test_deterministic_json_mapping_and_catalog():
    assert parse_json_mapping('{"max_iter": 50}', field_name="params") == {"max_iter": 50}
    catalog = classifier_catalog()
    assert catalog == classifier_catalog()
    upu = next(item for item in catalog if item["name"] == "upu")
    assert any(parameter["name"] == "reg_lambda" for parameter in upu["parameters"])
    loss = next(parameter for parameter in upu["parameters"] if parameter["name"] == "loss")
    assert loss["kind"] == "choice"
    assert loss["choices"] == ["double_hinge", "logistic", "squared"]


@pytest.mark.parametrize(
    ("annotation", "default", "kind", "nullable"),
    [
        (bool, True, "bool", False),
        (int, 5, "int", False),
        (float, 0.5, "float", False),
        ("float | None", None, "float", True),
        ("Literal['a', 'b']", "a", "choice", False),
        (object, None, "json", True),
    ],
)
def test_parameter_schema_selects_typed_editor(annotation, default, kind, nullable):
    parameter = inspect.Parameter(
        "value",
        inspect.Parameter.KEYWORD_ONLY,
        annotation=annotation,
        default=default,
    )
    schema = parameter_schema("value", parameter)
    assert schema["kind"] == kind
    assert schema["nullable"] is nullable
