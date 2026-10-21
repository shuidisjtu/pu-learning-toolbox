# tests/unit/experiment/test_survey_image_spec.py
"""The image backbone spec is chosen per dataset, and v1 (one flat spec) is read unchanged.

``survey_protocol_v2.json`` carries a single flat ``backbone_specs.image`` written for
CIFAR-10.  A later protocol may add ``per_dataset``: overrides merged over the shared
fields.  The runner then decides "is this an image dataset" from the catalog's modality
instead of comparing the dataset name with ``"cifar10"``.
"""

# ruff: noqa: N803, N806, S101

import copy
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.survey_execution import prepare_image_bundle
from pu_toolbox.experiment.survey_protocol import (
    PROTOCOL_PATH,
    ROLES,
    image_spec,
    load_protocol,
)

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/run_survey_experiment.py"
ADNI_SPEC = {"in_channels": 1, "input_size": [16, 16], "small_input_stem": True}


def _bundle(channels: int, size: int) -> DatasetBundle:
    rng = np.random.RandomState(3)
    parts = {}
    for index, role in enumerate(ROLES):
        parts[role] = DatasetPart(
            X=rng.randint(0, 256, size=(4, channels, size, size), dtype=np.uint8),
            labels=np.array([1, 1, 0, 0]),
            view="clean",
            indices=np.arange(index * 4, index * 4 + 4),
            for_selection=role != "test",
        )
    return DatasetBundle(**parts)


def _nested_protocol(tmp_path, per_dataset):
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["backbone_specs"]["image"]["per_dataset"] = per_dataset
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def runner_script():
    spec = importlib.util.spec_from_file_location("run_survey_experiment_spec", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset", ["cifar10", "adni", "mnist"])
def test_basic_the_flat_v1_spec_is_returned_whatever_the_dataset(dataset):
    protocol = load_protocol()

    assert image_spec(protocol, dataset) == protocol["backbone_specs"]["image"]


def test_basic_a_per_dataset_spec_overrides_only_what_it_names(tmp_path):
    protocol = load_protocol(_nested_protocol(tmp_path, {"adni": ADNI_SPEC}))

    merged = image_spec(protocol, "adni")

    assert merged["input_size"] == [16, 16] and merged["in_channels"] == 1
    assert merged["name"] == "resnet18" and merged["weights"] is None
    assert "per_dataset" not in merged


def test_edge_a_dataset_missing_from_per_dataset_is_an_error(tmp_path):
    protocol = load_protocol(_nested_protocol(tmp_path, {"adni": ADNI_SPEC}))

    with pytest.raises(ValueError, match="cifar10"):
        image_spec(protocol, "cifar10")


@pytest.mark.parametrize("bad", [[], {"adni": "x"}, {"adni": {"name": "vgg"}}])
def test_param_a_malformed_per_dataset_block_is_rejected_on_load(tmp_path, bad):
    with pytest.raises(ValueError, match="per_dataset"):
        load_protocol(_nested_protocol(tmp_path, bad))


def test_basic_prepare_image_bundle_uses_the_named_datasets_spec(tmp_path):
    protocol = load_protocol(_nested_protocol(tmp_path, {"adni": ADNI_SPEC}))

    prepared, _, manifest = prepare_image_bundle(_bundle(1, 16), protocol, 0, dataset="adni")

    assert prepared.train.X.shape[1:] == (1, 16, 16)
    assert manifest["input_size"] == [16, 16]
    assert manifest["dataset"] == "adni"


def test_edge_v1_still_refuses_an_image_that_is_not_its_locked_size():
    with pytest.raises(ValueError, match="input size"):
        prepare_image_bundle(_bundle(1, 16), load_protocol(), 0, dataset="adni")


def test_edge_a_spec_channel_count_the_data_does_not_have_is_an_error(tmp_path):
    wrong = {"adni": {**ADNI_SPEC, "in_channels": 3}}
    protocol = load_protocol(_nested_protocol(tmp_path, wrong))

    with pytest.raises(ValueError, match="channel"):
        prepare_image_bundle(_bundle(1, 16), protocol, 0, dataset="adni")


def test_determ_the_same_dataset_and_seed_give_the_same_manifest(tmp_path):
    protocol = load_protocol(_nested_protocol(tmp_path, {"adni": ADNI_SPEC}))
    source = _bundle(1, 16)
    changed = replace(source, test=replace(source.test, X=np.zeros_like(source.test.X)))

    first = prepare_image_bundle(source, copy.deepcopy(protocol), 0, dataset="adni")[2]
    second = prepare_image_bundle(changed, copy.deepcopy(protocol), 0, dataset="adni")[2]

    assert first == second


def test_basic_the_dataset_option_offers_every_catalog_dataset(runner_script):
    from pu_toolbox.experiment import survey_dataset_catalog

    assert set(runner_script.DATASET_CHOICES) == set(survey_dataset_catalog())


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("cifar10", "image"),
        ("adni", "image"),
        ("mnist", "image"),
        ("imdb", "text"),
        ("spambase", "tabular"),
    ],
)
def test_param_the_runner_reads_modality_from_the_catalog(runner_script, dataset, expected):
    assert runner_script.modality_of(dataset) == expected
