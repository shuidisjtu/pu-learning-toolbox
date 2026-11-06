"""Exercise copied and frozen encoder paths, including portable CPU predictors."""

import json
import pickle
import runpy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.experiment import adapt_image_bundle_to_features  # noqa: E402
from pu_toolbox.experiment.feature_adapter import encoder_state_sha256  # noqa: E402

pytestmark = pytest.mark.integration
EXAMPLE = Path(__file__).resolve().parents[2] / "examples/minimal/18_custom_encoders.py"


@pytest.fixture(scope="module")
def result():
    return runpy.run_path(str(EXAMPLE))["run_demo"]()


def test_basic_cnn_and_frozen_features_share_roles_and_respect_training_scope(result):
    record = result["record"]
    assert record["training_prior"]["value"] == pytest.approx(1 / 3)
    assert record["native_cnn"]["template_unchanged"]
    assert record["native_cnn"]["encoder_updated"]
    assert result["native"].encoder_ is not result["template"]
    assert result["features"].train.X.shape == (32, 4)
    adapter = record["cnn_feature_adapter"]
    assert adapter["encoder_state_sha256"] == encoder_state_sha256(result["template"])
    assert len(adapter["representation_sha256"]) == 64
    assert adapter["backbone_manifest"]["pretraining"] == "none"
    for role in ("train", "pu_val", "clean_val", "test"):
        np.testing.assert_array_equal(
            getattr(result["features"], role).indices, getattr(result["bundle"], role).indices
        )
    json.dumps(record, allow_nan=False)


def test_basic_edge_frozen_pickle_rebuilds_identical_test_features_and_predictions(result):
    encoder, classifier = pickle.loads(pickle.dumps((result["template"], result["frozen"])))
    adapted, record = adapt_image_bundle_to_features(
        result["bundle"],
        encoder,
        feature_version="reloaded-example-v1",
        backbone_manifest={"name": "tiny", "pretraining": "none"},
        batch_size=1,
    )
    np.testing.assert_allclose(adapted.test.X, result["features"].test.X, atol=1e-7)
    np.testing.assert_array_equal(
        classifier.predict(adapted.test.X), result["frozen"].predict(result["features"].test.X)
    )
    assert record["encoder_state_sha256"] == encoder_state_sha256(result["template"])


def test_determ_repeated_fit_starts_from_same_template_and_restores_native_predictions(result):
    demo = runpy.run_path(str(EXAMPLE))
    second = demo["run_demo"]()
    np.testing.assert_array_equal(
        second["native"].decision_function(second["bundle"].test.X),
        result["native"].decision_function(result["bundle"].test.X),
    )
    restored = pickle.loads(pickle.dumps(result["native"]))
    np.testing.assert_array_equal(
        restored.decision_function(result["bundle"].test.X),
        result["native"].decision_function(result["bundle"].test.X),
    )
    assert second["record"] == result["record"]


@pytest.mark.parametrize("backbone", ["cnn13_no_bn", "resnet18"])
def test_param_shared_factory_outputs_features_for_alternative_backbones(backbone):
    if backbone == "resnet18":
        pytest.importorskip("torchvision")
    demo = runpy.run_path(str(EXAMPLE))
    encoder = demo["make_encoder"](backbone).eval()
    before = encoder_state_sha256(encoder)
    with torch.no_grad():
        features = encoder(torch.zeros(2, 1, 32, 32))
    assert features.shape == (2, 512)
    assert torch.isfinite(features).all()
    assert encoder_state_sha256(encoder) == before
