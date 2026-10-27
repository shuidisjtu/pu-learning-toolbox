"""The unit script binds the clean-support part in the prepared image space.

``prepare_image_support`` is tested on its own; this pins the call site, which no
other test reaches -- the loader, the fitted preprocessing and the frozen encoder
belong to the runner, and the trainer must be bound with the converted part.
Training is replaced by a spy, so the cost is one ResNet-18 construction per run
and no epochs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment.support_data import prepare_support_dataset, save_support_dataset

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_survey_experiment as script  # noqa: E402

pytestmark = pytest.mark.unit

SUPPORT_SIZE = 10


class _SupportBoundError(Exception):
    """The trainer spy's stop: the run ends at the binding, before any epoch."""


def image_product(root, *, dataset="mnist", size=28, channels=1):
    rng = np.random.RandomState(11)
    images = rng.randint(0, 256, size=(60, channels, size, size), dtype=np.uint8)
    test_images = rng.randint(0, 256, size=(40, channels, size, size), dtype=np.uint8)
    bundle, support, manifest = prepare_support_dataset(
        images,
        np.tile([0, 1], 30),
        dataset=dataset,
        seed=0,
        support_size=SUPPORT_SIZE,
        support_seed=7,
        group_policy="not_applicable",
        source="test fixture, not a protocol product",
        X_test=test_images,
        y_test=np.tile([0, 1], 20),
    )
    save_support_dataset(bundle, support, manifest, root)
    return root


def tabular_product(root):
    rng = np.random.RandomState(0)
    features = rng.normal(size=(400, 5))
    truth = np.tile([0, 1], 200)
    features[:, 0] += truth
    bundle, support, manifest = prepare_support_dataset(
        features,
        truth,
        dataset="spambase",
        seed=0,
        support_size=40,
        support_seed=7,
        group_policy="not_applicable",
        source="test fixture, not a protocol product",
    )
    save_support_dataset(bundle, support, manifest, root)
    return root


def run_unit(monkeypatch, tmp_path, product, *, dataset, training_path, method, tag="once"):
    """Stop the runner at the trainer binding and return the bound support part."""
    captured = {}

    def spy(algorithm, view, support, manifest):
        captured["support"] = support
        raise _SupportBoundError

    monkeypatch.setattr(script, "_support_trainer", spy)
    argv = [
        str(product),
        "--dataset",
        dataset,
        "--training-path",
        training_path,
        "--method",
        method,
        "--seeds",
        "0",
        "--protocol",
        "survey-v3.0",
        "--c",
        "0.5",
        "--class-prior",
        "0.5",
        "--support-dir",
        str(product),
        "--split-ref",
        str(product / "support_manifest.json"),
        "--device",
        "cpu",
        "--adapter-cache",
        str(tmp_path / f"adapter-cache-{tag}"),
        "--out-dir",
        str(tmp_path / f"runs-{tag}"),
    ]
    with pytest.raises(_SupportBoundError):
        script.main(argv)
    return captured["support"]


def test_basic_native_cnn_support_is_bound_as_scaled_images(monkeypatch, tmp_path):
    product = image_product(tmp_path / "product")
    support = run_unit(
        monkeypatch,
        tmp_path,
        product,
        dataset="mnist",
        training_path="native_cnn",
        method="lagam",
    )
    assert support.X.dtype == np.float32
    assert support.X.shape == (SUPPORT_SIZE, 1, 28, 28)
    assert support.X.min() >= 0.0 and support.X.max() <= 1.0
    assert support.view == "clean" and support.for_selection is False
    assert len(support.indices) == SUPPORT_SIZE


def test_param_adapter_support_is_bound_as_encoder_features(monkeypatch, tmp_path):
    product = image_product(tmp_path / "product")
    support = run_unit(
        monkeypatch,
        tmp_path,
        product,
        dataset="mnist",
        training_path="cnn_feature_adapter",
        method="self_pu_full_meta",
    )
    assert support.X.dtype == np.float32
    assert support.X.ndim == 2 and support.X.shape == (SUPPORT_SIZE, 512)
    assert len(support.indices) == SUPPORT_SIZE


def test_edge_tabular_support_reaches_the_trainer_untransformed(monkeypatch, tmp_path):
    product = tabular_product(tmp_path / "product")
    support = run_unit(
        monkeypatch,
        tmp_path,
        product,
        dataset="spambase",
        training_path="native_2d",
        method="lagam",
    )
    from pu_toolbox.experiment.support_data import load_support_dataset

    _, loaded, _ = load_support_dataset(product)
    np.testing.assert_array_equal(support.X, loaded.X)
    assert support.X.dtype == np.float32


def test_determ_the_same_product_binds_the_same_support(monkeypatch, tmp_path):
    product = image_product(tmp_path / "product")
    first = run_unit(
        monkeypatch,
        tmp_path,
        product,
        dataset="mnist",
        training_path="native_cnn",
        method="lagam",
        tag="first",
    )
    second = run_unit(
        monkeypatch,
        tmp_path,
        product,
        dataset="mnist",
        training_path="native_cnn",
        method="lagam",
        tag="second",
    )
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.indices, second.indices)
