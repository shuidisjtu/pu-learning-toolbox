"""Preintegrated native CNN public pipelines, without formal survey claims."""

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import PipelineError  # noqa: E402
from pu_toolbox.workflows import PUPipeline  # noqa: E402

pytestmark = pytest.mark.integration


def images():
    values = np.random.RandomState(3).normal(size=(24, 3, 8, 8)).astype("float32")
    return values, np.tile(np.r_[np.ones(4, int), np.zeros(8, int)], 2)


def small_params(method):
    if method == "pan":
        return {"batch_size": 4}
    if method == "gradpu":
        return {"batch_size": 8, "hidden_dim": 4}
    if method == "robust_pu":
        return {"pretrain_epochs": 1, "episodes": 1, "batch_size": 8, "hidden_dim": 4}
    if method == "holistic_pu":
        return {"warmup_epochs": 3, "batch_size": 8, "hidden_dim": 4}
    if method == "genpu":
        return {"classifier_epochs": 1, "batch_size": 8, "hidden_dim": 4, "latent_dim": 2}
    if method == "split_pu":
        return {
            "teacher_epochs": 1,
            "split_epochs": 1,
            "student_epochs": 1,
            "rounds": 1,
            "batch_size": 8,
            "hidden_dim": 4,
        }
    return {
        "warmup_epochs": 1,
        "pu_epochs": 1,
        "depth": 1,
        "hidden_dim": 4,
        "positive_batch_size": 4,
        "unlabeled_batch_size": 8,
    }


@pytest.mark.parametrize(
    "method", ["pan", "pulda", "gradpu", "robust_pu", "split_pu", "holistic_pu", "genpu"]
)
def test_basic_cnn_pipeline_fits_and_reports_encoder_provenance(method):
    features, labels = images()
    pipe = PUPipeline(
        classifier=method,
        architecture="cnn",
        backbone="cnn13_no_bn" if method == "gradpu" else "cnn13",
        cv=2,
        max_epochs=1,
        classifier_params=small_params(method),
        random_state=3,
        device="cpu",
    )
    with pytest.warns(UserWarning, match=r"trained 2 times \(CV folds only; refit=False\)"):
        report = pipe.fit_evaluate(features, labels, refit=False, class_prior=0.4)
    assert report.provenance["architecture"] == "native_cnn"
    assert report.provenance["backbone"] == pipe.backbone
    assert report.provenance["encoder"] == {"backbone": pipe.backbone, "in_channels": 3}
    assert report.provenance["device"] == {"requested": "cpu", "resolved": "cpu"}


@pytest.mark.parametrize(
    "method", ["pan", "pulda", "gradpu", "robust_pu", "split_pu", "holistic_pu", "genpu"]
)
def test_determ_seeded_pipeline_fresh_estimators_roundtrip_without_template_leak(method):
    from pu_toolbox import build_encoder

    features, labels = images()
    pipe = PUPipeline(
        classifier=method,
        architecture="cnn",
        backbone="cnn13_no_bn" if method == "gradpu" else "cnn13",
        cv=2,
        max_epochs=1,
        classifier_params=small_params(method),
        random_state=3,
        device="cpu",
    )
    torch.manual_seed(3)
    pipe._encoder = build_encoder("cnn", backbone=pipe.backbone, in_channels=3)
    before = {key: value.clone() for key, value in pipe._encoder.state_dict().items()}
    first = pipe._fresh_estimator(pipe._classifier_cls, None, 0.4).fit(features, labels)
    second = pipe._fresh_estimator(pipe._classifier_cls, None, 0.4).fit(features, labels)
    np.testing.assert_array_equal(
        first.decision_function(features), second.decision_function(features)
    )
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(first)).decision_function(features),
        first.decision_function(features),
    )
    for key, value in pipe._encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_param_edge_cnn_not_silently_enabled_for_mlp_only_algorithms():
    with pytest.raises(PipelineError, match="cnn"):
        PUPipeline(classifier="dist_pu", architecture="cnn", cv=2, max_epochs=1)


@pytest.mark.parametrize("backbone", ["cnn13", "resnet18"])
def test_basic_genpu_pixel_tanh_recipe_runs_through_public_cnn_pipeline(backbone):
    features, labels = images()
    features = np.tanh(features)
    pipe = PUPipeline(
        classifier="genpu",
        architecture="cnn",
        backbone=backbone,
        cv=2,
        max_epochs=1,
        classifier_params={**small_params("genpu"), "generator_output": "tanh"},
        random_state=3,
        device="cpu",
    )
    with pytest.warns(UserWarning, match=r"trained 2 times \(CV folds only; refit=False\)"):
        report = pipe.fit_evaluate(features, labels, refit=False, class_prior=0.4)
    assert report.provenance["architecture"] == "native_cnn"
    assert report.provenance["backbone"] == backbone
    assert report.provenance["classifier_params"]["generator_output"] == "tanh"


@pytest.mark.parametrize("backbone", ["cnn13", "resnet18"])
def test_basic_holistic_reinitialization_is_available_through_public_pipeline(backbone):
    features, labels = images()
    params = {**small_params("holistic_pu"), "pseudo_pn_initialization": "reinitialize"}
    pipe = PUPipeline(
        classifier="holistic_pu",
        architecture="cnn",
        backbone=backbone,
        cv=2,
        max_epochs=1,
        classifier_params=params,
        random_state=3,
        device="cpu",
    )
    with pytest.warns(UserWarning, match=r"trained 2 times \(CV folds only; refit=False\)"):
        report = pipe.fit_evaluate(features, labels, refit=False)
    assert report.provenance["architecture"] == "native_cnn"
    assert report.provenance["backbone"] == backbone
    assert report.provenance["classifier_params"]["pseudo_pn_initialization"] == "reinitialize"


@pytest.mark.parametrize("initialization", ["continue", "reinitialize"])
def test_basic_holistic_lzo_recipe_is_explicit_in_pipeline_parameters(initialization):
    features, labels = images()
    params = {
        **small_params("holistic_pu"),
        "warmup_selection": "lzo_positive_loss",
        "lzo_validation_size": 5,
        "pseudo_pn_initialization": initialization,
    }
    pipe = PUPipeline(
        classifier="holistic_pu",
        architecture="cnn",
        backbone="cnn13",
        cv=2,
        max_epochs=1,
        classifier_params=params,
        random_state=3,
        device="cpu",
    )
    with pytest.warns(UserWarning, match=r"trained 2 times \(CV folds only; refit=False\)"):
        report = pipe.fit_evaluate(features, labels, refit=False)
    assert report.provenance["classifier_params"]["warmup_selection"] == "lzo_positive_loss"
    assert report.provenance["classifier_params"]["lzo_validation_size"] == 5
    assert report.provenance["classifier_params"]["pseudo_pn_initialization"] == initialization
