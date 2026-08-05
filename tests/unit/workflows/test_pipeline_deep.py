# ruff: noqa: E402, N803, N806

"""PUPipeline deep-algorithm integration tests (architecture selection)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.workflows import PipelineError, PUPipeline  # noqa: E402


def _image_data(n=24, channels=3, size=8, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.5, 0.3, size=(n, channels, size, size)).astype(np.float32)
    y_pu = np.concatenate([np.ones(8, dtype=int), np.zeros(n - 8, dtype=int)])
    return X, y_pu


def _table_data(n=40, seed=2):
    rng = np.random.RandomState(seed)
    X = np.vstack(
        [rng.normal(1.0, 0.3, size=(n // 2, 5)), rng.normal(-1.0, 0.3, size=(n // 2, 5))]
    ).astype(np.float32)
    y_pu = np.concatenate([np.ones(8, dtype=int), np.zeros(n - 8, dtype=int)])
    return X, y_pu


@pytest.mark.unit
class TestPipelineDeepValidation:
    def test_param_cnn_with_shallow_classifier_raises(self):
        with pytest.raises(PipelineError, match="cnn"):
            PUPipeline(classifier="upu", architecture="cnn")

    def test_param_cnn_with_auto_raises(self):
        with pytest.raises(PipelineError, match="cnn"):
            PUPipeline(classifier="auto", architecture="cnn")

    def test_param_2d_with_cnn_raises(self):
        X, y_pu = _table_data()
        pipe = PUPipeline(classifier="wconpu", architecture="cnn", cv=2)
        with pytest.raises(PipelineError, match="4-D"):
            pipe.fit_evaluate(X, y_pu, class_prior=0.3)

    def test_param_4d_with_mlp_raises(self):
        X, y_pu = _image_data()
        pipe = PUPipeline(classifier="wconpu", cv=2)
        with pytest.raises(PipelineError, match="4-D"):
            pipe.fit_evaluate(X, y_pu, class_prior=0.3)

    def test_param_invalid_architecture_raises(self):
        with pytest.raises(ValueError, match="architecture"):
            PUPipeline(architecture="rnn")

    def test_param_invalid_backbone_raises(self):
        with pytest.raises(ValueError, match="backbone"):
            PUPipeline(classifier="wconpu", architecture="cnn", backbone="vgg16")


@pytest.mark.unit
class TestPipelineDeepInstantiation:
    def test_basic_fresh_estimator_injects_encoder_and_prior(self):
        X, y_pu = _image_data()
        pipe = PUPipeline(classifier="wconpu", architecture="cnn", cv=2)
        pipe._encoder = __import__(
            "pu_toolbox.estimators.deep.vision", fromlist=["build_encoder"]
        ).build_encoder("cnn", backbone="cnn13", in_channels=3)
        clf = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
        assert clf.encoder is pipe._encoder
        assert clf.class_prior == 0.3

    def test_basic_fresh_estimator_injects_device(self):
        X, y_pu = _table_data()
        pipe = PUPipeline(classifier="wconpu", cv=2, device="cpu")
        clf = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
        assert clf.device == "cpu"

    def test_edge_injected_cnn_encoder_trains_on_4d(self):
        from pu_toolbox.estimators.deep.vision import build_encoder

        X, y_pu = _image_data()
        pipe = PUPipeline(classifier="wconpu", architecture="cnn", cv=2)
        pipe._encoder = build_encoder("cnn", backbone="cnn13", in_channels=3)
        clf = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
        clf.max_epochs = 2  # 测试提速；注入链路（encoder + prior + device）才是被测对象
        clf.fit(X, y_pu, class_prior=0.3)
        assert clf.predict(X).shape == (len(X),)


@pytest.mark.unit
class TestPipelineDeepAutoUnchanged:
    def test_basic_auto_on_table_never_selects_deep(self, rng):
        # 小数据 has_few 规则排除 DEEP_PU；auto 行为与重构前一致
        X, y_pu = _table_data(n=40)
        pipe = PUPipeline(classifier="auto", cv=2)
        report = pipe.fit_evaluate(X, y_pu)
        assert report.provenance["classifier_mode"] == "auto"
        assert report.final_model.backend.value != "torch"
