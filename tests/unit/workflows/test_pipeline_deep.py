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

    def test_full_fit_evaluate_completes_on_4d_images(self, monkeypatch):
        # Regression: 4-D + deep + cnn used to crash at the final
        # diagnostic stage (profiling requires a 2-D view).  The full
        # workflow must now complete end to end; training is shortened
        # by forcing max_epochs=1 at construction time.
        from pu_toolbox.estimators.deep.weighted_contrastive_pu import (
            WeightedContrastivePUClassifier,
        )

        original_init = WeightedContrastivePUClassifier.__init__

        # NOTE: no **kwargs here -- pipeline._missing_required_params would
        # misread a VAR_KEYWORD parameter as a required constructor arg.
        def fast_init(self, class_prior, *, encoder=None, device="cpu", max_epochs=1):
            original_init(self, class_prior, encoder=encoder, device=device, max_epochs=max_epochs)

        monkeypatch.setattr(WeightedContrastivePUClassifier, "__init__", fast_init)
        X, y_pu = _image_data()
        pipe = PUPipeline(classifier="wconpu", architecture="cnn", cv=2)
        report = pipe.fit_evaluate(X, y_pu, class_prior=0.3)
        assert report.final_model.predict(X).shape == (len(X),)
        assert report.diagnostic is not None


@pytest.mark.unit
class TestPipelineDeepAutoUnchanged:
    def test_basic_auto_on_table_never_selects_deep(self, rng):
        # 小数据 has_few 规则排除 DEEP_PU；auto 行为与重构前一致
        X, y_pu = _table_data(n=40)
        pipe = PUPipeline(classifier="auto", cv=2)
        report = pipe.fit_evaluate(X, y_pu)
        assert report.provenance["classifier_mode"] == "auto"
        assert report.final_model.backend.value != "torch"


class _StubClf:
    """Minimal classifier satisfying the pipeline contract (no training)."""

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def decision_function(self, X):
        return np.zeros(len(X))


@pytest.mark.unit
class TestPipelineDeepSeedReproducibility:
    def test_cnn_encoder_weights_follow_random_state(self, monkeypatch):
        """Same random_state -> identical encoder init; different -> different.

        The encoder is built inside fit_evaluate, so torch must be seeded
        from the pipeline's random_state there (regression for the
        same-seed reproducibility promise). Training is stubbed out so only
        the encoder initialization is exercised.
        """
        import pu_toolbox.estimators.deep.vision as vision

        real_build = vision.build_encoder
        captured = {}

        def capturing_build(*args, **kwargs):
            encoder = real_build(*args, **kwargs)
            captured["params"] = [p.detach().clone() for p in encoder.parameters()]
            return encoder

        monkeypatch.setattr(vision, "build_encoder", capturing_build)

        def stub_fresh(cls, instance, prior):
            return _StubClf()

        X, y_pu = _image_data()

        def run_pipe(seed):
            captured.clear()
            pipe = PUPipeline(classifier="wconpu", architecture="cnn", cv=2, random_state=seed)
            monkeypatch.setattr(pipe, "_fresh_estimator", stub_fresh)
            pipe.fit_evaluate(X, y_pu, class_prior=0.3)
            return captured["params"]

        first_42 = run_pipe(42)
        second_42 = run_pipe(42)
        seed_43 = run_pipe(43)

        assert len(first_42) > 0
        for p42, p42_again in zip(first_42, second_42, strict=False):
            assert torch.allclose(p42, p42_again)
        assert not all(torch.allclose(a, b) for a, b in zip(first_42, seed_43, strict=False))
