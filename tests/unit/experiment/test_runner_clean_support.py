"""Support-dependent estimators need a fifth, independent training partition."""

from dataclasses import replace
from functools import wraps

import pytest
from _support_checkpoint_helpers import inputs, model

from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, SCARGenerator
from pu_toolbox.experiment.tracking import EpochRecord, RunTrajectory

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["lagam", "pulns"])
def test_param_default_trainer_refuses_missing_support_before_fit(method, monkeypatch):
    parts, _ = inputs()
    estimator = model(method)
    original_fit = type(estimator).fit
    attempts = []

    @wraps(original_fit)
    def record_fit(self, *args, **kwargs):
        attempts.append(self)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(type(estimator), "fit", record_fit)
    runner = ExperimentRunner(protocols=[ProtocolOA()], config={"clean_support_provided": True})
    with pytest.raises(ValueError, match="requires independent clean support.*trainer"):
        runner.fit(estimator, *parts)
    assert not attempts


class _CustomSupportTrainer:
    def __init__(self, support):
        self.support = support
        self.fits = 0

    def fit(self, estimator, x, y, *, training_population_prior=None, val_pu=None):
        self.fits += 1
        estimator.fit(x, y, support_data=(self.support.X, self.support.labels))
        return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=estimator)


def test_basic_custom_trainer_with_independent_support_is_still_accepted():
    pytest.importorskip("torch")
    parts, support = inputs()
    trainer = _CustomSupportTrainer(support)
    result = ExperimentRunner(protocols=[ProtocolOA()], config={"trainer": trainer}).fit(
        model("lagam"), *parts
    )
    assert trainer.fits == 1
    assert set(result.test_metrics) == {"OA"}


def test_edge_custom_trainer_cannot_reuse_clean_val_as_support():
    parts, support = inputs()
    trainer = _CustomSupportTrainer(replace(support, indices=parts[2].indices[: len(support.X)]))
    with pytest.raises(ValueError, match="overlap.*support.*clean_val"):
        ExperimentRunner(protocols=[ProtocolOA()], config={"trainer": trainer}).fit(
            model("lagam"), *parts
        )
    assert trainer.fits == 0


def test_determ_missing_support_fails_before_seeded_label_generation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("missing support must be rejected before generating labels")

    monkeypatch.setattr(SCARGenerator, "generate", forbidden)
    parts, _ = inputs()
    errors = []
    for seed in (0, 7):
        runner = ExperimentRunner(seed=seed, protocols=[ProtocolOA()])
        with pytest.raises(ValueError, match="requires independent clean support") as caught:
            runner.fit(model("lagam"), *parts)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]
