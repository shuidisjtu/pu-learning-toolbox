# tests/unit/experiment/test_runner_class_prior.py
"""PA's population class prior: resolution order, provenance, and refusal.

Split out of ``test_runner.py`` (which reached the ≤15 tests-per-file limit)
because everything here turns on one contract from protocol §3.1: pi is the
weight of PA's positive term, so a run either supplies one or refuses.  A
default would not fail -- it would silently select by a different criterion and
leave a manifest that looks normal.
"""

# ruff: noqa: N803, N806, S101

import numpy as np
import pytest

from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import (
    FitTrainer,
    ProtocolOA,
    ProtocolPA,
    SARLBEAGenerator,
    SARLBEBGenerator,
    SCARGenerator,
)

pytestmark = pytest.mark.unit


def _make_part(x, labels, idx, fs=True):
    return DatasetPart(
        X=x[np.asarray(idx)],
        labels=labels[np.asarray(idx)],
        view="clean",
        indices=np.asarray(idx),
        for_selection=fs,
    )


def _make_bundle(seed=1):
    """Four-way bundle whose pu_val split receives a real positive.

    The runner fails loudly when the generated pu-view val set has no labeled
    positive, so pu_val must contain one under SCAR.  The test split mixes
    classes so the test AUC is defined and real.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(40, 3)
    y = np.array([1] * 2 + [0] * 16 + [1] * 8 + [0] * 4 + [1] * 1 + [0] * 5 + [1] * 1 + [0] * 3)
    return (
        _make_part(x, y, np.arange(28)),
        _make_part(x, y, np.arange(28, 32)),
        _make_part(x, y, np.arange(32, 36)),
        _make_part(x, y, np.arange(36, 40), fs=False),
    )


def _fit(**kwargs):
    train, pu_val, clean_val, test = _make_bundle()
    return ExperimentRunner(seed=0, **kwargs).fit(
        UPUClassifier(0.3, fit_intercept=False, random_state=0), train, pu_val, clean_val, test
    )


def test_param_pa_selection_requires_a_class_prior():
    """No pi anywhere is a refusal, never a default."""
    with pytest.raises(ValueError, match="class prior"):
        _fit(protocols=[ProtocolPA()], config={"c": 0.5})


def test_basic_selection_prior_falls_back_to_the_split_artifact():
    """A method that needs no prior still leaves PA one to select with."""
    config = {"c": 0.5, "split_ref": {"class_prior": {"population": 0.7}}}
    result = _fit(protocols=[ProtocolPA()], config=config)
    pa = result.manifest["selection"]["PA"]
    assert pa["metrics"]["class_prior"] == pytest.approx(0.7)
    assert pa["class_prior"] == {
        "population": pytest.approx(0.7),
        "source": "split_ref.class_prior.population",
    }
    assert result.selected_models["PA"].class_prior_ == pytest.approx(0.3)


def test_basic_runner_class_prior_wins_over_the_split_artifact():
    """The explicit population value takes precedence for PA selection."""
    config = {"c": 0.5, "split_ref": {"class_prior": {"population": 0.7}}}
    result = _fit(protocols=[ProtocolPA()], class_prior=0.2, config=config)
    pa = result.manifest["selection"]["PA"]
    assert pa["metrics"]["class_prior"] == pytest.approx(0.2)
    assert pa["class_prior"]["source"] == "runner.class_prior"


def test_basic_pa_emits_a_threshold_for_the_test_path():
    """PA picks a threshold, so the runner takes the affine+threshold test path.

    ``threshold=None`` used to mean PA fell back to ``est.predict`` -- the
    pre-R9 engineering behaviour.  This pins the switch so it cannot drift back.
    """
    result = _fit(class_prior=0.3, config={"c": 0.5})
    art = result.selections["PA"]
    assert art.threshold is not None
    assert art.metrics["val_score_min"] is not None
    assert art.metrics["val_score_scale"] is not None


def test_edge_split_ref_class_prior_without_population_is_refused():
    """A malformed record is not a prior -- it must refuse, not fall through.

    The record exists but cannot answer for the population rate; reading it as
    "no prior recorded" would be exactly the silent downgrade this contract
    exists to prevent.
    """
    config = {"c": 0.5, "split_ref": {"class_prior": {"train": 0.4}}}
    with pytest.raises(ValueError, match="class prior"):
        _fit(protocols=[ProtocolPA()], config=config)


@pytest.mark.parametrize(
    ("view", "expected_fit_prior", "expected_source"),
    [
        ("os", 0.26956521739130435, "runner.class_prior.os_to_native_unlabeled"),
        (None, 0.26956521739130435, "runner.class_prior.os_to_native_unlabeled"),
        ("ts", 0.4, "runner.class_prior"),
    ],
)
@pytest.mark.parametrize("use_fit_trainer", [False, True])
def test_param_upu_fits_with_its_risk_pool_prior_but_pa_uses_population(
    view, expected_fit_prior, expected_source, use_fit_trainer
):
    config = {"c": 0.5, "candidates": [{"class_prior": 0.25}, {"class_prior": 0.35}]}
    if view is not None:
        config["os_or_ts"] = view
    if use_fit_trainer:
        config["trainer"] = FitTrainer()
    result = _fit(class_prior=0.4, config=config)

    # Five of 28 training rows are marked; pu_val has a different marked fraction.
    assert result.manifest["generation"]["train"]["n_labeled"] == 5
    assert result.selected_models["PA"].class_prior_ == pytest.approx(expected_fit_prior)
    assert result.manifest["selection"]["PA"]["metrics"]["class_prior"] == 0.4
    settings = result.manifest["research_settings"]["prior"]
    assert settings["selection"]["PA"] == {"population": 0.4, "source": "runner.class_prior"}
    for entry in (settings["training"], *settings["candidates"]):
        assert entry["value"] == pytest.approx(expected_fit_prior)
        assert entry["source"] == expected_source


@pytest.mark.parametrize("view", ["TS", "", "case_control"])
def test_edge_runner_refuses_unknown_training_view_before_generation(view, monkeypatch):
    def never(*args, **kwargs):
        pytest.fail("invalid view must be refused before generating training data")

    monkeypatch.setattr(SCARGenerator, "generate", never)
    with pytest.raises(ValueError, match="os_or_ts"):
        _fit(class_prior=0.4, config={"os_or_ts": view})


def test_edge_upu_refuses_population_prior_below_train_label_mass():
    with pytest.raises(ValueError, match="labeled_fraction"):
        _fit(class_prior=0.1, config={"os_or_ts": "os", "c": 0.5})


class _IndependentTSDeclaration(SCARGenerator):
    """Exercise sampling routing independently of real case-control construction."""

    def generate(self, X, y_true, c, seed=None):
        labels, metadata = super().generate(X, y_true, c, seed)
        return labels, {**metadata, "input_sampling": "ts"}


def test_edge_runner_refuses_to_reconstruct_declared_independent_ts(monkeypatch):
    def never(*args, **kwargs):
        pytest.fail("independent TS must not enter OS-to-TS risk-pool reconstruction")

    monkeypatch.setattr(UPUClassifier, "fit", never)
    with pytest.raises(ValueError, match="independent TS"):
        _fit(generator=_IndependentTSDeclaration(), class_prior=0.4, config={"os_or_ts": "ts"})


def test_basic_declared_independent_ts_keeps_the_native_u_prior():
    result = _fit(
        generator=_IndependentTSDeclaration(),
        protocols=[ProtocolOA()],
        class_prior=0.4,
        config={"os_or_ts": "os", "c": 0.5},
    )
    assert result.selected_models["OA"].class_prior_ == pytest.approx(0.4)
    assert result.manifest["research_settings"]["sampling"]["input_sampling"] == "ts"


@pytest.mark.parametrize(
    ("generator", "reason"),
    [
        (SARLBEAGenerator(), "SAR"),
        (SARLBEBGenerator(), "SAR"),
        (_IndependentTSDeclaration(), "independent TS"),
    ],
)
def test_edge_os_pa_refuses_incompatible_validation_before_training(generator, reason, monkeypatch):
    def never(*args, **kwargs):
        pytest.fail("incompatible PA validation must be refused before training")

    monkeypatch.setattr(UPUClassifier, "fit", never)
    with pytest.raises(ValueError, match=f"PA.*{reason}"):
        _fit(generator=generator, class_prior=0.4, config={"os_or_ts": "os", "c": 0.5})


def test_basic_custom_trainer_keeps_the_population_input():
    class CustomTrainer(FitTrainer):
        def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
            assert class_prior == 0.4
            return super().fit(estimator, X, y, class_prior=0.3, os_or_ts=os_or_ts)

    result = _fit(class_prior=0.4, config={"os_or_ts": "os", "c": 0.5, "trainer": CustomTrainer()})
    assert result.selected_models["PA"].class_prior_ == pytest.approx(0.3)
    assert result.manifest["selection"]["PA"]["metrics"]["class_prior"] == pytest.approx(0.4)
