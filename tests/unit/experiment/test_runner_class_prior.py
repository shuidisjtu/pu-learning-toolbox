# tests/unit/experiment/test_runner_class_prior.py
"""PA's population class prior: resolution order, provenance, and refusal.

Split out of ``test_runner.py`` (which reached the ≤15 tests-per-file limit)
because everything here turns on one contract from protocol §3.1: pi is the
weight of PA's positive term, so a run either supplies one or refuses.  A
default would not fail -- it would silently select by a different criterion and
leave a manifest that looks normal.
"""

# ruff: noqa: N803, N806, S101

import json
from dataclasses import replace

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


def _fit(*, reverse_test_labels=False, **kwargs):
    train, pu_val, clean_val, test = _make_bundle()
    if reverse_test_labels:
        test = replace(test, labels=1 - test.labels)
    return ExperimentRunner(seed=0, **kwargs).fit(
        UPUClassifier(0.3, fit_intercept=False, random_state=0), train, pu_val, clean_val, test
    )


def test_param_pa_selection_requires_a_class_prior():
    """No pi anywhere is a refusal, never a default."""
    with pytest.raises(ValueError, match="class prior"):
        _fit(protocols=[ProtocolPA()], config={"c": 0.5})


@pytest.mark.parametrize("candidate_prior", [None, 0.35])
@pytest.mark.parametrize("explicit", [None, 0.2])
def test_param_selection_prior_prefers_explicit_before_the_split_artifact(
    candidate_prior, explicit
):
    """The split supplies PA's population prior; fitting keeps its own prior."""
    config = {"c": 0.5, "split_ref": {"class_prior": {"population": 0.7}}}
    if candidate_prior is not None:
        config["candidates"] = [{"risk_prior": candidate_prior}]
    result = _fit(protocols=[ProtocolPA()], validation_population_prior=explicit, config=config)
    expected_validation = 0.7 if explicit is None else explicit
    risk, pa = result.manifest["research_settings"]["prior"]
    assert result.selections["PA"].metrics["validation_population_prior"] == pytest.approx(
        expected_validation
    )
    assert pa == {
        "protocol": "PA",
        "value": pytest.approx(expected_validation),
        "source": (
            "split_ref.class_prior.population"
            if explicit is None
            else "runner.validation_population_prior"
        ),
        "consumer": "pa_risk",
        "partition": "pu_val",
        "target": "population",
    }
    expected_fit_prior = 0.3 if candidate_prior is None else candidate_prior
    assert result.selected_models["PA"].risk_prior_ == pytest.approx(expected_fit_prior)
    assert risk == {
        "candidate_index": 0,
        "consumer": "upu_risk",
        "value": expected_fit_prior,
        "source": "estimator.risk_prior" if candidate_prior is None else "candidate.risk_prior",
        "target": "native_unlabeled",
        "partition": "train",
    }


def test_basic_training_and_validation_priors_have_independent_inputs():
    result = _fit(
        training_population_prior=0.4,
        validation_population_prior=0.6,
        config={"c": 0.5, "split_ref": {"class_prior": {"population": 0.7}}},
    )

    settings = result.manifest["research_settings"]
    assert settings["schema_version"] == "2.0"
    risk, pa = settings["prior"]
    assert risk["value"] == pytest.approx(0.26956521739130435)
    assert result.selected_models["PA"].risk_prior_ == pytest.approx(risk["value"])
    assert risk["source"] == "runner.training_population_prior"
    assert risk["conversion"]["original_value"] == 0.4
    assert pa == {
        "protocol": "PA",
        "consumer": "pa_risk",
        "partition": "pu_val",
        "target": "population",
        "value": 0.6,
        "source": "runner.validation_population_prior",
    }
    assert "class_prior" not in result.manifest["selection"]["PA"]
    assert "validation_population_prior" not in result.manifest["selection"]["PA"]["metrics"]
    assert result.selections["PA"].metrics["validation_population_prior"] == 0.6


def test_basic_pa_emits_a_threshold_for_the_test_path():
    """PA picks a threshold, so the runner takes the affine+threshold test path.

    ``threshold=None`` used to mean PA fell back to ``est.predict`` -- the
    pre-R9 engineering behaviour.  This pins the switch so it cannot drift back.
    """
    result = _fit(training_population_prior=0.3, validation_population_prior=0.3, config={"c": 0.5})
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
    ("view", "expected_fit_prior"),
    [
        ("os", 0.26956521739130435),
        (None, 0.26956521739130435),
        ("ts", 0.4),
    ],
)
@pytest.mark.parametrize("use_fit_trainer", [False, True])
def test_param_upu_fits_with_its_risk_pool_prior_but_pa_uses_population(
    view, expected_fit_prior, use_fit_trainer
):
    config = {"c": 0.5, "candidates": [{"risk_prior": 0.25}, {"risk_prior": 0.35}]}
    if view is not None:
        config["os_or_ts"] = view
    if use_fit_trainer:
        config["trainer"] = FitTrainer()
    result = _fit(training_population_prior=0.4, validation_population_prior=0.4, config=config)

    # Five of 28 training rows are marked; pu_val has a different marked fraction.
    assert result.manifest["generation"]["train"]["n_labeled"] == 5
    assert result.selected_models["PA"].risk_prior_ == pytest.approx(expected_fit_prior)
    assert result.selections["PA"].metrics["validation_population_prior"] == 0.4
    *fit_priors, selection_prior = result.manifest["research_settings"]["prior"]
    assert selection_prior == {
        "protocol": "PA",
        "value": 0.4,
        "source": "runner.validation_population_prior",
        "consumer": "pa_risk",
        "partition": "pu_val",
        "target": "population",
    }
    assert len(fit_priors) == 2
    for index, binding in enumerate(fit_priors):
        assert binding["candidate_index"] == index
        assert binding["consumer"] == "upu_risk"
        assert binding["value"] == pytest.approx(expected_fit_prior)
        assert binding["source"] == "runner.training_population_prior"
        assert binding["partition"] == "train"
        if view == "ts":
            assert binding["target"] == "positive_union_unlabeled"
            assert "conversion" not in binding
        else:
            assert binding["target"] == "native_unlabeled"
            assert binding["conversion"] == {
                "rule": "population_to_native_unlabeled",
                "original_value": 0.4,
                "q": pytest.approx(5 / 28),
            }


def test_determ_test_labels_do_not_change_selection_or_prior_bindings():
    config = {"c": 0.5, "candidates": [{"reg_lambda": 0.1}, {"reg_lambda": 1}]}
    result = _fit(training_population_prior=0.4, validation_population_prior=0.4, config=config)
    changed = _fit(
        training_population_prior=0.4,
        validation_population_prior=0.4,
        config=config,
        reverse_test_labels=True,
    )

    assert result.manifest["selection"] == changed.manifest["selection"]
    assert (
        result.manifest["research_settings"]["prior"]
        == changed.manifest["research_settings"]["prior"]
    )
    assert result.test_metrics != changed.test_metrics


def test_edge_failed_fit_keeps_candidate_parameters_without_consumption(tmp_path):
    path = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="all candidate runs failed"):
        _fit(
            manifest_path=str(path),
            validation_population_prior=0.4,
            config={"c": 0.5, "candidates": [{"loss": "unknown"}]},
        )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["research_settings"]["schema_version"] == "2.0"
    assert manifest["research_settings"]["prior"] == []
    assert manifest["candidate_runs"][0]["params"] == {"loss": "unknown"}


@pytest.mark.parametrize("view", ["TS", "", "case_control"])
def test_edge_runner_refuses_unknown_training_view_before_generation(view, monkeypatch):
    def never(*args, **kwargs):
        pytest.fail("invalid view must be refused before generating training data")

    monkeypatch.setattr(SCARGenerator, "generate", never)
    with pytest.raises(ValueError, match="os_or_ts"):
        _fit(
            training_population_prior=0.4,
            validation_population_prior=0.4,
            config={"os_or_ts": view},
        )


def test_edge_upu_refuses_population_prior_below_train_label_mass():
    with pytest.raises(ValueError, match="labeled_fraction"):
        _fit(
            training_population_prior=0.1,
            validation_population_prior=0.1,
            config={"os_or_ts": "os", "c": 0.5},
        )


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
        _fit(
            generator=_IndependentTSDeclaration(),
            training_population_prior=0.4,
            validation_population_prior=0.4,
            config={"os_or_ts": "ts"},
        )


def test_basic_declared_independent_ts_keeps_the_native_u_prior():
    result = _fit(
        generator=_IndependentTSDeclaration(),
        protocols=[ProtocolOA()],
        training_population_prior=0.4,
        validation_population_prior=0.4,
        config={"os_or_ts": "os", "c": 0.5},
    )
    assert result.selected_models["OA"].risk_prior_ == pytest.approx(0.4)
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
        _fit(
            generator=generator,
            training_population_prior=0.4,
            validation_population_prior=0.4,
            config={"os_or_ts": "os", "c": 0.5},
        )


def test_basic_custom_trainer_keeps_the_population_input():
    class CustomTrainer(FitTrainer):
        def fit(
            self, estimator, X, y, *, training_population_prior=None, val_pu=None, os_or_ts=None
        ):
            assert training_population_prior == 0.4
            return super().fit(estimator, X, y, risk_prior=0.3, os_or_ts=os_or_ts)

    result = _fit(
        training_population_prior=0.4,
        validation_population_prior=0.4,
        config={"os_or_ts": "os", "c": 0.5, "trainer": CustomTrainer()},
    )
    assert result.selected_models["PA"].risk_prior_ == pytest.approx(0.3)
    assert result.selections["PA"].metrics["validation_population_prior"] == pytest.approx(0.4)
    binding = result.manifest["research_settings"]["prior"][0]
    assert binding == {
        "candidate_index": 0,
        "consumer": "upu_risk",
        "partition": "train",
        "target": "native_unlabeled",
        "source": "fit.risk_prior",
        "value": 0.3,
    }


def test_basic_custom_trainer_can_report_its_actual_consumer_and_source():
    class CustomTrainer(FitTrainer):
        def fit(
            self, estimator, X, y, *, training_population_prior=None, val_pu=None, os_or_ts=None
        ):
            trajectory = super().fit(estimator, X, y, risk_prior=0.28, os_or_ts=os_or_ts)
            (receipt,) = estimator._prior_consumptions_
            trajectory.prior_consumptions = ({**receipt, "source": "custom.heldout_estimate"},)
            return trajectory

    result = _fit(
        training_population_prior=0.4,
        validation_population_prior=0.5,
        config={"os_or_ts": "os", "c": 0.5, "trainer": CustomTrainer()},
    )
    train, pa = result.manifest["research_settings"]["prior"]
    assert (train["value"], train["source"]) == (0.28, "custom.heldout_estimate")
    assert pa["value"] == 0.5
