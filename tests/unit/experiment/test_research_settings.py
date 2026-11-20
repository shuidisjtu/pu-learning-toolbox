"""Research settings describe the generated roles and supplied training parameters."""

import json

import numpy as np
import pytest

from pu_toolbox.estimators.bias_aware.pusb_kernel import PUSBKernelClassifier
from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.experiment import (
    DatasetPart,
    ExperimentRunner,
    FitTrainer,
    ProtocolOA,
    research_settings,
)
from pu_toolbox.experiment.strategies import SARLBEAGenerator, SCARGenerator

pytestmark = pytest.mark.unit


def _fit(tmp_path, *, view="os", generator=None, prior=None, model=None):
    rng = np.random.RandomState(7)
    labels = np.tile([0, 1], 40)
    features = rng.normal(size=(80, 3)) + labels[:, None]
    parts = [
        DatasetPart(
            X=features[start : start + 20],
            labels=labels[start : start + 20],
            indices=np.arange(start, start + 20),
            view="clean",
            for_selection=start != 60,
        )
        for start in (0, 20, 40, 60)
    ]
    return ExperimentRunner(
        seed=7,
        training_population_prior=prior,
        validation_population_prior=prior,
        generator=generator or SCARGenerator(),
        protocols=[ProtocolOA()],
        manifest_path=str(tmp_path / "manifest.json"),
        config={"c": 0.5, "os_or_ts": view, "trainer": FitTrainer()},
    ).fit(
        model if model is not None else UPUClassifier(0.3, fit_intercept=False, random_state=7),
        *parts,
    )


@pytest.mark.parametrize(
    ("view", "pool"),
    [
        ("os", "native_unlabeled"),
        ("ts", "positive_union_unlabeled"),
    ],
)
def test_param_recorded_risk_and_prior_follow_the_training_view(tmp_path, view, pool):
    result = _fit(tmp_path, view=view)
    settings = result.manifest["research_settings"]
    assert settings["sampling"]["input_sampling"] == "os"
    assert settings["risk"]["pool"] == pool
    (fit_prior,) = settings["prior"]
    assert fit_prior == {
        "candidate_index": 0,
        "consumer": "upu_risk",
        "partition": "train",
        "value": 0.3,
        "target": pool,
        "source": "estimator.risk_prior",
    }
    assert settings["supervision"]["selection_roles"] == {"OA": "clean_val"}
    assert settings["representation"]["input_ndim"] == 2
    assert settings["labeling"]["train"]["sampling_policy"] == (
        "fixed_count_uniform_without_replacement"
    )
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["research_settings"] == settings


def test_basic_sar_settings_preserve_weight_policy_and_score_source(tmp_path):
    settings = _fit(tmp_path, generator=SARLBEAGenerator()).manifest["research_settings"]
    labeling = settings["labeling"]["train"]
    assert labeling["sampling_policy"] == "fixed_count_weighted_without_replacement"
    assert labeling["posterior_fit_on"] == "real_labels"
    assert labeling["n_labeled"] == 5
    assert "sar_lbe_a" in settings["summary"]
    assert "clean_val" in settings["summary"]


def test_edge_runner_prior_records_the_forwarded_value_and_source(tmp_path):
    manifest = _fit(tmp_path, prior=0.4).manifest
    settings = manifest["research_settings"]
    (fit_prior,) = settings["prior"]
    assert fit_prior["value"] == pytest.approx(0.2)
    assert fit_prior["source"] == "runner.training_population_prior"
    assert "class_prior" not in manifest["selection"]["OA"]


def test_determ_same_seed_has_identical_settings_with_or_without_disk(tmp_path):
    first = _fit(tmp_path).manifest["research_settings"]
    second = _fit(tmp_path).manifest["research_settings"]
    assert first == second


@pytest.mark.parametrize("table_prior", [None, 0.6])
def test_edge_param_pusb_independent_ts_requires_a_training_table_prior(tmp_path, table_prior):
    class DeclaredTS(SCARGenerator):
        def generate(self, X, y_true, c, seed=None):  # noqa: N803
            labels, metadata = super().generate(X, y_true, c, seed=seed)
            return labels, {**metadata, "input_sampling": "ts"}

    model = PUSBKernelClassifier(
        training_table_prior=table_prior,
        risk_prior=0.25,
        n_basis=6,
        cv=2,
        sigma_grid=(0.5,),
        reg_grid=(0.1,),
    )
    if table_prior is None:
        with pytest.raises(RuntimeError, match="all candidate runs failed"):
            _fit(tmp_path, generator=DeclaredTS(), prior=0.4, model=model)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["research_settings"]["prior"] == []
        assert "requires training_table_prior" in manifest["failures"][0]["errors"][0]["message"]
    else:
        result = _fit(tmp_path, generator=DeclaredTS(), prior=0.4, model=model)
        threshold, risk, cv = result.manifest["research_settings"]["prior"]
        assert (threshold["value"], threshold["source"]) == (0.6, "estimator.training_table_prior")
        assert risk["value"] == cv["value"] == 0.25


class _LegacyPriorModel:
    def fit(self, features, labels, *, class_prior=None):
        return self

    def get_params(self, deep=False):
        return {"class_prior": 0.3}


def _build_settings(model=None, *, config=None):
    return research_settings.build_research_settings(
        model=model if model is not None else _LegacyPriorModel(),
        trainer=FitTrainer(),
        config=config or {},
        context={},
        generation={"train": {"input_sampling": "os"}, "pu_val": {}},
        input_ndim=2,
        selection_roles={"OA": "clean_val"},
        label_view="pu",
    )


def test_basic_new_writes_never_infer_consumption_from_legacy_model_parameters():
    settings = _build_settings(config={"candidates": [{"class_prior": 0.2}]})
    assert settings["schema_version"] == "2.0"
    assert settings["prior"] == []


@pytest.mark.parametrize("name", ["neural_lbe", "rank_pruning"])
def test_param_native_os_methods_keep_their_estimator_defined_pool(name):
    from pu_toolbox.estimators.classic.rank_pruning import RankPruningClassifier
    from pu_toolbox.estimators.deep.lbe_neural import NeuralLBEClassifier

    models = {"neural_lbe": NeuralLBEClassifier(), "rank_pruning": RankPruningClassifier()}

    settings = _build_settings(models[name], config={"os_or_ts": "os"})

    assert settings["risk"]["pool"] == "estimator_defined"


def test_basic_cvir_candidate_inputs_do_not_create_prior_consumption():
    from pu_toolbox.estimators.risk.cvir import CVIRClassifier

    settings = _build_settings(
        CVIRClassifier(unlabeled_positive_prior=0.2, prior_source="study setting"),
        config={"candidates": [{}, {"unlabeled_positive_prior": 0.35}]},
    )

    assert settings["schema_version"] == "2.0"
    assert settings["prior"] == []


def test_basic_finalization_summarizes_only_recorded_prior_consumption():
    settings = _build_settings()
    settings["supervision"]["selection_roles"] = {"custom": "pu_val"}
    settings["prior"].append(
        {
            "candidate_index": 0,
            "consumer": "custom_risk",
            "value": 0.2,
            "partition": "train",
            "target": "native_unlabeled",
            "source": "custom_trainer.risk_prior",
        }
    )
    research_settings.finalize_research_settings(settings)
    assert "candidate 0 custom_risk=0.2" in settings["summary"]
    assert "custom=pu_val" in settings["summary"]
    assert "OA=clean_val" not in settings["summary"]


def test_determ_finalizing_settings_twice_preserves_the_same_summary():
    settings = _build_settings()

    research_settings.finalize_research_settings(settings)
    first = json.dumps(settings, sort_keys=True)
    research_settings.finalize_research_settings(settings)

    assert json.dumps(settings, sort_keys=True) == first
