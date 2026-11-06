"""Research settings describe the generated roles and supplied training parameters."""

import json

import numpy as np
import pytest

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


def _fit(tmp_path, *, view="os", generator=None, prior=None):
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
        class_prior=prior,
        generator=generator or SCARGenerator(),
        protocols=[ProtocolOA()],
        manifest_path=str(tmp_path / "manifest.json"),
        config={"c": 0.5, "os_or_ts": view, "trainer": FitTrainer()},
    ).fit(UPUClassifier(0.3, random_state=7), *parts)


@pytest.mark.parametrize(
    ("view", "pool", "target"),
    [
        ("os", "native_unlabeled", "native_unlabeled"),
        ("ts", "positive_union_unlabeled", "risk_pool"),
    ],
)
def test_param_recorded_risk_and_prior_follow_the_training_view(tmp_path, view, pool, target):
    result = _fit(tmp_path, view=view)
    settings = result.manifest["research_settings"]
    assert settings["sampling"]["input_sampling"] == "os"
    assert settings["risk"]["pool"] == pool
    assert settings["prior"]["training"] == {
        "value": 0.3,
        "target": target,
        "source": "estimator.class_prior",
        "declared_target": "native_unlabeled",
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
    settings = _fit(tmp_path, prior=0.4).manifest["research_settings"]
    assert settings["prior"]["training"]["value"] == 0.4
    assert settings["prior"]["training"]["source"] == "runner.class_prior"
    assert settings["prior"]["selection"]["OA"]["population"] == 0.4


def test_determ_same_seed_has_identical_settings_with_or_without_disk(tmp_path):
    first = _fit(tmp_path).manifest["research_settings"]
    second = _fit(tmp_path).manifest["research_settings"]
    assert first == second


def _build_settings(model=None, *, config=None, prior=None):
    return research_settings.build_research_settings(
        model=model if model is not None else UPUClassifier(0.3),
        trainer=FitTrainer(),
        config=config or {},
        context={},
        generation={"train": {"input_sampling": "os"}, "pu_val": {}},
        input_ndim=2,
        training_prior=prior,
        selection_prior=None,
        selection_prior_source=None,
        selection_roles={"OA": "clean_val"},
        label_view="pu",
    )


@pytest.mark.parametrize(
    ("prior", "candidates", "values", "sources"),
    [
        (
            None,
            [{"class_prior": 0.2}, {"class_prior": 0.4}],
            [0.2, 0.4],
            ["candidate.class_prior"] * 2,
        ),
        (0.45, [{"class_prior": 0.2}, {}], [0.45, 0.45], ["runner.class_prior"] * 2),
        (None, [{}], [0.3], ["estimator.class_prior"]),
    ],
)
def test_param_candidate_prior_inputs_follow_actual_override_order(
    prior, candidates, values, sources
):
    settings = _build_settings(config={"candidates": candidates}, prior=prior)

    inputs = settings["prior"]["candidates"]
    assert [item["candidate_index"] for item in inputs] == list(range(len(candidates)))
    assert [item["value"] for item in inputs] == values
    assert [item["source"] for item in inputs] == sources
    assert "base prior input=" in settings["summary"]
    assert all(
        f"candidate {index} prior input={value}" in settings["summary"]
        for index, value in enumerate(values)
    )


@pytest.mark.parametrize("name", ["neural_lbe", "rank_pruning"])
def test_param_native_os_methods_keep_their_estimator_defined_pool(name):
    from pu_toolbox.estimators.classic.rank_pruning import RankPruningClassifier
    from pu_toolbox.estimators.deep.lbe_neural import NeuralLBEClassifier

    models = {"neural_lbe": NeuralLBEClassifier(), "rank_pruning": RankPruningClassifier()}

    settings = _build_settings(models[name], config={"os_or_ts": "os"})

    assert settings["risk"]["pool"] == "estimator_defined"


def test_basic_cvir_candidate_inputs_record_constructor_alpha_and_source():
    from pu_toolbox.estimators.risk.cvir import CVIRClassifier

    settings = _build_settings(
        CVIRClassifier(unlabeled_positive_prior=0.2, prior_source="study setting"),
        config={"candidates": [{}, {"unlabeled_positive_prior": 0.35}]},
        prior=0.4,
    )

    candidates = settings["prior"]["candidates"]
    assert [item["value"] for item in candidates] == [0.4, 0.4]
    assert [item["unlabeled_positive_prior"] for item in candidates] == [0.2, 0.35]
    assert all(item["prior_source"] == "study setting" for item in candidates)


def test_edge_finalization_preserves_actual_candidate_provenance_and_current_roles():
    settings = _build_settings()
    settings["supervision"]["selection_roles"] = {"custom": "pu_val"}
    provenance = {
        "unlabeled_positive_prior": 0.2,
        "prior_source": "study setting",
        "risk_pool": "original_U",
        "population_prior_forwarded_to_fit": False,
        "population_prior_binding": {"population_prior": 0.4, "risk_pool_positive_prior": 0.2},
    }
    runs = [
        {"candidate_index": 0, "training_provenance": provenance},
        {"candidate_index": 1, "status": "excluded"},
    ]

    research_settings.finalize_research_settings(settings, runs)

    assert settings["prior"]["candidate_training"] == [
        {"candidate_index": 0, "training_provenance": provenance}
    ]
    assert "unlabeled_positive_prior=0.2" in settings["summary"]
    assert "prior_source=study setting" in settings["summary"]
    assert "risk_pool=original_U" in settings["summary"]
    assert "risk_pool_positive_prior=0.2" in settings["summary"]
    assert "population_prior_forwarded_to_fit=False" in settings["summary"]
    assert "custom=pu_val" in settings["summary"]
    assert "OA=clean_val" not in settings["summary"]
    provenance["population_prior_binding"]["population_prior"] = 0.9
    assert (
        settings["prior"]["candidate_training"][0]["training_provenance"][
            "population_prior_binding"
        ]["population_prior"]
        == 0.4
    )


def test_determ_finalizing_settings_twice_preserves_the_same_summary():
    settings = _build_settings()
    runs = [{"candidate_index": 0, "training_provenance": {"risk_pool": "original_U"}}]

    research_settings.finalize_research_settings(settings, runs)
    first = json.dumps(settings, sort_keys=True)
    research_settings.finalize_research_settings(settings, runs)

    assert json.dumps(settings, sort_keys=True) == first
