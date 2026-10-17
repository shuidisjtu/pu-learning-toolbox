# ruff: noqa: N806
"""Extra-support experiments require real role/label-budget evidence."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from pu_toolbox.estimators.deep.pulns import PULNSClassifier  # noqa: E402
from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart  # noqa: E402
from pu_toolbox.experiment.pulns_adapter import PULNSExtraSupportTrainer  # noqa: E402
from pu_toolbox.experiment.runner import ExperimentRunner  # noqa: E402
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA  # noqa: E402

pytestmark = pytest.mark.unit


def inputs():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 3)).astype("float32")
    y = np.tile([0, 1], 20)
    roles = tuple(
        DatasetPart(X[a:b].copy(), y[a:b].copy(), "clean", np.arange(a, b), i != 3)
        for i, (a, b) in enumerate(((0, 20), (20, 28), (28, 34), (34, 40)))
    )
    support = DatasetPart(
        rng.normal(size=(6, 3)).astype("float32"),
        np.tile([0, 1], 3),
        "clean",
        np.arange(40, 46),
        False,
    )
    return roles, support


def model():
    return PULNSClassifier(
        hidden_dim=4,
        pretrain_epochs=1,
        episodes=2,
        classifier_epochs=1,
        batch_size=8,
        device="cpu",
        random_state=0,
    )


def adapter(support, **kwargs):
    options = dict(
        dataset="spambase",
        source="external synthetic clean cohort",
        prior_estimation_indices=[],
        original_train_count=20,
        support_removed_from_train=False,
        group_policy="not_applicable",
        preparation_seed=0,
        sampling_rule="fixed external synthetic cohort, no resampling",
    )
    return PULNSExtraSupportTrainer(support, **(options | kwargs))


def run(trainer, roles, path=None):
    return ExperimentRunner(
        seed=0, protocols=[ProtocolOA()], manifest_path=path, config={"c": 0.5, "trainer": trainer}
    ).fit(model(), *roles)


def test_basic_manifest_prominently_accounts_extra_labels_and_preserves_pickle(tmp_path):
    import pickle

    roles, support = inputs()
    path = tmp_path / "manifest.json"
    result = run(adapter(support), roles, str(path))
    audit = result.manifest["extra_supervision"]
    assert audit["extra_supervision"] and audit["comparison_group"] == "extra_clean_support"
    assert audit["support_positive_labels"] == audit["support_negative_labels"] == 3
    assert audit["isolation_status"] == "all_role_ids_disjoint"
    assert not audit["prior_estimation_used"] and not audit["pa_eligible"]
    assert audit["role_counts"] == dict(
        train=20, pu_val=8, clean_val=6, test=6, prior_estimation=0, support=6
    )
    assert audit["support_label_uses"] == ["training_reward", "internal_best_classifier_selection"]
    assert audit["support_origin"] == "external"
    assert audit["support_source_verified"] is False
    assert audit["source_verification_scope"] == "caller_assertion_unverified"
    assert audit["role_pool_effect"]["train"] == {
        "original_count": 20,
        "remaining_count": 20,
        "reduced_count": 0,
    }
    assert audit["support_pool_effect_consistent"] is True
    assert audit["source_recipe_replay"] is False and not result.manifest["formal_eligible"]
    provenance = result.manifest["candidate_runs"][0]["training_provenance"]
    assert provenance["total_observed_training_label_budget"] == 11
    assert provenance["observed_labeled_positive_count"] == 5
    assert provenance["stage_optimizer_steps"]["policy"] == 2
    assert json.loads(path.read_text())["extra_supervision"] == audit
    selected = result.selected_models["OA"]
    restored = pickle.loads(pickle.dumps(selected))
    np.testing.assert_array_equal(
        selected.decision_function(roles[-1].X), restored.decision_function(roles[-1].X)
    )


@pytest.mark.parametrize("role", ["train", "pu_val", "clean_val", "test", "prior_estimation"])
def test_edge_same_sample_copy_in_any_role_is_rejected_before_fit(role, monkeypatch):
    roles, support = inputs()
    ids = support.indices.copy()
    ids[0] = {"train": 0, "pu_val": 20, "clean_val": 28, "test": 34, "prior_estimation": 100}[role]
    copied = DatasetPart(support.X.copy(), support.labels.copy(), "clean", ids, False)

    def never(*args, **kwargs):
        raise AssertionError("overlap must be caught before training")

    monkeypatch.setattr(PULNSClassifier, "fit", never)
    options = {"prior_estimation_indices": [100]} if role == "prior_estimation" else {}
    with pytest.raises(ValueError, match="overlap"):
        run(adapter(copied, **options), roles)


@pytest.mark.parametrize("protocols", [[ProtocolPA()], [ProtocolPA(), ProtocolOA()]])
def test_edge_extra_support_cannot_be_pu_only_pa(protocols):
    roles, support = inputs()
    with pytest.raises(ValueError, match="OA only"):
        runner = ExperimentRunner(protocols=protocols, config={"trainer": adapter(support)})
        runner.fit(model(), *roles)


def test_edge_group_isolation_and_unknown_adni_subjects():
    roles, support = inputs()
    with pytest.raises(ValueError, match="ADNI requires"):
        run(adapter(support, dataset="adni"), roles)
    names = ["train", "pu_val", "clean_val", "test"]
    groups = {name: part.indices + 1000 for name, part in zip(names, roles, strict=True)}
    groups.update(support=support.indices + 1000, prior_estimation=[])
    grouped = adapter(support, group_policy="grouped", role_groups=groups)
    result = run(grouped, roles)
    assert result.manifest["extra_supervision"]["group_ids_sha256"]
    groups["support"][0] = groups["test"][0]
    with pytest.raises(ValueError, match="overlap"):
        run(grouped, roles)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": ""},
        {"original_train_count": 99},
        {"support_removed_from_train": "yes"},
        {"group_policy": "unknown"},
        {"prior_estimation_indices": [float("nan")]},
        {"prior_estimation_indices": [100, 100]},
    ],
)
def test_param_incomplete_budget_identity_or_origin_rejected(kwargs):
    roles, support = inputs()
    with pytest.raises(ValueError):
        run(adapter(support, **kwargs), roles)


def test_determ_test_truth_does_not_affect_training_or_oa_selection():
    roles, support = inputs()
    first = run(adapter(support), roles)
    test = roles[-1]
    changed = (*roles[:-1], DatasetPart(test.X + 20, 1 - test.labels, "clean", test.indices, False))
    second = run(adapter(support), changed)
    assert first.manifest["selection"] == second.manifest["selection"]
    np.testing.assert_array_equal(
        first.selected_models["OA"].decision_function(roles[0].X),
        second.selected_models["OA"].decision_function(roles[0].X),
    )


def test_edge_support_mutation_after_preflight_is_rejected():
    roles, support = inputs()
    trainer = adapter(support)
    trainer.validate_experiment(model(), DatasetBundle(*roles), {}, [ProtocolOA()])
    support.labels[:] = 1 - support.labels
    with pytest.raises(ValueError, match="changed after"):
        trainer.fit(model(), roles[0].X, roles[0].labels)


def test_basic_train_pool_reduction_and_prior_role_are_recorded():
    roles, support = inputs()
    result = run(
        adapter(
            support,
            support_removed_from_train=True,
            original_train_count=26,
            prior_estimation_indices=[100, 101],
        ),
        roles,
    )
    audit = result.manifest["extra_supervision"]
    assert audit["support_removed_from_train"] and audit["original_train_count"] == 26
    assert audit["remaining_train_count"] == 20 and audit["prior_estimation_used"]
    assert audit["support_origin"] == "carved_from_original_pool"
    assert audit["role_pool_effect"]["train"] == {
        "original_count": 26,
        "remaining_count": 20,
        "reduced_count": 6,
    }
    assert audit["role_pool_effect"]["pu_val"]["reduced_count"] == 0
    assert audit["support_pool_effect_consistent"] is True


def test_basic_dataset_plan_covers_approved_mappings_and_keeps_admission_closed():
    from pathlib import Path

    from pu_toolbox.experiment.datasets import survey_dataset_catalog
    from pu_toolbox.experiment.method_ledger import load_ledger

    entry = load_ledger()["methods"]["pulns"]["extra_supervision_adapter"]
    root = Path(__file__).resolve().parents[3]
    plan = json.loads((root / entry["dataset_rules_ref"]).read_text())
    rows = {row["dataset"]: row for row in plan["datasets"]}
    assert rows.keys() == survey_dataset_catalog().keys()
    for name, spec in survey_dataset_catalog().items():
        assert rows[name]["survey_positive"] == list(spec.positive_classes)
        assert rows[name]["survey_negative"] == list(spec.negative_classes)
    assert {name for name, row in rows.items() if row["paper_covered"]} == {
        "mnist",
        "cifar10",
        "spambase",
    }
    assert plan["official_source_search"]["status"] == "not_found"
    assert not plan["formal_admission"]
    assert not plan["common_project_rule"]["formal_support_artifacts_materialized"]


def test_edge_incomplete_group_map_and_sar_generation_are_rejected():
    roles, support = inputs()
    with pytest.raises(ValueError, match="every role"):
        run(adapter(support, group_policy="grouped", role_groups={"support": []}), roles)
    with pytest.raises(ValueError, match="SCAR only"):
        adapter(support).validate_generation({"mechanism": "sar_lbe_a"}, {"mechanism": "scar"})


def test_basic_carved_support_records_multi_role_pool_effect():
    roles, support = inputs()
    original = {
        "train": 22,
        "pu_val": 10,
        "clean_val": 8,
        "test": 6,
        "prior_estimation": 0,
    }
    result = run(
        adapter(
            support,
            support_origin="carved_from_original_pool",
            original_role_counts=original,
            original_train_count=22,
            support_removed_from_train=True,
        ),
        roles,
    )
    effect = result.manifest["extra_supervision"]["role_pool_effect"]
    assert {name: row["reduced_count"] for name, row in effect.items()} == {
        "train": 2,
        "pu_val": 2,
        "clean_val": 2,
        "test": 0,
        "prior_estimation": 0,
    }
    assert sum(row["reduced_count"] for row in effect.values()) == 6
    assert result.manifest["extra_supervision"]["support_pool_effect_consistent"] is True


def test_edge_carved_pool_effect_must_equal_support_size():
    roles, support = inputs()
    with pytest.raises(ValueError, match="reductions must equal"):
        run(
            adapter(
                support,
                support_origin="carved_from_original_pool",
                original_role_counts={
                    "train": 21,
                    "pu_val": 9,
                    "clean_val": 7,
                    "test": 6,
                    "prior_estimation": 0,
                },
                original_train_count=21,
                support_removed_from_train=True,
            ),
            roles,
        )
    with pytest.raises(ValueError, match="original train reduction requires"):
        run(
            adapter(
                support,
                support_origin="carved_from_original_pool",
                original_role_counts={
                    "train": 22,
                    "pu_val": 10,
                    "clean_val": 8,
                    "test": 6,
                    "prior_estimation": 0,
                },
                original_train_count=22,
                support_removed_from_train=False,
            ),
            roles,
        )


def test_edge_external_support_cannot_reduce_train_or_declare_original_counts():
    roles, support = inputs()
    with pytest.raises(ValueError, match="external support cannot"):
        run(adapter(support, support_origin="external", support_removed_from_train=True), roles)
    with pytest.raises(ValueError, match="external support must not"):
        run(
            adapter(
                support,
                support_origin="external",
                original_role_counts={
                    "train": 20,
                    "pu_val": 8,
                    "clean_val": 6,
                    "test": 6,
                    "prior_estimation": 0,
                },
            ),
            roles,
        )


def test_basic_verified_source_requires_receipt_and_records_scope():
    roles, support = inputs()
    with pytest.raises(ValueError, match="receipt"):
        run(adapter(support, support_source_verified=True), roles)
    result = run(
        adapter(
            support,
            support_source_verified=True,
            support_source_receipt="cohort.json#sha256=abc",
        ),
        roles,
    )
    audit = result.manifest["extra_supervision"]
    assert audit["support_source_verified"] is True
    assert audit["source_verification_scope"] == "verified_receipt"
    assert audit["support_source_receipt"] == "cohort.json#sha256=abc"


def test_edge_support_role_flags_mutation_after_preflight_is_rejected():
    roles, support = inputs()
    trainer = adapter(support)
    trainer.validate_experiment(model(), DatasetBundle(*roles), {}, [ProtocolOA()])
    trainer.support = DatasetPart(support.X, support.labels, "pu", support.indices, False)
    with pytest.raises(ValueError, match="view or selection role changed"):
        trainer.fit(model(), roles[0].X, roles[0].labels)
