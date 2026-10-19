"""Full-meta Self-PU / LaGAM extra-label paths must not reuse selection truth."""

import json
import pickle

import numpy as np
import pytest

pytest.importorskip("torch")

from pu_toolbox.estimators.deep.lagam import LaGAMClassifier  # noqa: E402
from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier  # noqa: E402
from pu_toolbox.experiment.bundle import DatasetPart  # noqa: E402
from pu_toolbox.experiment.clean_support_adapters import (  # noqa: E402
    LaGAMCleanSupportTrainer,
    SelfPUCleanSupportTrainer,
)
from pu_toolbox.experiment.runner import ExperimentRunner  # noqa: E402
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA  # noqa: E402

pytestmark = pytest.mark.unit


def inputs():
    rng = np.random.RandomState(7)
    features = rng.normal(size=(60, 3)).astype("float32")
    truth = np.tile([0, 1], 30)
    parts = tuple(
        DatasetPart(features[a:b], truth[a:b], "clean", np.arange(a, b), i != 3)
        for i, (a, b) in enumerate(((0, 30), (30, 40), (40, 50), (50, 60)))
    )
    support = DatasetPart(
        rng.normal(size=(8, 3)).astype("float32"),
        np.tile([0, 1], 4),
        "clean",
        np.arange(60, 68),
        False,
    )
    return parts, support


def estimator(method):
    if method == "self_pu":
        return SelfPUClassifier(
            class_prior=0.5,
            hidden_dim=4,
            warmup_epochs=0,
            self_paced_start=0,
            self_paced_end=1,
            distill_start=2,
            max_epochs=3,
            batch_size=8,
            random_state=0,
            device="cpu",
            require_validation=True,
        )
    return LaGAMClassifier(
        hidden_dim=4,
        warmup_epochs=0,
        max_epochs=2,
        batch_size=8,
        support_batch_size=8,
        num_clusters=2,
        random_state=0,
        device="cpu",
    )


def trainer(method, support):
    cls = SelfPUCleanSupportTrainer if method == "self_pu" else LaGAMCleanSupportTrainer
    return cls(
        support,
        dataset="spambase",
        source="independent synthetic clean cohort",
        prior_estimation_indices=[],
        original_train_count=30,
        support_removed_from_train=False,
        preparation_seed=7,
        sampling_rule="fixed external cohort",
        group_policy="not_applicable",
    )


def run(method, parts, support, path=None):
    return ExperimentRunner(
        seed=0,
        class_prior=0.5,
        protocols=[ProtocolOA()],
        manifest_path=path,
        config={"c": 0.5, "trainer": trainer(method, support)},
    ).fit(estimator(method), *parts)


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_basic_actual_meta_training_and_saved_provenance(method, tmp_path):
    parts, support = inputs()
    path = tmp_path / "manifest.json"
    result = run(method, parts, support, str(path))
    metadata = result.manifest["candidate_runs"][0]["training_provenance"]
    assert metadata["method"] == method and metadata["all_role_isolation_verified"]
    assert metadata["support_positive_labels"] == metadata["support_negative_labels"] == 4
    assert metadata["total_observed_training_label_budget"] == 16  # round(15*.5) + 8
    assert metadata["formal_admission"] is False and metadata["pa_eligible"] is False
    assert result.manifest["extra_supervision"]["comparison_group"] == "extra_clean_support"
    assert json.loads(path.read_text())["extra_supervision"]["method"] == method
    fitted = result.selected_models["OA"]
    if method == "self_pu":
        assert fitted.calibration_mode_ == "clean_validation_meta"
        assert fitted.distillation_scope_ == "paper_population"
        assert len(fitted.reweight_history_) > 0
        assert fitted.teacher_selection_basis_ == "clean_validation_accuracy"
        assert metadata["meta_reweight_records"] == len(fitted.reweight_history_)
        assert metadata["active_meta_reweight_records"] > 0
        assert metadata["distillation_scope"] == "paper_population"
        assert any(row["calibration_active"] for row in fitted.reweight_history_)
    else:
        assert fitted.virtual_steps_ > 0
        assert metadata["support_label_uses"] == ["meta_label_disambiguation"]
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(
        fitted.decision_function(parts[-1].X), restored.decision_function(parts[-1].X)
    )


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
@pytest.mark.parametrize("role", [0, 1, 2, 3])
def test_edge_all_selection_or_train_support_overlap_refused(method, role, monkeypatch):
    parts, support = inputs()
    ids = support.indices.copy()
    ids[0] = parts[role].indices[0]
    invalid = DatasetPart(support.X.copy(), support.labels.copy(), "clean", ids, False)

    def never(*args, **kwargs):
        raise AssertionError("overlap must be rejected before actual fit")

    monkeypatch.setattr(type(estimator(method)), "fit", never)
    with pytest.raises(ValueError, match="overlap"):
        run(method, parts, invalid)


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_param_pa_and_ts_not_silently_supported(method):
    parts, support = inputs()
    with pytest.raises(ValueError, match="OA only"):
        ExperimentRunner(
            protocols=[ProtocolPA()], class_prior=0.5, config={"trainer": trainer(method, support)}
        ).fit(estimator(method), *parts)
    with pytest.raises(ValueError):
        ExperimentRunner(
            protocols=[ProtocolOA()],
            class_prior=0.5,
            config={"trainer": trainer(method, support), "os_or_ts": "ts"},
        ).fit(estimator(method), *parts)


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_edge_warmup_only_budget_refused_before_fit(method, monkeypatch):
    parts, support = inputs()
    model = estimator(method)
    if method == "self_pu":
        model.self_paced_start = model.max_epochs
    else:
        model.warmup_epochs = model.max_epochs

    def never(*args, **kwargs):
        raise AssertionError("inactive meta budget must be rejected before fit")

    monkeypatch.setattr(type(model), "fit", never)
    adapter = trainer(method, support)
    with pytest.raises(ValueError, match="budget"):
        ExperimentRunner(
            protocols=[ProtocolOA()],
            class_prior=0.5,
            config={"trainer": adapter},
        ).fit(model, *parts)
    assert not adapter._prepared


def test_basic_selfpu_meta_receives_only_explicit_support(monkeypatch):
    parts, support = inputs()
    actual = SelfPUClassifier._meta_weights
    calls = []

    def inspect(self, model, positive_loss, ce_losses, pu_losses, val_x, val_y):
        np.testing.assert_array_equal(val_x.detach().cpu().numpy(), support.X)
        np.testing.assert_array_equal(val_y.detach().cpu().numpy(), support.labels)
        calls.append(True)
        return actual(self, model, positive_loss, ce_losses, pu_losses, val_x, val_y)

    monkeypatch.setattr(SelfPUClassifier, "_meta_weights", inspect)
    run("self_pu", parts, support)
    assert calls


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_determ_test_truth_and_features_do_not_affect_meta_or_selection(method):
    parts, support = inputs()
    first = run(method, parts, support)
    original = parts[-1]
    changed = (
        *parts[:-1],
        DatasetPart(original.X + 20, 1 - original.labels, "clean", original.indices, False),
    )
    second = run(method, changed, support)
    assert first.manifest["selection"] == second.manifest["selection"]
    np.testing.assert_array_equal(
        first.selected_models["OA"].decision_function(parts[0].X),
        second.selected_models["OA"].decision_function(parts[0].X),
    )
