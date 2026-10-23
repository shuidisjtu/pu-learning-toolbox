# ruff: noqa: N803, N806
"""Full-meta TS support is real native training, never a PU-only PA row."""

import copy
import functools
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from _support_checkpoint_helpers import inputs, model  # noqa: E402

from pu_toolbox.estimators.deep import self_pu as native  # noqa: E402
from pu_toolbox.experiment.bundle import DatasetBundle  # noqa: E402
from pu_toolbox.experiment.runner import ExperimentRunner  # noqa: E402
from pu_toolbox.experiment.strategies import (  # noqa: E402
    ProtocolOA,
    ProtocolPA,
    ProtocolPACleanSupport,
)
from pu_toolbox.experiment.support_checkpoints import (  # noqa: E402
    SelfPUCleanSupportCheckpointTrainer,
    SelfPUTSCleanSupportCheckpointTrainer,
)

pytestmark = pytest.mark.unit


def trainer(support, cls=SelfPUTSCleanSupportCheckpointTrainer):
    return cls(
        support,
        dataset="spambase",
        source="fixed independent synthetic support",
        prior_estimation_indices=[],
        original_train_count=30,
        support_removed_from_train=False,
        preparation_seed=7,
        sampling_rule="external synthetic cohort",
        group_policy="not_applicable",
    )


def settings(adapter, **kwargs):
    return (
        dict(
            c=0.5,
            os_or_ts="ts",
            trainer=adapter,
            checkpoint_bytes_per_component=65536,
            reclaim_unselected_checkpoints=True,
        )
        | kwargs
    )


def run(directory, parts=None, support=None, **kwargs):
    if parts is None:
        parts, support = inputs()
    return ExperimentRunner(
        seed=0,
        class_prior=0.5,
        protocols=[ProtocolPACleanSupport(), ProtocolOA()],
        config=settings(trainer(support), **kwargs),
        manifest_path=str(directory / "manifest.json"),
    ).fit(model("self_pu"), *parts)


def test_basic_actual_ts_meta_roles_both_teachers_cost_and_dual_selection(tmp_path, monkeypatch):
    parts, support = inputs()
    original, calls = native.SelfPUClassifier.fit, []
    marginal, domains = native._marginal_negative_risk, []
    trust_update, populations = native.TrustedSetManager.update, []

    @functools.wraps(original)
    def audit(self, X, y, **kwargs):
        result = original(self, X, y, **kwargs)
        calls.append((X.copy(), y.copy(), kwargs, self))
        return result

    def blend(U, P, **kwargs):
        domains.append(
            (len(P), kwargs["n_unlabeled_role"], kwargs["include_positive_in_unlabeled"])
        )
        return marginal(U, P, **kwargs)

    def population(manager, probabilities, target):
        populations.append(manager.n_unlabeled)
        return trust_update(manager, probabilities, target)

    monkeypatch.setattr(native.SelfPUClassifier, "fit", audit)
    monkeypatch.setattr(native, "_marginal_negative_risk", blend)
    monkeypatch.setattr(native.TrustedSetManager, "update", population)
    result = run(tmp_path, parts, support)
    assert len(calls) == 1 and not result.failures
    features, observed, kwargs, fitted = calls[0]
    np.testing.assert_array_equal(features, parts[0].X)
    assert set(kwargs) == {"class_prior", "validation_data", "os_or_ts", "epoch_callback"}
    assert kwargs["os_or_ts"] == "ts" and kwargs["validation_data"][0] is support.X
    np.testing.assert_array_equal(kwargs["validation_data"][1], support.labels)
    assert all(positive > 0 and untrusted > 0 and active for positive, untrusted, active in domains)
    assert set(populations) == {len(observed) - observed.sum()}
    for trusted in fitted.trusted_indices_.values():
        assert not set(trusted["indices"]).intersection(np.flatnonzero(observed))
    assert fitted.calibration_mode_ == "clean_validation_meta"
    facts = result.manifest["candidate_runs"][0]["training_provenance"]
    assert (
        facts["active_meta_reweight_records"] == 6
        and facts["distillation_scope"] == "paper_population"
    )
    assert facts["optimizer_steps"] == 6 and facts["student_optimizer_steps"] == {"1": 3, "2": 3}
    assert facts["total_observed_training_label_budget"] == observed.sum() + len(support.X)
    assert facts["calibration_applied"] and not facts["population_mixture_restoration_verified"]
    assert result.manifest["run_view"] == "ts-compatible" and not result.manifest["formal_eligible"]
    assert set(result.selected_models) == {"PA+clean-support", "OA"}
    assert not result.manifest["extra_supervision"]["pa_eligible"]
    rows = result.manifest["candidate_runs"][0]["epoch_checkpoints"]
    assert len(rows) == 6 and {r["component"] for r in rows} == {"teacher_1", "teacher_2"}
    assert 1 <= len(list((tmp_path / "checkpoints").rglob("*.pt"))) <= 2


@pytest.mark.parametrize("seed", [0, 7])
def test_determ_bridge_is_identical_to_same_full_meta_ts_native_fit(seed, tmp_path):
    parts, support = inputs()
    fitted = model("self_pu").set_params(random_state=seed)
    adapter = trainer(support)
    adapter.validate_experiment(fitted, DatasetBundle(*parts), settings(adapter), [ProtocolOA()])
    adapter.bind_checkpoint_directory(tmp_path)
    plain = copy.deepcopy(fitted)
    observed = np.r_[np.ones(7, int), np.zeros(23, int)]
    plain.fit(
        parts[0].X,
        observed,
        class_prior=0.5,
        validation_data=(support.X, support.labels),
        os_or_ts="ts",
    )
    adapter.fit(fitted, parts[0].X, observed, class_prior=0.5, os_or_ts="ts", val_pu=object())
    assert plain.history_ == fitted.history_ and plain.training_history_ == fitted.training_history_
    assert plain.reweight_history_ == fitted.reweight_history_
    for component in ("student_1_", "student_2_", "teacher_1_", "teacher_2_"):
        for name, value in getattr(plain, component).state_dict().items():
            torch.testing.assert_close(
                value, getattr(fitted, component).state_dict()[name], rtol=0, atol=0
            )
    np.testing.assert_array_equal(
        plain.decision_function(parts[3].X), fitted.decision_function(parts[3].X)
    )


@pytest.mark.parametrize("role", [1, 2, 3])
def test_edge_external_selection_and_test_changes_leave_all_training_weights_unchanged(
    role, tmp_path
):
    parts, support = inputs()
    original = run(tmp_path / "one", parts, support)
    changed = list(parts)
    changed[role] = replace(parts[role], X=parts[role].X + 5, labels=1 - parts[role].labels)
    other = run(tmp_path / "two", tuple(changed), support)
    one, two = (result.manifest["candidate_runs"][0] for result in (original, other))
    assert one["training_provenance"] == two["training_provenance"]
    assert [r["sha256"] for r in one["epoch_checkpoints"]] == [
        r["sha256"] for r in two["epoch_checkpoints"]
    ]


@pytest.mark.parametrize("view", [None, "os", "TS"])
def test_param_new_ts_adapter_requires_explicit_ts_and_old_os_still_rejects_ts(view):
    parts, support = inputs()
    adapter = trainer(support)
    with pytest.raises(ValueError, match="explicit os_or_ts"):
        adapter.validate_experiment(
            model("self_pu"),
            DatasetBundle(*parts),
            settings(adapter, os_or_ts=view),
            [ProtocolOA()],
        )
    old = trainer(support, SelfPUCleanSupportCheckpointTrainer)
    with pytest.raises(ValueError, match="OS only"):
        old.validate_experiment(
            model("self_pu"), DatasetBundle(*parts), settings(old), [ProtocolOA()]
        )


def test_edge_wrong_view_after_preflight_or_changed_support_refused_before_any_checkpoint(tmp_path):
    parts, support = inputs()
    adapter = trainer(support)
    fitted = model("self_pu")
    adapter.validate_experiment(fitted, DatasetBundle(*parts), settings(adapter), [ProtocolOA()])
    adapter.bind_checkpoint_directory(tmp_path)
    with pytest.raises(ValueError, match="explicit os_or_ts"):
        adapter.fit(fitted, parts[0].X, parts[0].labels, os_or_ts="os")
    support.labels[:] = 1 - support.labels
    with pytest.raises(ValueError, match="changed after"):
        adapter.fit(fitted, parts[0].X, parts[0].labels, os_or_ts="ts")
    assert not list(tmp_path.rglob("*.pt"))


def test_edge_plain_pa_and_no_meta_fallback_cannot_enter_ts_support(tmp_path):
    parts, support = inputs()
    adapter = trainer(support)
    fitted = model("self_pu")
    for protocols, override in (
        ([ProtocolPA()], {}),
        ([ProtocolOA()], {"require_validation": False}),
    ):
        with pytest.raises(ValueError):
            adapter.validate_experiment(
                fitted, DatasetBundle(*parts), settings(adapter, candidates=[override]), protocols
            )
    assert not adapter._prepared and not list(tmp_path.rglob("*.pt"))


def test_edge_warmup_is_paid_but_only_meta_active_teachers_are_selectable(tmp_path):
    result = run(
        tmp_path,
        candidates=[
            {
                "warmup_epochs": 1,
                "self_paced_start": 1,
                "self_paced_end": 2,
                "distill_start": 3,
            }
        ],
    )
    facts = result.manifest["candidate_runs"][0]["training_provenance"]
    assert facts["captured_meta_epochs"] == [2, 3]
    assert facts["executed_epochs"] == 3 and facts["optimizer_steps"] == 6
    assert facts["active_meta_reweight_records"] == 4
    assert len(result.manifest["candidate_runs"][0]["epoch_checkpoints"]) == 4


def test_edge_oversize_trajectory_refusal_revokes_model_and_keeps_evidence(tmp_path):
    parts, support = inputs()
    adapter, fitted = trainer(support), model("self_pu")
    adapter.validate_experiment(
        fitted,
        DatasetBundle(*parts),
        settings(adapter, checkpoint_bytes_per_component=1),
        [ProtocolOA()],
    )
    adapter.bind_checkpoint_directory(tmp_path)
    with pytest.raises(ValueError, match="actual checkpoint exceeds"):
        adapter.fit(fitted, parts[0].X, parts[0].labels, os_or_ts="ts", class_prior=0.5)
    assert not fitted._is_fitted and not hasattr(fitted, "training_provenance_")
    assert len(list(tmp_path.rglob("*.pt"))) == 6


def test_param_formal_registry_or_sar_cannot_be_approved_by_ts_adapter():
    parts, support = inputs()
    adapter = trainer(support)
    for option in ({"survey_protocol": {}}, {"recipe_registry": {}}):
        with pytest.raises(ValueError):
            adapter.validate_experiment(
                model("self_pu"), DatasetBundle(*parts), settings(adapter, **option), [ProtocolOA()]
            )
        assert not adapter._prepared
    with pytest.raises(ValueError, match="SCAR only"):
        adapter.validate_generation({"mechanism": "scar"}, {"mechanism": "sar_lbe_a"})
