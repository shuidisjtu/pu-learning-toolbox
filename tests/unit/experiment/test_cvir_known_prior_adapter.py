# ruff: noqa: N806
"""Known alpha_U must not be overwritten by selection pi or a view flag."""

import numpy as np
import pytest

pytest.importorskip("torch")

from pu_toolbox.estimators.risk.cvir import CVIRClassifier  # noqa: E402
from pu_toolbox.experiment.bundle import DatasetPart  # noqa: E402
from pu_toolbox.experiment.cvir_adapter import CVIRKnownPriorTrainer  # noqa: E402
from pu_toolbox.experiment.method_ledger import load_ledger  # noqa: E402
from pu_toolbox.experiment.runner import ExperimentRunner  # noqa: E402
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA  # noqa: E402
from pu_toolbox.experiment.training_views import resolve_training_view  # noqa: E402

pytestmark = pytest.mark.unit


def model(**kwargs):
    options = dict(
        unlabeled_positive_prior=0.25,
        prior_source="synthetic distribution parameter",
        hidden_dim=4,
        warm_start_epochs=1,
        max_epochs=2,
        batch_size=8,
        device="cpu",
        random_state=0,
    )
    return CVIRClassifier(**(options | kwargs))


def parts():
    X = np.random.RandomState(1).normal(size=(40, 2)).astype("float32")
    y = np.tile([0, 1], 20)
    return tuple(
        DatasetPart(X[a:b], y[a:b], "clean", np.arange(a, b), i != 3)
        for i, (a, b) in enumerate(((0, 20), (20, 28), (28, 34), (34, 40)))
    )


def test_basic_manifest_separates_alpha_from_population_pi(tmp_path):
    runner = ExperimentRunner(
        class_prior=0.4,
        protocols=[ProtocolPA(), ProtocolOA()],
        config={"c": 0.5, "trainer": CVIRKnownPriorTrainer()},
        manifest_path=str(tmp_path / "manifest.json"),
    )
    result = runner.fit(model(), *parts())
    provenance = result.manifest["candidate_runs"][0]["training_provenance"]
    assert provenance["unlabeled_positive_prior"] == 0.25
    assert provenance["runner_fit_population_prior_ignored"] == 0.4
    assert provenance["population_prior_forwarded_to_fit"] is False
    assert provenance["known_prior"] and provenance["prior_source"]
    assert provenance["source_verified"] is False
    assert provenance["empirical_unlabeled_positive_fraction"] is None
    assert provenance["n_unlabeled_candidates"] == 15
    assert provenance["positive_rows_added_to_candidates"] == 0
    assert provenance["path"] == "explicit_os_adaptation"
    assert provenance["calibration_applied"] is False
    assert result.manifest["run_view"] == "os-compatible"
    assert not result.manifest["formal_eligible"]
    assert result.manifest["selection"]["PA"]["class_prior"]["population"] == 0.4
    assert len(result.manifest["candidate_runs"][0]["epoch_checkpoints"]) == 3


@pytest.mark.parametrize("alpha", [None, True, "0.25", 0, 1, float("nan"), 0.25 + 0.1j])
def test_param_invalid_alpha_is_refused(alpha):
    part = parts()[0]
    with pytest.raises(ValueError, match="unlabeled_positive_prior"):
        model(unlabeled_positive_prior=alpha).fit(part.X, part.labels)


def test_edge_native_ts_input_keeps_only_original_u_as_candidates():
    part = parts()[0]
    fitted = model(
        input_sampling="ts",
        unlabeled_positive_prior=0.4,
        prior_source="known marginal population parameter",
    ).fit(part.X, part.labels)
    assert fitted.training_provenance_["path"] == "native_ts_input"
    assert not fitted.training_provenance_["calibration_applied"]
    assert fitted.training_provenance_["n_unlabeled_candidates"] == 10
    assert fitted.history_["selected_negative_count"] == [6, 6]
    assert set(fitted.selected_negative_indices_) <= set(np.flatnonzero(part.labels == 0))
    assert set(fitted.training_provenance_["unlabeled_roles"]) == {
        "warm_start",
        "ranking",
        "negative_retention",
        "weighted_loss",
    }


@pytest.mark.parametrize(
    "overrides,view", [({"input_sampling": "ts"}, None), ({"prior_source": None}, None), ({}, "ts")]
)
def test_edge_os_runner_refuses_unsupported_paths_before_fit(overrides, view, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("preflight must reject without training")

    monkeypatch.setattr(CVIRClassifier, "_fit_impl", never)
    runner = ExperimentRunner(
        class_prior=0.4,
        protocols=[ProtocolOA()],
        config={"c": 0.5, "trainer": CVIRKnownPriorTrainer(), "os_or_ts": view},
    )
    with pytest.raises(ValueError):
        runner.fit(model(**overrides), *parts())
    with pytest.raises(ValueError, match="calibration_hooked=false"):
        resolve_training_view(
            load_ledger(), "cvir", "ts", is_oracle=False, estimator_class=CVIRClassifier
        )


def test_determ_prior_provenance_does_not_change_training():
    part = parts()[0]
    first = model().fit(part.X, part.labels)
    second = model(prior_source="different description").fit(part.X, part.labels)
    np.testing.assert_array_equal(first.decision_function(part.X), second.decision_function(part.X))


def test_edge_native_ts_requires_traceable_prior_and_sar_is_not_admitted():
    part = parts()[0]
    with pytest.raises(ValueError, match="prior_source"):
        model(input_sampling="ts", prior_source=None).fit(part.X, part.labels)
    with pytest.raises(ValueError, match="SCAR only"):
        CVIRKnownPriorTrainer().validate_generation(
            {"mechanism": "sar_lbe_a"}, {"mechanism": "scar"}
        )
    trajectory = CVIRKnownPriorTrainer().fit(
        model(unlabeled_positive_prior=np.float32(0.25)), part.X, part.labels
    )
    assert trajectory.model.training_provenance_["unlabeled_positive_prior"] == 0.25
