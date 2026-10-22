"""Population-to-risk prior conversion, isolated roles, snapshots and guards."""

# ruff: noqa: E402, N803, N806

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pu_toolbox.estimators.risk.cvir import CVIRClassifier
from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.cvir_adapter import CVIRPopulationPriorTrainer, cvir_risk_prior
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA

pytestmark = pytest.mark.unit


def model(**kwargs):
    return CVIRClassifier(
        **(
            dict(hidden_dim=4, warm_start_epochs=1, max_epochs=2, batch_size=8, device="cpu")
            | kwargs
        )
    )


def parts():
    X = np.random.RandomState(1).normal(size=(80, 3)).astype("float32")
    X[:, 0] = np.arange(80)
    y = np.tile([0, 1], 40)
    return tuple(
        DatasetPart(X[a:b].copy(), y[a:b].copy(), "clean", np.arange(a, b), i != 3)
        for i, (a, b) in enumerate(((0, 40), (40, 56), (56, 68), (68, 80)))
    )


def trainer(**kwargs):
    return CVIRPopulationPriorTrainer(
        population_prior_source="synthetic known-prior fixture", **kwargs
    )


@pytest.mark.math
@pytest.mark.parametrize("n,p,pi", [(100, 10, 0.4), (19, 6, 0.7), (101, 13, 0.4), (100, 1, 0.1)])
def test_basic_prior_conversion_uses_actual_count_not_requested_c(n, p, pi):
    expected = (pi * n - p) / (n - p)
    assert cvir_risk_prior(pi, n_samples=n, n_labeled=p, os_or_ts="os") == pytest.approx(expected)
    assert cvir_risk_prior(pi, n_samples=n, n_labeled=p, os_or_ts="ts") == pi
    # Restoring the known labeled mass recovers the canonical prior exactly.
    assert (expected * (n - p) + p) / n == pytest.approx(pi)


@pytest.mark.parametrize("view", ["os", "ts"])
def test_basic_runner_binds_correct_prior_isolates_roles_and_preserves_default_config(
    tmp_path, monkeypatch, view
):
    original = CVIRClassifier.fit
    calls = []

    def audit(self, X, y, **kwargs):
        calls.append((X.copy(), y.copy(), self.unlabeled_positive_prior, dict(kwargs)))
        return original(self, X, y, **kwargs)

    # Preserve the public callback/view signature used by the checkpoint writer.
    import functools

    monkeypatch.setattr(CVIRClassifier, "fit", functools.wraps(original)(audit))
    inputs = parts()
    before = [(part.X.copy(), part.labels.copy()) for part in inputs]
    base = model()
    adapter = trainer()
    result = ExperimentRunner(
        class_prior=0.4,
        seed=7,
        protocols=[ProtocolPA(), ProtocolOA()],
        config={
            "c": 0.5,
            "trainer": adapter,
            "os_or_ts": view,
            "checkpoint_bytes_per_component": 65536,
            "reclaim_unselected_checkpoints": True,
        },
        manifest_path=str(tmp_path / "manifest.json"),
    ).fit(base, *inputs)
    assert len(calls) == 1 and not result.failures
    X, observed, alpha, kwargs = calls[0]
    assert set(X[:, 0]) == set(inputs[0].X[:, 0])
    assert observed.sum() == 10 and inputs[0].labels.sum() == 20
    assert set(kwargs) == {"epoch_callback", "os_or_ts"}
    assert kwargs["os_or_ts"] == view
    assert alpha == pytest.approx(0.4 if view == "ts" else 0.2)
    assert base.unlabeled_positive_prior is base.prior_source is None
    candidate = result.manifest["candidate_runs"][0]
    provenance = candidate["training_provenance"]
    binding = provenance["population_prior_binding"]
    assert binding["n_train"] == 40 and binding["n_labeled_positive"] == 10
    assert binding["population_prior"] == 0.4
    assert binding["risk_pool_positive_prior"] == pytest.approx(alpha)
    assert binding["source_verified"] is binding["uses_hidden_training_labels"] is False
    assert provenance["population_prior_forwarded_to_fit"] is False
    assert provenance["n_unlabeled_candidates"] == (40 if view == "ts" else 30)
    assert result.manifest["calibration_applied"] is (view == "ts")
    assert not result.manifest["formal_eligible"]
    assert result.manifest["selection"]["PA"]["class_prior"]["population"] == 0.4
    assert set(result.selected_models) == {"PA", "OA"}
    assert len(candidate["epoch_checkpoints"]) == 3
    for checkpoint in candidate["epoch_checkpoints"]:
        assert Path(checkpoint["path"]).is_relative_to(tmp_path / "checkpoints")
    for part, (X_before, labels_before) in zip(inputs, before, strict=True):
        np.testing.assert_array_equal(part.X, X_before)
        np.testing.assert_array_equal(part.labels, labels_before)


@pytest.mark.parametrize("view", ["os", "ts"])
def test_determ_fit_matches_manually_bound_prior_and_adapter_is_reusable(tmp_path, view):
    X = parts()[0].X
    y = np.r_[np.ones(10, int), np.zeros(30, int)]
    alpha = cvir_risk_prior(0.4, n_samples=40, n_labeled=10, os_or_ts=view)
    fitted = model(random_state=7)
    adapter = trainer(checkpoint_dir=tmp_path)
    expected = model(
        random_state=7, unlabeled_positive_prior=alpha, prior_source="explicit synthetic binding"
    ).fit(X, y, os_or_ts=view)
    for _ in range(2):
        trajectory = adapter.fit(fitted, X, y, class_prior=0.4, os_or_ts=view)
        assert fitted.unlabeled_positive_prior is fitted.prior_source is None
        assert fitted.unlabeled_positive_prior_ == pytest.approx(alpha)
        np.testing.assert_array_equal(fitted.decision_function(X), expected.decision_function(X))
        np.testing.assert_allclose(
            trajectory.checkpoints[-1].restore().decision_function(X),
            expected.decision_function(X),
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.parametrize("prior", [None, True, np.bool_(False), "0.4", 0, 1, np.nan, np.inf, 0.4j])
def test_param_invalid_population_prior(prior):
    with pytest.raises(ValueError, match="population prior"):
        cvir_risk_prior(prior, n_samples=40, n_labeled=10, os_or_ts="os")


@pytest.mark.parametrize("prior", [0.1, 0.25])
def test_edge_impossible_or_zero_os_prior_is_not_clipped(prior):
    with pytest.raises(ValueError, match="refusing to clip"):
        cvir_risk_prior(prior, n_samples=40, n_labeled=10, os_or_ts="os")


@pytest.mark.parametrize("n,p", [(0, 0), (10, 0), (10, 10), (10, 11), (True, 1), (10, 1.0)])
def test_edge_invalid_pool_counts(n, p):
    with pytest.raises(ValueError, match="positive P and U"):
        cvir_risk_prior(0.4, n_samples=n, n_labeled=p, os_or_ts="ts")
    with pytest.raises(ValueError, match="os_or_ts"):
        cvir_risk_prior(0.4, n_samples=40, n_labeled=10, os_or_ts="TS")


@pytest.mark.parametrize(
    "params",
    [{"unlabeled_positive_prior": 0.3}, {"prior_source": "alpha"}, {"input_sampling": "ts"}],
)
def test_param_candidate_cannot_override_prior_contract_before_any_fit(params, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid candidate reached training")

    monkeypatch.setattr(CVIRClassifier, "_fit_impl", forbidden)
    with pytest.raises(ValueError):
        ExperimentRunner(
            class_prior=0.4,
            config={
                "c": 0.5,
                "trainer": trainer(),
                "candidates": [{}, params],
                "checkpoint_bytes_per_component": 65536,
            },
        ).fit(model(), *parts())


def test_edge_sar_formal_and_disabled_checkpoints_are_refused():
    adapter = trainer()
    with pytest.raises(ValueError, match="SCAR only"):
        adapter.validate_generation({"mechanism": "scar"}, {"mechanism": "sar_lbe_a"})
    for config, message in [
        ({"survey_protocol": {}}, "exploratory"),
        ({"capture_epoch_checkpoints": False}, "completed epoch"),
        ({"checkpoint_bytes_per_component": True}, "checkpoint_bytes"),
    ]:
        with pytest.raises(ValueError, match=message):
            adapter.validate_experiment(model(), None, config, [])


def test_basic_largest_candidate_budget_controls_disk_preflight(tmp_path):
    adapter = trainer()
    adapter.validate_experiment(
        model(),
        None,
        {"checkpoint_bytes_per_component": 65536, "candidates": [{}, {"max_epochs": 4}]},
        [],
    )
    assert adapter.checkpoint_epoch_bound() == 5
    adapter.bind_checkpoint_directory(tmp_path)
    assert adapter.checkpoint_dir == tmp_path


def test_edge_underestimated_checkpoint_fails_and_retains_evidence(tmp_path):
    adapter = trainer(checkpoint_dir=tmp_path)
    adapter.validate_experiment(model(), None, {"checkpoint_bytes_per_component": 1}, [])
    X = parts()[0].X
    y = np.r_[np.ones(10, int), np.zeros(30, int)]
    fitted = model(random_state=7)
    with pytest.raises(ValueError, match="exceeded declared"):
        adapter.fit(fitted, X, y, class_prior=0.4, os_or_ts="ts")
    assert fitted.unlabeled_positive_prior is fitted.prior_source is None
    assert list(tmp_path.glob("attempt-*/*"))


def test_edge_failed_fit_restores_constructor_prior_binding(monkeypatch):
    fitted = model()

    def fail(*args, **kwargs):
        raise KeyboardInterrupt("fit stopped")

    monkeypatch.setattr(CVIRClassifier, "_fit_impl", fail)
    with pytest.raises(KeyboardInterrupt, match="fit stopped"):
        trainer().fit(fitted, parts()[0].X, parts()[0].labels, class_prior=0.75, os_or_ts="ts")
    assert fitted.unlabeled_positive_prior is fitted.prior_source is None


@pytest.mark.parametrize("source", [None, "", " ", 123])
def test_param_missing_source_and_nonbinary_training_labels_are_rejected(source):
    with pytest.raises(ValueError, match="population_prior_source"):
        CVIRPopulationPriorTrainer(population_prior_source=source)
    with pytest.raises(ValueError, match="binary observed PU labels"):
        trainer().fit(model(), parts()[0].X, np.full(40, 2), class_prior=0.4)
