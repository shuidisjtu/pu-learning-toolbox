# tests/unit/experiment/test_runner_oracle.py
"""The PN-oracle path: what the runner hands the trainer, and what it writes.
Every way the oracle must *refuse* to run lives in
``test_runner_oracle_guards.py``.  Keeping the refusals separate is what lets
this module read as a description of the success path, and lets that one
assert the trainer was never called without the contrast looking odd.
"""

import numpy as np
import pytest
from _runner_oracle_helpers import (
    PNLogisticRegression,
    RecordingOracle,
    RecordingPU,
    make_bundle,
    run_oracle,
)
from sklearn.linear_model import LogisticRegression

from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.experiment.manifest import load_manifest
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import (
    CleanLabelGenerator,
    ProtocolOA,
    SCARGenerator,
    SupervisedTrainer,
)

pytestmark = pytest.mark.unit


def test_basic_oracle_trains_on_true_labels():
    """Guard the core defect: every real positive must reach the oracle.

    With SCAR at c=0.1 only ~1 of the 12 train positives would be labelled;
    the oracle must instead receive all 12.
    """
    bundle = make_bundle()
    n_true_positives = int(np.sum(bundle[0].labels == 1))
    assert n_true_positives == 12

    _, recorder = run_oracle(bundle, c=0.1)

    assert recorder.calls, "the oracle trainer was never called"
    assert recorder.calls[0]["n_positive_labels"] == n_true_positives


def test_basic_oracle_records_pn_oracle_generation(tmp_path):
    """Manifests and results must read as PN oracle: OA only, no SCAR metadata."""
    manifest_path = tmp_path / "manifest.json"

    result, _ = run_oracle(make_bundle(), manifest_path=str(manifest_path))

    assert set(result.test_metrics) == {"OA"}
    manifest = load_manifest(manifest_path)
    assert set(manifest["test_results"]) == {"OA"}
    assert manifest["generation"]["train"]["mechanism"] == "pn_oracle"
    assert manifest["generation"]["train"]["c_realized"] == 1.0


def test_edge_oracle_runs_without_labeled_positive_in_pu_val():
    """The PU-view positive check belongs to generated views, not the oracle."""
    result, recorder = run_oracle(make_bundle(pu_val_has_positive=False))

    assert set(result.test_metrics) == {"OA"}
    assert recorder.calls[0]["n_positive_labels"] == 12


def test_edge_oracle_trains_with_the_trainer_the_guard_approved():
    """A config that swaps trainers mid-run must not swap the one that trains."""

    class FlippingConfig(dict):
        def __init__(self, first, second):
            # Non-empty: the runner keeps the config it is given only if truthy.
            super().__init__({"c": 0.1})
            self._first, self._second, self.lookups = first, second, 0

        def get(self, key, default=None):
            if key == "trainer":
                self.lookups += 1
                return self._first if self.lookups == 1 else self._second
            return super().get(key, default)

    approved, swapped_in = RecordingOracle(), RecordingPU()
    runner = ExperimentRunner(
        seed=0,
        generator=CleanLabelGenerator(),
        protocols=[ProtocolOA()],
        config=FlippingConfig(approved, swapped_in),
    )
    runner.fit(PNLogisticRegression(max_iter=200), *make_bundle())

    assert approved.calls  # the guard-approved trainer is the one that trained
    assert swapped_in.calls == 0


def test_clean_view_rejects_pu_estimator_even_with_supervised_trainer():
    """F2: the trainer cannot turn a PU risk estimator into a PN oracle."""
    estimator = NonNegativePUClassifier(class_prior=0.3, max_epochs=2, random_state=0, device="cpu")
    recorder = RecordingOracle()
    runner = ExperimentRunner(
        generator=CleanLabelGenerator(),
        protocols=[ProtocolOA()],
        config={"trainer": recorder},
    )
    with pytest.raises(ValueError, match="label_semantics='pu'.*requires 'pn'"):
        runner.fit(estimator, *make_bundle())
    assert recorder.calls == []


def test_pu_view_rejects_pn_estimator_before_training():
    runner = ExperimentRunner(
        generator=SCARGenerator(), protocols=[ProtocolOA()], config={"c": 0.3}
    )
    with pytest.raises(ValueError, match="label_semantics='pn'.*requires 'pu'"):
        runner.fit(PNLogisticRegression(max_iter=200), *make_bundle())


def test_clean_view_requires_explicit_pn_declaration():
    runner = ExperimentRunner(
        generator=CleanLabelGenerator(),
        protocols=[ProtocolOA()],
        config={"trainer": SupervisedTrainer()},
    )
    with pytest.raises(ValueError, match="label_semantics='pu'.*requires 'pn'"):
        runner.fit(LogisticRegression(max_iter=200), *make_bundle())


def test_determ_oracle_metrics_are_c_independent():
    """c only drives PU marking, so the oracle's numbers must not move with c."""
    low, _ = run_oracle(make_bundle(), c=0.1)
    high, _ = run_oracle(make_bundle(), c=0.9)

    assert low.test_metrics == high.test_metrics
