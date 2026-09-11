# tests/unit/experiment/test_runner_oracle.py

# ruff: noqa: N803, N806, S101

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.manifest import load_manifest
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import (
    CleanLabelGenerator,
    DeepFitTrainer,
    ProtocolOA,
    ProtocolPA,
    SCARGenerator,
    SupervisedTrainer,
)

pytestmark = pytest.mark.unit


def make_part(x, labels, idx, fs=True):
    return DatasetPart(
        X=x[np.asarray(idx)],
        labels=labels[np.asarray(idx)],
        view="clean",
        indices=np.asarray(idx),
        for_selection=fs,
    )


def make_bundle(*, pu_val_has_positive=True):
    """Four-way bundle for the PN-oracle path.

    train (0-27) holds 12 real positives; clean_val and test mix both classes
    so OA thresholding and accuracy are defined.  ``pu_val_has_positive=False``
    builds the boundary case where a PU-view positive check would have raised.
    """
    rng = np.random.RandomState(1)
    x = rng.randn(40, 3)
    pu_val_labels = [1, 1, 0, 0] if pu_val_has_positive else [0, 0, 0, 0]
    y = np.array(
        [1] * 12
        + [0] * 16  # train (0-27): 12 real positives
        + pu_val_labels  # pu_val (28-31)
        + [1, 1, 0, 0]  # clean_val (32-35)
        + [1, 1, 0, 0]  # test (36-39)
    )
    train = make_part(x, y, np.arange(28))
    pu_val = make_part(x, y, np.arange(28, 32))
    clean_val = make_part(x, y, np.arange(32, 36))
    test = make_part(x, y, np.arange(36, 40), fs=False)
    return train, pu_val, clean_val, test


class RecordingOracle(SupervisedTrainer):
    """SupervisedTrainer that records what the runner hands it."""

    def __init__(self):
        self.calls = []

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        self.calls.append(
            {
                "n_positive_labels": int(np.sum(np.asarray(y) == 1)),
                "class_prior": class_prior,
            }
        )
        return super().fit(estimator, X, y, class_prior=class_prior)


class RecordingPU(DeepFitTrainer):
    """A PU trainer (no real-label declaration) that counts fits."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        self.calls += 1
        return super().fit(estimator, X, y, class_prior=class_prior, val_pu=val_pu)


def run_oracle(bundle, *, protocols=None, c=0.1, recorder=None, manifest_path=None):
    """Run the oracle path with an injected recording trainer."""
    train, pu_val, clean_val, test = bundle
    recorder = recorder if recorder is not None else RecordingOracle()
    runner = ExperimentRunner(
        seed=0,
        generator=CleanLabelGenerator(),
        protocols=[ProtocolOA()] if protocols is None else protocols,
        config={"trainer": recorder, "c": c},
        manifest_path=manifest_path,
    )
    result = runner.fit(LogisticRegression(max_iter=200), train, pu_val, clean_val, test)
    return result, recorder


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


def test_param_oracle_rejects_pa_protocol():
    """A clean view must make ProtocolPA fail loudly, never emit a fake PA row."""
    train, pu_val, clean_val, test = make_bundle()
    runner = ExperimentRunner(
        seed=0,
        generator=CleanLabelGenerator(),
        protocols=[ProtocolPA()],
        config={"trainer": SupervisedTrainer(), "c": 0.1},
    )
    with pytest.raises(ValueError, match="PU view"):
        runner.fit(LogisticRegression(max_iter=200), train, pu_val, clean_val, test)


def test_param_oracle_rejects_pu_view_generator():
    """SCAR + SupervisedTrainer is the exact mis-wiring that shipped the fake oracle.

    Without this guard the pairing trains on marked labels and reports a
    plausible-looking but wrong upper bound instead of failing.
    """
    train, pu_val, clean_val, test = make_bundle()
    recorder = RecordingOracle()
    runner = ExperimentRunner(
        seed=0,
        generator=SCARGenerator(),
        protocols=[ProtocolOA()],
        config={"trainer": recorder, "c": 0.3},
    )
    with pytest.raises(ValueError, match="clean label view"):
        runner.fit(LogisticRegression(max_iter=200), train, pu_val, clean_val, test)
    # the guard must fire before training, not after a wasted candidate run
    assert recorder.calls == []


def test_param_oracle_rejects_misdeclared_clean_view():
    """A generator declaring "clean" must actually emit the real labels.

    Trusting the declaration would let it produce a silent fake oracle whose
    manifest still reads mechanism='pn_oracle'.
    """

    class LyingCleanGenerator(CleanLabelGenerator):
        def generate(self, X, y_true, c, seed=None):
            return SCARGenerator().generate(X, y_true, c, seed)  # marked, not real

    train, pu_val, clean_val, test = make_bundle()
    runner = ExperimentRunner(
        seed=0,
        generator=LyingCleanGenerator(),
        protocols=[ProtocolOA()],
        config={"trainer": RecordingOracle(), "c": 0.3},
    )
    with pytest.raises(ValueError, match="must carry the real labels"):
        runner.fit(LogisticRegression(max_iter=200), train, pu_val, clean_val, test)


def test_param_oracle_rejects_unknown_output_view():
    """A typo like "Clean" must fail closed instead of skipping every guard."""

    class TypoGenerator(SCARGenerator):
        output_view = "Clean"

    train, pu_val, clean_val, test = make_bundle()
    runner = ExperimentRunner(
        seed=0,
        generator=TypoGenerator(),
        protocols=[ProtocolOA()],
        config={"trainer": RecordingOracle(), "c": 0.3},
    )
    with pytest.raises(ValueError, match="expected 'pu' or 'clean'"):
        runner.fit(LogisticRegression(max_iter=200), train, pu_val, clean_val, test)


def test_param_oracle_rejects_class_prior():
    """A prior on a real-label run would stop the oracle being an upper bound."""
    train, pu_val, clean_val, test = make_bundle()
    runner = ExperimentRunner(
        seed=0,
        generator=CleanLabelGenerator(),
        protocols=[ProtocolOA()],
        class_prior=0.3,
        config={"trainer": RecordingOracle(), "c": 0.3},
    )
    with pytest.raises(ValueError, match="must not receive a prior"):
        runner.fit(LogisticRegression(max_iter=200), train, pu_val, clean_val, test)


def test_param_oracle_rejects_pu_trainer_on_clean_view():
    """A PU trainer on the clean view trains the wrong objective — silently.

    Its loss reads label 0 as "unlabeled", so on the oracle view every real
    negative becomes unlabeled, and the run still writes a pn_oracle manifest.
    This is the original defect read from the other side: the guard used to
    check only "PU view + supervised trainer".
    """
    train, pu_val, clean_val, test = make_bundle()
    estimator = NonNegativePUClassifier(class_prior=0.3, max_epochs=2, random_state=0, device="cpu")
    recorder = RecordingPU()
    runner = ExperimentRunner(
        seed=0,
        generator=CleanLabelGenerator(),
        protocols=[ProtocolOA()],
        config={"trainer": recorder, "c": 0.1},
    )
    with pytest.raises(ValueError, match="trains_on_real_labels"):
        runner.fit(estimator, train, pu_val, clean_val, test)
    # the guard must fire before training, not after a wasted candidate run
    assert recorder.calls == 0


def test_param_oracle_rejects_default_trainer_on_clean_view():
    """Forgetting ``config['trainer']`` must not fall back to the PU default."""
    train, pu_val, clean_val, test = make_bundle()
    runner = ExperimentRunner(
        seed=0,
        generator=CleanLabelGenerator(),
        protocols=[ProtocolOA()],
        config={"c": 0.1},  # no trainer -> ExperimentRunner's PU-view default
    )
    with pytest.raises(ValueError, match="trains_on_real_labels"):
        runner.fit(LogisticRegression(max_iter=200), train, pu_val, clean_val, test)


def test_param_oracle_rejects_pu_view_carrying_oracle_mechanism():
    """Mechanism and view are two reports of the same fact; they must agree.

    A generator that keeps the oracle's mechanism while declaring a PU view
    skips every clean-view guard and is still recorded as an oracle row — with
    real labels fed to a PU objective. Verified reachable before this check.
    """

    class QuietOracle(CleanLabelGenerator):
        output_view = "pu"  # lies: the labels below are still the real ones

    train, pu_val, clean_val, test = make_bundle()
    runner = ExperimentRunner(
        seed=0,
        generator=QuietOracle(),
        protocols=[ProtocolOA()],
        config={"trainer": RecordingPU(), "c": 0.1},
    )
    with pytest.raises(ValueError, match="implies real labels"):
        runner.fit(LogisticRegression(max_iter=200), train, pu_val, clean_val, test)


def test_param_oracle_rejects_trainer_class_not_instance():
    """A class satisfies the guard through its class attributes and never trains."""
    train, pu_val, clean_val, test = make_bundle()
    runner = ExperimentRunner(
        seed=0,
        generator=CleanLabelGenerator(),
        protocols=[ProtocolOA()],
        config={"trainer": SupervisedTrainer, "c": 0.1},
    )
    with pytest.raises(ValueError, match="must be an instance"):
        runner.fit(LogisticRegression(max_iter=200), train, pu_val, clean_val, test)


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
    runner.fit(LogisticRegression(max_iter=200), *make_bundle())

    assert approved.calls  # the guard-approved trainer is the one that trained
    assert swapped_in.calls == 0


def test_determ_oracle_metrics_are_c_independent():
    """c only drives PU marking, so the oracle's numbers must not move with c."""
    low, _ = run_oracle(make_bundle(), c=0.1)
    high, _ = run_oracle(make_bundle(), c=0.9)

    assert low.test_metrics == high.test_metrics
