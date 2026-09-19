# tests/unit/experiment/test_runner_oracle_guards.py

# ruff: noqa: N803, N806, S101

"""The PN-oracle wiring guards: every mis-configuration that must fail closed.

Split from the behaviour suite along the question each answers.  That module
shows what the oracle *does* when it is wired correctly; this one pins every
way it must refuse to run.  The two are separate because a guard is only
worth anything if it fires *instead of* training -- so most of these also
assert the trainer was never called, which is a claim about a non-event and
reads badly next to the success-path tests.
"""

import pytest
from _runner_oracle_helpers import (
    PNLogisticRegression,
    RecordingOracle,
    RecordingPU,
    make_bundle,
)
from sklearn.linear_model import LogisticRegression

from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import (
    CleanLabelGenerator,
    ProtocolOA,
    ProtocolPA,
    SCARGenerator,
    SupervisedTrainer,
)

pytestmark = pytest.mark.unit


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
        runner.fit(PNLogisticRegression(max_iter=200), train, pu_val, clean_val, test)


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
