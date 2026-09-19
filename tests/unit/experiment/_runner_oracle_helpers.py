# ruff: noqa: N803, N806, S101

"""Shared fixtures/builders for the PN-oracle runner test modules.

Not collected by pytest (``python_files = ["test_*.py"]``).  The behaviour
suite and the guard suite must construct *identical* bundles: a guard that
passes on a bundle the behaviour tests never build proves nothing, and a
bundle that drifts between the two would hide exactly that.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import (
    CleanLabelGenerator,
    DeepFitTrainer,
    ProtocolOA,
    SupervisedTrainer,
)


class PNLogisticRegression(LogisticRegression):
    """Test-only supervised estimator that declares the meaning of fit labels."""

    label_semantics = "pn"


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
    result = runner.fit(PNLogisticRegression(max_iter=200), train, pu_val, clean_val, test)
    return result, recorder
