# tests/unit/experiment/test_runner.py

# ruff: noqa: N803, N806, S101

import numpy as np
import pytest
import torch

from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import DeepFitTrainer, ProtocolOA, SCARGenerator
from pu_toolbox.experiment.tracking import EpochRecord, RunTrajectory, SelectionArtifact

pytestmark = pytest.mark.unit


def make_part(x, labels, idx, fs=True):
    return DatasetPart(
        X=x[np.asarray(idx)],
        labels=labels[np.asarray(idx)],
        view="clean",
        indices=np.asarray(idx),
        for_selection=fs,
    )


def make_bundle(seed=1):
    """Four-way bundle whose PU-val split receives real positives.

    The runner fails loudly when the generated pu-view val set has no
    labeled positive (Task-8 controller resolution), so pu_val must
    contain real positives under SCAR.  The test split mixes classes so
    the test AUC is defined and real.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(40, 3)
    # pos: 0-1, 18-25, 30, 36 ; neg: 2-17, 26-29, 31-35, 37-39
    y = np.array([1] * 2 + [0] * 16 + [1] * 8 + [0] * 4 + [1] * 1 + [0] * 5 + [1] * 1 + [0] * 3)
    train = make_part(x, y, np.arange(28))
    pu_val = make_part(x, y, np.arange(28, 32))
    clean_val = make_part(x, y, np.arange(32, 36))
    test = make_part(x, y, np.arange(36, 40), fs=False)
    return train, pu_val, clean_val, test


def test_pa_never_receives_clean_labels():
    train, pu_val, clean_val, test = make_bundle()
    capture = {}

    class FakePA:
        def select(self, trajectories, val_part, threshold_candidates=None):
            capture["view"] = val_part.view
            capture["indices"] = val_part.indices.copy()
            capture["labels"] = val_part.labels.copy()
            return SelectionArtifact(
                protocol="PA", run_index=0, epoch=None, threshold=None, metrics={}
            )

    # c=0.15 keeps the generated train PU view above the estimator gate
    # (MIN_POSITIVE_SAMPLES=2) on this tiny synthetic split.
    runner = ExperimentRunner(
        seed=3, generator=SCARGenerator(), protocols=[FakePA(), FakePA()], config={"c": 0.15}
    )
    runner.fit(UPUClassifier(0.3, random_state=0), train, pu_val, clean_val, test)

    # PA reads the PU view ...
    assert capture["view"] == "pu"
    # ... and it is the pu_val split (never clean_val / test).
    assert np.array_equal(capture["indices"], pu_val.indices)
    # PA-view labels are the GENERATED PU labels: a {0, 1} subset ...
    assert set(np.unique(capture["labels"])).issubset({0, 1})
    # ... never the real clean_val labels.
    assert not np.array_equal(capture["labels"], clean_val.labels)


def test_runner_raises_when_pu_view_has_no_labeled_positive():
    """Task-8 controller resolution: fail loudly, never silently degrade to
    ProtocolPA's -1.0 sentinel."""
    rng = np.random.RandomState(0)
    x = rng.randn(40, 3)
    y = np.array([1] * 6 + [0] * 34)  # all real positives live in train (0-5)
    train = make_part(x, y, np.arange(28))
    pu_val = make_part(x, y, np.arange(28, 32))  # no real positive -> all-zero PU view
    clean_val = make_part(x, y, np.arange(32, 36))
    test = make_part(x, y, np.arange(36, 40), fs=False)
    runner = ExperimentRunner(seed=0, generator=SCARGenerator(), class_prior=6 / 40)
    with pytest.raises(ValueError, match="labeled positive"):
        runner.fit(UPUClassifier(0.15, random_state=0), train, pu_val, clean_val, test)


def test_architecture_capability_preflight_blocks_before_training():
    train, pu_val, clean_val, test = make_bundle()

    class _NeverTrainer:
        called = False

        def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
            self.called = True
            raise AssertionError("capability gate must run before training")

    trainer = _NeverTrainer()
    runner = ExperimentRunner(config={"c": 0.5, "trainer": trainer})

    def as_image_part(part):
        return DatasetPart(
            X=part.X[:, None, None, :],
            labels=part.labels,
            view=part.view,
            indices=part.indices,
            for_selection=part.for_selection,
        )

    with pytest.raises(ValueError, match=r"train input ndim=4.*input_ndims=\[2\]"):
        runner.fit(
            UPUClassifier(0.3, random_state=0),
            *(as_image_part(part) for part in (train, pu_val, clean_val, test)),
        )
    assert not trainer.called

    runner = ExperimentRunner(
        config={"architecture": "cnn", "c": 0.5, "trainer": trainer}
    )
    with pytest.raises(ValueError, match="does not support architecture='cnn'"):
        runner.fit(UPUClassifier(0.3, random_state=0), train, pu_val, clean_val, test)
    assert not trainer.called


def test_end_to_end_small_pu(tmp_path):
    rng = np.random.RandomState(0)
    x = rng.randn(60, 2)
    # pos: 0-11, 28-31, 44-45, 54-55 ; neg: 12-27, 32-43, 46-53, 56-59
    y = np.array([1] * 12 + [0] * 16 + [1] * 4 + [0] * 12 + [1] * 2 + [0] * 8 + [1] * 2 + [0] * 4)
    train = make_part(x, y, np.arange(40))
    pu_val = make_part(x, y, np.arange(40, 45))  # index 44: one real positive
    clean_val = make_part(x, y, np.arange(45, 50))  # index 45: one real positive
    test = make_part(x, y, np.arange(50, 60), fs=False)  # indices 54-55: two positives

    runner = ExperimentRunner(
        seed=0,
        generator=SCARGenerator(),
        class_prior=20 / 60,
        threshold_candidates=np.linspace(0, 1, 6),
        manifest_path=str(tmp_path / "artifacts" / "run.json"),
    )
    res = runner.fit(
        UPUClassifier(0.33, random_state=0),
        train,
        pu_val,
        clean_val,
        test,
    )
    assert "PA" in res.selections and "OA" in res.selections
    assert set(res.test_metrics.keys()) >= {"PA", "OA"}
    assert res.failures == []
    # Manifest was written to a non-existent nested dir (runner creates parents).
    assert (tmp_path / "artifacts" / "run.json").exists()
    # Independent test evaluation ran with real labels (both classes present):
    assert 0.0 <= res.test_metrics["PA"]["accuracy"] <= 1.0
    assert 0.0 <= res.test_metrics["OA"]["accuracy"] <= 1.0
    assert np.isfinite(res.test_metrics["PA"]["auc"]) and np.isfinite(res.test_metrics["OA"]["auc"])


def test_auc_records_reason_when_test_has_one_class():
    train, pu_val, clean_val, test = make_bundle()
    one_class_test = DatasetPart(
        X=test.X,
        labels=np.zeros_like(test.labels),
        view=test.view,
        indices=test.indices,
        for_selection=False,
    )
    runner = ExperimentRunner(seed=0, class_prior=0.3, config={"c": 0.5})

    res = runner.fit(
        UPUClassifier(0.3, random_state=0),
        train,
        pu_val,
        clean_val,
        one_class_test,
    )

    for metrics in res.test_metrics.values():
        assert np.isnan(metrics["auc"])
        assert "only one class" in metrics["auc_unavailable_reason"]


def test_auc_does_not_hide_model_scoring_errors():
    train, pu_val, clean_val, test = make_bundle()

    class _BrokenScoreModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def decision_function(self, X):
            raise RuntimeError("broken score path")

    class _FakeTrainer:
        def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
            return RunTrajectory(epochs=[], model=_BrokenScoreModel())

    class _FixedProtocol:
        def select(self, trajectories, val_part, threshold_candidates=None):
            return SelectionArtifact(
                protocol="custom", run_index=0, epoch=None, threshold=None, metrics={}
            )

    runner = ExperimentRunner(
        seed=0,
        protocols=[_FixedProtocol()],
        config={"c": 0.5, "trainer": _FakeTrainer()},
    )
    with pytest.raises(RuntimeError, match="broken score path"):
        runner.fit(UPUClassifier(0.3, random_state=0), train, pu_val, clean_val, test)


class _RecordingTrainer(DeepFitTrainer):
    """Records every trajectory so the deep-path invariant can be asserted."""

    def __init__(self):
        super().__init__()
        self.trajectories = []

    def fit(self, estimator, x, y, *, class_prior=None, val_pu=None):
        traj = super().fit(estimator, x, y, class_prior=class_prior, val_pu=val_pu)
        self.trajectories.append(traj)
        return traj


class _SmallCNN(torch.nn.Module):
    """Tiny CNN on (1, 8, 8) images — pushes nnPU down the deep path."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x))).view(x.shape[0], -1)


def test_end_to_end_cnn_smoke(tmp_path):
    """§7: one CNN method must run end-to-end (nnPU + small images)."""
    rng = np.random.RandomState(0)
    n = 24
    x = rng.randn(n, 1, 8, 8).astype("float32")
    y = np.zeros(n, dtype=int)
    # 9 positives spread over train (0-11), pu_val (12-15), clean_val (16-19)
    # and test (20-23), so both the generated train PU view and the PU view VAL
    # set clear the estimator-agnostic MIN_POSITIVE_SAMPLES=2 gate and test
    # AUROC is defined (both classes present).
    y[[0, 1, 8, 12, 13, 14, 15, 18, 20]] = 1
    train = make_part(x, y, np.arange(12))
    pu_val = make_part(x, y, np.arange(12, 16))
    clean_val = make_part(x, y, np.arange(16, 20))
    test = make_part(x, y, np.arange(20, 24), fs=False)

    trainer = _RecordingTrainer()
    model = NonNegativePUClassifier(
        encoder=_SmallCNN(),
        max_epochs=2,
        patience=1,
        random_state=0,
        device="cpu",
    )
    runner = ExperimentRunner(
        seed=0,
        generator=SCARGenerator(),
        class_prior=9 / 24,
        threshold_candidates=np.linspace(0, 1, 5),
        config={"c": 0.5, "trainer": trainer, "candidates": [{"max_epochs": 2}]},
        manifest_path=str(tmp_path / "cnn.json"),
    )
    res = runner.fit(model, train, pu_val, clean_val, test)

    assert res.selections["OA"].protocol == "OA"
    assert res.test_metrics["OA"]["accuracy"] >= 0  # ran, metrics present
    # Deep path proved: best_epoch comes only from history_["val_risk"] (not None).
    assert res.selections["OA"].epoch is not None
    # Controller resolution: best_epoch is the 1-based position inside
    # traj.epochs (epochs[best_epoch - 1] is the best record), never an
    # assumption about the estimator's epoch labels.
    for traj in trainer.trajectories:
        if traj.epochs:
            assert traj.best_epoch is not None
            assert 1 <= traj.best_epoch <= len(traj.epochs)


def test_oa_test_eval_reuses_val_side_transform():
    """F1 fix: OA test evaluation must apply the VAL-side min-max affine
    constants recorded at selection time. Re-normalising with the test
    set's own min/max applies different affine constants and silently
    shifts the threshold, so the reported test accuracy is wrong."""
    x_test = np.zeros((10, 2))
    x_test[6:, 0] = [0.0, 0.1, 0.2, 0.3]  # test decision scores live in [0.0, 0.3]
    # labels[6:10] = [1, 1, 0, 0].
    y_test = np.concatenate([np.zeros(6, dtype=int), np.array([1, 1, 0, 0])])
    x_other = np.zeros((6, 2))  # rows 0-5 shared by train/pu_val/clean_val
    y_other = np.array([1, 1, 1, 1, 0, 1])  # per-part segment controls the labels
    train = make_part(x_other, y_other, np.arange(2))
    pu_val = make_part(x_other, y_other, np.arange(2, 4))  # real positive for SCAR
    clean_val = make_part(x_other, y_other, np.arange(4, 6))
    test = make_part(x_test, y_test, np.arange(6, 10), fs=False)

    class _ScoreModel:
        """decision_function = first column (test scores are [0.0, 0.3])."""

        def decision_function(self, X):
            return X[:, 0]

        def predict(self, X):
            return (X[:, 0] >= 0.0).astype(int)

    class _FakeTrainer:
        """Returns a single trajectory holding ``_ScoreModel`` (never fits)."""

        def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
            return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=_ScoreModel())

    class _FixedOA(ProtocolOA):
        """OA artifact with known val-side constants: val scores live in
        [0.8, 1.0] (min=0.8, scale=0.2), threshold mid-grid."""

        def select(self, trajectories, val_part, threshold_candidates=None):
            return SelectionArtifact(
                protocol="OA",
                run_index=0,
                epoch=None,
                threshold=0.5,
                metrics={
                    "val_accuracy": 1.0,
                    "val_score_min": 0.8,
                    "val_score_scale": 0.2,
                },
            )

    runner = ExperimentRunner(
        seed=0,
        generator=SCARGenerator(),
        protocols=[_FixedOA()],
        config={"c": 0.5, "trainer": _FakeTrainer()},
    )
    res = runner.fit(UPUClassifier(0.3, random_state=0), train, pu_val, clean_val, test)
    acc = res.test_metrics["OA"]["accuracy"]

    # Fixed val transform: (s - 0.8) / 0.2 lands in [-4.0, -2.5], all below
    # 0.5 — every test point is predicted negative, accuracy = 0.5.
    s_test = x_test[6:, 0]
    y_test_part = y_test[6:]
    val_transform_pred = ((s_test - 0.8) / 0.2 >= 0.5).astype(int)
    assert acc == float(np.mean(val_transform_pred == y_test_part))
    assert acc == 0.5  # all-negative predictions against two real positives
    # The drift the fix prevents: the test's own min-max would predict
    # [0, 0, 1, 1] here (accuracy 0.0), so the assertion discriminates the
    # two transforms.
    test_transform_pred = ((s_test - s_test.min()) / np.ptp(s_test) >= 0.5).astype(int)
    assert float(np.mean(test_transform_pred == y_test_part)) != acc
