# ruff: noqa: N803, N806, S101

"""The run manifest must name the training view the run was bound to.

Protocol §2.3 makes ``os_or_ts`` a property of the run, not of the method, so
the value belongs in the manifest: aggregation gates need it to keep os and ts
runs out of one leaderboard group, and a refused calibrated run has to leave a
trace rather than look like a run nobody asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier  # noqa: E402
from pu_toolbox.experiment.bundle import DatasetPart  # noqa: E402
from pu_toolbox.experiment.manifest import load_manifest  # noqa: E402
from pu_toolbox.experiment.runner import ExperimentRunner  # noqa: E402
from pu_toolbox.experiment.strategies import DeepFitTrainer, SCARGenerator  # noqa: E402

pytestmark = pytest.mark.unit


def _make_part(x, labels, idx, fs=True):
    return DatasetPart(
        X=x[np.asarray(idx)],
        labels=labels[np.asarray(idx)],
        view="clean",
        indices=np.asarray(idx),
        for_selection=fs,
    )


def _bundle():
    """Four-way split with enough SCAR-labeled positives for nnPU's own gate.

    ``c=0.5`` is used below because nnPU requires at least two labeled
    positives in the validation view, and the default ``c`` would round the
    six pu_val positives down to one.
    """
    rng = np.random.RandomState(0)
    x = rng.randn(60, 2)
    y = np.zeros(60, dtype=int)
    y[:12] = 1  # train positives
    y[28:32] = 1  # pu_val positives
    y[44:46] = 1  # pu_val positives
    y[54:56] = 1  # test positives
    return (
        _make_part(x, y, np.arange(28)),
        _make_part(x, y, [28, 29, 30, 31, 44, 45, 46, 47]),
        _make_part(x, y, np.arange(48, 54)),
        _make_part(x, y, np.arange(54, 60), fs=False),
    )


def _nnpu():
    return NonNegativePUClassifier(
        model=torch.nn.Linear(2, 1), max_epochs=2, batch_size=8, random_state=0
    )


class _PlainEstimator:
    """Stand-in for an estimator whose ``fit`` has no view hook.

    Pinned to a stand-in rather than a real method on purpose: which methods
    still lack ``os_or_ts`` is current state, and the runner's gate is about
    the signature, not about any particular estimator.  The only thing that
    has to hold is that the gate is reached -- ``_declared_epoch_components``
    runs first and reads ``epoch_components`` off this object.
    """

    def fit(self, X, y, *, class_prior=None):  # noqa: N803
        raise AssertionError("the view gate must refuse this run before any training.")


def _runner(tmp_path, name, **config):
    return ExperimentRunner(
        seed=0,
        generator=SCARGenerator(),
        class_prior=20 / 48,
        threshold_candidates=np.linspace(0, 1, 4),
        manifest_path=str(tmp_path / name),
        config={"c": 0.5, **config},
    )


class TestManifestViewFields:
    def test_calibrated_run_is_recorded_as_ts(self, tmp_path):
        """nnpu declares the hook, so the ts request reaches training and is recorded."""
        train, pu_val, clean_val, test = _bundle()
        res = _runner(tmp_path, "ts.json", os_or_ts="ts").fit(
            _nnpu(), train, pu_val, clean_val, test
        )
        assert res.manifest["run_view"] == "ts-compatible"
        assert res.manifest["calibration_applied"] is True

    def test_uncalibrated_run_is_recorded_as_os(self, tmp_path):
        train, pu_val, clean_val, test = _bundle()
        res = _runner(tmp_path, "os.json", os_or_ts="os").fit(
            _nnpu(), train, pu_val, clean_val, test
        )
        assert res.manifest["run_view"] == "os-compatible"
        assert res.manifest["calibration_applied"] is False

    def test_absent_config_defaults_to_os(self, tmp_path):
        """Every pre-existing run keeps reporting the OS view it already used."""
        train, pu_val, clean_val, test = _bundle()
        res = _runner(tmp_path, "default.json").fit(_nnpu(), train, pu_val, clean_val, test)
        assert res.manifest["run_view"] == "os-compatible"
        assert res.manifest["calibration_applied"] is False

    def test_ts_for_an_estimator_without_the_hook_fails_loud(self, tmp_path):
        """A method that cannot honour the view must not be silently recorded as os."""
        train, pu_val, clean_val, test = _bundle()
        with pytest.raises(ValueError, match="does not accept os_or_ts"):
            _runner(tmp_path, "bad.json", os_or_ts="ts").fit(
                _PlainEstimator(), train, pu_val, clean_val, test
            )

    def test_failed_run_still_records_the_view(self, tmp_path):
        """A run that never produced a trajectory is still traceable to its view."""

        class _FailingTrainer(DeepFitTrainer):
            def fit(self, estimator, X, y, **kwargs):  # noqa: N803
                raise RuntimeError("boom")

        train, pu_val, clean_val, test = _bundle()
        path = tmp_path / "failed.json"
        runner = ExperimentRunner(
            seed=0,
            generator=SCARGenerator(),
            class_prior=20 / 60,
            threshold_candidates=np.linspace(0, 1, 4),
            manifest_path=str(path),
            config={"os_or_ts": "ts", "trainer": _FailingTrainer()},
        )
        with pytest.raises(RuntimeError, match="all candidate runs failed"):
            runner.fit(_nnpu(), train, pu_val, clean_val, test)
        manifest = load_manifest(path)
        assert manifest["run_view"] == "ts-compatible"
        assert manifest["calibration_applied"] is True
