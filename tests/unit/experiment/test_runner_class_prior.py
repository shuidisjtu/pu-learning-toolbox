# tests/unit/experiment/test_runner_class_prior.py
"""PA's population class prior: resolution order, provenance, and refusal.

Split out of ``test_runner.py`` (which reached the ≤15 tests-per-file limit)
because everything here turns on one contract from protocol §3.1: pi is the
weight of PA's positive term, so a run either supplies one or refuses.  A
default would not fail -- it would silently select by a different criterion and
leave a manifest that looks normal.
"""

# ruff: noqa: N803, N806, S101

import numpy as np
import pytest

from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolPA

pytestmark = pytest.mark.unit


def _make_part(x, labels, idx, fs=True):
    return DatasetPart(
        X=x[np.asarray(idx)],
        labels=labels[np.asarray(idx)],
        view="clean",
        indices=np.asarray(idx),
        for_selection=fs,
    )


def _make_bundle(seed=1):
    """Four-way bundle whose pu_val split receives a real positive.

    The runner fails loudly when the generated pu-view val set has no labeled
    positive, so pu_val must contain one under SCAR.  The test split mixes
    classes so the test AUC is defined and real.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(40, 3)
    y = np.array([1] * 2 + [0] * 16 + [1] * 8 + [0] * 4 + [1] * 1 + [0] * 5 + [1] * 1 + [0] * 3)
    return (
        _make_part(x, y, np.arange(28)),
        _make_part(x, y, np.arange(28, 32)),
        _make_part(x, y, np.arange(32, 36)),
        _make_part(x, y, np.arange(36, 40), fs=False),
    )


def _fit(**kwargs):
    train, pu_val, clean_val, test = _make_bundle()
    return ExperimentRunner(seed=0, **kwargs).fit(
        UPUClassifier(0.3, random_state=0), train, pu_val, clean_val, test
    )


def test_param_pa_selection_requires_a_class_prior():
    """No pi anywhere is a refusal, never a default."""
    with pytest.raises(ValueError, match="class prior"):
        _fit(protocols=[ProtocolPA()], config={"c": 0.5})


def test_basic_selection_prior_falls_back_to_the_split_artifact():
    """A method that needs no prior still leaves PA one to select with."""
    config = {"c": 0.5, "split_ref": {"class_prior": {"population": 0.7}}}
    result = _fit(protocols=[ProtocolPA()], config=config)
    pa = result.manifest["selection"]["PA"]
    assert pa["metrics"]["class_prior"] == pytest.approx(0.7)
    assert pa["class_prior"] == {
        "population": pytest.approx(0.7),
        "source": "split_ref.class_prior.population",
    }


def test_basic_runner_class_prior_wins_over_the_split_artifact():
    """The run's own prior is the one training used, so selection uses it too.

    Otherwise one run would carry two different constants and
    ``--allow-prior-override`` would only half-apply.
    """
    config = {"c": 0.5, "split_ref": {"class_prior": {"population": 0.7}}}
    result = _fit(protocols=[ProtocolPA()], class_prior=0.2, config=config)
    pa = result.manifest["selection"]["PA"]
    assert pa["metrics"]["class_prior"] == pytest.approx(0.2)
    assert pa["class_prior"]["source"] == "runner.class_prior"


def test_basic_pa_emits_a_threshold_for_the_test_path():
    """PA picks a threshold, so the runner takes the affine+threshold test path.

    ``threshold=None`` used to mean PA fell back to ``est.predict`` -- the
    pre-R9 engineering behaviour.  This pins the switch so it cannot drift back.
    """
    result = _fit(class_prior=0.3, config={"c": 0.5})
    art = result.selections["PA"]
    assert art.threshold is not None
    assert art.metrics["val_score_min"] is not None
    assert art.metrics["val_score_scale"] is not None


def test_edge_split_ref_class_prior_without_population_is_refused():
    """A malformed record is not a prior -- it must refuse, not fall through.

    The record exists but cannot answer for the population rate; reading it as
    "no prior recorded" would be exactly the silent downgrade this contract
    exists to prevent.
    """
    config = {"c": 0.5, "split_ref": {"class_prior": {"train": 0.4}}}
    with pytest.raises(ValueError, match="class prior"):
        _fit(protocols=[ProtocolPA()], config=config)
