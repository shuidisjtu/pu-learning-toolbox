# tests/unit/experiment/test_strategies_selection.py

# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.strategies import (
    ProtocolOA,
    ProtocolPA,
    proxy_accuracy,
    select_threshold,
)
from pu_toolbox.experiment.tracking import EpochRecord, RunTrajectory

pytestmark = pytest.mark.unit


class FakeModel:
    def decision_function(self, X):
        return X[:, 0]


def _traj(value):
    return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=value)


def test_select_threshold_picks_best_accuracy():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.0, -0.1, -0.2, -0.3])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    best_thr, best_acc = select_threshold(scores, labels, np.array([0.2, 0.5, 0.9]))
    assert best_thr == 0.2  # fully separable below 0.9; tie at 0.2/0.5 -> lowest picks first
    assert best_acc == 1.0


def test_basic_protocoloa_uses_real_labels():
    X_val = np.array([[0.1], [0.9], [-0.3]])
    val_part = DatasetPart(X=X_val, labels=np.array([0, 1, 0]), view="clean", indices=np.arange(3))
    traj = _traj(FakeModel())
    art = ProtocolOA().select([traj], val_part)
    assert art.protocol == "OA"
    assert art.threshold is not None
    assert art.metrics["val_accuracy"] >= 0.5


def test_protocolpa_marks_pu_view_usage():
    X_val = np.array([[0.1], [0.9], [-0.3]])
    val_part = DatasetPart(X=X_val, labels=np.array([1, 0, 0]), view="pu", indices=np.arange(3))
    traj = _traj(FakeModel())
    art = ProtocolPA().select([traj], val_part, class_prior=0.5)
    assert art.protocol == "PA"
    assert art.threshold is not None
    assert "val_proxy_accuracy" in art.metrics


def test_param_protocolpa_rejects_clean_view():
    val_part = DatasetPart(
        X=np.array([[0.1]]), labels=np.array([1]), view="clean", indices=np.arange(1)
    )
    traj = _traj(FakeModel())
    with pytest.raises(ValueError):
        ProtocolPA().select([traj], val_part)


def test_edge_protocoloa_empty_trajectories():
    val_part = DatasetPart(
        X=np.array([[0.1], [0.9], [-0.3]]),
        labels=np.array([0, 1, 0]),
        view="clean",
        indices=np.arange(3),
    )
    with pytest.raises(ValueError, match="at least one trajectory"):
        ProtocolOA().select([], val_part)


# --- PA = Proxy Accuracy (Wang et al. 2026, Definition 1, OS branch) ---------
#
# PA(f) = (2*pi/n'_P) * sum_{D'_P} I(f(x) >= theta)
#       + (1/(n'_P+n'_U)) * sum_{D'_P u D'_U} I(f(x) < theta)
#
# The second sum runs over EVERY validation sample; the first is weighted by
# 2*pi/n'_P, which makes PA = ACC + pi (a constant shift), hence Proposition 1.

#: Hand-computed fixture.  Scores are already min-max normalised (min=0, max=1),
#: so the protocol's per-checkpoint normalisation is the identity here.
#: Two labeled positives (1.0, 0.2); nine unlabeled (eight at 0.5, one at 0.0).
_PA_SCORES = np.array([1.0, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
_PA_LABELS = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def _pa_view():
    return DatasetPart(
        X=_PA_SCORES.reshape(-1, 1), labels=_PA_LABELS, view="pu", indices=np.arange(11)
    )


def test_proxy_accuracy_matches_paper_definition():
    """The OS formula, evaluated by hand at two thresholds (pi=0.5, n'_P=2, N=11)."""
    # theta=0.6: only the 1.0 positive clears it (A=1); ten samples fall below (B=10).
    assert proxy_accuracy(_PA_SCORES, _PA_LABELS, 0.6, 0.5) == pytest.approx(31 / 22)
    # theta=0.2: both positives clear it (A=2); one sample falls below (B=1).
    assert proxy_accuracy(_PA_SCORES, _PA_LABELS, 0.2, 0.5) == pytest.approx(1 + 1 / 11)


def test_protocolpa_threshold_moves_with_class_prior():
    """Same val view, two priors, two different argmax thresholds.

    pi enters the weight of the positive term (2*pi/n'_P), so it changes the
    argmax rather than merely rescaling it -- a proxy that ignored pi would
    return the same threshold twice and fail here.
    """
    art_low = ProtocolPA().select([_traj(FakeModel())], _pa_view(), class_prior=0.5)
    art_high = ProtocolPA().select([_traj(FakeModel())], _pa_view(), class_prior=0.9)
    assert art_low.threshold == pytest.approx(0.6)
    assert art_low.metrics["val_proxy_accuracy"] == pytest.approx(31 / 22)
    # theta=0.1 and theta=0.2 hit exactly the same ten samples (the 0.2 positive
    # clears both), so they score identically and the plateau resolves to 0.1.
    assert art_high.threshold == pytest.approx(0.1)
    assert art_high.metrics["val_proxy_accuracy"] == pytest.approx(9 / 5 + 1 / 11)


def test_protocolpa_tie_keeps_lowest_threshold():
    """A plateau resolves to the earliest candidate, per selection_spec.tie_breaking."""
    # theta in [0.6, 1.0] all score identically (A=1, B=10) -> 0.6 wins.
    art = ProtocolPA().select([_traj(FakeModel())], _pa_view(), class_prior=0.5)
    assert art.threshold == pytest.approx(0.6)


def test_protocolpa_requires_class_prior():
    with pytest.raises(ValueError, match="class prior"):
        ProtocolPA().select([_traj(FakeModel())], _pa_view())


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan")])
def test_protocolpa_rejects_invalid_class_prior(bad):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        ProtocolPA().select([_traj(FakeModel())], _pa_view(), class_prior=bad)
