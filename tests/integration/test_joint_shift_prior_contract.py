"""Regression tests for independently consumed domain priors."""

import numpy as np
import pytest
from sklearn.base import clone

pytest.importorskip("torch")

from pu_toolbox.estimators.research import (  # noqa: E402
    DynamicJointShiftPUClassifier,
    JointShiftPUBaseline,
    JointShiftPUClassifier,
)

pytestmark = pytest.mark.integration


def _data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 3))
    y = np.tile([0, 1, 0, 0], 6)
    return x, y, x + 0.3, y.copy()


@pytest.mark.parametrize("kind", ["joint", "dynamic", "trpu", "tepu", "fine_tune", "mmd"])
def test_basic_determ_domain_prior_consumptions_match_actual_calls(kind, monkeypatch):
    import pu_toolbox.estimators.research.dynamic_joint_shift as dynamic
    import pu_toolbox.estimators.research.joint_shift as joint
    import pu_toolbox.estimators.research.joint_shift_baselines as baseline

    seen = []
    if kind == "joint":
        original = joint._calibrate_prior

        def spy(scores, prior):
            seen.append(prior)
            return original(scores, prior)

        monkeypatch.setattr(joint, "_calibrate_prior", spy)
        model = JointShiftPUClassifier(max_iter=1, n_cv_folds=2, random_state=1)
    elif kind == "dynamic":
        original = dynamic.paper_importance_weight_objective

        def spy(**kwargs):
            seen.extend([kwargs["source_domain_prior"], kwargs["target_domain_prior"]])
            return original(**kwargs)

        monkeypatch.setattr(dynamic, "paper_importance_weight_objective", spy)
        model = DynamicJointShiftPUClassifier(max_epochs=1, hidden_dim=4, feature_dim=3)
    else:
        original = baseline.paper_pu_risk

        def spy(*args, **kwargs):
            seen.append(kwargs["risk_prior"])
            return original(*args, **kwargs)

        monkeypatch.setattr(baseline, "paper_pu_risk", spy)
        model = JointShiftPUBaseline(strategy=kind, max_epochs=1, hidden_dim=4, feature_dim=3)
    x, y, xt, yt = _data()
    model.fit(x, y, X_target=xt, y_target_pu=yt, source_domain_prior=0.3, target_domain_prior=0.7)
    assert 0.3 in seen and 0.7 in seen
    records = model._prior_consumptions_
    assert {(r["target"], r["value"], r["source"]) for r in records} == {
        ("source_domain", 0.3, "fit.source_domain_prior"),
        ("target_domain", 0.7, "fit.target_domain_prior"),
    }
    assert all(r["partition"] == "train" and r["consumer"] for r in records)
    second = clone(model).fit(
        x, y, X_target=xt, y_target_pu=yt, source_domain_prior=0.3, target_domain_prior=0.7
    )
    assert second._prior_consumptions_ == records


@pytest.mark.parametrize(
    "model",
    [
        JointShiftPUClassifier(max_iter=1),
        JointShiftPUBaseline(strategy="trpu", max_epochs=1),
        DynamicJointShiftPUClassifier(max_epochs=1),
    ],
)
@pytest.mark.parametrize("source,target", [(None, 0.7), (0.3, None), (0, 0.7), (0.3, 1)])
def test_param_edge_domain_priors_are_both_required_and_validated(model, source, target):
    x, y, xt, yt = _data()
    with pytest.raises(ValueError):
        model.fit(
            x,
            y,
            X_target=xt,
            y_target_pu=yt,
            source_domain_prior=source,
            target_domain_prior=target,
        )
