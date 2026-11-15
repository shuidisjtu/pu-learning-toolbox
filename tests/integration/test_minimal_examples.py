"""Run every public minimal script and check direct-estimator prior contracts."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pu_toolbox.metrics import pu_zero_one_risk

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples/minimal").glob("*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_basic_param_scripts_run_with_default_arguments(example, tmp_path):
    if example.stem in {"03_nnpu", "10_self_pu", "13_dynamic_joint_shift", "18_custom_encoders"}:
        pytest.importorskip("torch")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip(), "The tutorial should demonstrate its result."


def test_basic_upu_validation_uses_population_os_risk():
    demo = runpy.run_path(str(ROOT / "examples/minimal/02_upu.py"))
    expected_prior = (0.5 - demo["y_pu"].mean()) / (1 - demo["y_pu"].mean())
    assert demo["clf"].class_prior_ == pytest.approx(expected_prior)
    expected_risk = pu_zero_one_risk(
        demo["y_pu"], demo["clf"].decision_function(demo["X"]), 0.5, input_sampling="os"
    )
    assert demo["risk"] == pytest.approx(expected_risk)


def test_determ_upu_repeats_predictions_and_risk():
    first = runpy.run_path(str(ROOT / "examples/minimal/02_upu.py"))
    second = runpy.run_path(str(ROOT / "examples/minimal/02_upu.py"))
    np.testing.assert_array_equal(first["y_pred"], second["y_pred"])
    assert first["risk"] == second["risk"]


def test_basic_nnpu_consumes_native_unlabeled_prior():
    pytest.importorskip("torch")
    demo = runpy.run_path(str(ROOT / "examples/minimal/03_nnpu.py"))
    expected = (0.5 - demo["y_pu"].mean()) / (1 - demo["y_pu"].mean())
    assert demo["clf"].class_prior_ == pytest.approx(expected)
    assert np.isfinite(demo["risk"])


def test_edge_self_pu_converts_only_training_label_mass(monkeypatch):
    pytest.importorskip("torch")
    from pu_toolbox.estimators.deep import SelfPUClassifier

    observed = []
    original_fit = SelfPUClassifier.fit

    def audited_fit(self, features, labels, **kwargs):
        model = original_fit(self, features, labels, **kwargs)
        expected = (0.3 - np.mean(labels == 1)) / (1 - np.mean(labels == 1))
        observed.append((model.class_prior_, expected))
        return model

    monkeypatch.setattr(SelfPUClassifier, "fit", audited_fit)
    demo = runpy.run_path(str(ROOT / "examples/minimal/10_self_pu.py"))
    demo["main"]()
    assert len(observed) == 1
    assert observed[0][0] == pytest.approx(observed[0][1])


def test_basic_pnu_uses_unlabeled_risk():
    demo = runpy.run_path(str(ROOT / "examples/minimal/04_pnu.py"))
    model = demo["clf"]
    assert 0 < model.eta_ < 1
    components = model.risk_components_
    assert components["pnu_risk"] == pytest.approx(
        (1 - model.eta_) * components["pn_risk"] + model.eta_ * components["pu_risk"]
    )


def test_basic_joint_shift_uses_independent_population_pools(monkeypatch):
    pytest.importorskip("torch")
    from pu_toolbox import preprocessing
    from pu_toolbox.estimators.research import DynamicJointShiftPUClassifier

    draws, fitted = [], []
    generate = preprocessing.make_gaussian_case_control_data
    original_fit = DynamicJointShiftPUClassifier.fit

    def draw(*args, **kwargs):
        result = generate(*args, **kwargs)
        draws.append(result)
        return result

    def fit(self, features, labels, **kwargs):
        fitted.append((features, labels, kwargs))
        return original_fit(self, features, labels, **kwargs)

    monkeypatch.setattr(preprocessing, "make_gaussian_case_control_data", draw)
    monkeypatch.setattr(DynamicJointShiftPUClassifier, "fit", fit)
    demo = runpy.run_path(str(ROOT / "examples/minimal/13_dynamic_joint_shift.py"))
    demo["main"]()
    assert len(draws) == 2
    assert len(fitted) == 1
    features, labels, kwargs = fitted[0]
    for index, (features_drawn, labels_drawn, truth, metadata) in enumerate(draws):
        assert metadata["input_sampling"] == "ts"
        assert metadata["sampling_policy"] == "independent_generative"
        assert np.all(truth[labels_drawn == 1] == 1)
        assert len(np.unique(metadata["sample_ids"])) == len(truth)
        assert (
            kwargs["class_prior" if index == 0 else "target_class_prior"]
            == metadata["population_class_prior"]
        )
        np.testing.assert_array_equal(labels if index == 0 else kwargs["y_target_pu"], labels_drawn)
        np.testing.assert_array_equal(
            features if index == 0 else kwargs["X_target"], features_drawn
        )


@pytest.mark.parametrize(
    ("recommendation", "has_weighted", "use_weighted"),
    [
        ("reweight_recommended", True, True),
        ("no_clear_benefit", True, False),
        ("audit_only", True, False),
        ("collect_target_data", False, False),
    ],
)
def test_edge_shift_review_respects_comparison_decision(
    monkeypatch, recommendation, has_weighted, use_weighted
):
    from pu_toolbox import diagnostics, workflows

    baseline, weighted = object(), object()
    comparison = SimpleNamespace(
        recommendation=recommendation,
        baseline=SimpleNamespace(final_model=baseline),
        weighted=SimpleNamespace(final_model=weighted) if has_weighted else None,
    )
    monkeypatch.setattr(workflows.ShiftAwarePUPipeline, "compare", lambda *a, **kw: comparison)
    observed = []

    def record_review(model, *args, **kwargs):
        observed.append(model)
        return SimpleNamespace(summary={"coverage": 1.0}, query_indices=np.array([], dtype=int))

    monkeypatch.setattr(diagnostics, "analyze_pu_uncertainty", record_review)
    demo = runpy.run_path(str(ROOT / "examples/minimal/12_shift_decision_tools.py"))
    demo["main"]()
    assert observed == [weighted if use_weighted else baseline]
