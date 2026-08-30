# ruff: noqa: N802, N803, N806
"""Provenance field tests: architecture/backbone/device/encoder
(dual_architecture_plan.md §5 阶段 1)."""

from __future__ import annotations

import pytest

from pu_toolbox.preprocessing.data_profiler import PUDataProfile
from pu_toolbox.workflows._reporting import build_pipeline_report
from pu_toolbox.workflows.report import PriorInfo

pytestmark = [pytest.mark.unit]


def _report(*, architecture, backbone, device, encoder_in_channels):
    profile = PUDataProfile(
        summary={"n_samples": 40, "n_features": 5, "positive_fraction": 0.25},
        feature_statistics={},
        selection_diagnostic={
            "separability_auc": None,
            "is_identifying": False,
            "status": "inconclusive",
        },
        issues=(),
        assumption_hints=(),
    )
    return build_pipeline_report(
        profile=profile,
        prior_info=PriorInfo(value=0.3, source="user", method_requires_prior=True),
        recommendation=None,
        cv_metrics={},
        classifier_name="wconpu",
        auto_mode=False,
        classifier_cls=None,
        skipped_candidates=[],
        y_true=None,
        splitter=None,
        n_splits=2,
        final_model=None,
        diagnostic=None,
        random_state=42,
        classifier_params={},
        sample_weight=None,
        architecture=architecture,
        backbone=backbone,
        device=device,
        encoder_in_channels=encoder_in_channels,
    )


@pytest.mark.unit
def test_mlp_provenance_reports_native_mlp_without_backbone_or_encoder():
    report = _report(architecture="mlp", backbone=None, device="auto", encoder_in_channels=None)
    p = report.provenance
    assert p["architecture"] == "native_mlp"
    assert p["backbone"] is None
    assert p["encoder"] is None
    assert p["device"]["requested"] == "auto"
    assert p["device"]["resolved"] in {"cpu", "cuda"}


@pytest.mark.unit
def test_cnn_provenance_reports_native_cnn_with_backbone_and_encoder_summary():
    report = _report(architecture="cnn", backbone="cnn13", device="cpu", encoder_in_channels=3)
    p = report.provenance
    assert p["architecture"] == "native_cnn"
    assert p["backbone"] == "cnn13"
    assert p["encoder"] == {"backbone": "cnn13", "in_channels": 3}
    assert p["device"] == {"requested": "cpu", "resolved": "cpu"}
