"""Portable result handoff never averages missing seeds or doubles repairs."""

import json

import pytest

from scripts import export_survey_queue_handoff as exporter

pytestmark = pytest.mark.unit


def rows(seeds):
    return [
        dict(
            dataset="mnist",
            profile="nnpu",
            training_path="native_cnn",
            view="ts",
            comparability_group="shared",
            mechanism="scar",
            c="0.1",
            selection="PA",
            protocol_sha256="p",
            recipe_registry_sha256="r",
            source_family="baseline",
            seed=seed,
            accuracy_pct=80.0 + seed,
            auc=0.9,
            f1=None,
            precision=0.8,
            recall=0.8,
        )
        for seed in seeds
    ]


def test_basic_determ_complete_group_has_sample_std_and_no_admission():
    complete, incomplete = exporter.summarize(rows(range(5)))
    assert (complete, incomplete) == exporter.summarize(rows(range(5)))
    assert not incomplete
    assert complete[0]["accuracy_pct_mean"] == 82.0
    assert complete[0]["accuracy_pct_std_ddof1"] == pytest.approx(2.5**0.5)
    assert complete[0]["f1_mean"] is None
    assert not complete[0]["formal_eligible"]


@pytest.mark.parametrize("seeds", [[0], [0, 1, 2, 3], [1, 2, 3, 4]])
def test_param_edge_missing_seed_never_gets_mean(seeds):
    complete, incomplete = exporter.summarize(rows(seeds))
    assert not complete
    assert "accuracy_pct_mean" not in incomplete[0]
    assert incomplete[0]["status"] == "incomplete_no_mean_or_conclusion"


def test_edge_duplicate_seed_and_mixed_sources_are_not_pooled():
    with pytest.raises(ValueError, match="duplicate seed"):
        exporter.summarize(rows([0, 0]))
    data = rows(range(5))
    data[0]["source_family"] = "corrected"
    complete, incomplete = exporter.summarize(data)
    assert not complete and len(incomplete) == 2


def test_basic_correction_replaces_only_in_export(tmp_path, monkeypatch):
    key = ("mnist", "lagam", "cnn_feature_adapter", "scar", "0.1", 0)
    baseline = tmp_path / "old"
    repair = tmp_path / "new"
    for root in (baseline, repair):
        root.mkdir()
        (root / "manifest.json").write_text("unchanged")
    jobs = [
        dict(
            label=name,
            root=str(root),
            expected_runs=1,
            expected_keys=[list(key)],
            campaign_id=campaign,
            status="completed",
        )
        for name, root, campaign in [("old", baseline, None), ("new", repair, "repair")]
    ]
    (tmp_path / "queue_status.json").write_text(json.dumps(dict(jobs=jobs, updated_at="snapshot")))
    (tmp_path / "batch_evidence.json").write_text(
        json.dumps(dict(protocol_sha256="p", previous_manifest_paths=[]))
    )

    def read(path, protocol):
        accuracy = 0.9 if path.parent == repair else 0.6
        manifest = dict(
            selection={"OA": {}},
            test_results={"OA": {"accuracy": accuracy}},
            protocol_sha256=protocol,
        )
        return dict(key=key, path=str(path), sha256="hash", manifest=manifest)

    monkeypatch.setattr(exporter, "load_record", read)
    data, replacements, summary = exporter.collect(tmp_path, lambda job: True)
    assert len(data) == 1 and data[0]["accuracy_pct"] == 90.0
    assert len(replacements) == 1
    assert summary["baseline_completed_fits"] == summary["effective_completed_fits"] == 1
    assert summary["corrected_fits"] == 1
    assert (baseline / "manifest.json").read_text() == "unchanged"
