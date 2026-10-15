"""CLI/UI/search entry points consume the same PU label meaning contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.progress import CancellationToken

pytestmark = pytest.mark.unit


@pytest.fixture
def data_files(tmp_path):
    feature = tmp_path / "X.csv"
    labels = tmp_path / "y.csv"
    feature.write_text("x,z\n" + "".join(f"{i},{i % 3}\n" for i in range(40)))
    labels.write_text("label\n" + "".join(f"{i % 2}\n" for i in range(40)))
    return feature, labels


def test_basic_cli_run_rejects_pnu_before_fit(data_files, tmp_path, monkeypatch):
    from pu_toolbox.cli import build_parser
    from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

    register_all_builtin_methods()
    monkeypatch.setattr(get_algorithm("pnu"), "fit", lambda *a, **k: pytest.fail("fit reached"))
    features, labels = data_files
    args = build_parser().parse_args(
        [
            "run",
            "--data",
            str(features),
            "--labels",
            str(labels),
            "--classifier",
            "pnu",
            "--class-prior",
            "0.4",
            "--cv",
            "2",
            "--out-dir",
            str(tmp_path / "result"),
        ]
    )
    with pytest.raises(ValueError, match="label_semantics='pnu'"):
        args.func(args)
    assert not (tmp_path / "result/report.json").exists()


@pytest.mark.parametrize("meaning", ["pn", "pnu", "broken"])
def test_param_sensitivity_direct_fit_is_guarded(meaning, data_files, tmp_path, monkeypatch):
    from pu_toolbox.cli import sensitivity

    class WrongClassifier(BasePUClassifier):
        label_semantics = meaning

        def fit(self, *args, **kwargs):
            pytest.fail("sensitivity bypassed semantic guard")

        def _predict(self, features):
            return np.ones(len(features))

        def _decision_function(self, features):
            return np.zeros(len(features))

    monkeypatch.setattr(
        sensitivity, "get_metadata", lambda name: SimpleNamespace(requires_class_prior=False)
    )
    monkeypatch.setattr(sensitivity, "get_algorithm", lambda name: WrongClassifier)
    features, labels = data_files
    args = SimpleNamespace(
        data=str(features),
        labels=str(labels),
        classifier="custom",
        seed=42,
        class_priors="0.3,0.5",
        label_propensities=None,
        out_dir=str(tmp_path / "result"),
    )
    with pytest.raises(ValueError, match="label_semantics"):
        sensitivity.run_sensitivity(args)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("mode", ["pipeline", "comparison", "tuning"])
def test_param_ui_dispatch_rejects_mismatch_without_history_or_fit(mode, monkeypatch):
    from pu_toolbox.registry import get_algorithm, register_all_builtin_methods
    from pu_toolbox.ui.execution import execute_analysis
    from pu_toolbox.workflows import PipelineError

    register_all_builtin_methods()
    monkeypatch.setattr(get_algorithm("pnu"), "fit", lambda *a, **k: pytest.fail("fit reached"))
    with pytest.raises((ValueError, PipelineError), match="label_semantics='pnu'"):
        execute_analysis(
            X=np.arange(80).reshape(40, 2),
            y_pu=np.tile([0, 1], 20),
            y_true=None,
            class_prior=0.4,
            classifier="pnu",
            classifier_params={},
            tuning_grid={"reg_lambda": [0.1]} if mode == "tuning" else {},
            comparison_classifiers=["upu", "pnu"] if mode == "comparison" else [],
            scoring="pu_zero_one_risk",
            pipeline_params={"cv": 2},
            cancellation_token=CancellationToken(),
            progress_callback=lambda event: None,
        )


def test_basic_ui_catalog_marks_pnu_unavailable_for_pu_uploads():
    from pu_toolbox.ui.parameters import classifier_catalog

    catalog = {row["name"]: row for row in classifier_catalog()}
    assert catalog["pnu"]["label_semantics"] == "pnu"
    assert not catalog["pnu"]["ui_ready"]
    assert catalog["upu"]["label_semantics"] == "pu" and catalog["upu"]["ui_ready"]


@pytest.mark.parametrize("mode", ["manual", "comparison"])
def test_edge_ui_config_import_rejects_pnu_without_mutating_widget_state(mode):
    from pu_toolbox.run_config import RunConfiguration
    from pu_toolbox.ui import apply_run_configuration, classifier_catalog

    catalog = {row["name"]: row for row in classifier_catalog()}
    config = RunConfiguration(
        classifier="pnu" if mode == "manual" else "auto",
        comparison_classifiers=("upu", "pnu") if mode == "comparison" else (),
    )
    state = {"existing": "preserve"}
    with pytest.raises(ValueError, match="label_semantics='pnu'"):
        apply_run_configuration(state, config, catalog)
    assert state == {"existing": "preserve"}


def test_determ_cli_labels_column_preserves_alias_semantics(capsys):
    from pu_toolbox.cli import build_parser
    from pu_toolbox.registry import list_algorithms

    args = build_parser().parse_args(["list-methods"])
    args.func(args)
    first = capsys.readouterr().out
    args.func(args)
    assert capsys.readouterr().out == first
    assert "Labels" in first
    rows = {row.split()[0]: row.split()[-2] for row in first.splitlines()[2:]}
    for metadata in list_algorithms():
        if metadata.name in rows:
            assert rows[metadata.name] == metadata.label_semantics
            for alias in metadata.aliases:
                assert rows[alias] == metadata.label_semantics
    assert rows["pnu"] == "pnu" and rows["self_pu"] == "pu"


def test_edge_recommender_excludes_non_pu_methods_and_records_filter():
    from pu_toolbox.advisor import recommend_methods

    result = recommend_methods(
        np.arange(80).reshape(40, 2),
        np.tile([0, 1], 20),
        class_prior=0.4,
        top_k=100,
    )
    assert all(candidate.name != "pnu" for candidate in result.candidates)
    assert result.filters_applied["label_semantics"] == "pu (main training input)"
