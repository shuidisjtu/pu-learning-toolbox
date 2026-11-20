"""Independent CLI entry points enforce the PU label meaning contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from pu_toolbox.core.base import BasePUClassifier

pytestmark = pytest.mark.unit


@pytest.fixture
def data_files(tmp_path):
    feature = tmp_path / "X.csv"
    labels = tmp_path / "y.csv"
    feature.write_text("x,z\n" + "".join(f"{i},{i % 3}\n" for i in range(40)))
    labels.write_text("label\n" + "".join(f"{i % 2}\n" for i in range(40)))
    return feature, labels


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
        sensitivity, "get_metadata", lambda name: SimpleNamespace(required_prior_parameters=())
    )
    monkeypatch.setattr(sensitivity, "get_algorithm", lambda name: WrongClassifier)
    features, labels = data_files
    args = SimpleNamespace(
        data=str(features),
        labels=str(labels),
        classifier="custom",
        seed=42,
        population_priors="0.3,0.5",
        label_propensities=None,
        out_dir=str(tmp_path / "result"),
    )
    with pytest.raises(ValueError, match="label_semantics"):
        sensitivity.run_sensitivity(args)
    assert not (tmp_path / "result").exists()


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
