# ruff: noqa: E402, N802, N803, N806, E501

"""CLI run-subcommand deep-architecture tests (.npy input + flags)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.cli.run import build_run_parser, run_run


def _write_image(tmp_path, n=16, channels=3, size=8, seed=4):
    rng = np.random.RandomState(seed)
    array = rng.normal(0.5, 0.3, size=(n, channels, size, size)).astype(np.float32)
    data_path = tmp_path / "images.npy"
    np.save(data_path, array)
    labels = np.concatenate([np.ones(5, dtype=int), np.zeros(n - 5, dtype=int)])
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("labels\n" + "\n".join(str(v) for v in labels) + "\n")
    return data_path, labels_path


def _make_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="subcommand")
    build_run_parser(sub)
    return parser


def _run_cli(parser, argv):
    args = parser.parse_args(argv)
    try:
        run_run(args)
        return 0
    except SystemExit as exc:
        return exc.code


@pytest.mark.unit
class TestRunDeepInput:
    def test_basic_npy_with_cnn_wconpu_reaches_pipeline(self, tmp_path, monkeypatch):
        from pu_toolbox.cli import run as cli_run

        data_path, labels_path = _write_image(tmp_path)
        captured: dict = {}

        class FakePipe:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def fit_evaluate(self, X, y_pu, *, y_true=None, class_prior=None):
                assert X.ndim == 4 and X.shape[0] == 16
                return type("R", (), {"save": lambda *a: None, "summary": lambda *a: ""})()

        monkeypatch.setattr(cli_run, "PUPipeline", FakePipe)
        parser = _make_parser()
        code = _run_cli(
            parser,
            [
                "run",
                "--data",
                str(data_path),
                "--labels",
                str(labels_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--architecture",
                "cnn",
                "--backbone",
                "resnet18",
                "--device",
                "cpu",
                "--classifier",
                "wconpu",
            ],
        )
        assert code == 0
        assert captured["architecture"] == "cnn"
        assert captured["backbone"] == "resnet18"
        assert captured["device"] == "cpu"

    def test_param_cnn_with_shallow_classifier_exits_one(self, tmp_path, capsys):
        data_path, labels_path = _write_image(tmp_path)
        parser = _make_parser()
        code = _run_cli(
            parser,
            [
                "run",
                "--data",
                str(data_path),
                "--labels",
                str(labels_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--architecture",
                "cnn",
                "--classifier",
                "upu",
            ],
        )
        assert code == 1

    def test_param_backbone_with_mlp_exits_one(self, tmp_path, capsys):
        data_path, labels_path = _write_image(tmp_path)
        parser = _make_parser()
        code = _run_cli(
            parser,
            [
                "run",
                "--data",
                str(data_path),
                "--labels",
                str(labels_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--architecture",
                "mlp",
                "--backbone",
                "cnn13",
                "--classifier",
                "upu",
            ],
        )
        assert code == 1

    def test_edge_4d_npy_with_mlp_exits_one(self, tmp_path, capsys):
        data_path, labels_path = _write_image(tmp_path)
        parser = _make_parser()
        code = _run_cli(
            parser,
            [
                "run",
                "--data",
                str(data_path),
                "--labels",
                str(labels_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--architecture",
                "mlp",
                "--classifier",
                "upu",
            ],
        )
        assert code == 1

    def test_param_invalid_npy_dimension_exits_one(self, tmp_path, capsys):
        rng = np.random.RandomState(0)
        path = tmp_path / "flat.npy"
        np.save(path, rng.rand(10, 5).astype(np.float32))
        labels_path = tmp_path / "labels.csv"
        labels_path.write_text("labels\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n")
        parser = _make_parser()
        code = _run_cli(
            parser,
            [
                "run",
                "--data",
                str(path),
                "--labels",
                str(labels_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--classifier",
                "upu",
            ],
        )
        assert code == 1
