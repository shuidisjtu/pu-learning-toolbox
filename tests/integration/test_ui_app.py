"""Streamlit page smoke tests (enabled by the optional ``ui`` extra)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration

APP_PATH = Path(__file__).resolve().parents[2] / "pu_toolbox" / "ui" / "app.py"


def _run_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=20)
    return app


def test_basic_ui_renders_upload_workflow():
    app = _run_app()
    assert not app.exception
    assert app.title[0].value == "PU Learning Toolbox"
    assert len(app.get("file_uploader")) == 3


def test_param_ui_exposes_expected_upload_types():
    app = _run_app()
    labels = [u.label for u in app.get("file_uploader")]
    assert labels == [
        "特征数据",
        "PU 标签（1/0 单列 CSV）",
        "真实标签（可选，1/0 单列 CSV）",
    ]


def test_edge_ui_waits_for_required_uploads_without_exception():
    app = _run_app()
    assert not app.exception
    assert any("请先上传" in message.value for message in app.info)


def test_deterministic_ui_initial_render_is_stable():
    first = _run_app()
    second = _run_app()
    assert [title.value for title in first.title] == [title.value for title in second.title]
    assert len(first.get("file_uploader")) == len(second.get("file_uploader"))
