# ruff: noqa: N803, S101

"""End-to-end UI run writes to the process-level history (D9 wiring)."""

import io
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pu_toolbox.ui import history  # noqa: E402

pytestmark = pytest.mark.integration

APP_PATH = Path(__file__).resolve().parents[2] / "pu_toolbox" / "ui" / "app.py"


def _csv_bytes(values: np.ndarray, header: str) -> bytes:
    buffer = io.StringIO()
    buffer.write(header + "\n")
    np.savetxt(buffer, values, fmt="%g", delimiter=",")
    return buffer.getvalue().encode("utf-8")


def test_basic_ui_run_writes_history_entry(tmp_path: Path):
    rng = np.random.RandomState(42)
    x = rng.randn(40, 5)
    y_pu = np.zeros(40, dtype=int)
    y_pu[:8] = 1
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    history.clear_for_tests()

    # 1) render once first — the widget tree is empty before the first run,
    #    so file_uploader would be an empty list and set_value would fail.
    app.run()
    assert not app.exception

    # 2) upload feature + label files; 3) re-run to process the uploads.
    # (streamlit >= 1.44 expects (filename, content, mime_type) tuples.)
    app.file_uploader[0].set_value(("X.csv", _csv_bytes(x, "f0,f1,f2,f3,f4"), "text/csv"))
    app.file_uploader[1].set_value(("y_pu.csv", _csv_bytes(y_pu, "label"), "text/csv"))
    app.run()
    assert not app.exception

    # 3) press the start-analysis button (label resolved from app.py at
    # implementation time; adjust if the wording differs).
    start = next(b for b in app.button if "分析" in b.label or "运行" in b.label)
    start.click()
    app.run()

    # 4) background thread finishes after a few script re-runs; poll the
    # process-level history instead of the session (refresh-safe by design).
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        entries = history.snapshot()
        if entries:
            assert entries[0]["状态"] == "completed"
            return
        time.sleep(1.0)
        app.run()
    pytest.fail("UI run did not write a history entry within 60s")
