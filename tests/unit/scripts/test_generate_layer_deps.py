"""Tests for the layer-dependency table generator.

The generator's job is to make ``docs/dev/architecture.md`` §2.1 impossible to
drift: the block is a pure function of the source tree.  These tests drive it
against synthetic trees so they never depend on the real repository layout.

Category names are load-bearing: ``scripts/check_test_quality.py`` classifies by
name substring, and ``generate`` is a *basic* keyword -- a test name that says
"generator" silently counts as basic coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import generate_layer_deps as gld  # noqa: E402

pytestmark = pytest.mark.unit

_TWO_LAYERS = {"Low": ("pkg/low",), "High": ("pkg/high",)}


def _write(root: Path, rel: str, source: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_basic_two_layers_linked_by_one_import_are_reported(tmp_path):
    _write(tmp_path, "pkg/high/a.py", "from pkg.low.b import thing\n")
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    assert gld.measure_edges(tmp_path, _TWO_LAYERS) == {"High": {"Low": "module"}}


def test_basic_edges_from_several_files_in_one_layer_are_merged(tmp_path):
    """A layer is many files; one file's edges must not overwrite another's."""
    _write(tmp_path, "pkg/high/a.py", "import pkg.low.b\n")
    _write(tmp_path, "pkg/high/z.py", "import pkg.top.c\n")
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    _write(tmp_path, "pkg/top/c.py", "thing = 1\n")
    layers = {"Low": ("pkg/low",), "Top": ("pkg/top",), "High": ("pkg/high",)}
    assert gld.measure_edges(tmp_path, layers) == {"High": {"Low": "module", "Top": "module"}}


def test_basic_layer_with_no_outgoing_import_renders_a_dash(tmp_path):
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    _write(tmp_path, "pkg/high/a.py", "value = 1\n")
    edges = gld.measure_edges(tmp_path, _TWO_LAYERS)
    block = gld.render_block(edges, _TWO_LAYERS)
    assert "| High（`high/`） | — |" in block


def test_basic_stale_document_makes_check_mode_fail(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "pkg/high/a.py", "import pkg.low.b\n")
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    doc = tmp_path / "architecture.md"
    doc.write_text(
        f"intro\n{gld.BEGIN_MARK}\nstale content\n{gld.END_MARK}\ntail\n",
        encoding="utf-8",
        newline="",
    )
    monkeypatch.setattr(gld, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gld, "ARCHITECTURE_MD", doc)
    monkeypatch.setattr(gld, "LAYERS", _TWO_LAYERS)
    assert gld.main(["--check"]) == 1
    assert "stale" in capsys.readouterr().out

    assert gld.main(["--update"]) == 0
    assert gld.main(["--check"]) == 0
    # The patched layer map is the one that took effect.  A default argument
    # bound at import time would silently keep the real map, and the row below
    # would never appear.
    assert "| High（`high/`） | Low |" in doc.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "source",
    [
        "import pkg.low.b\n",
        "from pkg import low\n",
        "from pkg.low import b\n",
        "from ..low import b\n",
    ],
)
def test_param_every_import_form_is_detected_in_isolation(tmp_path, source):
    """One tree per form, so a broken sibling cannot be covered for.

    ``from ..low import b`` is the form that matters most: resolving a relative
    import needs the *containing* package, and using the module's own dotted
    name instead silently resolves it one level too deep -- a miss that a tree
    holding an absolute import alongside it will happily hide.
    """
    _write(tmp_path, "pkg/high/a.py", source)
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    assert gld.measure_edges(tmp_path, _TWO_LAYERS) == {"High": {"Low": "module"}}


def test_edge_function_local_import_is_marked_as_such(tmp_path):
    _write(tmp_path, "pkg/high/a.py", "def f():\n    import pkg.low.b\n")
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    assert gld.measure_edges(tmp_path, _TWO_LAYERS) == {"High": {"Low": "function"}}


def test_edge_facade_and_top_level_helpers_are_not_measured(tmp_path):
    """The umbrella facade reaches every layer by design; it is not a layer."""
    _write(tmp_path, "pkg/__init__.py", "import pkg.low.b\n")
    _write(tmp_path, "pkg/run_config.py", "import pkg.low.b\n")
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    assert gld.measure_edges(tmp_path, _TWO_LAYERS) == {}


def test_edge_import_of_the_same_layer_is_not_a_layer_edge(tmp_path):
    _write(tmp_path, "pkg/low/a.py", "from pkg.low.b import thing\n")
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    assert gld.measure_edges(tmp_path, _TWO_LAYERS) == {}


def test_edge_unparseable_source_fails_loudly(tmp_path):
    """A silent skip would under-report edges -- the exact failure this guards."""
    _write(tmp_path, "pkg/high/a.py", "def broken(:\n")
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    with pytest.raises(SyntaxError):
        gld.measure_edges(tmp_path, _TWO_LAYERS)


def test_edge_missing_markers_are_reported(tmp_path):
    with pytest.raises(ValueError):
        gld.replace_block("no markers here\n", "block\n")


def test_determ_measurement_is_order_stable(tmp_path):
    _write(tmp_path, "pkg/high/a.py", "import pkg.low.b\n")
    _write(tmp_path, "pkg/high/z.py", "import pkg.low.b\n")
    _write(tmp_path, "pkg/low/b.py", "thing = 1\n")
    first = gld.measure_edges(tmp_path, _TWO_LAYERS)
    assert first == gld.measure_edges(tmp_path, _TWO_LAYERS)
    assert gld.render_block(first, _TWO_LAYERS) == gld.render_block(first, _TWO_LAYERS)


def test_determ_update_is_idempotent(tmp_path):
    text = f"head\n{gld.BEGIN_MARK}\nold\n{gld.END_MARK}\ntail\n"
    block = gld.render_block({}, _TWO_LAYERS)
    once = gld.replace_block(text, block)
    assert gld.replace_block(once, block) == once
    # The file's own line ending is reused, not hardcoded.
    assert gld.render_block({}, _TWO_LAYERS, newline="\r\n").count("\r\n") > 0
