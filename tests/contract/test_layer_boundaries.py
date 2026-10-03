"""Layer-boundary ratchet: no lower layer may reach the survey experiment layer.

``docs/dev/architecture.md`` §2.1 declares ``experiment/`` a leaf entry point --
"not depended on by any layer".  This module pins that direction two ways,
because each alone has a blind spot:

* an AST scan catches *function-local* imports, which never show up in an
  import graph because nothing executes them at module load.  It is limited to
  what an AST can see: an import built from a runtime string
  (``importlib.import_module("pu_toolbox.experiment")``) leaves no trace, and
  no static scan can recover it;
* a subprocess import graph catches *transitive* pulls -- a lower-layer module
  reaching ``experiment`` through some third package -- and does execute
  dynamic imports, but only when they run at module load; a dynamic import
  inside a function escapes both layers.

The rule is quoted from §2.1; this file does not invent one.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _PROJECT_ROOT / "pu_toolbox"

# Lower layers per architecture.md §2.  Orchestration (workflows/, cli/) and the
# user layer may legitimately reach experiment/ and are deliberately absent.
_LOWER_LAYERS = (
    "core",
    "preprocessing",
    "registry",
    "advisor",
    "utils",
    "prior",
    "losses",
    "estimators",
    "metrics",
    "model_selection",
    "diagnostics",
)

_FORBIDDEN = "pu_toolbox.experiment"


def _lower_layer_sources() -> list[Path]:
    """Every .py file under the lower layers, sorted for reproducibility."""
    files: list[Path] = []
    for layer in _LOWER_LAYERS:
        root = _PACKAGE_ROOT / layer
        if root.is_dir():
            files.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


def _imports_experiment(source: str, filename: str, *, package: str) -> list[tuple[int, str]]:
    """Return (lineno, target) for every *static* import of the experiment layer.

    Walks the whole tree, so a function-local import counts too.  ``package``
    is the dotted package of the file being scanned; it is needed to resolve
    relative imports (``from ..experiment import x``).

    Limited to what an AST can see: a dynamic import built from a string
    (``importlib.import_module("pu_toolbox.experiment")``) is invisible here
    and is only caught by the subprocess layer when it runs at import time.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                anchor = ".".join([*base, node.module]) if node.module else ".".join(base)
            else:
                anchor = node.module or ""
            if anchor:
                # ``from X import a`` may import the submodule X.a, so each
                # alias is a candidate too.  This is what catches
                # ``from pu_toolbox import experiment``.
                candidates = [anchor, *(f"{anchor}.{alias.name}" for alias in node.names)]
        match = next(
            (c for c in candidates if c == _FORBIDDEN or c.startswith(_FORBIDDEN + ".")),
            None,
        )
        if match:
            hits.append((node.lineno, match))
    return hits


def _package_of(path: Path) -> str:
    """Dotted package of a file under pu_toolbox/, e.g. pu_toolbox.core."""
    rel = path.relative_to(_PACKAGE_ROOT)
    parts = list(rel.parts[:-1])
    return ".".join(["pu_toolbox", *parts])


_CHILD = r"""
import importlib, pathlib, sys

ROOT = pathlib.Path("pu_toolbox")
LAYERS = sys.argv[1:]
FORBIDDEN = "pu_toolbox.experiment"


def module_names():
    out = []
    for layer in LAYERS:
        base = ROOT / layer
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            parts = list(path.relative_to(ROOT).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            out.append("pu_toolbox." + ".".join(parts))
    return out


errors = []
violations = []
for name in module_names():
    before = set(sys.modules)
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report and keep scanning
        errors.append(f"IMPORT-ERROR: {name}: {type(exc).__name__}: {exc}")
    for mod in sorted(set(sys.modules) - before):
        if mod == FORBIDDEN or mod.startswith(FORBIDDEN + "."):
            violations.append(f"{name} pulled {mod}")
for line in errors:
    print(line)
for line in violations:
    print(line)
"""


def test_basic_no_lower_layer_source_imports_the_experiment_layer():
    """Static scan over every lower-layer file, at any nesting level."""
    violations = []
    for path in _lower_layer_sources():
        found = _imports_experiment(
            path.read_text(encoding="utf-8"),
            str(path),
            package=_package_of(path),
        )
        violations.extend(
            f"{path.relative_to(_PROJECT_ROOT)}:{lineno} -> {target}" for lineno, target in found
        )
    assert not violations, "lower layer reached the survey layer: " + "; ".join(violations)


def test_basic_subprocess_import_graph_does_not_pull_the_experiment_layer():
    """Transitive check: one child process, per-module sys.modules diff."""
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, *_LOWER_LAYERS],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("IMPORT-ERROR:")]
    violations = [line for line in lines if not line.startswith("IMPORT-ERROR:")]
    assert not errors, "lower-layer module failed to import:\n" + "\n".join(errors)
    assert not violations, "lower layer reached the survey layer:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "import pu_toolbox.experiment\n",
        "import pu_toolbox.experiment.runner\n",
        "from pu_toolbox.experiment import runner\n",
        "def f():\n    from pu_toolbox.experiment import runner\n",
        "from ..experiment import runner\n",
        "from ..experiment.runner import run\n",
        "from pu_toolbox import experiment\n",
        "def f():\n    from pu_toolbox import experiment\n",
    ],
)
def test_param_scanner_flags_every_forbidden_import_form(source):
    assert _imports_experiment(source, "synthetic.py", package="pu_toolbox.core")


def test_edge_scanner_ignores_mentions_in_comments_and_strings():
    source = '# from pu_toolbox.experiment import runner\nDOC = "pu_toolbox.experiment"\n'
    assert _imports_experiment(source, "synthetic.py", package="pu_toolbox.core") == []


def test_determ_scan_of_real_sources_is_reproducible():
    """Pin purity/order-stability, not a live defect.

    Scans the real lower-layer sources forwards and in reverse; the per-file
    results must match.  The implementation sorts its file list and keeps no
    cross-file state, so a mismatch could only mean that property regressed --
    there is no known defect this test catches today.
    """

    def scan(paths: list[Path]) -> dict[str, list[tuple[int, str]]]:
        return {
            str(p): _imports_experiment(
                p.read_text(encoding="utf-8"), str(p), package=_package_of(p)
            )
            for p in paths
        }

    sources = _lower_layer_sources()
    assert scan(sources) == scan(list(reversed(sources)))
