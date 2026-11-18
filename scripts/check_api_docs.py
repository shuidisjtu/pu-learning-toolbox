r"""Check the API reference covers every public API symbol (anti-drift gate).

A public export or registered method must appear in a contract page under
``docs/user/reference/api/``. The navigation index is required but does not
count toward symbol coverage.

Symbol sources (both parsed statically, no package import -- CI must not
need torch or other optional deps):

- ``pu_toolbox/__init__.py`` ``__all__``  (package root public exports)
- ``pu_toolbox/registry/builtin_methods.py`` ``AlgorithmMetadata(name=...)``
  (active registered method names; deprecated aliases like ``pe`` are
  intentionally not checked -- they are not canonical names)

The check is name-presence based and intentionally shallow: the reference
doc is the human-facing contract, docstrings are the behavioural truth
source (ADR-0013); this gate only guards against forgotten/renamed symbols.
Either symbol source coming back empty is itself a failure -- a renamed or
unparsable file must not shrink the scan into a vacuous pass.

Run:  uv run python scripts/check_api_docs.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_ROOT / "docs" / "user" / "reference" / "api"
REQUIRED_PAGES = ("index.md", "classifiers.md", "tools.md", "errors.md")


def extract_all_exports(root: Path) -> list[str]:
    """Statically read the ``__all__`` string list of ``pu_toolbox/__init__.py``."""
    init_path = root / "pu_toolbox" / "__init__.py"
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except OSError:
        return []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            continue
        if isinstance(node.value, ast.List):
            return [
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
    return []


def extract_registered_names(root: Path) -> list[str]:
    """Statically read ``AlgorithmMetadata(name=...)`` entries from the registry."""
    reg_path = root / "pu_toolbox" / "registry" / "builtin_methods.py"
    try:
        tree = ast.parse(reg_path.read_text(encoding="utf-8"))
    except OSError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "AlgorithmMetadata"):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "name"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                names.append(kw.value.value)
    return names


def scan_text(text: str, symbols: list[str]) -> list[str]:
    """Return the symbols that do not appear in *text* (case-sensitive)."""
    return [s for s in symbols if s not in text]


def main(argv: list[str] | None = None) -> int:
    del argv  # CLI takes no arguments; kept for a consistent script signature
    exports = extract_all_exports(PROJECT_ROOT)
    registered = extract_registered_names(PROJECT_ROOT)
    if not exports:
        print(
            "Could not parse __all__ from pu_toolbox/__init__.py; refusing to pass empty scan.",
            file=sys.stderr,
        )
        return 1
    if not registered:
        print(
            "Could not parse any AlgorithmMetadata(name=...) entry from "
            "pu_toolbox/registry/builtin_methods.py; refusing to pass empty scan.",
            file=sys.stderr,
        )
        return 1
    symbols = sorted(set(exports + registered))
    try:
        pages = {path: path.read_text(encoding="utf-8") for path in sorted(API_DIR.rglob("*.md"))}
    except (OSError, UnicodeError) as exc:
        print(f"Could not read API reference under {API_DIR}: {exc}", file=sys.stderr)
        return 1
    if not pages:
        print(f"No Markdown pages found under {API_DIR}.", file=sys.stderr)
        return 1
    for name in REQUIRED_PAGES:
        if not pages.get(API_DIR / name, "").strip():
            print(f"Required API reference page is missing or empty: {name}", file=sys.stderr)
            return 1
    # Navigation mentions alone do not document a public symbol.
    contracts = "\n".join(text for path, text in pages.items() if path.name != "index.md")
    missing = scan_text(contracts, symbols)
    if missing:
        for s in missing:
            print(f"  API doc missing symbol: {s}")
        print(f"Missing {len(missing)} of {len(symbols)} symbols")
        return 1
    print(f"All {len(symbols)} public symbols covered by the API reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
