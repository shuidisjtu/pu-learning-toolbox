"""Public export contract: a declared name must actually exist.

``__all__`` is what ``from pu_toolbox.experiment import *`` reads, and nothing in
the repository checked it against the module's own namespace: ``check_api_docs.py``
scans the package root's ``__all__`` plus the builtin registry, and no test used
the star form.  A name could therefore sit in a package's ``__all__`` with no
matching import while every gate passed and the star import raised
``AttributeError``.

The walk is over package ``__init__`` files on purpose.  That is where a
re-export is declared in one list and bound in another, so the two can drift in
a single edit; a module that defines its own names declares and binds them
adjacent to each other.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import pu_toolbox

pytestmark = pytest.mark.contract


def declared_but_absent(modules: Mapping[str, Any]) -> dict[str, list[str]]:
    """Per module, the names ``__all__`` declares that the module does not have.

    A module with no ``__all__`` declares nothing and is not a finding: the
    attribute's absence means "export everything public", not "export a name
    that is missing".
    """
    violations: dict[str, list[str]] = {}
    for name, module in sorted(modules.items()):
        declared = list(getattr(module, "__all__", None) or [])
        absent = [item for item in declared if not hasattr(module, item)]
        if absent:
            violations[name] = absent
    return violations


def _package_modules() -> dict[str, Any]:
    """Every subpackage of :mod:`pu_toolbox`, keyed by its dotted name."""
    root = Path(pu_toolbox.__file__).parent
    modules: dict[str, Any] = {}
    for init_file in sorted(root.rglob("__init__.py")):
        relative = init_file.parent.relative_to(root)
        dotted = ".".join(("pu_toolbox", *relative.parts))
        modules[dotted] = importlib.import_module(dotted)
    return modules


def test_basic_every_declared_export_exists():
    assert declared_but_absent(_package_modules()) == {}


def test_param_invalid_declaration_is_reported():
    class Stub:
        __all__ = ["bound", "unbound"]
        bound = object()

    assert declared_but_absent({"stub": Stub}) == {"stub": ["unbound"]}


def test_edge_empty_and_undeclared_exports_are_not_violations():
    class Empty:
        __all__: list[str] = []

    assert declared_but_absent({"empty": Empty, "undeclared": object()}) == {}


def test_determ_the_walk_reports_the_same_violations_twice():
    first = declared_but_absent(_package_modules())
    second = declared_but_absent(_package_modules())
    assert first == second == {}
