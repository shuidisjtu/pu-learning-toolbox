"""Pin the one upward edge that static scanners structurally cannot see.

``registry`` is a Core-layer module, and Core is declared to have no upward
dependencies.  ``registry/builtin_methods.py`` reaches Algorithms and
Estimation anyway -- through ``importlib.import_module`` on a string, which an
AST scan cannot match: it only looks at Import/ImportFrom nodes.

The edge is a ruled exception, not a defect: looking algorithms up by name is
what keeps the experiment layer decoupled from the algorithm family.  But an
exception that exists only as prose is an unverified claim, so what the
exception *means* is pinned here.  Note what is deliberately not asserted: the
call adds no new modules to ``sys.modules``, because the umbrella facade in
``pu_toolbox/__init__.py`` already imported all of them.  The observable is the
registry, not the module cache.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pu_toolbox import registry
from pu_toolbox.core.exceptions import RegistryError

pytestmark = pytest.mark.contract

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_basic_registry_starts_empty_until_registration_is_called():
    """The edge fires on the call, never on the import.

    A separate process, because the registry is module-global and any earlier
    test in this session would already have populated it.
    """
    child = (
        "import pu_toolbox.registry as r;"
        "print(len(r.get_algorithm_registry()), end=' ');"
        "r.register_all_builtin_methods();"
        "print(len(r.get_algorithm_registry()))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    before, after = (int(value) for value in completed.stdout.split())
    assert before == 0
    assert after > 0


def test_edge_registered_classes_are_defined_in_the_higher_layers():
    """Core reaching Algorithms and Estimation, stated as the exception does.

    ``list_algorithms()`` yields metadata, not names, so the name is read off
    each entry before asking for the bound class.

    Walking every registered name also asserts a second, deliberate thing:
    nothing sits in the metadata table without a bound class.  It is asserted
    separately rather than left as a side effect of the loop, so the two
    failures say different things.  If a legitimately ``api_only`` entry is
    ever added, this is the line to revisit -- that is a registry question, not
    a question about the dynamic edge.
    """
    registry.register_all_builtin_methods()
    modules: set[str] = set()
    unbound: list[str] = []
    for entry in registry.list_algorithms():
        try:
            modules.add(registry.get_algorithm(entry.name).__module__)
        except RegistryError:
            unbound.append(entry.name)
    assert modules, "no algorithm class was bound"
    assert not unbound, f"registered without a bound class: {unbound}"
    outside = sorted(
        name
        for name in modules
        if not name.startswith(("pu_toolbox.estimators", "pu_toolbox.prior"))
    )
    assert not outside, f"registered classes outside the higher layers: {outside}"


def test_determ_registration_is_idempotent():
    registry.register_all_builtin_methods()
    first = registry.list_algorithms()
    registry.register_all_builtin_methods()
    assert registry.list_algorithms() == first


def test_param_unknown_method_name_is_rejected():
    registry.register_all_builtin_methods()
    with pytest.raises(RegistryError, match="unknown_method_name"):
        registry.get_algorithm("unknown_method_name")
