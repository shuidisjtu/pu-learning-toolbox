"""Every candidate estimator keeps the shared failed-fit guard and its role isolation.

Reads estimator source with ``ast`` only: no estimator import, no torch, no source hash.
The runtime behavior of the guard is covered by test_fit_state.py and the per-method
fit-failure tests; this file only pins that no candidate silently loses the decorator or
grows a clean-label parameter.
"""

import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "pu_toolbox/experiment/method_ledger.json"
ESTIMATORS = ROOT / "pu_toolbox/estimators"
GUARD = "invalidate_failed_fit"
CLEAN_ROLES = {"y_true", "test_data", "clean_val"}
MODULES = {
    "pulda": "risk/pulda.py",
    "puet": "risk/puet.py",
    "gradpu": "deep/grad_pu.py",
    "robust_pu": "deep/robust_pu.py",
    "split_pu": "deep/split_pu.py",
    "genpu": "deep/gen_pu.py",
    "holistic_pu": "deep/holistic_pu.py",
    "lagam": "deep/lagam.py",
    "pulns": "deep/pulns.py",
    "tedn": "deep/tedn.py",
    "vpu": "risk/vpu.py",
    "lbe_neural": "deep/lbe_neural.py",
    "pan": "deep/pan.py",
    "cvir": "risk/cvir.py",
    "rp": "classic/rank_pruning.py",
    "p3mix": "deep/p3mix.py",
}


def _class_name(method: str) -> str:
    methods = json.loads(LEDGER.read_text(encoding="utf-8"))["methods"]
    if method == "tedn":
        return "TEDnClassifier"
    return methods[method]["class"]


def _module_tree(method: str) -> ast.Module:
    return ast.parse((ESTIMATORS / MODULES[method]).read_text(encoding="utf-8"))


def _fit(method: str) -> ast.FunctionDef:
    name = _class_name(method)
    classes = [
        n for n in _module_tree(method).body if isinstance(n, ast.ClassDef) and n.name == name
    ]
    assert len(classes) == 1, f"{method}: candidate class {name} missing or duplicated"
    fits = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "fit"]
    assert len(fits) == 1, f"{method}: {name} must declare exactly one fit"
    return fits[0]


@pytest.mark.parametrize("method", sorted(MODULES))
def test_basic_fit_is_wrapped_by_the_shared_guard_exactly_once(method):
    decorators = _fit(method).decorator_list
    guards = [d for d in decorators if isinstance(d, ast.Name) and d.id == GUARD]
    assert len(guards) == 1, f"{method}: fit must carry {GUARD} exactly once"


@pytest.mark.parametrize("method", sorted(MODULES))
def test_param_fit_declares_no_clean_label_role_and_no_catch_all(method):
    args = _fit(method).args
    names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    assert not names & CLEAN_ROLES, f"{method}: fit declares a clean role {names & CLEAN_ROLES}"
    assert args.vararg is None and args.kwarg is None, f"{method}: fit must name its label roles"


@pytest.mark.parametrize("method", sorted(MODULES))
def test_edge_guard_is_imported_directly_from_the_shared_module(method):
    imports = [
        n
        for n in _module_tree(method).body
        if isinstance(n, ast.ImportFrom) and n.level == 3 and n.module == "core.fit_state"
    ]
    imported = {a.name for n in imports for a in n.names if a.asname is None}
    assert GUARD in imported, f"{method}: {GUARD} must come from core.fit_state, unaliased"


def test_determ_candidate_set_matches_the_guarded_module_list():
    guarded = sorted(
        str(p.relative_to(ESTIMATORS)).replace("\\", "/")
        for p in ESTIMATORS.rglob("*.py")
        if f"@{GUARD}" in p.read_text(encoding="utf-8")
    )
    missing = sorted(set(MODULES.values()) - set(guarded))
    assert not missing, f"listed candidates without a guarded fit: {missing}"
