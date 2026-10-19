"""Read-only AST/source binding for candidate failed-fit guards, not admission.

Print strict JSON, or compare it with an engineering draft using --check.
No estimator imports, training, artifact loading or file writes occur here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = "pu_toolbox/experiment/method_ledger.json"
GUARD = "pu_toolbox/core/fit_state.py"
CHECKER = "scripts/check_candidate_fit_contracts.py"
METHODS = {
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


def require(condition, message):
    """Reject missing evidence; do not infer permission from code availability."""
    if not condition:
        raise ValueError(message)


def repo_file(root, reference):
    """Only regular, repository-contained relative file references are evidence."""
    require(isinstance(reference, str) and reference, "missing repository reference")
    relative = Path(reference)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe reference")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "missing/escaping file")
    return path


def source_digest(path):
    """Normalize line endings so Linux/Windows checkouts share an identity."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def fit_signature(klass):
    """Describe declared parameters without importing a deep-learning runtime."""
    fits = [node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "fit"]
    require(len(fits) == 1, "candidate must declare exactly one fit")
    fit = fits[0]
    require(
        sum(isinstance(d, ast.Name) and d.id == "invalidate_failed_fit" for d in fit.decorator_list)
        == 1,
        "missing/duplicate failed-fit guard",
    )
    require(fit.args.kwarg is None and fit.args.vararg is None, "fit must declare label roles")
    positional = [*fit.args.posonlyargs, *fit.args.args]
    require(positional and positional[0].arg == "self", "fit lacks self")
    defaults = [None] * (len(positional) - len(fit.args.defaults)) + list(fit.args.defaults)
    parameters = []
    groups = (
        (positional, defaults, len(fit.args.posonlyargs)),
        (fit.args.kwonlyargs, fit.args.kw_defaults, None),
    )
    for arguments, values, positional_only_count in groups:
        for index, (arg, value) in enumerate(zip(arguments, values, strict=True)):
            if arg.arg == "self":
                continue
            require(arg.arg not in {"y_true", "test_data", "clean_val"}, "undeclared clean role")
            parameters.append(
                {
                    "name": arg.arg,
                    "kind": "keyword_only"
                    if positional_only_count is None
                    else (
                        "positional_only"
                        if index < positional_only_count
                        else "positional_or_keyword"
                    ),
                    "required": value is None,
                    **({"default": ast.literal_eval(value)} if value is not None else {}),
                }
            )
    return parameters


def build_contracts(*, root=ROOT):
    """Bind current guard declarations and ledger facts, never authorize a run."""
    facts = json.loads(repo_file(root, LEDGER).read_text(encoding="utf-8"))
    guard = repo_file(root, GUARD)
    methods = {}
    for method, module in METHODS.items():
        if method == "tedn":
            entry = facts["methods"]["cvir"]["independent_extensions"]["tedn"]
            route_entry = facts["methods"]["tedn"]
            for field in ("calibration_applied", "calibration_hooked"):
                require(
                    entry.get(field) == route_entry.get(field),
                    "TEDn independent-extension calibration drift: " + field,
                )
            class_name = "TEDnClassifier"
            ledger_ref = LEDGER + "#methods.cvir.independent_extensions.tedn"
        else:
            entry = facts["methods"][method]
            route_entry = entry
            class_name = entry["class"]
            ledger_ref = LEDGER + "#methods." + method
        implementation = "pu_toolbox/estimators/" + module
        path = repo_file(root, implementation)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        require(
            any(
                isinstance(node, ast.ImportFrom)
                and node.level == 3
                and node.module == "core.fit_state"
                and any(
                    alias.name == "invalidate_failed_fit" and alias.asname is None
                    for alias in node.names
                )
                for node in tree.body
            ),
            "missing direct shared guard import: " + method,
        )
        classes = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        require(len(classes) == 1, "missing/duplicate candidate class: " + method)
        review_ref = entry.get("fit_failure_review_ref")
        review = repo_file(root, review_ref)
        calibration_hooked = route_entry.get("calibration_hooked")
        require(type(calibration_hooked) is bool, "missing/invalid calibration_hooked: " + method)
        calibration_applied = route_entry["calibration_applied"]
        require(type(calibration_applied) is bool, "bad calibration flag: " + method)
        require(
            not calibration_applied or calibration_hooked,
            "calibration_applied=true without an explicit fit hook: " + method,
        )
        run_view = route_entry.get("run_view", entry.get("run_view"))
        if not calibration_hooked and run_view == "ts-compatible":
            require(
                route_entry.get("calibration_applicability") == "not_applicable_native_ts",
                "uncalibrated TS-compatible method lacks native-TS applicability: " + method,
            )
        methods[method] = {
            "class": class_name,
            "implementation_ref": implementation,
            "implementation_sha256": source_digest(path),
            "fit_parameters": fit_signature(classes[0]),
            "ledger_ref": ledger_ref,
            "fit_failure_review_ref": review_ref,
            "review_document_sha256": source_digest(review),
            "run_view": run_view or entry.get("native_sampling_assumption"),
            "calibration_applied": calibration_applied,
            "calibration_hooked": calibration_hooked,
            "formal_admission": False,
        }
        if calibration_hooked:
            require(
                any(
                    parameter["name"] == "os_or_ts"
                    for parameter in methods[method]["fit_parameters"]
                ),
                "calibration_hooked=true without a public os_or_ts fit parameter: " + method,
            )
    result = {
        "schema_version": "candidate-fit-contracts-engineering-draft-1",
        "scope": "AST declarations and source identity; runtime behavior belongs to tests",
        "source_digest_policy": "utf8_text_universal_newlines_sha256",
        "guard_ref": GUARD,
        "guard_sha256": source_digest(guard),
        "checker_ref": CHECKER,
        "checker_sha256": source_digest(repo_file(root, CHECKER)),
        "invalidates_before_argument_binding": True,
        "exception_boundary": "BaseException_reraise_no_retry",
        "rollback_of_models_or_external_artifacts": False,
        "formal_admission": False,
        "owner_signatures": [],
        "methods": methods,
    }
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    """Emit a receipt or verify a recorded draft without modifying any files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_contracts()
        if args.check is not None:
            recorded = json.loads(args.check.read_text(encoding="utf-8"))
            require(recorded == report, "candidate lifecycle evidence drift")
            print(
                json.dumps(
                    {"ok": True, "methods_checked": list(METHODS), "formal_admission": False}
                )
            )
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
