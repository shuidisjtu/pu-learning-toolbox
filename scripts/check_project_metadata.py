"""Check compatibility and packaging metadata for cross-file drift."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PYTHON = ("3.10", "3.11", "3.12")
DEV_ONLY_DEPENDENCIES = {"build", "pytest", "pytest-cov", "ruff"}


def _dependency_name(specifier: str) -> str:
    return re.split(r"[<>=!~;\s\[]", specifier, maxsplit=1)[0].lower()


def _check(condition: bool, message: str, issues: list[str]) -> None:
    if not condition:
        issues.append(message)


def main() -> int:
    pyproject_path = ROOT / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    ruff = pyproject["tool"]["ruff"]
    hatch = pyproject["tool"]["hatch"]["build"]["targets"]
    extras = project["optional-dependencies"]
    issues: list[str] = []

    classifiers = {
        value.rsplit("::", maxsplit=1)[-1].strip()
        for value in project["classifiers"]
        if value.startswith("Programming Language :: Python :: 3.")
    }
    _check(
        classifiers == set(SUPPORTED_PYTHON),
        "Python classifiers must match SUPPORTED_PYTHON and the CI matrix.",
        issues,
    )
    _check(
        project["requires-python"] == ">=3.10",
        "requires-python must remain >=3.10 while Python 3.10 is supported.",
        issues,
    )
    _check(
        ruff["target-version"] == "py310",
        "ruff target-version must match the minimum supported Python.",
        issues,
    )

    recommended = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    _check(
        recommended in SUPPORTED_PYTHON,
        ".python-version must select a supported interpreter.",
        issues,
    )

    workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    for version in SUPPORTED_PYTHON:
        _check(
            f'"{version}"' in workflow,
            f"CI matrix is missing Python {version}.",
            issues,
        )
    _check(
        'uv sync --python "${{ matrix.python-version }}"' in workflow,
        "CI must pass the matrix interpreter explicitly to uv sync.",
        issues,
    )
    _check(
        'uv run --python "${{ matrix.python-version }}"' in workflow,
        "CI must pass the matrix interpreter explicitly to uv run.",
        issues,
    )

    extra_names = {
        name: {_dependency_name(specifier) for specifier in dependencies}
        for name, dependencies in extras.items()
    }
    _check(
        "torch" in extra_names["research"],
        "The research extra must declare torch explicitly.",
        issues,
    )
    _check(
        not (extra_names["all"] & DEV_ONLY_DEPENDENCIES),
        "The all extra must contain runtime features only, not developer tools.",
        issues,
    )
    _check(
        extra_names["dev"] >= DEV_ONLY_DEPENDENCIES,
        "The dev extra must include build, pytest, pytest-cov, and ruff.",
        issues,
    )

    _check(
        hatch["wheel"]["packages"] == ["pu_toolbox"],
        "The wheel target must explicitly package only pu_toolbox.",
        issues,
    )
    _check(
        "/CONTRIBUTING.md" in hatch["sdist"]["include"],
        "The sdist must include CONTRIBUTING.md.",
        issues,
    )
    _check(
        not (ROOT / "MANIFEST.in").exists(),
        "MANIFEST.in is not authoritative under Hatchling and should not exist.",
        issues,
    )

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    _check("uv.lock" in gitignore, "uv.lock policy must be explicit in .gitignore.", issues)
    requirements_header = (ROOT / "requirements.txt").read_text(encoding="utf-8")[:500]
    _check(
        "authoritative dependency spec is pyproject.toml" in requirements_header,
        "requirements.txt must identify pyproject.toml as authoritative.",
        issues,
    )

    if issues:
        print("Project metadata consistency check failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print(
        "Project metadata consistency check passed: Python matrix, extras, "
        "Hatchling, and dependency policy are aligned."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
