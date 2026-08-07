"""Audit assigned-method paper configurations before expensive execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def runtime_capabilities() -> dict[str, Any]:
    """Collect runtime facts used by the locked official configurations."""
    packages = {}
    for name in ("numpy", "scikit-learn", "torch", "torchvision"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
    except ImportError:
        cuda_available = False
        cuda_device_count = 0
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "commands": {"matlab": shutil.which("matlab")},
    }


def _version_matches(actual: str | None, required: str) -> bool:
    """Compare locked versions while allowing omitted patch components."""
    if actual is None:
        return False
    required_parts = required.split(".")
    return actual.split(".")[: len(required_parts)] == required_parts


def _source_record(config_path: Path, reference: str) -> dict[str, Any]:
    try:
        relative_path, fragment = reference.split("#sources/", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"invalid source_lock reference in {config_path}: {reference!r}") from exc
    lock_path = (config_path.parent / relative_path).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    try:
        return lock["sources"][fragment]
    except KeyError as exc:
        raise ValueError(f"source_lock target is missing for {config_path}: {fragment}") from exc


def _resource_blockers(
    method: str,
    config: dict[str, Any],
    requirements: dict[str, Any],
    source: dict[str, Any],
    *,
    source_roots: dict[str, Path],
    data_roots: dict[str, Path],
) -> list[str]:
    blockers = []
    source_root = source_roots.get(method)
    if requirements.get("source_checkout_required"):
        if source_root is None:
            blockers.append("Official source checkout was not supplied")
        elif not source_root.is_dir():
            blockers.append(f"Official source checkout is not a directory: {source_root}")
        else:
            required_files = requirements.get("required_source_files")
            if required_files is None:
                entrypoint = config.get("runner", {}).get("entrypoint")
                required_files = [entrypoint] if entrypoint else []
            for relative in required_files:
                source_file = source_root / relative
                if not source_file.is_file():
                    blockers.append(f"Official source file is missing: {relative}")
                    continue
                expected_hash = config.get("dataset_sha256", {}).get(source_file.name)
                if expected_hash:
                    actual_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
                    if actual_hash != expected_hash:
                        blockers.append(
                            f"Official bundled dataset hash differs: {relative} "
                            f"(expected {expected_hash}, found {actual_hash})"
                        )
            expected_commit = source.get("commit")
            if expected_commit:
                result = subprocess.run(
                    ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                actual_commit = result.stdout.strip() if result.returncode == 0 else None
                if actual_commit != expected_commit:
                    blockers.append(
                        f"Official source commit differs: expected {expected_commit}, "
                        f"found {actual_commit or 'not a Git checkout'}"
                    )
    if requirements.get("data_root_required"):
        data_root = data_roots.get(method)
        if data_root is None:
            blockers.append("Official dataset root was not supplied")
        elif not data_root.is_dir():
            blockers.append(f"Official dataset root is not a directory: {data_root}")
    return blockers


def _runtime_blockers(requirements: dict[str, Any], capabilities: dict[str, Any]) -> list[str]:
    blockers = []
    if requirements.get("cuda_required") and not capabilities["cuda_available"]:
        blockers.append("No usable CUDA device is available")
    for command in requirements.get("required_commands", []):
        if not capabilities["commands"].get(command):
            blockers.append(f"Required command is unavailable: {command}")
    runtime = requirements.get("historical_runtime", {})
    required_python = runtime.get("python")
    if required_python and not _version_matches(capabilities["python_version"], required_python):
        blockers.append(
            f"Python version differs from lock: required {required_python}, "
            f"found {capabilities['python_version']}"
        )
    for package, required in runtime.get("packages", {}).items():
        actual = capabilities["packages"].get(package)
        if not _version_matches(actual, required):
            blockers.append(
                f"Package version differs from lock: {package} requires {required}, "
                f"found {actual or 'not installed'}"
            )
    return blockers


def audit_locked_configs(
    config_dir: str | Path,
    *,
    source_roots: dict[str, Path] | None = None,
    data_roots: dict[str, Path] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return official-execution and toolbox-replication blockers per method."""
    source_roots = {} if source_roots is None else source_roots
    data_roots = {} if data_roots is None else data_roots
    capabilities = runtime_capabilities() if capabilities is None else capabilities
    reports = []
    seen_methods = set()
    for path in sorted(Path(config_dir).glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("schema_version") != 1 or config.get("protocol") != "paper_like":
            raise ValueError(f"invalid paper-like configuration: {path}")
        method = config.get("method")
        if not isinstance(method, str) or not method or method in seen_methods:
            raise ValueError(f"method must be a unique non-empty string: {path}")
        seen_methods.add(method)
        requirements = config.get("execution_requirements")
        if not isinstance(requirements, dict):
            raise ValueError(f"execution_requirements must be an object: {path}")

        source = _source_record(path, config["source_lock"])
        official_blockers = []
        if source.get("status") != "locked":
            official_blockers.append(
                f"Immutable official source is not locked: status={source.get('status', 'missing')}"
            )
        official_blockers.extend(
            _resource_blockers(
                method,
                config,
                requirements,
                source,
                source_roots=source_roots,
                data_roots=data_roots,
            )
        )
        official_blockers.extend(_runtime_blockers(requirements, capabilities))
        implementation_gaps = list(requirements.get("toolbox_implementation_gaps", []))
        toolbox_blockers = [*official_blockers, *implementation_gaps]
        reports.append(
            {
                "method": method,
                "config": str(path),
                "locked_status": config.get("status"),
                "source_lock_status": source.get("status"),
                "ready_for_official_execution": not official_blockers,
                "ready_for_toolbox_replication": not toolbox_blockers,
                "official_execution_blockers": official_blockers,
                "toolbox_implementation_gaps": implementation_gaps,
                "toolbox_replication_blockers": toolbox_blockers,
            }
        )
    if not reports:
        raise ValueError(f"no paper-like configurations found in {config_dir}")
    unknown_source_methods = sorted(set(source_roots) - seen_methods)
    unknown_data_methods = sorted(set(data_roots) - seen_methods)
    if unknown_source_methods:
        raise ValueError(f"source roots reference unknown methods: {unknown_source_methods}")
    if unknown_data_methods:
        raise ValueError(f"data roots reference unknown methods: {unknown_data_methods}")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "capabilities": capabilities,
        "source_roots": {name: str(path) for name, path in sorted(source_roots.items())},
        "data_roots": {name: str(path) for name, path in sorted(data_roots.items())},
        "all_ready_for_official_execution": bool(reports)
        and all(item["ready_for_official_execution"] for item in reports),
        "all_ready_for_toolbox_replication": bool(reports)
        and all(item["ready_for_toolbox_replication"] for item in reports),
        "methods": reports,
    }


def _parse_method_paths(values: list[str], option: str) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must use METHOD=PATH, got {value!r}")
        method, raw_path = value.split("=", maxsplit=1)
        if not method or not raw_path or method in parsed:
            raise ValueError(f"{option} contains an empty or duplicate method: {value!r}")
        parsed[method] = Path(raw_path).expanduser().resolve()
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--data-root", action="append", default=[], metavar="METHOD=PATH")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        source_roots = _parse_method_paths(args.source_root, "--source-root")
        data_roots = _parse_method_paths(args.data_root, "--data-root")
        report = audit_locked_configs(
            args.config_dir,
            source_roots=source_roots,
            data_roots=data_roots,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
