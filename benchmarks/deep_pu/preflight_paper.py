"""Audit locked paper-like configurations against the current runtime."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def runtime_capabilities(*, edm_backend: str | None = None) -> dict[str, Any]:
    """Collect only capabilities relevant to the locked deep-PU runs."""
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
    except ImportError:
        cuda_available = False
        cuda_device_count = 0
    return {
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "edm_backend": edm_backend,
    }


def audit_locked_configs(
    config_dir: str | Path,
    *,
    accepted_datasets: set[str] | None = None,
    edm_backend: str | None = None,
) -> dict[str, Any]:
    """Return method-level blockers without attempting an expensive run."""
    accepted = set() if accepted_datasets is None else accepted_datasets
    capabilities = runtime_capabilities(edm_backend=edm_backend)
    reports = []
    for path in sorted(Path(config_dir).glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        requirements = config.get("execution_requirements", {})
        blockers = list(requirements.get("implementation_gaps", []))
        if requirements.get("cuda_required") and not capabilities["cuda_available"]:
            blockers.append("No usable CUDA device is available")
        if "conditional_edm" in requirements.get("external_backends", []) and not edm_backend:
            blockers.append("No conditional EDM backend was supplied")
        for dataset in requirements.get("restricted_datasets", []):
            if dataset not in accepted:
                blockers.append(f"Dataset access has not been confirmed: {dataset}")
        reports.append(
            {
                "method": config.get("method"),
                "config": str(path),
                "locked_status": config.get("status"),
                "ready_for_full_run": not blockers,
                "blockers": blockers,
            }
        )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "capabilities": capabilities,
        "accepted_restricted_datasets": sorted(accepted),
        "all_ready": bool(reports) and all(item["ready_for_full_run"] for item in reports),
        "methods": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edm-backend")
    parser.add_argument("--accept-dataset", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_locked_configs(
        args.config_dir,
        accepted_datasets=set(args.accept_dataset),
        edm_backend=args.edm_backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
